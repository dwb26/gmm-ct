"""GMM-CT model: forward physics and 4-stage reconstruction pipeline.

Pipeline stages inside GMM_reco.fit():
  1. _stage_trajectory_optimization   – multi-start L-BFGS on peak heights
  2. _stage_omega_initialization       – residual-sinogram grid search
  3. _stage_alpha_initialization       – NNLS for attenuation coefficients
  4. _stage_multistart_joint           – multi-start L-BFGS on full projections
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchmin import minimize
from tqdm.auto import tqdm

from .utils import NewtonRaphsonLBFGS
from .structures import PeakData

logger = logging.getLogger(__name__)


class GMM_reco:
    """Reconstruct GMM parameters from CT projection data.

    Parameters
    ----------
    d : Spatial dimensionality (2 for 2-D problems).
    N : Number of Gaussian components.
    sources : X-ray source positions.
    receivers : Receiver positions, one list per source.
    x0s : Known initial positions for each Gaussian.
    a0s : Known accelerations for each Gaussian.
    omega_min, omega_max : Angular velocity search bounds (Hz).
    device : Computation device (auto-detected when None).
    output_dir : Directory for diagnostic plots (default: ``'data/results/'``).
    N_traj_trials : Multi-start trials for Stage 1 (default: max(20, 2·N)).
    N_omega_inits : Multi-start trials for Stage 2 (default: 5).
    save_diagnostics : Save diagnostic plots at the end of Stage 1 (default: True).
    """
    def __init__(
        self, 
        d: int, 
        N: int, 
        sources: list[torch.Tensor], 
        receivers: list[list[torch.Tensor]],
        x0s: list[torch.Tensor], 
        a0s: list[torch.Tensor],
        omega_min: float, 
        omega_max: float, 
        device: torch.device | None = None, 
        output_dir: str | None = None,
        N_traj_trials: int | None = None, 
        N_omega_inits: int | None = None,
        save_diagnostics: bool = True,
    ):
        self.d = d
        self.N = N
        self.x0s = x0s
        self.a0s = a0s
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.N_traj_trials = N_traj_trials
        self.N_omega_inits = N_omega_inits
        self.save_diagnostics = save_diagnostics
        self.t_observable = []

        # Device
        self.device = (
            torch.device(device) if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.output_dir = Path(output_dir) if output_dir else Path('data/results')
        if self.save_diagnostics:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Precomputed constant
        self.sqrt_pi = math.sqrt(math.pi)

        # Move geometry to device
        self.sources = [
            s.to(self.device, dtype=torch.float64) if isinstance(s, torch.Tensor)
            else torch.tensor(s, dtype=torch.float64, device=self.device)
            for s in sources
        ]
        self.receivers = [
            [
                r.to(self.device, dtype=torch.float64) if isinstance(r, torch.Tensor)
                else torch.tensor(r, dtype=torch.float64, device=self.device)
                for r in rec_list
            ]
            for rec_list in receivers
        ]
        self.n_sources = len(self.sources)
        self.n_rcvrs = len(self.receivers[0])

    @classmethod
    def from_config(cls, cfg):
        """Instantiate GMM_reco from ReconstructConfig."""
        device = torch.device(
            cfg.device if cfg.device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        sources, receivers = cfg.geometry.to_tensors(device)
        x0s, a0s = cfg.physics.to_tensors(cfg.n_gaussians, device)
        omega_min, omega_max = cfg.physics.omega_range

        return cls(
            d=cfg.geometry.dimensionality,
            N=cfg.n_gaussians,
            sources=sources,
            receivers=receivers,
            x0s=x0s,
            a0s=a0s,
            omega_min=omega_min,
            omega_max=omega_max,
            device=device,
            output_dir=cfg.output.directory,
            N_traj_trials=cfg.reconstruction.N_trajectory_trials,
            N_omega_inits=cfg.reconstruction.N_omega_inits,
            save_diagnostics=cfg.output.save_plots,
        )

    # ==================================================================
    # Pipeline Entry Point
    # ==================================================================

    def fit(
        self, 
        proj_data: list[torch.Tensor], 
        t: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        """Execute full 4-stage optimization pipeline."""
        self.t = t.to(self.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device)
        self.proj_data = self.process_projections(self._to_device(proj_data))

        stage_bar = tqdm(
            total=4, desc="GMM-CT fit", unit="stage", leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]"
        )

        # Stage 1: Trajectory Optimization
        stage_bar.set_description("GMM-CT  [1/4 trajectory]")
        soln_dict, best_res = self._stage_trajectory_optimization(t, proj_data)
        self.theta_pre_stage1_5 = self._clone_dict(soln_dict)
        stage_bar.update(1)

        # Stage 1.5a: Grid Search for Angular Velocities (ω)
        stage_bar.set_description("GMM-CT  [2/4 ω search]")
        soln_dict = self._stage_omega_initialization(soln_dict)
        stage_bar.update(1)

        # Stage 1.5b: NNLS for Amplitudes (α)
        stage_bar.set_description("GMM-CT  [3/4 NNLS α]")
        soln_dict = self._stage_alpha_initialization(soln_dict)
        self.theta_pre_stage2 = self._clone_dict(soln_dict)
        stage_bar.update(1)

        # Stage 2: Multi-start Joint Refinement
        stage_bar.set_description("GMM-CT  [4/4 joint opt.]")
        soln_dict = self._stage_multistart_joint(soln_dict, warm_start=True)
        stage_bar.update(1)

        stage_bar.set_description("GMM-CT  [done]")
        stage_bar.close()

        return soln_dict

    # ==================================================================
    # Forward model
    # ==================================================================
    
    def generate_projections(
        self, 
        t: torch.Tensor, 
        theta_dict: dict[str, list[torch.Tensor]], 
        loss_type: str | None = None
    ) -> list[torch.Tensor]:
        """Vectorized forward calculation of X-ray projection series."""
        if loss_type is not None:
            theta_dict = {**self.theta_fixed, **theta_dict}

        rot_mats = self._compute_rotation_matrices(t, theta_dict)  # [N, T, d, d]
        trajs = self._compute_trajectories(t, theta_dict)          # [N, T, d]

        projs = [
            torch.zeros(len(t), self.n_rcvrs, dtype=torch.float64, device=self.device)
            for _ in range(self.n_sources)
        ]
        EPS = 1e-10

        for n_s, source in enumerate(self.sources):
            receivers = torch.stack(self.receivers[n_s]) # [R, d]
            r_minus_s = receivers - source               # [R, d]
            r_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True) # [R, d]

            for n in range(self.N):
                alpha_n = theta_dict["alphas"][n].squeeze()
                U_n = theta_dict["U_skews"][n]           # [d, d]
                
                # Batch transform shape matrix: U_n_t = U_n @ R_n(t)^T
                U_n_t = torch.matmul(U_n, rot_mats[n].transpose(-1, -2)) # [T, d, d]

                # Project ray directions & trajectory offset
                # U_r_hat: [T, R, d], U_r: [T, R, d], U_traj: [T, 1, d]
                U_r_hat = torch.matmul(r_hat, U_n_t.transpose(-1, -2))
                U_r = torch.matmul(r_minus_s, U_n_t.transpose(-1, -2))
                s_minus_mu = source - trajs[n] # [T, d]
                U_traj = torch.matmul(s_minus_mu.unsqueeze(1), U_n_t.transpose(-1, -2))

                norm_r_hat = torch.norm(U_r_hat, dim=-1) # [T, R]
                quotient = (self.sqrt_pi * alpha_n) / (norm_r_hat + EPS)

                inner_prod_sq = torch.sum(U_r * U_traj, dim=-1)**2 # [T, R]
                divisor = torch.sum(U_r**2, dim=-1) + EPS
                subtractor = torch.sum(U_traj**2, dim=-1)

                exp_arg = (inner_prod_sq / divisor) - subtractor
                projs[n_s] += quotient * torch.exp(exp_arg)

        return projs
    
    def _compute_rotation_matrices(
        self, 
        t: torch.Tensor,
        theta: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute rotation matrix stacks for all Gaussians across time vector t."""
        T = len(t) if t.dim() > 0 else 1
        t_vec = t.reshape(-1)
        two_pi = 2.0 * math.pi
        
        rot_stack = []
        for n in range(self.N):
            omega_n = theta["omegas"][n]
            R_n = torch.eye(self.d, dtype=torch.float64, device=self.device).repeat(T, 1, 1)
            
            for idx, omega in enumerate(omega_n):
                i, j = torch.combinations(torch.arange(self.d, device=self.device), r=2)[idx]
                angles = two_pi * omega * t_vec
                cos_a, sin_a = torch.cos(angles), torch.sin(angles)
                
                R_plane = torch.eye(self.d, dtype=torch.float64, device=self.device).repeat(T, 1, 1)
                R_plane[:, i, i] = cos_a
                R_plane[:, i, j] = -sin_a
                R_plane[:, j, i] = sin_a
                R_plane[:, j, j] = cos_a
                
                R_n = torch.bmm(R_n, R_plane)
            rot_stack.append(R_n)
            
        return torch.stack(rot_stack)   # [N, T, d, d]
        
    
    def _compute_trajectories(
        self,
        t: torch.Tensor,
        theta: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        """Compute position vectors for all Gaussians across time vector t."""
        t_vec = t.unsqueeze(-1) if t.dim() > 0 else t   # [T, 1]
        trajs = []
        for n in range(self.N):
            x0, v0, a0 = theta["x0s"][n], theta["v0s"][n], theta["a0s"][n]
            trajs.append(x0 + v0 * t_vec + 0.5 * a0 * (t_vec**2))
        return torch.stack(trajs)   # [N, T, d]
        
    def process_projections(
        self, 
        projections: list[torch.Tensor]
    ) -> torch.Tensor:
        """Flatten multi-source projection lists to a unified 2D tensor."""
        return projections[0] if self.n_sources == 1 else torch.cat(projections, dim=0)



    # ==================================================================
    # Stage 1 – trajectory optimization
    # ==================================================================

    def _stage_trajectory_optimization(
        self, 
        t: torch.Tensor, 
        proj_data: list[torch.Tensor],
    ) -> dict[str, list[torch.Tensor]]:
        """Multi-start L-BFGS to estimate initial velocities v0."""
        logger.info("Stage 1: Trajectory optimization")

        N_traj_trials = self.N_traj_trials or max(20, 2 * self.N)
        logger.info("Running %d trajectory multi-start trials", N_traj_trials)

        errors, results = [], []
        trial_bar = tqdm(range(N_traj_trials), desc='  trials', unit='trial', leave=False)
        for n_trial in trial_bar:
            logger.info("Trial %d/%d", n_trial + 1, N_traj_trials)
            self.theta_dict_init = self.initialize_parameters(t, proj_data)
            [v0_n.requires_grad_(True) for v0_n in self.theta_dict_init['v0s']]

            theta_tensor_init = self.map_from_dict_to_tensor(self.theta_dict_init, mode='trajectory')
            res_trial = minimize(
                self._loss_trajectory, 
                x0=theta_tensor_init, 
                method='l-bfgs',
                tol=1e-8, 
                options={'gtol': 1e-8, 'max_iter': 1500, 'disp': False},
            )
            errors.append(res_trial.fun)
            results.append(res_trial)
            trial_bar.set_postfix({'best': f'{min(errors):.3e}'})

        best_res = results[np.argmin(np.array(errors))]
        soln_dict = self.construct_soln_dict(best_res)

        if 'v0s' not in soln_dict:
            raise RuntimeError(
                f"Trajectory optimization failed — no v0s in result. "
                f"Got keys: {list(soln_dict.keys())}"
            )

        soln_dict["v0s"] = [v0_n.clone().detach() for v0_n in soln_dict["v0s"]]

        # Optional diagnostic plots
        # if self.save_diagnostics:
        #     from .visualization.diagnostics import (
        #         plot_trajectory_estimations,
        #         plot_heights_by_assignment,
        #         plot_raw_receiver_heights,
        #         plot_assignment_quality,
        #         plot_gmm_and_projections,
        #         plot_trajectory_fitting,
        #     )
        #     plot_trajectory_estimations(self, best_res)
        #     plot_raw_receiver_heights(self)
        #     plot_heights_by_assignment(self)
        #     plot_assignment_quality(self, best_res)
        #     plot_gmm_and_projections(self, best_res, theta_true=getattr(self, 'theta_true', None))
        #     plot_trajectory_fitting(self, best_res)

        soln_dict = self.refine_initial_velocities_via_newton_raphson(soln_dict, best_res)

        soln_dict["omegas"] = [omega.clone().detach() for omega in self.theta_dict_init["omegas"]]
        soln_dict["alphas"] = [alpha.clone().detach() for alpha in self.theta_dict_init["alphas"]]
        soln_dict["U_skews"] = self.initialize_anisotropic_U_skews(soln_dict["v0s"])

        return soln_dict
    
    
    # ==================================================================
    # Initialization Routines
    # ==================================================================
    def initialize_parameters(
        self, 
        t: torch.Tensor, 
        proj_data: list[torch.Tensor],
    ) -> None:
        """Initialize all GMM parameters before Stage 1 optimization."""
        v0s = self.initialize_initial_velocities(t, proj_data)
        return {
            "alphas": [torch.tensor([12.5], dtype=torch.float64, device=self.device) for _ in range(self.N)],
            'omegas': [torch.zeros(size=(1,), dtype=torch.float64, device=self.device) for _ in range(self.N)],
            'U_skews': self.initialize_anisotropic_U_skews(v0s),
            'x0s': self.x0s,
            'v0s': v0s,
            'a0s': self.a0s,
        }

    def initialize_initial_velocities(
        self, 
        t: torch.Tensor, 
        proj_data: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Detect projection peaks and create random v0 starting points."""
        self.peak_data = PeakData(self.N, self.device)
        proj_data_array = proj_data[0] if isinstance(proj_data, list) else proj_data
        
        self._detect_all_peaks(proj_data_array, self.receivers[0], t)
        self.peak_data.finalize_detections()
        self._create_legacy_aliases()

        # Sample v0 ~ N([1, 1], 1.5²·I) for each Gaussian
        v0s = []
        for _ in range(self.N):
            v0 = torch.tensor([1.0, 1.0], dtype=torch.float64, device=self.device)
            v0 = v0 + 1.5 * torch.randn(2, dtype=torch.float64, device=self.device)
            v0.requires_grad_(True)
            v0s.append(v0)
            
        return v0s
    
    def _detect_all_peaks(
        self, 
        proj_data: list[torch.Tensor], 
        receivers: list[list[torch.Tensor]], 
        t: torch.Tensor
    ) -> None:
        """Detect peaks across all time steps via 3-point sliding window."""
        for time_idx, time_val in enumerate(t):
            detected_heights = []
            gaussian_idx = 0
            projection = proj_data[time_idx]
            
            # Two-dimensional, noiseless peak detection method
            for offset in range(self.n_rcvrs - 2):
                idx_center = self.n_rcvrs - 2 - offset
                if projection[idx_center + 1] < projection[idx_center] > projection[idx_center - 1]:
                    receiver_pos = receivers[idx_center]
                    self.peak_data.add_peak_detection(
                        time_idx=time_idx,
                        time_val=time_val,
                        receiver_idx=idx_center,
                        receiver_pos=receiver_pos,
                        peak_val=projection[idx_center],
                        gaussian_idx=gaussian_idx,
                    )
                    detected_heights.append(receiver_pos[1])
                    gaussian_idx += 1
                    if gaussian_idx >= self.N:
                        break
                    
            self.peak_data.add_time_detections(time_val.item(), detected_heights)
    
    # ==================================================================
    # Trajectory Traversal & Hungarian Loss
    # ==================================================================
    
    def _loss_trajectory(self, theta_tensor: torch.Tensor) -> torch.Tensor:
        """Stage 1 loss: L1 distance between predicted and observed peak heights.

        Uses the Hungarian algorithm for optimal peak-to-Gaussian assignment.
        """
        theta_dict = self.map_from_tensor_to_dict(theta_tensor, mode='trajectory')
        self.t_observable = self.t[self.peak_data.observable_indices]
        
        r_maxs_list = self.map_velocities_to_maximising_receivers(theta_dict)
        self._assign_peaks_hungarian(r_maxs_list)
        return self._compute_trajectory_loss(r_maxs_list)
    
    def map_velocities_to_maximising_receivers(
        self, 
        theta_dict: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Map v0 parameters to predicted ray-intersection receiver coordinates."""
        r_maxs_list = []
        s = self.sources[0]
        r0_x = self.receivers[0][0][0]
        EPS = 1e-10
            
        for n in range(self.N):
            v0_n = theta_dict['v0s'][n]
            x0_n, a0_n = self.theta_fixed['x0s'][n], self.theta_fixed['a0s'][n]
                
            # Vectorized center trajectories over observable times
            t_obs = self.t_observable.unsqueeze(-1)                 # [T_obs, 1]
            c_n = x0_n + v0_n * t_obs + 0.5 * a0_n * (t_obs**2)     # [T_obs, d]
            
            denom = s[0] - c_n[:, 0]
            denom_safe = torch.where(
                torch.abs(denom) < EPS,
                torch.sign(denom) * EPS + (denom == 0).float() * EPS,
                denom,
            )
            lambda_t = (r0_x - s[0]) / denom_safe               # [T_obs]
            r_maxs_n = s + lambda_t.unsqueeze(-1) * (s - c_n)   # [T_obs, d]
            r_maxs_list.append(torch.stack(r_maxs_n))

        return r_maxs_list
    
    def _assign_peaks_hungarian(self, r_maxs_list):
        """Assign detected peaks to predicted trajectories via the Hungarian algorithm."""
        self.assigned_curve_data = [[] for _ in range(self.N)]
        heights_dict = self.peak_data.get_heights_dict_non_empty()

        for time_idx, time_val in enumerate(self.t_observable):
            observed_heights = heights_dict.get(time_val.item(), [])
            if not observed_heights:
                continue
            
            # Vectorized cost matrix construction
            obs_tensor = torch.tensor(observed_heights, dtype=torch.float64, device=self.device).unsqueeze(1)   # [H, 1]
            pred_tensor = torch.stack([r_maxs_list[g][time_idx, 1] for g in range(self.N)]).unsqueeze(0)        # [1, N]
            
            dist_matrix = torch.abs(obs_tensor - pred_tensor)
            dist_matrix = torch.where(torch.isnan(dist_matrix) | torch.isinf(dist_matrix), 1e10, dist_matrix)

            row_indices, col_indices = linear_sum_assignment(dist_matrix.cpu().numpy())
            for h_idx, g_idx in zip(row_indices, col_indices):
                self.assigned_curve_data[g_idx].append((time_idx, observed_heights[h_idx]))
    
    def _compute_trajectory_loss(self, r_maxs_list):
        """Compute L1 loss between predicted and assigned receiver heights."""
        loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        for k in range(self.N):
            assignments_k = self.assigned_curve_data[k]
            if not assignments_k:
                continue
            time_indices = [item[0] for item in assignments_k]
            observed_heights = torch.stack([item[1] for item in assignments_k])
            predicted_heights = r_maxs_list[k][time_indices, 1]
            loss += torch.norm(predicted_heights - observed_heights, p=1)
        return loss
    
    # ==================================================================
    # Velocity Refinement
    # ==================================================================

    def refine_initial_velocities_via_newton_raphson(
        self, 
        soln_dict: list[torch.Tensor], 
        res: dict,
    ) -> dict[str, list[torch.Tensor]]:
        """Refine v0 via Newton-Raphson root finding."""
        r_maxs_list = self.map_velocities_to_maximising_receivers(self.map_from_tensor_to_dict(res.x))
        self._assign_peaks_to_trajectories(r_maxs_list)

        # Build format expected by diagnostic plots
        self.assigned_curve_data = [
            [
                (torch.where(self.t_observable == time_val)[0][0].item(), torch.tensor(height, device=self.device))
                for time_val, height in zip(*self.peak_data.get_assignment_data(g))
                if len(torch.where(self.t_observable == time_val)[0]) > 0
            ]
            for g in range(self.N)
        ]
        self.assigned_peak_values = self.peak_data.assigned_values

        soln_dict["v0s"] = [v0.clone().detach() for v0 in self._newton_raphson_refinement(soln_dict)]
        return soln_dict

    def _assign_peaks_to_trajectories(self, r_maxs_list: list[torch.Tensor]) -> None:
        """Assign peaks to trajectories via nearest-neighbour matching."""
        for time_idx, detected_heights in enumerate(self.peak_data.get_heights_sorted_by_time()):
            for height in detected_heights:
                distances = [torch.abs(trajectory[time_idx, 1] - height).item() for trajectory in r_maxs_list]
                gaussian_idx = np.argmin(distances)

                receiver_heights = torch.tensor([r[1].item() for r in self.receivers[0]], dtype=torch.float64, device=self.device)
                receiver_idx = int(torch.argmin(torch.abs(receiver_heights - height)).item())

                self.peak_data.add_optimal_assignment(
                    gaussian_idx,
                    self.t_observable[time_idx].item(),
                    height,
                    self.proj_data[time_idx, receiver_idx].item(),
                )

    def _newton_raphson_refinement(self, soln_dict: dict[str, list[torch.Tensor]]) -> list[torch.Tensor]:
        """Refine v0_n for n = 1, ..., N via Newton-Raphson on the optimal peak assignments."""
        v0s_refined = []
        r0_x = self.receivers[0][0][0]

        for gaussian_idx in range(self.N):
            times, heights = self.peak_data.get_assignment_data(gaussian_idx)
            t_obs = torch.tensor(times, dtype=torch.float64, device=self.device)
            receivers_n = [torch.tensor([r0_x, h], dtype=torch.float64, device=self.device) for h in heights]
            
            v0_n_refined = NewtonRaphsonLBFGS(
                self.isotropic_derivative_function_over_all_times,
                soln_dict['v0s'][gaussian_idx],
                t_obs, receivers_n, self.sources[0],
                soln_dict['x0s'][gaussian_idx],
                soln_dict['a0s'][gaussian_idx],
            )
            v0s_refined.append(v0_n_refined.requires_grad_(True))

        return v0s_refined
    
    # ==================================================================
    # Internal Helpers & Placeholders
    # ==================================================================
    
    def _create_legacy_aliases(self):
        """Set model attributes expected by diagnostic plotting functions."""
        self.t_obs_by_cluster = self.peak_data.times
        self.maximising_rcvrs = self.peak_data.receiver_positions
        self.maximising_inds = self.peak_data.receiver_indices
        self.peak_values = self.peak_data.peak_values
        self.observable_indices = self.peak_data.observable_indices



    # ==================================================================
    # Stage 1.5 – omega grid search
    # ==================================================================

    def _stage_omega_initialization(self, soln_dict):
        """Per-Gaussian omega estimation via residual-sinogram grid search.

        For each Gaussian k, subtracts all other Gaussians' contributions from
        the observed sinogram, then sweeps a uniform grid of omega candidates
        and keeps the one that minimises the residual norm.
        """
        n_planes = math.comb(self.d, 2)
        n_grid = 200

        logger.info(
            "Stage 1.5: Residual-sinogram ω grid search (%d plane(s), %d candidates)",
            n_planes, n_grid,
        )

        theta_true = getattr(self, 'theta_true', None)
        if theta_true is not None and 'omegas' in theta_true:
            for k, omega_true_k in enumerate(theta_true['omegas']):
                logger.debug("  Gaussian %d: ω_true = [%s] Hz", k,
                             ', '.join(f'{w.item():.4f}' for w in omega_true_k.flatten()))

        proj_obs = self.proj_data
        t = self.t
        omega_candidates = torch.linspace(
            self.omega_min, self.omega_max, n_grid,
            dtype=torch.float64, device=self.device,
        )

        gaussian_bar = tqdm(
            range(self.N),
            desc='  Gaussians',
            unit='ρ',
            leave=False,
        )
        for k in gaussian_bar:
            gaussian_bar.set_description(f'  ρ{k + 1}/{self.N}')
            # Residual sinogram: observed minus all other Gaussians
            bg_dict = {key: list(vals) for key, vals in soln_dict.items()}
            bg_dict['alphas'] = [
                torch.zeros(1, dtype=torch.float64, device=self.device) if j == k
                else soln_dict['alphas'][j].clone()
                for j in range(self.N)
            ]
            with torch.no_grad():
                proj_resid = proj_obs - self.process_projections(
                    self.generate_projections(t, bg_dict)
                )

            omega_k = soln_dict['omegas'][k].clone()

            for plane_idx in range(n_planes):
                best_loss = float('inf')
                best_val = omega_k[plane_idx].clone()

                for omega_val in omega_candidates:
                    test_omega_k = omega_k.clone()
                    test_omega_k[plane_idx] = omega_val

                    test_dict = {key: list(vals) for key, vals in soln_dict.items()}
                    test_dict['omegas'] = [
                        test_omega_k if j == k else soln_dict['omegas'][j].clone()
                        for j in range(self.N)
                    ]
                    test_dict['alphas'] = [
                        soln_dict['alphas'][k].clone() if j == k
                        else torch.zeros(1, dtype=torch.float64, device=self.device)
                        for j in range(self.N)
                    ]

                    with torch.no_grad():
                        proj_k = self.process_projections(
                            self.generate_projections(t, test_dict)
                        )

                    loss = torch.norm(proj_resid - proj_k).item()
                    if loss < best_loss:
                        best_loss = loss
                        best_val = omega_val.clone()

                omega_k[plane_idx] = best_val

            soln_dict['omegas'][k] = omega_k

            omega_str = ', '.join(f'{w.item():.4f}' for w in soln_dict['omegas'][k])
            if theta_true is not None and 'omegas' in theta_true:
                omega_true_k = theta_true['omegas'][k]
                logger.info("  Gaussian %d: ω_est = [%s] Hz | ω_true = [%s] Hz", k,
                            omega_str,
                            ', '.join(f'{w.item():.4f}' for w in omega_true_k.flatten()))
            else:
                logger.info("  Gaussian %d: ω = [%s] Hz", k, omega_str)

        return soln_dict

    # ==================================================================
    # Stage 1.5b – alpha NNLS
    # ==================================================================

    def _stage_alpha_initialization(self, soln_dict):
        """Initialise attenuation coefficients via non-negative least squares.

        With trajectories, shapes and omegas fixed, the forward model is linear
        in alphas.  Solves ``min_{α≥0} ‖Φα − p_obs‖₂²`` in closed form.
        """
        logger.info("Stage 1.5b: NNLS alpha initialisation")

        t_obs = self.t[self.peak_data.observable_indices]
        p_obs = self.proj_data[self.peak_data.observable_indices]
        T_obs, R = p_obs.shape

        Phi = torch.zeros(T_obs * R, self.N, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for k in range(self.N):
                unit_dict = {
                    'alphas': [
                        torch.ones(1, dtype=torch.float64, device=self.device) if kk == k
                        else torch.zeros(1, dtype=torch.float64, device=self.device)
                        for kk in range(self.N)
                    ],
                    'U_skews': soln_dict['U_skews'],
                    'omegas': soln_dict['omegas'],
                    'x0s': soln_dict['x0s'],
                    'v0s': soln_dict['v0s'],
                    'a0s': soln_dict['a0s'],
                }
                proj_k = self.generate_projections(t_obs, unit_dict)
                Phi[:, k] = self.process_projections(proj_k).reshape(-1)

        if not torch.isfinite(Phi).all():
            logger.warning("Non-finite values in basis matrix; skipping alpha init.")
            return soln_dict

        p_vec_t = p_obs.reshape(-1, 1)
        result = torch.linalg.lstsq(Phi, p_vec_t, driver='gelsd')
        alpha_hat = result.solution.squeeze(1).clamp(min=0.0)
        residual = torch.norm(Phi @ alpha_hat - p_obs.reshape(-1)).item()

        soln_dict['alphas'] = [
            alpha_hat[k].reshape(1).detach().clone() for k in range(self.N)
        ]

        theta_true = getattr(self, 'theta_true', None)
        if theta_true is not None and 'alphas' in theta_true:
            logger.info("  α_est  = [%s]", ', '.join(f'{alpha_hat[k].item():.3f}' for k in range(self.N)))
            logger.info("  α_true = [%s]", ', '.join(f'{theta_true["alphas"][k].item():.3f}' for k in range(self.N)))
        else:
            logger.info("  α = %s", [f'{alpha_hat[k].item():.3f}' for k in range(self.N)])
        logger.info("  NNLS residual ‖Φα − p_obs‖₂ = %.4e", residual)

        return soln_dict

    # ==================================================================
    # Stage 2 – multi-start joint optimization
    # ==================================================================

    def _stage_multistart_joint(self, soln_dict, warm_start=False):
        """Multi-start L-BFGS on full projections (Huber loss) to refine α, U, ω.

        Trial 0 uses the omegas from ``soln_dict`` when ``warm_start=True``;
        subsequent trials draw random omega candidates.
        """
        logger.info("Stage 2: Multi-start joint optimization")
        n_trials = self.N_omega_inits or 5
        logger.info("Running %d trials", n_trials)

        initial_alphas = [a.clone().detach() for a in soln_dict['alphas']]
        initial_U_skews = [U.clone().detach() for U in soln_dict['U_skews']]
        omega_min = self.omega_min - 0.01
        omega_max = self.omega_max + 0.01

        all_losses, all_results = [], []

        joint_bar = tqdm(
            range(n_trials),
            desc='  trials',
            unit='trial',
            leave=False,
        )
        for trial_idx in joint_bar:
            if warm_start and trial_idx == 0:
                initial_omegas = [omega.clone().detach() for omega in soln_dict['omegas']]
            else:
                initial_omegas = [
                    torch.tensor(
                        [np.random.uniform(omega_min, omega_max)],
                        dtype=torch.float64, device=self.device,
                    )
                    for _ in range(self.N)
                ]

            test_dict = {
                'alphas': [a.clone().requires_grad_(True) for a in initial_alphas],
                'U_skews': [U.clone().requires_grad_(True) for U in initial_U_skews],
                'omegas': [omega.requires_grad_(True) for omega in initial_omegas],
                'x0s': soln_dict['x0s'],
                'v0s': soln_dict['v0s'],
                'a0s': soln_dict['a0s'],
            }
            self.theta_fixed = {
                'x0s': [x0.clone() for x0 in soln_dict['x0s']],
                'v0s': [v0.clone() for v0 in soln_dict['v0s']],
                'a0s': [a0.clone() for a0 in soln_dict['a0s']],
            }

            theta_tensor = self.map_from_dict_to_tensor(test_dict, mode='joint')
            res = minimize(
                self._loss_joint, x0=theta_tensor, method='l-bfgs',
                tol=1e-10, options={'gtol': 1e-10, 'max_iter': 1000, 'disp': False},
            )

            result_dict = self.construct_soln_dict(res)
            final_loss = res.fun.item()
            all_losses.append(final_loss)
            all_results.append(result_dict)
            joint_bar.set_postfix({'best': f'{min(all_losses):.3e}'})

            logger.info("  Trial %d/%d: loss = %.6e, ω = %s",
                        trial_idx + 1, n_trials, final_loss,
                        [f'{omega.item():.3f}' for omega in result_dict['omegas']])

        best_trial_idx = int(np.argmin(all_losses))
        best_result = all_results[best_trial_idx]
        best_loss = all_losses[best_trial_idx]

        soln_dict['alphas'] = [alpha.clone().detach() for alpha in best_result['alphas']]
        soln_dict['omegas'] = [omega.clone().detach() for omega in best_result['omegas']]
        soln_dict['U_skews'] = [U.clone().detach() for U in best_result['U_skews']]

        logger.info("Multi-start complete — best trial: %d, loss: %.6e",
                    best_trial_idx + 1, best_loss)
        logger.info("Best ω: %s", [f'{omega.item():.4f}' for omega in soln_dict['omegas']])

        return soln_dict


    def _generate_peak_pattern_for_omega(self, alpha, U_skew, omega, x0, v0, a0, times, gaussian_idx):
        """Generate predicted projection peaks for a single Gaussian at a given omega."""
        device = self.device
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
        source = self.sources[0]
        receiver_line = self.receivers[0]
        peak_values = []

        for t_n in times:
            mu_t = x0 + v0 * t_n + 0.5 * a0 * t_n ** 2
            angle = 2 * torch.pi * omega * t_n
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            R_t = torch.stack([torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])])
            U_rot = U_skew @ R_t.T

            projections = []
            for receiver in receiver_line:
                r_minus_s = receiver - source
                r_minus_s_hat = r_minus_s / torch.norm(r_minus_s)
                U_r_hat = U_rot @ r_minus_s_hat
                U_r = U_rot @ r_minus_s
                U_mu = U_rot @ (source - mu_t)

                norm_term = torch.norm(U_r_hat)
                quotient = sqrt_pi * alpha / (norm_term + 1e-10)
                inner_prod_sq = torch.dot(U_r.squeeze(), U_mu) ** 2
                exp_arg = inner_prod_sq / (torch.norm(U_r) ** 2 + 1e-10) - torch.norm(U_mu) ** 2
                projections.append(quotient * torch.exp(exp_arg))

            peak_values.append(torch.max(torch.stack(projections)))

        return torch.stack(peak_values)


    def initialize_anisotropic_U_skews(self, v0s, eps=1.0):
        """Initialise U_skew as diag(30, 15) + small upper-triangular noise.

        The 4:1 aspect ratio ensures Gaussians have a detectable rotation
        signature in the projections.  Noise on the off-diagonal helps the
        optimizer recover the true off-diagonal shape.
        """
        diag_vals = torch.tensor([30.0, 15.0], dtype=torch.float64, device=self.device)
        U_skews = []
        for _ in range(self.N):
            U_k = torch.diag(diag_vals).clone()
            if eps > 0:
                rows, cols = torch.triu_indices(self.d, self.d, offset=1, device=self.device)
                noise = eps * torch.randn(len(rows), dtype=torch.float64, device=self.device)
                U_k[rows, cols] = U_k[rows, cols] + noise
            U_skews.append(U_k)
        return U_skews

    def _loss_joint(self, theta_tensor):
        """Stage 2 loss: Huber loss between simulated and observed projections."""
        loss_func = nn.HuberLoss(delta=0.3)
        has_v0_fixed = 'v0s' in self.theta_fixed
        mode = 'joint' if has_v0_fixed else 'joint_with_v0'

        theta_dict = self.map_from_tensor_to_dict(theta_tensor, mode=mode)
        for key, value in self.theta_fixed.items():
            if key not in theta_dict:
                theta_dict[key] = value

        sim_projs = self.generate_projections(self.t_observable, theta_dict)
        sim_projs_processed = self.process_projections(sim_projs)
        proj_data_observable = self.proj_data[self.peak_data.observable_indices]

        return loss_func(proj_data_observable, sim_projs_processed)

    def _sup_projection_error(self, result_dict):
        """Compute ``max_{t,r} |sim(t,r) − obs(t,r)|`` at observable time points."""
        with torch.no_grad():
            sim_projs = self.generate_projections(self.t_observable, result_dict)
            sim_proc = self.process_projections(sim_projs)
            obs_proc = self.proj_data[self.peak_data.observable_indices]
            return torch.max(torch.abs(sim_proc - obs_proc)).item()


    def isotropic_derivative_function(self, v0, *args):
        """Isotropic projection derivative used as the root-finding objective."""
        t_n, r, s, x0, a0 = args
        r1, r2 = r[0], r[1]
        s1, s2 = s[0], s[1]
        d1, d2 = r1 - s1, r2 - s2
        norm_n_sq = d1 ** 2 + d2 ** 2
        c_n = s - x0 - v0 * t_n - 0.5 * a0 * t_n ** 2
        h_k = d1 * c_n[0] - s2 * c_n[1]
        R_k_l = 2 * norm_n_sq * c_n[1] * (c_n[1] * r2 + h_k)
        R_k_r = -2 * d2 * (c_n[1] * r2 + h_k) ** 2
        return (R_k_l + R_k_r) / norm_n_sq ** 2

    def isotropic_derivative_function_over_all_times(self, v0, *args):
        """Sum of |isotropic derivatives| across all time points."""
        t, r, s, x0, a0 = args
        R_all = torch.zeros(1, dtype=torch.float64, device=self.device)
        for n, t_n in enumerate(t):
            R_all += torch.abs(self.isotropic_derivative_function(v0, t_n, r[n], s, x0, a0))
        return R_all


    # ==================================================================
    # Parameter serialization (dict ↔ flat tensor for L-BFGS)
    # ==================================================================

    def map_from_dict_to_tensor(self, theta_dict, mode='trajectory'):
        """Pack parameters into a flat tensor for L-BFGS.

        Parameters
        ----------
        theta_dict : dict
        mode : {'trajectory', 'joint', 'joint_with_v0'}

        Returns
        -------
        torch.Tensor
        """
        d, N = self.d, self.N
        tensor_rows = []

        if mode == "trajectory":
            self.theta_fixed = {
                'alphas': [alpha.clone() for alpha in theta_dict['alphas']],
                'U_skews': [U.clone() for U in theta_dict['U_skews']],
                'omegas': [omega.clone() for omega in theta_dict['omegas']],
                'x0s': [x0.clone() for x0 in theta_dict['x0s']],
                'a0s': [a0.clone() for a0 in theta_dict['a0s']],
            }
            for k in range(N):
                v0_n = theta_dict['v0s'][k]
                v0_k_0 = torch.log(torch.abs(v0_n[0]) + 1e-8)
                tensor_rows.append(torch.stack([v0_k_0, v0_n[1]]))

        elif mode in ("joint", "joint_with_v0"):
            if not hasattr(self, 'theta_fixed') or self.theta_fixed is None:
                self.theta_fixed = {
                    'x0s': [x0.clone() for x0 in theta_dict['x0s']],
                    'a0s': [a0.clone() for a0 in theta_dict['a0s']],
                }
                if mode == "joint":
                    self.theta_fixed['v0s'] = [v0.clone() for v0 in theta_dict['v0s']]

            for k in range(N):
                row_parts = []

                if mode == "joint_with_v0":
                    v0_n = theta_dict['v0s'][k]
                    row_parts.append(torch.log(torch.abs(v0_n[0]) + 1e-8).reshape(-1))
                    row_parts.append(v0_n[1].reshape(-1))

                # Alpha – log transform
                row_parts.append(torch.log(theta_dict["alphas"][k].clone()).reshape(-1))

                # U_skew – log-transform diagonal, keep upper triangle
                U_skew_copy = theta_dict["U_skews"][k].clone()
                EPS = 1e-8
                diag_logged = torch.log(torch.clamp(torch.diagonal(U_skew_copy), min=EPS))
                U_no_diag = U_skew_copy - torch.diag(torch.diagonal(U_skew_copy))
                U_with_log_diag = U_no_diag + torch.diag(diag_logged)
                triu_idx = torch.triu_indices(d, d, device=U_skew_copy.device)
                row_parts.append(U_with_log_diag[triu_idx[0], triu_idx[1]].reshape(-1))

                # Omega – logit reparametrisation to keep ω ∈ (ω_min, ω_max)
                theta_fixed_keys = list(getattr(self, 'theta_fixed', {}).keys())
                if 'omegas' in theta_dict and 'omegas' not in theta_fixed_keys:
                    omega_k = theta_dict["omegas"][k].clone()
                    omega_range = self.omega_max - self.omega_min
                    p = torch.clamp((omega_k - self.omega_min) / omega_range, 1e-6, 1.0 - 1e-6)
                    row_parts.append(torch.log(p / (1.0 - p)).reshape(-1))

                tensor_rows.append(torch.cat(row_parts))

        return tensor_rows[0] if len(tensor_rows) == 1 else torch.stack(tensor_rows)

    def map_from_tensor_to_dict(self, theta_tensor, mode='trajectory'):
        """Unpack a flat tensor back to a parameter dict (inverse of the above)."""
        d, N = self.d, self.N
        theta_dict = {}

        if mode == "trajectory":
            v0s = []
            if N == 1:
                theta_tensor = theta_tensor.squeeze(0)
                v0s.append(torch.stack([torch.exp(theta_tensor[0]), theta_tensor[1]]))
            else:
                for k in range(N):
                    v0s.append(torch.stack([torch.exp(theta_tensor[k, 0]), theta_tensor[k, 1]]))
            theta_dict['v0s'] = v0s

        elif mode in ("joint", "joint_with_v0"):
            alphas, U_skews, omegas, v0s = [], [], [], []
            n_U_params = d * (d + 1) // 2
            rows = [theta_tensor[k] for k in range(N)] if N > 1 else [theta_tensor]

            for row_k in rows:
                idx = 0

                if mode == "joint_with_v0":
                    v0s.append(torch.stack([torch.exp(row_k[idx]), row_k[idx + 1]]))
                    idx += 2

                # Alpha
                alphas.append(torch.exp(torch.clamp(row_k[idx], -5, 5)).unsqueeze(0))
                idx += 1

                # U_skew
                U_skew_vals = row_k[idx: idx + n_U_params]
                U_skew = torch.zeros((d, d), dtype=theta_tensor.dtype, device=theta_tensor.device)
                triu_indices = torch.triu_indices(d, d)
                U_skew[triu_indices[0], triu_indices[1]] = U_skew_vals
                diag_mask = torch.eye(d, dtype=torch.bool, device=theta_tensor.device)
                U_skew_final = U_skew.clone()
                U_skew_final[diag_mask] = torch.exp(torch.clamp(U_skew[diag_mask], -4, 4))
                U_skews.append(U_skew_final)
                idx += n_U_params

                # Omega – inverse sigmoid
                if len(row_k) > idx:
                    z_omega = row_k[idx]
                    omega = self.omega_min + (self.omega_max - self.omega_min) * torch.sigmoid(z_omega)
                    omegas.append(omega.unsqueeze(0) if omega.dim() == 0 else omega)

            theta_dict['alphas'] = alphas
            theta_dict['U_skews'] = U_skews
            if omegas:
                theta_dict['omegas'] = omegas
            if mode == "joint_with_v0" and v0s:
                theta_dict['v0s'] = v0s

        return theta_dict

    def construct_soln_dict(self, res):
        """Build a full parameter dict from an optimization result."""
        theta_tensor = res.x
        tensor_size = (
            theta_tensor.numel() if theta_tensor.dim() == 1
            else theta_tensor.shape[0] * theta_tensor.shape[1]
        )
        params_per_gaussian = tensor_size // self.N if self.N > 0 else tensor_size
        has_v0_fixed = hasattr(self, 'theta_fixed') and 'v0s' in self.theta_fixed

        if params_per_gaussian == 2:
            mode = 'trajectory'
        elif params_per_gaussian == 4 and hasattr(self, 'theta_fixed') and 'omegas' in self.theta_fixed:
            mode = 'joint'
        elif params_per_gaussian == 7 or (params_per_gaussian >= 5 and not has_v0_fixed):
            mode = 'joint_with_v0'
        elif params_per_gaussian >= 4:
            mode = 'joint'
        else:
            raise ValueError(f"Cannot determine mode: {params_per_gaussian} params per Gaussian")

        soln_dict = self.map_from_tensor_to_dict(theta_tensor, mode=mode)
        for key, value in self.theta_fixed.items():
            if key not in soln_dict:
                soln_dict[key] = value.copy()
        return soln_dict

    # ==================================================================
    # Optional utility methods (not part of the main pipeline)
    # ==================================================================

    def _eval_joint_loss(self, soln_dict):
        """Evaluate the joint projection loss at the current parameters."""
        self.theta_fixed = {
            'x0s': [x0.clone() for x0 in soln_dict['x0s']],
            'v0s': [v0.clone() for v0 in soln_dict['v0s']],
            'a0s': [a0.clone() for a0 in soln_dict['a0s']],
        }
        test_dict = {
            'alphas': [a.clone().requires_grad_(False) for a in soln_dict['alphas']],
            'U_skews': [U.clone().requires_grad_(False) for U in soln_dict['U_skews']],
            'omegas': [w.clone().requires_grad_(False) for w in soln_dict['omegas']],
            'x0s': soln_dict['x0s'], 'v0s': soln_dict['v0s'], 'a0s': soln_dict['a0s'],
        }
        theta_tensor = self.map_from_dict_to_tensor(test_dict, mode='joint')
        with torch.no_grad():
            return self._loss_joint(theta_tensor).item()

    def _optimize_joint(self, soln_dict, max_iter=300):
        """Additional joint optimization of ω, U_skew, α (not part of main pipeline)."""
        logger.info("Optimizing omega, U_skew, alpha jointly...")
        for key in ('alphas', 'U_skews', 'omegas'):
            soln_dict[key] = [p.requires_grad_(True) for p in soln_dict[key]]

        self.theta_fixed = {
            'x0s': [x0.clone() for x0 in soln_dict['x0s']],
            'a0s': [a0.clone() for a0 in soln_dict['a0s']],
        }
        if not soln_dict['v0s'][0].requires_grad:
            self.theta_fixed['v0s'] = [v0.clone() for v0 in soln_dict['v0s']]

        theta_tensor = self.map_from_dict_to_tensor(soln_dict, mode='joint')
        res = minimize(
            self._loss_joint, x0=theta_tensor, method='l-bfgs',
            tol=1e-8, options={'gtol': 1e-8, 'max_iter': max_iter, 'disp': False},
        )
        result_dict = self.construct_soln_dict(res)
        for key in ('alphas', 'U_skews', 'omegas'):
            soln_dict[key] = [p.clone().detach() for p in result_dict[key]]
        logger.info("Joint optimization: loss = %.6e (%d iters)", res.fun.item(), res.nit)
        return soln_dict

    def _fine_grid_search_omega(self, soln_dict, current_sup_error, omega_range=3.0, omega_step=0.1):
        """Fine ±omega_range Hz grid search around the current omega estimate."""
        logger.info("Grid search: ±%.1f Hz with %.2f Hz steps", omega_range, omega_step)
        self.theta_fixed = {
            'x0s': soln_dict['x0s'], 'v0s': soln_dict['v0s'], 'a0s': soln_dict['a0s'],
        }
        best_sup_err = current_sup_error
        best_omegas = [omega.clone() for omega in soln_dict['omegas']]

        for k in range(self.N):
            omega_current = soln_dict['omegas'][k].item()
            omega_lo = omega_current - omega_range
            omega_hi = omega_current + omega_range
            n_points = int((omega_hi - omega_lo) / omega_step) + 1
            omega_candidates = np.linspace(omega_lo, omega_hi, n_points)

            sup_errors = []
            for omega_test in omega_candidates:
                test_dict = {
                    key: [p.clone().requires_grad_(False) for p in soln_dict[key]]
                    for key in ('alphas', 'U_skews', 'omegas')
                }
                test_dict.update({
                    'x0s': soln_dict['x0s'], 'v0s': soln_dict['v0s'], 'a0s': soln_dict['a0s'],
                })
                test_dict['omegas'][k] = torch.tensor([omega_test], dtype=torch.float64, device=self.device)
                sup_errors.append(self._sup_projection_error(test_dict))

            min_idx = np.argmin(sup_errors)
            if sup_errors[min_idx] < best_sup_err:
                best_omegas[k] = torch.tensor([omega_candidates[min_idx]], dtype=torch.float64, device=self.device)
                best_sup_err = sup_errors[min_idx]
                logger.info("  Gaussian %d: ω %.4f → %.4f Hz (Δsup = %.6e)",
                            k, omega_current, omega_candidates[min_idx],
                            current_sup_error - sup_errors[min_idx])

        soln_dict['omegas'] = [omega.clone().detach() for omega in best_omegas]
        return soln_dict

    # ==================================================================
    # Internal utilities
    # ==================================================================

    def _clone_dict(self, d):
        """Deep-clone a parameter dict (lists of tensors)."""
        return {
            key: (
                [v.clone().detach() for v in val] if isinstance(val, list)
                else val.clone().detach() if isinstance(val, torch.Tensor) else val
            )
            for key, val in d.items()
        }

    def _to_device(self, obj):
        """Recursively move tensors / nested structures to ``self.device``."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, list):
            return [self._to_device(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj









        # print(f"t_observable is = {self.t_observable}")
        # print(f"The maximising receievers are: {self.maximising_rcvrs}")
        # print(f"t_obs_by_cluster = {self.t_obs_by_cluster}")
    #     if len(self.t_observable) == 0:
    #         self.t_observable = torch.tensor([0.6040, 0.6141, 0.6242, 0.6342, 0.6443, 0.6544, 0.6644, 0.6745, 0.6846,
    #                                           0.6946, 0.7047, 0.7148, 0.7248, 0.7349, 0.7450, 0.7550, 0.7651, 0.7752,
    #                                           0.7852, 0.7953, 0.8054, 0.8154, 0.8255, 0.8356, 0.8456, 0.8557, 0.8658,
    #                                           0.8758, 0.8859, 0.8960, 0.9060, 0.9161, 0.9262, 0.9362, 0.9463, 0.9564,
    #                                           0.9664, 0.9765, 0.9866, 0.9966, 1.0067, 1.0168, 1.0268, 1.0369, 1.0470,
    #                                           1.0570, 1.0671, 1.0772, 1.0872, 1.0973, 1.1074, 1.1174, 1.1275, 1.1376,
    #                                           1.1477, 1.1577, 1.1678, 1.1779, 1.1879, 1.1980, 1.2081, 1.2181, 1.2282],
    #                                          dtype=torch.float64)
    #         # self.maximising_rcvrs = [[torch.tensor([4.0000, 0.9370], dtype=torch.float64), torch.tensor([4.0000, 0.8110], dtype=torch.float64), 
    #         #                           torch.tensor([4.0000, 0.7165], dtype=torch.float64), torch.tensor([4.0000, 0.6220], dtype=torch.float64), 
    #         #                           torch.tensor([4.0000, 0.4961], dtype=torch.float64), torch.tensor([4.0000, 0.3701], dtype=torch.float64), 
    #         #                           torch.tensor([4.0000, 0.2756], dtype=torch.float64), torch.tensor([4.0000, 0.1496], dtype=torch.float64), 
    #         #                           torch.tensor([4.0000, 0.0236], dtype=torch.float64), torch.tensor([ 4.0000, -0.0709], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -0.1969], dtype=torch.float64), torch.tensor([ 4.0000, -0.2913], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -0.4173], dtype=torch.float64), torch.tensor([ 4.0000, -0.5433], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -0.6693], dtype=torch.float64), torch.tensor([ 4.0000, -0.7953], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -0.9213], dtype=torch.float64), torch.tensor([ 4.0000, -1.0157], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -1.1417], dtype=torch.float64), torch.tensor([ 4.0000, -1.2677], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -1.3937], dtype=torch.float64), torch.tensor([ 4.0000, -1.5197], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -1.6457], dtype=torch.float64), torch.tensor([ 4.0000, -1.7717], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -1.8976], dtype=torch.float64), torch.tensor([ 4.0000, -2.0236], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -2.1496], dtype=torch.float64), torch.tensor([ 4.0000, -2.2756], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -2.4016], dtype=torch.float64), torch.tensor([ 4.0000, -2.5276], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -2.6535], dtype=torch.float64), torch.tensor([ 4.0000, -2.8110], dtype=torch.float64), 
    #         #                           torch.tensor([ 4.0000, -2.9370], dtype=torch.float64)]]
    #         self.maximising_rcvrs = [[torch.tensor([4.0000, 0.9055], dtype=torch.float64), torch.tensor([4.0000, 0.8110], dtype=torch.float64), torch.tensor([4.0000, 0.7165], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.6220], dtype=torch.float64), torch.tensor([4.0000, 0.4961], dtype=torch.float64), torch.tensor([4.0000, 0.3701], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.2756], dtype=torch.float64), torch.tensor([4.0000, 0.1496], dtype=torch.float64), torch.tensor([4.0000, 0.0551], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.0709], dtype=torch.float64), torch.tensor([ 4.0000, -0.1969], dtype=torch.float64), torch.tensor([ 4.0000, -0.2913], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.4173], dtype=torch.float64), torch.tensor([ 4.0000, -0.5433], dtype=torch.float64), torch.tensor([ 4.0000, -0.6693], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.7953], dtype=torch.float64), torch.tensor([ 4.0000, -0.9213], dtype=torch.float64), torch.tensor([ 4.0000, -1.0472], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.1732], dtype=torch.float64), torch.tensor([ 4.0000, -1.2992], dtype=torch.float64), torch.tensor([ 4.0000, -1.4252], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.5512], dtype=torch.float64), torch.tensor([ 4.0000, -1.6772], dtype=torch.float64), torch.tensor([ 4.0000, -1.8031], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.9291], dtype=torch.float64), torch.tensor([ 4.0000, -2.0551], dtype=torch.float64), torch.tensor([ 4.0000, -2.2126], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.3386], dtype=torch.float64), torch.tensor([ 4.0000, -2.4646], dtype=torch.float64), torch.tensor([ 4.0000, -2.6220], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.7480], dtype=torch.float64), torch.tensor([ 4.0000, -2.9055], dtype=torch.float64), torch.tensor([ 4.0000, -1.8031], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.8976], dtype=torch.float64), torch.tensor([ 4.0000, -2.0236], dtype=torch.float64), torch.tensor([ 4.0000, -2.1181], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.2126], dtype=torch.float64), torch.tensor([ 4.0000, -2.3386], dtype=torch.float64), torch.tensor([ 4.0000, -2.4331], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.5276], dtype=torch.float64), torch.tensor([ 4.0000, -2.6535], dtype=torch.float64), torch.tensor([ 4.0000, -2.7480], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.8740], dtype=torch.float64), torch.tensor([ 4.0000, -2.9685], dtype=torch.float64), torch.tensor([ 4.0000, -2.3386], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.4646], dtype=torch.float64), torch.tensor([ 4.0000, -2.6220], dtype=torch.float64), torch.tensor([ 4.0000, -2.7480], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.8740], dtype=torch.float64), torch.tensor([ 4.0000, -1.9921], dtype=torch.float64), torch.tensor([ 4.0000, -2.0866], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.2126], dtype=torch.float64), torch.tensor([ 4.0000, -2.3701], dtype=torch.float64), torch.tensor([ 4.0000, -2.4961], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.6220], dtype=torch.float64), torch.tensor([ 4.0000, -2.7480], dtype=torch.float64), torch.tensor([ 4.0000, -2.8740], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.4646], dtype=torch.float64), torch.tensor([ 4.0000, -2.5591], dtype=torch.float64), torch.tensor([ 4.0000, -2.6535], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.7480], dtype=torch.float64), torch.tensor([ 4.0000, -2.8425], dtype=torch.float64), torch.tensor([ 4.0000, -2.9370], dtype=torch.float64)], 
    #                                  [torch.tensor([4.0000, 0.9685], dtype=torch.float64), torch.tensor([4.0000, 0.8740], dtype=torch.float64), torch.tensor([4.0000, 0.8110], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.7165], dtype=torch.float64), torch.tensor([4.0000, 0.6220], dtype=torch.float64), torch.tensor([4.0000, 0.5276], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.4646], dtype=torch.float64), torch.tensor([4.0000, 0.3701], dtype=torch.float64), torch.tensor([4.0000, 0.2756], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.1811], dtype=torch.float64), torch.tensor([4.0000, 0.0866], dtype=torch.float64), torch.tensor([ 4.0000, -0.0079], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.1024], dtype=torch.float64), torch.tensor([ 4.0000, -0.1969], dtype=torch.float64), torch.tensor([ 4.0000, -0.2913], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.3858], dtype=torch.float64), torch.tensor([ 4.0000, -0.4803], dtype=torch.float64), torch.tensor([ 4.0000, -0.5748], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.6693], dtype=torch.float64), torch.tensor([ 4.0000, -0.7638], dtype=torch.float64), torch.tensor([ 4.0000, -0.8898], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.9843], dtype=torch.float64), torch.tensor([ 4.0000, -1.0787], dtype=torch.float64), torch.tensor([ 4.0000, -1.1732], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.2677], dtype=torch.float64), torch.tensor([ 4.0000, -1.3937], dtype=torch.float64), torch.tensor([ 4.0000, -1.4882], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.5827], dtype=torch.float64), torch.tensor([ 4.0000, -1.6772], dtype=torch.float64), torch.tensor([ 4.0000, -0.7953], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.9213], dtype=torch.float64), torch.tensor([ 4.0000, -1.0472], dtype=torch.float64), torch.tensor([ 4.0000, -1.1732], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.2992], dtype=torch.float64), torch.tensor([ 4.0000, -1.4252], dtype=torch.float64), torch.tensor([ 4.0000, -1.5512], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.6772], dtype=torch.float64), torch.tensor([ 4.0000, -1.8031], dtype=torch.float64), torch.tensor([ 4.0000, -1.9291], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.0866], dtype=torch.float64), torch.tensor([ 4.0000, -2.2126], dtype=torch.float64), torch.tensor([ 4.0000, -1.2992], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.3937], dtype=torch.float64), torch.tensor([ 4.0000, -1.4882], dtype=torch.float64), torch.tensor([ 4.0000, -1.7402], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.8661], dtype=torch.float64), torch.tensor([ 4.0000, -1.7402], dtype=torch.float64), torch.tensor([ 4.0000, -1.8346], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.9291], dtype=torch.float64), torch.tensor([ 4.0000, -1.9921], dtype=torch.float64), torch.tensor([ 4.0000, -2.0866], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -2.1811], dtype=torch.float64), torch.tensor([ 4.0000, -2.2756], dtype=torch.float64), torch.tensor([ 4.0000, -2.3701], dtype=torch.float64)], 
    #                                  [torch.tensor([4.0000, 0.9055], dtype=torch.float64), torch.tensor([4.0000, 0.8425], dtype=torch.float64), torch.tensor([4.0000, 0.7480], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.6535], dtype=torch.float64), torch.tensor([4.0000, 0.4961], dtype=torch.float64), torch.tensor([4.0000, 0.3701], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.2441], dtype=torch.float64), torch.tensor([4.0000, 0.1181], dtype=torch.float64), torch.tensor([4.0000, 0.0236], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.1024], dtype=torch.float64), torch.tensor([ 4.0000, -0.1969], dtype=torch.float64), torch.tensor([ 4.0000, -0.3228], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.4488], dtype=torch.float64), torch.tensor([ 4.0000, -0.5748], dtype=torch.float64), torch.tensor([ 4.0000, -0.6693], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.2283], dtype=torch.float64), torch.tensor([ 4.0000, -0.2913], dtype=torch.float64), torch.tensor([ 4.0000, -0.3228], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.3858], dtype=torch.float64), torch.tensor([ 4.0000, -0.4803], dtype=torch.float64), torch.tensor([ 4.0000, -0.6063], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.7323], dtype=torch.float64), torch.tensor([ 4.0000, -0.8583], dtype=torch.float64), torch.tensor([ 4.0000, -0.9528], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.0157], dtype=torch.float64), torch.tensor([ 4.0000, -1.1102], dtype=torch.float64), torch.tensor([ 4.0000, -1.2047], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -1.5512], dtype=torch.float64), torch.tensor([ 4.0000, -1.6457], dtype=torch.float64)], 
    #                                  [torch.tensor([4.0000, 0.6220], dtype=torch.float64), torch.tensor([4.0000, 0.5591], dtype=torch.float64), torch.tensor([4.0000, 0.4646], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.4016], dtype=torch.float64), torch.tensor([4.0000, 0.3071], dtype=torch.float64), torch.tensor([4.0000, 0.2441], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.1496], dtype=torch.float64), torch.tensor([4.0000, 0.0866], dtype=torch.float64), torch.tensor([ 4.0000, -0.0079], dtype=torch.float64), 
    #                                   torch.tensor([ 4.0000, -0.0709], dtype=torch.float64), torch.tensor([ 4.0000, -0.1654], dtype=torch.float64)], 
    #                                  [torch.tensor([4.0000, 0.9055], dtype=torch.float64), torch.tensor([4.0000, 0.8110], dtype=torch.float64), torch.tensor([4.0000, 0.7165], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.6220], dtype=torch.float64), torch.tensor([4.0000, 0.4961], dtype=torch.float64), torch.tensor([4.0000, 0.4016], dtype=torch.float64), 
    #                                   torch.tensor([4.0000, 0.2756], dtype=torch.float64)]]
            
    #         self.t_obs_by_cluster = [torch.tensor([0.6040, 0.6141, 0.6242, 0.6342, 0.6443, 0.6544, 0.6644, 0.6745, 0.6846,
    #                                                0.6946, 0.7047, 0.7148, 0.7248, 0.7349, 0.7450, 0.7550, 0.7651, 0.7752,
    #                                                0.7852, 0.7953, 0.8054, 0.8154, 0.8255, 0.8356, 0.8456, 0.8557, 0.8658,
    #                                                0.8758, 0.8859, 0.8960, 0.9060, 0.9161, 0.9262, 0.9362, 0.9463, 0.9564,
    #                                                0.9664, 0.9765, 0.9866, 0.9966, 1.0067, 1.0168, 1.0268, 1.0369, 1.0470,
    #                                                1.0570, 1.0671, 1.0772, 1.0872, 1.0973, 1.1074, 1.1174, 1.1275, 1.1376,
    #                                                1.1477, 1.1577, 1.1678, 1.1779, 1.1879, 1.1980, 1.2081, 1.2181, 1.2282], dtype=torch.float64), 
    #                                  torch.tensor([0.6342, 0.6443, 0.6544, 0.6644, 0.6745, 0.6846, 0.6946, 0.7047, 0.7148,
    #                                                0.7248, 0.7349, 0.7450, 0.7550, 0.7651, 0.7752, 0.7852, 0.7953, 0.8054,
    #                                                0.8154, 0.8255, 0.8356, 0.8456, 0.8557, 0.8658, 0.8758, 0.8859, 0.8960,
    #                                                0.9060, 0.9161, 0.9262, 0.9362, 0.9463, 0.9564, 0.9664, 0.9765, 0.9866,
    #                                                0.9966, 1.0067, 1.0168, 1.0268, 1.0369, 1.0470, 1.0570, 1.0671, 1.0772,
    #                                                1.0872, 1.0973, 1.1074, 1.1174, 1.1275, 1.1376, 1.1477, 1.1577, 1.1678], dtype=torch.float64), 
    #                                  torch.tensor([0.7752, 0.7852, 0.7953, 0.8054, 0.8154, 0.8255, 0.8356, 0.8456, 0.8557,
    #                                                0.8658, 0.8758, 0.8859, 0.8960, 0.9060, 0.9161, 0.9262, 0.9362, 0.9463,
    #                                                0.9564, 0.9664, 0.9765, 0.9866, 0.9966, 1.0067, 1.0168, 1.0268, 1.0369,
    #                                                1.0772, 1.0872], dtype=torch.float64), 
    #                                  torch.tensor([0.8154, 0.8255, 0.8356, 0.8456, 0.8557, 0.8658, 0.8758, 0.8859, 0.8960,
    #                                                0.9060, 0.9161], dtype=torch.float64), 
    #                                  torch.tensor([0.8356, 0.8456, 0.8557, 0.8658, 0.8758, 0.8859, 0.8960],
    #    dtype=torch.float64)]
    
    
    
    
    
    
    
    
# ==========================================================================
# PeakData – container for peak detection and assignment results
# ==========================================================================

# class PeakData:
#     """Store peak-detection and trajectory-assignment data.

#     Attributes
#     ----------
#     observable_indices : list of int
#         Indices into the full time array where peaks were found.
#     receiver_heights_by_time : dict
#         ``{time_val: [heights]}`` – detected peak heights at each time.
#     times, receiver_positions, receiver_indices, peak_values : list of list
#         Per-Gaussian sequential detection results (used by diagnostic plots).
#     assigned_times, assigned_heights, assigned_values : list of list
#         Per-Gaussian optimal assignments (set by Hungarian / nearest-neighbour).
#     """

#     def __init__(self, n_gaussians: int, device: torch.device):
#         self.N = n_gaussians
#         self.device = device

#         # Raw detection (per time point)
#         self.observable_indices: list = []
#         self.receiver_heights_by_time: dict = {}

#         # Sequential assignment (per Gaussian) – used by diagnostic plots
#         self.times = [[] for _ in range(n_gaussians)]
#         self.receiver_positions = [[] for _ in range(n_gaussians)]
#         self.receiver_indices = [[] for _ in range(n_gaussians)]
#         self.peak_values = [[] for _ in range(n_gaussians)]

#         # Optimal assignment (per Gaussian) – used by Newton-Raphson refinement
#         self.assigned_times = [[] for _ in range(n_gaussians)]
#         self.assigned_heights = [[] for _ in range(n_gaussians)]
#         self.assigned_values = [[] for _ in range(n_gaussians)]

#     def add_peak_detection(self, time_idx, time_val, receiver_idx, receiver_pos,
#                            peak_val, gaussian_idx):
#         """Record one detected peak from the sequential bottom-to-top scan."""
#         self.times[gaussian_idx].append(time_val)
#         self.receiver_positions[gaussian_idx].append(receiver_pos)
#         self.receiver_indices[gaussian_idx].append(receiver_idx)
#         self.peak_values[gaussian_idx].append(peak_val)

#         if gaussian_idx == 0 and time_idx not in self.observable_indices:
#             self.observable_indices.append(time_idx)

#     def add_time_detections(self, time_val, detected_heights):
#         """Record all peak heights found at one time point."""
#         if detected_heights:
#             self.receiver_heights_by_time[time_val] = detected_heights

#     def finalize_detections(self):
#         """Convert accumulated per-Gaussian lists to tensors."""
#         for k in range(self.N):
#             vals = self.times[k]
#             self.times[k] = (
#                 torch.tensor(vals, dtype=torch.float64, device=self.device)
#                 if vals
#                 else torch.tensor([], dtype=torch.float64, device=self.device)
#             )

#     def add_optimal_assignment(self, gaussian_idx, time_val, height, value):
#         """Record one peak-to-trajectory assignment from Hungarian or NN."""
#         self.assigned_times[gaussian_idx].append(time_val)
#         self.assigned_heights[gaussian_idx].append(height)
#         self.assigned_values[gaussian_idx].append(value)

#     def get_assignment_data(self, gaussian_idx):
#         """Return ``(times, heights)`` for the optimal assignment of Gaussian k."""
#         return self.assigned_times[gaussian_idx], self.assigned_heights[gaussian_idx]

#     def get_heights_dict_non_empty(self):
#         """Return ``{time: heights}`` filtered to times with detections."""
#         return {t: h for t, h in self.receiver_heights_by_time.items() if h}

#     def get_heights_sorted_by_time(self):
#         """Return detected heights sorted bottom-to-top at each time point."""
#         return [sorted(h) for h in self.receiver_heights_by_time.values()]



    # def generate_projections(self, t, theta_dict, loss_type=None):
    #     """Compute X-ray projections for all sources and time steps.

    #     Parameters
    #     ----------
    #     t : torch.Tensor
    #         Time vector.
    #     theta_dict : dict
    #         Parameter dict with keys
    #         ``'alphas', 'U_skews', 'omegas', 'x0s', 'v0s', 'a0s'``.
    #     loss_type : str, optional
    #         When set, merges ``self.theta_fixed`` into *theta_dict*
    #         (used internally during optimization).

    #     Returns
    #     -------
    #     list of torch.Tensor
    #         Projections for each source, shape ``(n_times, n_receivers)``.
    #     """
    #     rot_mat_funcs = self.construct_rotation_matrix_funcs()
    #     traj_funcs = self.construct_trajectory_funcs()
    #     projs = [
    #         torch.zeros(len(t), self.n_rcvrs, dtype=torch.float64, device=self.device)
    #         for _ in range(self.n_sources)
    #     ]

    #     if loss_type is not None:
    #         complete_theta_dict = theta_dict.copy()
    #         for key, value in self.theta_fixed.items():
    #             if key not in complete_theta_dict:
    #                 complete_theta_dict[key] = value
    #         theta_dict = complete_theta_dict

    #     EPS = 1e-10

    #     for n_t, t_n in enumerate(t):
    #         rot_mat_of_t = rot_mat_funcs(t_n, theta_dict)
    #         traj_of_t = traj_funcs(t_n, theta_dict)

    #         for n_s, s in enumerate(self.sources):
    #             receivers_ns = self.receivers[n_s]
    #             r = torch.stack(receivers_ns)

    #             r_minus_s = r - s
    #             r_minus_s_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True)

    #             for k in range(self.N):
    #                 alpha_k = theta_dict['alphas'][k].squeeze()
    #                 U_k = theta_dict['U_skews'][k]
    #                 R_k_of_t = rot_mat_of_t[k]
    #                 mu_k_of_t = traj_of_t[k]
    #                 new_U_k = U_k @ R_k_of_t.mT

    #                 U_r_hat = new_U_k @ r_minus_s_hat.T
    #                 U_r = new_U_k @ r_minus_s.T
    #                 U_traj = new_U_k @ (s - mu_k_of_t).unsqueeze(1)

    #                 norm_term = torch.norm(U_r_hat, dim=0)
    #                 quotient_term = self.sqrt_pi * alpha_k / (norm_term + EPS)

    #                 inner_prod_sq = torch.sum(U_r * U_traj, dim=0) ** 2
    #                 divisor = torch.norm(U_r, dim=0) ** 2 + EPS
    #                 subtractor = torch.norm(U_traj, dim=0) ** 2

    #                 exp_arg = inner_prod_sq / divisor - subtractor
    #                 projs[n_s][n_t] += quotient_term * torch.exp(exp_arg)

    #     return projs
    
    
    
        # def construct_rotation_matrix_funcs(self):
    #     """Return a callable ``f(t, theta) -> list of (d, d) rotation matrices``."""
    #     two_pi = 2 * torch.pi

    #     def all_rot_mat_funcs(t, theta):
    #         rot_matrices = []
    #         for k in range(self.N):
    #             omegas_k = theta['omegas'][k]
    #             kth_rot_mat = torch.eye(self.d, dtype=torch.float64, device=self.device)
    #             for n_rots, omega in enumerate(omegas_k):
    #                 i, j = torch.combinations(
    #                     torch.arange(self.d, device=self.device), r=2
    #                 )[n_rots]
    #                 rot_mat = torch.eye(self.d, dtype=torch.float64, device=self.device)
    #                 rot_mat[i, i] = torch.cos(two_pi * omega * t)
    #                 rot_mat[i, j] = -torch.sin(two_pi * omega * t)
    #                 rot_mat[j, i] = torch.sin(two_pi * omega * t)
    #                 rot_mat[j, j] = torch.cos(two_pi * omega * t)
    #                 kth_rot_mat = kth_rot_mat @ rot_mat
    #             rot_matrices.append(kth_rot_mat)
    #         return rot_matrices

    #     return all_rot_mat_funcs

    # def construct_trajectory_funcs(self):
    #     """Return a callable ``f(t, theta) -> list of position tensors``.

    #     Trajectory: ``μ_k(t) = x0_n + v0_k·t + ½·a0_k·t²``
    #     """
    #     def all_traj_funcs(t, theta):
    #         trajectories = []
    #         for k in range(self.N):
    #             x0, v0, a0 = theta['x0s'][k], theta['v0s'][k], theta['a0s'][k]
    #             if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
    #                 trajectories.append(x0 + v0 * t + 0.5 * a0 * t ** 2)
    #             else:
    #                 t_r = t.unsqueeze(1)
    #                 trajectories.append(x0 + v0 * t_r + 0.5 * a0 * t_r ** 2)
    #         return trajectories

    #     return all_traj_funcs

    # def process_projections(self, projections):
    #     """Flatten multi-source projections to a single ``(n_times, n_rcvrs)`` tensor."""
    #     if self.n_sources == 1:
    #         return projections[0]
    #     return torch.cat(projections, dim=0)