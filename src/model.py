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
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchmin import minimize

from .utils import NewtonRaphsonLBFGS
from .structures import PeakData

logger = logging.getLogger(__name__)

class GMM_reco:
    """Reconstruct GMM parameters from CT projection data."""
    
    def __init__(
        self, 
        d: int,                                 # Spatial dimensionality (e.g. 2 for 2-D problems).
        N: int,                                 # Number of Gaussian components.
        sources: list[torch.Tensor],            # X-ray source positions.
        receivers: list[list[torch.Tensor]],    # Receiver positions, one list per source.
        x0s: list[torch.Tensor],                # Initial positions for each Gaussian.
        a0s: list[torch.Tensor],                # Accelerations for each Gaussian.
        omega_min: float,                       # Minimum angular velocity (Hz).
        omega_max: float,                       # Maximum angular velocity (Hz).
        exp_dir: str,                           # Directory for diagnostic plots.
        device: str = "cpu",                    # Computation device (auto-detected when None).
        n_traj_trials: int | None = None,       # Multi-start trials for Stage 1 (default: max(20, 2·N)).
        n_omega_inits: int | None = None,       # Multi-start trials for Stage 2 (default: 5).
        save_diagnostics: bool = True,          # Save diagnostic plots at the end of Stage 1
        fitted_gaussian_peaks: dict[float, torch.Tensor] | None = None,
        peak_detection_records: pd.DataFrame | None = None,
    ):
        self.d = d
        self.N = N
        self.x0s = x0s
        self.a0s = a0s
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.n_traj_trials = n_traj_trials
        self.n_omega_inits = n_omega_inits
        self.save_diagnostics = save_diagnostics
        self.t_observable = []
        self.fitted_gaussian_peaks = fitted_gaussian_peaks
        self.peak_detection_records = peak_detection_records

        # Device
        self.device = (
            torch.device(device) if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.exp_dir = Path(exp_dir) if exp_dir else Path('data/results')
        if self.save_diagnostics:
            self.exp_dir.mkdir(parents=True, exist_ok=True)

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
        x0s, a0s = cfg.physics.to_tensors(cfg.reco_n_gaussians, device)
        omega_min, omega_max = cfg.physics.omega_range

        return cls(
            d=cfg.geometry.dimensionality,
            N=cfg.reco_n_gaussians,
            sources=sources,
            receivers=receivers,
            x0s=x0s,
            a0s=a0s,
            omega_min=omega_min,
            omega_max=omega_max,
            device=device,
            exp_dir=cfg.output.directory,
            n_traj_trials=cfg.reconstruction.n_trajectory_trials,
            n_omega_inits=cfg.reconstruction.n_omega_inits,
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

        # Stage 1: Trajectory Optimization
        soln_dict = self._stage_trajectory_optimization(t, proj_data)
        self.theta_pre_stage1_5 = self._clone_dict(soln_dict)

        # Stage 1.5a: Grid Search for Angular Velocities (ω)
        soln_dict = self._stage_omega_initialization(soln_dict)

        # Stage 1.5b: NNLS for Amplitudes (α)
        soln_dict = self._stage_alpha_initialization(soln_dict)
        self.theta_pre_stage2 = self._clone_dict(soln_dict)

        # Stage 2: Multi-start Joint Refinement
        soln_dict = self._stage_multistart_joint(soln_dict, warm_start=True)

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

        rot_mats = self._compute_2d_rotation_matrices(t, theta_dict)    # [N, T, d, d]
        trajs = self._compute_trajectories(t, theta_dict)               # [N, T, d]

        projs = [
            torch.zeros(len(t), self.n_rcvrs, dtype=torch.float64, device=self.device)
            for _ in range(self.n_sources)
        ]
        EPS = 1e-10

        for n_s, source in enumerate(self.sources):
            receivers = torch.stack(self.receivers[n_s])                    # [R, d]
            r_minus_s = receivers - source                                  # [R, d]
            r_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True)  # [R, d]

            for n in range(self.N):
                alpha_n = theta_dict["alphas"][n].squeeze()
                U_n = theta_dict["U_skews"][n]                              # [d, d]
                
                # Batch transform shape matrix: U_n_t = U_n @ R_n(t)^T
                U_n_t = torch.matmul(U_n, rot_mats[n].transpose(-1, -2))    # [T, d, d]

                # Project ray directions & trajectory offset
                # U_r_hat: [T, R, d], U_r: [T, R, d], U_traj: [T, 1, d]
                U_r_hat = torch.matmul(r_hat, U_n_t.transpose(-1, -2))
                U_r = torch.matmul(r_minus_s, U_n_t.transpose(-1, -2))
                s_minus_mu = source - trajs[n] # [T, d]
                U_traj = torch.matmul(s_minus_mu.unsqueeze(1), U_n_t.transpose(-1, -2))

                norm_r_hat = torch.norm(U_r_hat, dim=-1) # [T, R]
                quotient = (self.sqrt_pi * alpha_n) / (norm_r_hat + EPS)

                inner_prod_sq = torch.sum(U_r * U_traj, dim=-1)**2          # [T, R]
                divisor = torch.sum(U_r**2, dim=-1) + EPS
                subtractor = torch.sum(U_traj**2, dim=-1)

                exp_arg = (inner_prod_sq / divisor) - subtractor
                projs[n_s] += quotient * torch.exp(exp_arg)

        return projs
    
    def _compute_2d_rotation_matrices(
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
        
        self.theta_fixed = {
        'x0s': [x0.clone().detach() for x0 in self.x0s],
        'a0s': [a0.clone().detach() for a0 in self.a0s],
        }

        n_traj_trials = self.n_traj_trials or max(20, 2 * self.N)
        logger.info("Running %d trajectory multi-start trials", n_traj_trials)

        errors, results = [], []
        for n_trial in range(n_traj_trials):
            logger.info("Trial %d/%d", n_trial + 1, n_traj_trials)
            self.theta_dict_init = self.initialize_parameters(t, proj_data)
            for v0_n in self.theta_dict_init["v0s"]:
                v0_n.requires_grad_(True)

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

        best_res = results[np.argmin(np.array(errors))]
        soln_dict = self.construct_soln_dict(best_res)

        if 'v0s' not in soln_dict:
            raise RuntimeError(
                f"Trajectory optimization failed — no v0s in result. "
                f"Got keys: {list(soln_dict.keys())}"
            )

        soln_dict["v0s"] = [v0_n.clone().detach() for v0_n in soln_dict["v0s"]]
        logger.info(f"The estimates v0s are {soln_dict["v0s"]}")
        
        if self.save_diagnostics:
            self._plot_stage1_diagnostics(best_res)

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
        
        if self.fitted_gaussian_peaks is None:
            self._detect_all_peaks(proj_data_array, self.receivers[0], t)
        else:
            pdr = self.peak_detection_records.copy()
            
            for time_val, detected_heights in self.fitted_gaussian_peaks.items():
                t_val_float = time_val.item() if isinstance(time_val, torch.Tensor) else float(time_val)
                self.peak_data.add_time_detections(t_val_float, detected_heights)
                
                sub_df = pdr[
                    torch.isclose(torch.tensor(pdr['time_val'].values, dtype=torch.float64), 
                                  torch.tensor(t_val_float, dtype=torch.float64)
                                  ).numpy()
                    ].sort_values(by='gaussian_idx')                
                for n_r, row in enumerate(sub_df.itertuples()):
                    self.peak_data.add_peak_detection(
                        time_idx=int(row.time_idx),
                        time_val=row.time_val,
                        receiver_idx=int(row.receiver_idx),
                        receiver_pos=row.receiver_pos,  # Sub-pixel fitted mean (mu) or grid pos
                        peak_val=row.peak_val,
                        gaussian_idx=int(row.gaussian_idx),
                    )
                
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
            r_maxs_list.append(r_maxs_n)

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

            row_indices, col_indices = linear_sum_assignment(dist_matrix.cpu().detach().numpy())
            for h_idx, g_idx in zip(row_indices, col_indices):
                self.assigned_curve_data[g_idx].append((time_idx, observed_heights[h_idx]))
    
    def _compute_trajectory_loss(self, r_maxs_list):
        """Compute L1 loss between predicted and assigned receiver heights."""
        loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        for k in range(self.N):
            assignments_k = self.assigned_curve_data[k]
            if not assignments_k:
                continue
            time_indices = [int(item[0]) for item in assignments_k]
            observed_heights = torch.stack([
                item[1] if isinstance(item[1], torch.Tensor) 
                else torch.tensor(item[1], dtype=torch.float64, device=self.device) 
                for item in assignments_k
            ])
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
        
        if self.save_diagnostics:
            self._plot_assignment_diagnostics()

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
    
    def isotropic_derivative_function_over_all_times(
        self, 
        v0: torch.Tensor, 
        *args,
    ) -> torch.Tensor:
        """Sum of absolute isotropic projection derivatives across all observed time points."""
        t_obs, r_list, s, x0, a0 = args

        # Convert list of 1D receiver tensors to a single 2D tensor: [T, 2]
        r = torch.stack(r_list) if isinstance(r_list, list) else r_list

        # Broadened shapes across T time points
        t = t_obs.unsqueeze(-1)                              # [T, 1]
        d = r - s                                            # [T, 2]
        d1, d2 = d[:, 0], d[:, 1]
        norm_sq = torch.sum(d**2, dim=-1)                   # [T]

        # Center offsets: [T, 2]
        c_n = s - x0 - v0 * t - 0.5 * a0 * (t**2)

        h_k = d1 * c_n[:, 0] - s[1] * c_n[:, 1]
        term_inner = c_n[:, 1] * r[:, 1] + h_k

        R_k_l = 2.0 * norm_sq * c_n[:, 1] * term_inner
        R_k_r = -2.0 * d2 * (term_inner**2)

        R_k = (R_k_l + R_k_r) / (norm_sq**2)

        # Return total scalar absolute derivative sum across time steps
        return torch.sum(torch.abs(R_k))
    

    # ==================================================================
    # Stage 1.5 – omega grid search
    # ==================================================================

    def _stage_omega_initialization(
        self, 
        soln_dict: dict[str, list[torch.tensor]],
        n_grid: int = 200,
    ) -> dict[str, list[torch.tensor]]:
        """Per-Gaussian omega estimation via residual-sinogram grid search.

        For each Gaussian k, subtracts all other Gaussians' contributions from
        the observed sinogram, then sweeps a uniform grid of omega candidates
        and keeps the one that minimises the residual norm.
        """
        n_gaussians = len(soln_dict['alphas'])
        n_planes = math.comb(self.d, 2)
        logger.info(
            "Stage 1.5a: Residual-sinogram ω grid search (%d plane(s), %d candidates)",
            n_planes, n_grid,
        )
        
        theta_true = getattr(self, 'theta_true', None)
        if theta_true is not None and 'omegas' in theta_true:
            for k, omega_true_k in enumerate(theta_true['omegas']):
                logger.debug(
                    "  Gaussian %d: ω_true = [%s] Hz", 
                    k, ', '.join(f'{w.item():.4f}' for w in omega_true_k.flatten())
                )
                
        proj_obs = self.proj_data
        t = self.t
        omega_candidates = torch.linspace(
            self.omega_min, self.omega_max, n_grid,
            dtype=torch.float64, device=self.device,
        )
        
        for n in range(n_gaussians):
            # Compute background projection (all Gaussians EXCEPT n)
            bg_dict = {key: list(vals) for key, vals in soln_dict.items()}
            bg_dict["alphas"] = [
                torch.zeros(1, dtype=torch.float64, device=self.device) if j == n
                else soln_dict["alphas"][j]
                for j in range(n_gaussians)
            ]
            
            with torch.no_grad():
                proj_bg = self.process_projections(self.generate_projections(t, bg_dict))
                proj_resid_n = proj_obs - proj_bg
                
            omega_n = soln_dict["omegas"][n].clone()
            
            for plane_idx in range(n_planes):
                best_loss_n = torch.norm(proj_resid_n).item()
                best_val_n = omega_n[plane_idx].clone()
                
                for omega_val in omega_candidates:
                    test_omega_n = omega_n.clone()
                    test_omega_n[plane_idx] = omega_val
                    
                    test_dict = {
                        "alphas": [soln_dict["alphas"][n]],
                        "U_skews": [soln_dict["U_skews"][n]],
                        "omegas": [test_omega_n],
                        "x0s": [soln_dict["x0s"][n]],
                        "v0s": [soln_dict["v0s"][n]],
                        "a0s": [soln_dict["a0s"][n]],
                    }
                    
                    orig_N = self.N
                    try:
                        self.N = 1
                        with torch.no_grad():
                            proj_n = self.process_projections(self.generate_projections(t, test_dict))
                    finally:
                        self.N = orig_N
                    
                    loss_n = torch.norm(proj_resid_n - proj_n).item()
                    if loss_n < best_loss_n:
                        best_loss_n = loss_n
                        best_val_n = omega_val
                        
                omega_n[plane_idx] = best_val_n
                
            soln_dict["omegas"][n] = omega_n
            
            omega_str = ', '.join(f'{w.item():.4f}' for w in soln_dict['omegas'][n])
            if theta_true is not None and 'omegas' in theta_true:
                omega_true_str = ', '.join(f'{w.item():.4f}' for w in theta_true['omegas'][n].flatten())
                logger.info("  Gaussian %d: ω_est = [%s] Hz | ω_true = [%s] Hz", n, omega_str, omega_true_str)
            else:
                logger.info("  Gaussian %d: ω = [%s] Hz", n, omega_str)

        return soln_dict
        
    # ==================================================================
    # Stage 1.5b – alpha NNLS
    # ==================================================================

    def _stage_alpha_initialization(self, soln_dict):
        """Initialise attenuation coefficients via non-negative least squares.

        With trajectories, shapes and omegas fixed, the forward model is linear
        in alphas.  Solves ``min_{α≥0} ‖Φα − p_obs‖₂²`` in closed form.
        """
        logger.info("Stage 1.5b: NNLS alpha initialization")
        
        # Restrict to observable time steps and peak data
        t_obs = self.t[self.peak_data.observable_indices]
        p_obs = self.proj_data[self.peak_data.observable_indices]
        T_obs, R = p_obs.shape
        
        Phi = torch.zeros(T_obs * R, self.N, dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            orig_N = self.N
            self.N = 1
            for n in range(orig_N):
                single_dict = {
                    "alphas": [torch.ones(1, dtype=torch.float64, device=self.device)],
                    "U_skews": [soln_dict["U_skews"][n]],
                    "omegas": [soln_dict["omegas"][n]],
                    "x0s": [soln_dict["x0s"][n]],
                    "v0s": [soln_dict["v0s"][n]],
                    "a0s": [soln_dict["a0s"][n]],
                }
                proj_n = self.generate_projections(t_obs, single_dict)
                Phi[:, n] = self.process_projections(proj_n).reshape(-1)
            self.N = orig_N
            
        if not torch.isfinite(Phi).all():
            logger.warning("Non-finite values in basis matrix Φ; skipping alpha initialization.")
            return soln_dict
        
        p_vec = p_obs.reshape(-1, 1)
        
        # Solve least squares and project onto non-negative orthant (α ≥ 0)
        sol = torch.linalg.lstsq(Phi, p_vec, driver='gelsd')
        alpha_hat = sol.solution.squeeze(1).clamp(min=1e-3)
        residual = torch.norm(Phi @ alpha_hat.unsqueeze(1) - p_vec).item()
        
        soln_dict["alphas"] = [
            alpha_hat[n].reshape(1).detach().clone() for n in range(self.N)
        ]
        
        theta_true = getattr(self, 'theta_true', None)
        if theta_true is not None and 'alphas' in theta_true:
            est_str = ', '.join(f'{alpha_hat[n].item():.3f}' for n in range(self.N))
            true_str = ', '.join(f'{theta_true["alphas"][n].item():.3f}' for n in range(self.N))
            logger.info("  α_est  = [%s]", est_str)
            logger.info("  α_true = [%s]", true_str)
        else:
            logger.info("  α = %s", [f'{alpha_hat[n].item():.3f}' for n in range(self.N)])
        logger.info("  NNLS residual ‖Φα − p_obs‖₂ = %.4e", residual)

        return soln_dict

    # ==================================================================
    # Stage 2 – multi-start joint optimization
    # ==================================================================

    def _stage_multistart_joint(
        self, 
        soln_dict: dict[str, list[torch.Tensor]], 
        warm_start: bool = True,
    ) -> dict[str, list[torch.Tensor]]:
        """Multi-start L-BFGS joint optimization to refine α, U_skew, and ω."""
        logger.info("Stage 2: Multi-start joint optimization")
        n_trials = self.n_omega_inits or 5
        logger.info("Running %d optimization trial(s)", n_trials)

        initial_alphas = [a.clone().detach() for a in soln_dict['alphas']]
        initial_U_skews = [U.clone().detach() for U in soln_dict['U_skews']]
        omega_min = self.omega_min - 0.01
        omega_max = self.omega_max + 0.01

        # Explicitly lock stage fixed variables (pure state management)
        self.theta_fixed = {
            'x0s': [x0.clone().detach() for x0 in soln_dict['x0s']],
            'v0s': [v0.clone().detach() for v0 in soln_dict['v0s']],
            'a0s': [a0.clone().detach() for a0 in soln_dict['a0s']],
        }

        all_losses, all_results = [], []
        for trial_idx in range(n_trials):
            if warm_start and trial_idx == 0:
                initial_omegas = [w.clone().detach() for w in soln_dict['omegas']]
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
                'omegas': [w.requires_grad_(True) for w in initial_omegas],
                'x0s': self.theta_fixed['x0s'],
                'v0s': self.theta_fixed['v0s'],
                'a0s': self.theta_fixed['a0s'],
            }

            theta_tensor = self.map_from_dict_to_tensor(test_dict, mode='joint')

            res = minimize(
                self._loss_joint,
                x0=theta_tensor,
                method='l-bfgs',
                tol=1e-10,
                options={'gtol': 1e-10, 'max_iter': 1000, 'disp': False},
            )

            result_dict = self.construct_soln_dict(res, mode='joint')
            final_loss = res.fun.item()

            all_losses.append(final_loss)
            all_results.append(result_dict)

            logger.info(
                "  Trial %d/%d: loss = %.6e | ω = [%s] Hz",
                trial_idx + 1, n_trials, final_loss,
                ', '.join(f'{w.item():.3f}' for w in result_dict['omegas']),
            )

        best_idx = int(np.argmin(all_losses))
        best_result = all_results[best_idx]
        best_loss = all_losses[best_idx]

        soln_dict['alphas'] = [a.clone().detach() for a in best_result['alphas']]
        soln_dict['omegas'] = [w.clone().detach() for w in best_result['omegas']]
        soln_dict['U_skews'] = [U.clone().detach() for U in best_result['U_skews']]

        logger.info("Multi-start complete — best trial: %d, loss: %.6e", best_idx + 1, best_loss)
        logger.info("Best ω = [%s] Hz", ', '.join(f'{w.item():.4f}' for w in soln_dict['omegas']))

        return soln_dict

    def _loss_joint(self, theta_tensor: torch.Tensor) -> torch.Tensor:
        
        has_v0_fixed = hasattr(self, 'theta_fixed') and 'v0s' in self.theta_fixed
        mode = 'joint' if has_v0_fixed else 'joint_with_v0'

        theta_dict = self.map_from_tensor_to_dict(theta_tensor, mode=mode)
        for key, value in getattr(self, 'theta_fixed', {}).items():
            if key not in theta_dict:
                theta_dict[key] = value

        sim_projs = self.generate_projections(self.t_observable, theta_dict)
        sim_projs_processed = self.process_projections(sim_projs)
        proj_data_observable = self.proj_data[self.peak_data.observable_indices]

        return F.huber_loss(sim_projs_processed, proj_data_observable, delta=0.3)

    def construct_soln_dict(
        self, 
        res, 
        mode: str | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        """Construct a full parameter dictionary from an optimization result."""
        theta_tensor = res.x if isinstance(res.x, torch.Tensor) else torch.tensor(res.x, device=self.device)

        if mode is None:
            tensor_size = theta_tensor.numel()
            params_per_gaussian = tensor_size // self.N if self.N > 0 else tensor_size
            has_v0_fixed = hasattr(self, 'theta_fixed') and 'v0s' in self.theta_fixed

            if params_per_gaussian == 2:
                mode = 'trajectory'
            elif params_per_gaussian >= 4 and has_v0_fixed:
                mode = 'joint'
            else:
                mode = 'joint_with_v0'

        soln_dict = self.map_from_tensor_to_dict(theta_tensor, mode=mode)

        for key, value in getattr(self, 'theta_fixed', {}).items():
            if key not in soln_dict:
                soln_dict[key] = [v.clone() for v in value]

        return soln_dict

    # ==================================================================
    # Parameter Serialization (Pure Functions Without Side-Effects)
    # ==================================================================

    def map_from_dict_to_tensor(
        self, 
        theta_dict: dict[str, list[torch.Tensor]], 
        mode: str = 'trajectory',
    ) -> torch.Tensor:
        """Pack parameters into a flat tensor for L-BFGS (pure function)."""
        d, N = self.d, self.N
        tensor_rows = []

        if mode == "trajectory":
            for k in range(N):
                v0_n = theta_dict['v0s'][k]
                v0_k_0 = torch.log(torch.abs(v0_n[0]) + 1e-8)
                tensor_rows.append(torch.stack([v0_k_0, v0_n[1]]))

        elif mode in ("joint", "joint_with_v0"):
            fixed_keys = list(getattr(self, 'theta_fixed', {}).keys())

            for k in range(N):
                row_parts = []

                if mode == "joint_with_v0":
                    v0_n = theta_dict['v0s'][k]
                    row_parts.append(torch.log(torch.abs(v0_n[0]) + 1e-8).reshape(-1))
                    row_parts.append(v0_n[1].reshape(-1))

                # Alpha – log transform
                row_parts.append(torch.log(theta_dict["alphas"][k] + 1e-8).reshape(-1))

                # U_skew – log-transform diagonal, extract upper triangle
                U_skew = theta_dict["U_skews"][k].clone()
                diag_idx = torch.arange(d, device=U_skew.device)
                U_skew[diag_idx, diag_idx] = torch.log(torch.clamp(U_skew[diag_idx, diag_idx], min=1e-8))

                triu_r, triu_c = torch.triu_indices(d, d, device=U_skew.device)
                row_parts.append(U_skew[triu_r, triu_c].reshape(-1))

                # Omega – logit reparameterization to enforce ω ∈ (omega_min, omega_max)
                if 'omegas' in theta_dict and 'omegas' not in fixed_keys:
                    omega_k = theta_dict["omegas"][k]
                    omega_range = self.omega_max - self.omega_min
                    norm_omega = torch.clamp((omega_k - self.omega_min) / omega_range, 1e-6, 1.0 - 1e-6)
                    row_parts.append(torch.logit(norm_omega).reshape(-1))

                tensor_rows.append(torch.cat(row_parts))

        return tensor_rows[0] if len(tensor_rows) == 1 else torch.stack(tensor_rows)

    def map_from_tensor_to_dict(
        self, 
        theta_tensor: torch.Tensor, 
        mode: str = 'trajectory',
    ) -> dict[str, list[torch.Tensor]]:
        """Unpack a flat tensor back to a parameter dictionary."""
        d, N = self.d, self.N
        theta_dict = {}

        if mode == "trajectory":
            v0s = []
            rows = [theta_tensor[n] for n in range(N)] if (N > 1 and theta_tensor.dim() > 1) else [theta_tensor]
            for row in rows:
                v0s.append(torch.stack([torch.exp(row[0]), row[1]]))
            theta_dict['v0s'] = v0s

        elif mode in ("joint", "joint_with_v0"):
            alphas, U_skews, omegas, v0s = [], [], [], []
            n_U_params = d * (d + 1) // 2
            rows = [theta_tensor[n] for n in range(N)] if (N > 1 and theta_tensor.dim() > 1) else [theta_tensor]

            for row_n in rows:
                idx = 0

                if mode == "joint_with_v0":
                    v0s.append(torch.stack([torch.exp(row_n[idx]), row_n[idx + 1]]))
                    idx += 2

                # Alpha
                alphas.append(torch.exp(torch.clamp(row_n[idx], -5.0, 5.0)).unsqueeze(0))
                idx += 1

                # U_skew
                U_vals = row_n[idx: idx + n_U_params]
                U_skew = torch.zeros((d, d), dtype=theta_tensor.dtype, device=theta_tensor.device)
                triu_r, triu_c = torch.triu_indices(d, d, device=theta_tensor.device)
                U_skew[triu_r, triu_c] = U_vals

                diag_idx = torch.arange(d, device=theta_tensor.device)
                U_skew[diag_idx, diag_idx] = torch.exp(torch.clamp(U_skew[diag_idx, diag_idx], -4.0, 4.0))
                U_skews.append(U_skew)
                idx += n_U_params

                # Omega
                if len(row_n) > idx:
                    z_omega = row_n[idx]
                    omega = self.omega_min + (self.omega_max - self.omega_min) * torch.sigmoid(z_omega)
                    omegas.append(omega.unsqueeze(0) if omega.dim() == 0 else omega)

            theta_dict['alphas'] = alphas
            theta_dict['U_skews'] = U_skews
            if omegas:
                theta_dict['omegas'] = omegas
            if mode == "joint_with_v0" and v0s:
                theta_dict['v0s'] = v0s

        return theta_dict


    # ==================================================================
    # Internal utilities
    # ==================================================================
    
    def _create_legacy_aliases(self):
        """Set model attributes expected by diagnostic plotting functions."""
        self.t_obs_by_cluster = self.peak_data.times
        self.maximising_rcvrs = self.peak_data.receiver_positions
        self.maximising_inds = self.peak_data.receiver_indices
        self.peak_values = self.peak_data.peak_values
        self.observable_indices = self.peak_data.observable_indices
        
    def _plot_stage1_diagnostics(self, best_res: dict) -> None:
        from .visualization.diagnostics import (
            # plot_assignment_quality,
            plot_gmm_and_projections,
            plot_heights_by_assignment,
            # plot_raw_receiver_heights,
            # plot_trajectory_estimations,
            # plot_trajectory_fitting,
        )
        # plot_assignment_quality(model=self, res=best_res)
        plot_gmm_and_projections(model=self, res=best_res, theta_true=getattr(self, "theta_true", None))
        plot_heights_by_assignment(self)
        # plot_raw_receiver_heights(self)
        # plot_trajectory_estimations(model=self, res=best_res)
        # plot_trajectory_fitting(model=self, res=best_res)
        
    def _plot_assignment_diagnostics(self):
        from .visualization.diagnostics import plot_heights_by_assignment
        plot_heights_by_assignment(self)

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