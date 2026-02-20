"""
GMM-CT reconstruction pipeline.

Implements the multi-stage reconstruction of Gaussian Mixture Model parameters
from X-ray CT projection data:

  1. Trajectory estimation  (v0 via multi-start L-BFGS)
  2. Joint optimization     (omega, alpha, U_skew via multi-start L-BFGS)
  3. Omega grid search      (fine grid refinement)
  4. Final joint refinement  (polish all morphology parameters)
"""

import torch
import torch.nn as nn
from torchmin import minimize
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import numpy as np

from .forward_model import ForwardModelMixin
from .initialization import InitializationMixin
from .solvers import NewtonRaphsonLBFGS


class GMM_reco(ForwardModelMixin, InitializationMixin):
    """
    Gaussian mixture model reconstruction from CT projection data.

    Fits GMM parameters via a 4-stage pipeline:

      1. Multi-start trajectory optimization (v0 estimation).
      2. Multi-start joint optimization (omega, alpha, U_skew).
      3. Fine grid search for omega refinement.
      4. Final joint refinement.

    Parameters
    ----------
    d : int
        Dimensionality of the problem.
    N : int
        Number of Gaussians.
    sources : list of torch.Tensor
        Source locations.
    receivers : list of list of torch.Tensor
        Receiver locations per source.
    x0s : list of torch.Tensor
        Known initial positions for all Gaussians.
    a0s : list of torch.Tensor
        Known accelerations for all Gaussians.
    omega_min, omega_max : float
        Angular velocity bounds for initialization.
    device : str, optional
        Computation device (``'cuda'``, ``'cpu'``, or ``None`` for
        auto-detection).
    output_dir : str or Path, optional
        Directory for saving diagnostic plots (default: ``'plots/'``).
    N_traj_trials : int, optional
        Number of multi-start trials for trajectory optimization.
    n_omega_inits : int, optional
        Number of random multi-start trials for omega search.
    """

    def __init__(self, d, N, sources, receivers, x0s, a0s,
                 omega_min, omega_max, device=None, output_dir=None,
                 N_traj_trials=None, n_omega_inits=None):
        self.d = d
        self.N = N

        # Known physical parameters
        self.x0s = x0s
        self.a0s = a0s
        self.omega_min = omega_min
        self.omega_max = omega_max

        # Search configuration
        self.N_traj_trials = N_traj_trials
        self.n_omega_inits = n_omega_inits
        # self.use_fft_omega = True

        # Device management
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Output directory for diagnostic plots
        if output_dir is None:
            self.output_dir = Path('plots')
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Precompute constants used in projections
        self.sqrt_pi = torch.sqrt(
            torch.tensor(torch.pi, dtype=torch.float64, device=self.device)
        )

        # Move sources and receivers to device
        if isinstance(sources, list):
            self.sources = [
                s.to(self.device) if hasattr(s, 'to')
                else torch.tensor(s, device=self.device)
                for s in sources
            ]
        else:
            self.sources = (
                sources.to(self.device) if hasattr(sources, 'to')
                else torch.tensor(sources, device=self.device)
            )

        if isinstance(receivers, list):
            self.receivers = [
                [
                    r.to(self.device) if hasattr(r, 'to')
                    else torch.tensor(r, device=self.device)
                    for r in rec_list
                ]
                for rec_list in receivers
            ]
        else:
            self.receivers = (
                receivers.to(self.device) if hasattr(receivers, 'to')
                else torch.tensor(receivers, device=self.device)
            )

        self.n_sources = len(sources)
        if isinstance(receivers, list) and len(receivers) > 0:
            self.n_rcvrs = len(receivers[0])
        else:
            self.n_rcvrs = len(receivers) if hasattr(receivers, '__len__') else 1

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg):
        """Create a ``GMM_reco`` instance from a :class:`ReconstructConfig`.

        Parameters
        ----------
        cfg : gmm_ct.config.yaml_config.ReconstructConfig
            Parsed reconstruction configuration (typically loaded from YAML).

        Returns
        -------
        GMM_reco
        """
        device = torch.device(
            cfg.device
            if cfg.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
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
            N_traj_trials=cfg.reconstruction.n_trajectory_trials,
            n_omega_inits=cfg.reconstruction.n_omega_inits,
        )

    # ==================================================================
    # Main fitting pipeline
    # ==================================================================

    def fit(self, proj_data, t):
        """
        Fit GMM parameters via the 4-stage optimization pipeline.

        Parameters
        ----------
        proj_data : list of torch.Tensor
            Observed projection measurements.
        t : torch.Tensor
            Time vector.

        Returns
        -------
        dict
            Optimized parameter dictionary with keys
            ``'x0s'``, ``'v0s'``, ``'a0s'``, ``'alphas'``,
            ``'U_skews'``, ``'omegas'``.
        """
        self.t = (
            t.to(self.device) if hasattr(t, 'to')
            else torch.tensor(t, device=self.device)
        )
        self.proj_data = self.process_projections(self._to_device(proj_data))

        # Stage 1: Trajectory optimization
        soln_dict = self._stage_trajectory_optimization(t, proj_data)

        # Stage 2: Multi-start joint optimization (rotation + morphology)
        soln_dict, best_loss = self._stage_multistart_joint(soln_dict)

        # Stage 3: Fine grid search for omega refinement
        print(f"\n{'='*50}")
        print("Fine grid search around multi-start solution "
              "(±3 Hz, 0.1 Hz steps)...")
        print(f"{'='*50}")
        soln_dict = self._fine_grid_search_omega(
            soln_dict, best_loss, omega_range=3.0, omega_step=0.1,
        )

        # Stage 4: Final joint refinement
        print(f"\n{'='*50}")
        print("Final joint refinement...")
        print(f"{'='*50}")
        soln_dict = self._optimize_joint(soln_dict, max_iter=200)

        print(f"\n{'='*50}")
        print("Optimization complete!")
        print(f"{'='*50}")
        print(f"Final ω: "
              f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")

        return soln_dict

    # ==================================================================
    # Pipeline stages
    # ==================================================================

    def _stage_trajectory_optimization(self, t, proj_data):
        """Stage 1: Multi-start trajectory optimization to estimate v0."""
        print(f"\n{'='*50}")
        print(f"\n{'='*50}")
        print("Starting trajectory optimization...")
        print(f"{'='*50}")

        N_traj_trials = self.N_traj_trials or max(10, 2 * self.N)
        errors, results = [], []

        print(f"Running {N_traj_trials} trajectory multi-start trials")
        for n_trial in range(N_traj_trials):
            print(f"\nTrial {n_trial + 1}/{N_traj_trials}")

            self.theta_dict_init = self.initialize_parameters(t, proj_data)
            [v0_k.requires_grad_(True) for v0_k in self.theta_dict_init['v0s']]

            theta_tensor_init = self.map_from_dict_to_tensor(
                self.theta_dict_init, mode='trajectory',
            )
            res_trial = minimize(
                self._loss_trajectory, x0=theta_tensor_init, method='l-bfgs',
                tol=1e-8, options={'gtol': 1e-8, 'max_iter': 1500, 'disp': True},
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

        soln_dict["v0s"] = [v0_k.clone().detach() for v0_k in soln_dict["v0s"]]

        # Diagnostic plots (lazy import to avoid circular dependency)
        from ..visualization.diagnostics import (
            plot_trajectory_estimations,
            plot_heights_by_assignment,
            plot_raw_receiver_heights,
        )
        plot_trajectory_estimations(self, best_res)
        plot_raw_receiver_heights(self)
        plot_heights_by_assignment(self)

        soln_dict = self.refine_initial_velocities_via_newton_raphson(
            soln_dict, best_res,
        )

        soln_dict["omegas"] = [
            omega.clone().detach() for omega in self.theta_dict_init["omegas"]
        ]
        soln_dict["alphas"] = [
            alpha.clone().detach() for alpha in self.theta_dict_init["alphas"]
        ]

        # Initialize anisotropic U_skews aligned with velocity direction
        print(f"\nInitializing anisotropic Gaussians aligned with velocity...")
        soln_dict["U_skews"] = self.initialize_anisotropic_U_skews(
            soln_dict["v0s"],
        )
        print("✓ Anisotropic U_skews initialized.")

        return soln_dict

    def _stage_multistart_joint(self, soln_dict):
        """Stage 2: Multi-start joint optimization of rotation + morphology."""
        print(f"\n{'='*50}")
        print("Starting multi-start joint optimization...")
        print(f"{'='*50}")

        n_random_starts = self.n_omega_inits or 5
        print(f"Running {n_random_starts} random omega initializations")

        initial_alphas = [alpha.clone().detach() for alpha in soln_dict['alphas']]
        initial_U_skews = [U.clone().detach() for U in soln_dict['U_skews']]
        omega_min = self.omega_min - 0.01
        omega_max = self.omega_max + 0.01

        all_losses, all_results = [], []

        for trial_idx in range(n_random_starts):
            initial_omegas = [
                torch.tensor(
                    [np.random.uniform(omega_min, omega_max)],
                    dtype=torch.float64, device=self.device,
                )
                for _ in range(self.N)
            ]

            test_dict = {
                'alphas': [alpha.clone().requires_grad_(True)
                           for alpha in initial_alphas],
                'U_skews': [U.clone().requires_grad_(True)
                            for U in initial_U_skews],
                'omegas': [omega.requires_grad_(True)
                           for omega in initial_omegas],
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
                tol=1e-10, options={'gtol': 1e-10, 'max_iter': 1000,
                                    'disp': False},
            )

            result_dict = self.construct_soln_dict(res)
            final_loss = res.fun.item()
            all_losses.append(final_loss)
            all_results.append(result_dict)

            final_omegas = [omega.item() for omega in result_dict['omegas']]
            print(f"  Trial {trial_idx + 1}/{n_random_starts}: "
                  f"loss = {final_loss:.6e}, "
                  f"ω = {[f'{w:.3f}' for w in final_omegas]}")

        best_trial_idx = np.argmin(all_losses)
        best_result = all_results[best_trial_idx]
        best_loss = all_losses[best_trial_idx]

        soln_dict['alphas'] = [
            alpha.clone().detach() for alpha in best_result['alphas']
        ]
        soln_dict['omegas'] = [
            omega.clone().detach() for omega in best_result['omegas']
        ]
        soln_dict['U_skews'] = [
            U.clone().detach() for U in best_result['U_skews']
        ]

        print(f"\n{'='*50}")
        print(f"Multi-start complete! Best trial: {best_trial_idx + 1}")
        print(f"Best loss: {best_loss:.6e}")
        print(f"Best ω: "
              f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")
        print(f"{'='*50}")

        return soln_dict, best_loss

    # ==================================================================
    # Optimization helpers
    # ==================================================================

    def _optimize_joint(self, soln_dict, max_iter=300):
        """
        Joint optimization of omega, U_skew, alpha (and optionally v0).

        Parameters
        ----------
        soln_dict : dict
            Current solution with all parameters.
        max_iter : int
            Maximum L-BFGS iterations.

        Returns
        -------
        dict
            Updated solution with refined parameters.
        """
        print("\n  Optimizing omega, U_skew, alpha jointly...")

        soln_dict["alphas"] = [
            a.requires_grad_(True) for a in soln_dict['alphas']
        ]
        soln_dict["U_skews"] = [
            U.requires_grad_(True) for U in soln_dict['U_skews']
        ]
        soln_dict["omegas"] = [
            w.requires_grad_(True) for w in soln_dict['omegas']
        ]

        self.theta_fixed = {
            'x0s': [x0.clone() for x0 in soln_dict['x0s']],
            'a0s': [a0.clone() for a0 in soln_dict['a0s']],
        }
        if not soln_dict['v0s'][0].requires_grad:
            self.theta_fixed['v0s'] = [
                v0.clone() for v0 in soln_dict['v0s']
            ]

        theta_tensor = self.map_from_dict_to_tensor(soln_dict, mode='joint')
        print(f"  Optimizing {theta_tensor.numel()} parameters "
              f"(including omega)...")

        res = minimize(
            self._loss_joint, x0=theta_tensor, method='l-bfgs',
            tol=1e-8, options={'gtol': 1e-8, 'max_iter': max_iter,
                               'disp': False},
        )

        result_dict = self.construct_soln_dict(res)
        soln_dict['alphas'] = [
            alpha.clone().detach() for alpha in result_dict['alphas']
        ]
        soln_dict['U_skews'] = [
            U.clone().detach() for U in result_dict['U_skews']
        ]
        soln_dict['omegas'] = [
            omega.clone().detach() for omega in result_dict['omegas']
        ]

        print(f"  Joint optimization: loss = {res.fun.item():.6e} "
              f"({res.nit} iterations)")
        print(f"  Refined ω: "
              f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")

        return soln_dict

    def _fine_grid_search_omega(self, soln_dict, current_loss,
                                omega_range=3.0, omega_step=0.1):
        """
        Fine grid search around current omega estimate.

        Evaluates projection loss on a grid ``±omega_range`` Hz with
        ``omega_step`` spacing, keeping morphology (alpha, U_skew) fixed.

        Parameters
        ----------
        soln_dict : dict
            Current solution.
        current_loss : float
            Current projection loss (baseline).
        omega_range : float
            Half-width of search window (Hz).
        omega_step : float
            Grid spacing (Hz).

        Returns
        -------
        dict
            Solution with refined omega if improvement was found.
        """
        print(f"\n  Searching ±{omega_range} Hz with {omega_step} Hz "
              f"steps...")

        self.theta_fixed = {
            'x0s': soln_dict['x0s'],
            'v0s': soln_dict['v0s'],
            'a0s': soln_dict['a0s'],
        }

        best_loss = current_loss
        best_omegas = [omega.clone() for omega in soln_dict['omegas']]

        for k in range(self.N):
            omega_current = soln_dict['omegas'][k].item()
            omega_lo = omega_current - omega_range
            omega_hi = omega_current + omega_range
            n_points = int((omega_hi - omega_lo) / omega_step) + 1
            omega_candidates = np.linspace(omega_lo, omega_hi, n_points)

            losses = []
            for omega_test in omega_candidates:
                test_dict = {
                    'alphas': [a.clone().requires_grad_(False)
                               for a in soln_dict['alphas']],
                    'U_skews': [U.clone().requires_grad_(False)
                                for U in soln_dict['U_skews']],
                    'omegas': [w.clone().requires_grad_(False)
                               for w in soln_dict['omegas']],
                    'x0s': soln_dict['x0s'],
                    'v0s': soln_dict['v0s'],
                    'a0s': soln_dict['a0s'],
                }
                test_dict['omegas'][k] = torch.tensor(
                    [omega_test], dtype=torch.float64, device=self.device,
                )
                theta_tensor = self.map_from_dict_to_tensor(
                    test_dict, mode='joint',
                )
                losses.append(self._loss_joint(theta_tensor).item())

            min_idx = np.argmin(losses)
            min_loss = losses[min_idx]
            best_omega_k = omega_candidates[min_idx]

            if min_loss < best_loss:
                improvement = best_loss - min_loss
                print(f"  Gaussian {k}: ω {omega_current:.4f} → "
                      f"{best_omega_k:.4f} Hz "
                      f"(Δloss = {improvement:.6e})")
                best_omegas[k] = torch.tensor(
                    [best_omega_k], dtype=torch.float64, device=self.device,
                )
                best_loss = min_loss
            else:
                print(f"  Gaussian {k}: No improvement found "
                      f"(keeping ω = {omega_current:.4f} Hz)")

        soln_dict['omegas'] = [
            omega.clone().detach() for omega in best_omegas
        ]

        if best_loss < current_loss:
            improvement = current_loss - best_loss
            print(f"\n  ✓ Grid search improved loss: {current_loss:.6e} → "
                  f"{best_loss:.6e} (Δ = {improvement:.6e})")
        else:
            print(f"\n  Grid search: No improvement found")

        return soln_dict

    # ==================================================================
    # Loss functions
    # ==================================================================

    def _loss_trajectory(self, theta_tensor):
        """
        Loss for trajectory optimization (Phase 1).

        L2 distance between predicted and observed peak heights, using the
        Hungarian algorithm for optimal assignment.
        """
        theta_dict = self.map_from_tensor_to_dict(
            theta_tensor, mode='trajectory',
        )
        self.t_observable = self.t[self.peak_data.observable_indices]
        r_maxs_list = self.map_velocities_to_maximising_receivers(theta_dict)
        self._assign_peaks_hungarian(r_maxs_list)
        return self._compute_trajectory_loss(r_maxs_list)

    def _loss_joint(self, theta_tensor):
        """
        Loss for joint rotation + morphology optimization (Phase 2).

        SmoothL1 loss between observed and simulated projections.
        """
        loss_func = nn.SmoothL1Loss(beta=0.3)
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

    # ==================================================================
    # Peak assignment
    # ==================================================================

    def _assign_peaks_hungarian(self, r_maxs_list):
        """
        Assign detected peaks to trajectories using the Hungarian algorithm.

        Parameters
        ----------
        r_maxs_list : list of torch.Tensor
            Predicted receiver positions per Gaussian, shape ``(n_times, 2)``.
        """
        self.assigned_curve_data = [[] for _ in range(self.N)]
        heights_dict = self.peak_data.get_heights_dict_non_empty()

        for time_idx, time_val in enumerate(self.t_observable):
            observed_heights = heights_dict[time_val.item()]

            dist_matrix = torch.zeros(
                len(observed_heights), self.N,
                dtype=torch.float64, device=self.device,
            )

            for height_idx, height in enumerate(observed_heights):
                for gaussian_idx in range(self.N):
                    predicted = r_maxs_list[gaussian_idx][time_idx, 1]
                    distance = torch.abs(predicted - height)
                    if torch.isnan(distance) or torch.isinf(distance):
                        dist_matrix[height_idx, gaussian_idx] = 1e10
                    else:
                        dist_matrix[height_idx, gaussian_idx] = distance.item()

            row_indices, col_indices = linear_sum_assignment(
                dist_matrix.cpu().numpy(),
            )
            for height_idx, gaussian_idx in zip(row_indices, col_indices):
                self.assigned_curve_data[gaussian_idx].append(
                    (time_idx, observed_heights[height_idx])
                )

    def _compute_trajectory_loss(self, r_maxs_list):
        """Compute L2 loss between predicted and assigned receiver heights."""
        loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        for k in range(self.N):
            assignments_k = self.assigned_curve_data[k]
            if not assignments_k:
                continue

            time_indices = [item[0] for item in assignments_k]
            observed_heights = torch.stack([item[1] for item in assignments_k])
            predicted_heights = r_maxs_list[k][time_indices, 1]

            # loss += torch.norm(predicted_heights - observed_heights, p=2)
            loss += torch.norm(predicted_heights - observed_heights, p=1)

        return loss

    # ==================================================================
    # Parameter serialization (dict <-> tensor)
    # ==================================================================

    def map_from_dict_to_tensor(self, theta_dict, mode='trajectory'):
        """
        Convert parameter dictionary to flattened tensor for optimization.

        Parameters
        ----------
        theta_dict : dict
            Parameter dictionary.
        mode : str, {'trajectory', 'joint', 'joint_with_v0'}
            Optimization phase — determines which parameters are packed.

        Returns
        -------
        torch.Tensor
            Flattened parameter tensor.
        """
        d, N = self.d, self.N
        tensor_rows = []

        if mode == "trajectory":
            # Fix all non-v0 parameters
            self.theta_fixed = {
                'alphas': [alpha.clone().requires_grad_(True)
                           for alpha in theta_dict['alphas']],
                'U_skews': [U.clone().requires_grad_(True)
                            for U in theta_dict['U_skews']],
                'omegas': [omega.clone().requires_grad_(True)
                           for omega in theta_dict['omegas']],
                'x0s': [x0.clone().requires_grad_(True)
                        for x0 in theta_dict['x0s']],
                'a0s': [a0.clone().requires_grad_(True)
                        for a0 in theta_dict['a0s']],
            }

            for k in range(N):
                v0_k = theta_dict['v0s'][k]
                # Log transform first component for stability
                v0_k_0 = torch.log(torch.abs(v0_k[0]) + 1e-8)
                tensor_rows.append(torch.stack([v0_k_0, v0_k[1]]))

        elif mode in ("joint", "joint_with_v0"):
            # Initialize theta_fixed if not already set
            if not hasattr(self, 'theta_fixed') or self.theta_fixed is None:
                self.theta_fixed = {
                    'x0s': [x0.clone() for x0 in theta_dict['x0s']],
                    'a0s': [a0.clone() for a0 in theta_dict['a0s']],
                }
                if mode == "joint":
                    self.theta_fixed['v0s'] = [
                        v0.clone() for v0 in theta_dict['v0s']
                    ]

            for k in range(N):
                row_parts = []

                # Optional v0 (joint_with_v0 mode)
                if mode == "joint_with_v0":
                    v0_k = theta_dict['v0s'][k]
                    row_parts.append(
                        torch.log(torch.abs(v0_k[0]) + 1e-8).reshape(-1)
                    )
                    row_parts.append(v0_k[1].reshape(-1))

                # Alpha (log-transformed)
                alpha_k = theta_dict["alphas"][k].clone()
                row_parts.append(torch.log(alpha_k).reshape(-1))

                # U_skew: log-transform diagonal, pack upper triangle
                U_skew_copy = theta_dict["U_skews"][k].clone()
                EPS = 1e-8
                diag_clamped = torch.clamp(
                    torch.diagonal(U_skew_copy), min=EPS,
                )
                diag_logged = torch.log(diag_clamped)

                U_skew_no_diag = U_skew_copy - torch.diag(
                    torch.diagonal(U_skew_copy)
                )
                U_skew_with_logged_diag = U_skew_no_diag + torch.diag(
                    diag_logged
                )

                triu_idx = torch.triu_indices(d, d, device=U_skew_copy.device)
                U_skew_vals = U_skew_with_logged_diag[
                    triu_idx[0], triu_idx[1]
                ].reshape(-1)
                row_parts.append(U_skew_vals)

                # Omega (if not fixed)
                theta_fixed_keys = list(
                    getattr(self, 'theta_fixed', {}).keys()
                )
                if ('omegas' in theta_dict
                        and 'omegas' not in theta_fixed_keys):
                    row_parts.append(
                        theta_dict["omegas"][k].clone().reshape(-1)
                    )

                tensor_rows.append(torch.cat(row_parts))

        if len(tensor_rows) == 1:
            return tensor_rows[0]
        return torch.stack(tensor_rows)

    def map_from_tensor_to_dict(self, theta_tensor, mode='trajectory'):
        """
        Convert flattened parameter tensor back to dictionary format.

        Inverse of :meth:`map_from_dict_to_tensor`.

        Parameters
        ----------
        theta_tensor : torch.Tensor
            Flattened parameter tensor.
        mode : str, {'trajectory', 'joint', 'joint_with_v0'}
            Optimization phase.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        d, N = self.d, self.N
        theta_dict = {}

        if mode == "trajectory":
            v0s = []
            if N == 1:
                theta_tensor = theta_tensor.squeeze(0)
                v0s.append(torch.stack([
                    torch.exp(theta_tensor[0]), theta_tensor[1],
                ]))
            else:
                for k in range(N):
                    v0s.append(torch.stack([
                        torch.exp(theta_tensor[k, 0]), theta_tensor[k, 1],
                    ]))
            theta_dict['v0s'] = v0s

        elif mode in ("joint", "joint_with_v0"):
            alphas, U_skews, omegas, v0s = [], [], [], []
            n_U_params = d * (d + 1) // 2

            rows = (
                [theta_tensor[k] for k in range(N)] if N > 1
                else [theta_tensor]
            )

            for row_k in rows:
                idx = 0

                # v0 (optional)
                if mode == "joint_with_v0":
                    v0_0 = torch.exp(row_k[idx])
                    v0_1 = row_k[idx + 1]
                    v0s.append(torch.stack([v0_0, v0_1]))
                    idx += 2

                # Alpha (clamped exp)
                alpha_logged_clamped = torch.clamp(
                    row_k[idx], min=-5, max=5,
                )
                alphas.append(
                    torch.exp(alpha_logged_clamped).unsqueeze(0)
                )
                idx += 1

                # U_skew
                U_skew_vals = row_k[idx: idx + n_U_params]
                if U_skew_vals.numel() != n_U_params:
                    raise ValueError(
                        f"Expected {n_U_params} U_skew values, "
                        f"got {U_skew_vals.numel()}. "
                        f"Row has {row_k.numel()} elements, "
                        f"idx={idx}, mode={mode}"
                    )
                U_skew = torch.zeros(
                    (d, d), dtype=theta_tensor.dtype,
                    device=theta_tensor.device,
                )
                triu_indices = torch.triu_indices(d, d)
                U_skew[triu_indices[0], triu_indices[1]] = U_skew_vals

                diag_mask = torch.eye(
                    d, dtype=torch.bool, device=theta_tensor.device,
                )
                diag_clamped = torch.clamp(
                    U_skew[diag_mask], min=-5, max=10,
                )
                U_skew_final = U_skew.clone()
                U_skew_final[diag_mask] = torch.exp(diag_clamped)
                U_skews.append(U_skew_final)
                idx += n_U_params

                # Omega (if present in tensor)
                if len(row_k) > idx:
                    omega = row_k[idx]
                    omegas.append(
                        omega.unsqueeze(0) if omega.dim() == 0 else omega
                    )

            theta_dict['alphas'] = alphas
            theta_dict['U_skews'] = U_skews
            if omegas:
                theta_dict['omegas'] = omegas
            if mode == "joint_with_v0" and v0s:
                theta_dict['v0s'] = v0s

        return theta_dict

    def construct_soln_dict(self, res):
        """
        Construct solution dictionary from an optimization result.

        Automatically determines the mode based on tensor dimensions
        and the contents of ``theta_fixed``.
        """
        theta_tensor = res.x

        tensor_size = (
            theta_tensor.numel() if len(theta_tensor.shape) == 1
            else theta_tensor.shape[0] * theta_tensor.shape[1]
        )
        params_per_gaussian = (
            tensor_size // self.N if self.N > 0 else tensor_size
        )
        has_v0_fixed = (
            hasattr(self, 'theta_fixed') and 'v0s' in self.theta_fixed
        )

        if params_per_gaussian == 2:
            mode = 'trajectory'
        elif (params_per_gaussian == 4
              and hasattr(self, 'theta_fixed')
              and 'omegas' in self.theta_fixed):
            mode = 'joint'
        elif (params_per_gaussian == 7
              or (params_per_gaussian >= 5 and not has_v0_fixed)):
            mode = 'joint_with_v0'
        elif params_per_gaussian >= 4:
            mode = 'joint'
        else:
            raise ValueError(
                f"Cannot determine mode: "
                f"{params_per_gaussian} params per Gaussian"
            )

        soln_dict = self.map_from_tensor_to_dict(theta_tensor, mode=mode)
        for key, value in self.theta_fixed.items():
            if key not in soln_dict:
                soln_dict[key] = value.copy()
        return soln_dict

    # ==================================================================
    # Trajectory refinement
    # ==================================================================

    def map_velocities_to_maximising_receivers(self, theta_dict):
        """
        Map trajectory parameters to maximizing receiver positions over time.

        Given ``mu_k(t)`` and source ``s``, computes the receiver ``r`` that
        maximises the projection using
        ``r = s + lambda * (s - c)`` where
        ``lambda = (r_0 - s_0) / (s_0 - c_0)``.

        Returns
        -------
        list of torch.Tensor
            One ``(n_times, d)`` tensor per Gaussian.
        """
        r_maxs_list = []
        s = self.sources[0]
        r0 = self.receivers[0][0][0]
        EPS = 1e-10

        for k in range(self.N):
            v0_k = theta_dict['v0s'][k]
            x0_k = self.theta_fixed['x0s'][k]
            a0_k = self.theta_fixed['a0s'][k]

            r_maxs_k = []
            for t_n in self.t_observable:
                c_k = x0_k + v0_k * t_n + 0.5 * a0_k * t_n ** 2
                numerator = r0 - s[0]
                denominator = s[0] - c_k[0]

                denominator_safe = torch.where(
                    torch.abs(denominator) < EPS,
                    torch.sign(denominator) * EPS + (denominator == 0) * EPS,
                    denominator,
                )
                lambda_t = numerator / denominator_safe
                r_maxs_k.append(s + lambda_t * (s - c_k))

            r_maxs_list.append(torch.stack(r_maxs_k))

        return r_maxs_list

    def refine_initial_velocities_via_newton_raphson(self, soln_dict, res):
        """
        Refine v0 via Newton-Raphson with optimal peak assignment.

        1. Assigns detected peaks to predicted trajectories (nearest-neighbour).
        2. Optimises v0 to minimise deviation from assigned peaks.

        Parameters
        ----------
        soln_dict : dict
            Current solution with v0s, x0s, a0s.
        res : OptimizeResult
            Trajectory optimization result.

        Returns
        -------
        dict
            Updated solution with refined v0s.
        """
        r_maxs_list = self.map_velocities_to_maximising_receivers(
            self.map_from_tensor_to_dict(res.x),
        )
        self._assign_peaks_to_trajectories(r_maxs_list)

        # Build legacy assignment format for diagnostic plots
        self.assigned_curve_data = []
        for gaussian_idx in range(self.N):
            times, heights = self.peak_data.get_assignment_data(gaussian_idx)
            data_k = []
            for time_val, height in zip(times, heights):
                time_idx = torch.where(self.t_observable == time_val)[0]
                if len(time_idx) > 0:
                    data_k.append((
                        time_idx[0].item(),
                        torch.tensor(height, device=self.device),
                    ))
            self.assigned_curve_data.append(data_k)

        self.assigned_peak_values = self.peak_data.assigned_values

        from ..visualization.diagnostics import plot_heights_by_assignment
        plot_heights_by_assignment(self)

        v0s_refined = self._newton_raphson_refinement(soln_dict)
        soln_dict["v0s"] = [v0.clone().detach() for v0 in v0s_refined]
        return soln_dict

    def _assign_peaks_to_trajectories(self, r_maxs_list):
        """Assign peaks to trajectories using nearest-neighbour matching."""
        proj_data = self.proj_data

        for time_idx, detected_heights in enumerate(
            self.peak_data.get_heights_sorted_by_time()
        ):
            for height in detected_heights:
                distances = [
                    torch.abs(trajectory[time_idx, 1] - height).item()
                    for trajectory in r_maxs_list
                ]
                gaussian_idx = np.argmin(distances)

                receiver_heights = torch.tensor(
                    [r[1].item() for r in self.receivers[0]],
                    dtype=torch.float64, device=self.device,
                )
                receiver_idx = torch.argmin(
                    torch.abs(receiver_heights - height)
                ).item()
                peak_value = proj_data[time_idx, receiver_idx].item()

                self.peak_data.add_optimal_assignment(
                    gaussian_idx,
                    self.t_observable[time_idx].item(),
                    height,
                    peak_value,
                )

    def _newton_raphson_refinement(self, soln_dict):
        """Refine v0 via Newton-Raphson on assigned peak data."""
        v0s_refined = []
        r0 = self.receivers[0][0][0]

        for gaussian_idx in range(self.N):
            times, heights = self.peak_data.get_assignment_data(gaussian_idx)

            t_obs = torch.tensor(
                times, dtype=torch.float64, device=self.device,
            )
            receivers = [
                torch.tensor(
                    [r0, h], dtype=torch.float64, device=self.device,
                )
                for h in heights
            ]

            v0_k_refined = NewtonRaphsonLBFGS(
                self.isotropic_derivative_function_over_all_times,
                soln_dict['v0s'][gaussian_idx],
                t_obs, receivers, self.sources[0],
                soln_dict['x0s'][gaussian_idx],
                soln_dict['a0s'][gaussian_idx],
            )
            v0s_refined.append(v0_k_refined.requires_grad_(True))

        return v0s_refined

    # ==================================================================
    # Isotropic derivative (used by Newton-Raphson refinement)
    # ==================================================================

    def isotropic_derivative_function(self, v0, *args):
        """Isotropic projection derivative for root-finding."""
        t_n, r, s, x0, a0 = args

        r1, r2 = r[0], r[1]
        s1, s2 = s[0], s[1]
        d1, d2 = r1 - s1, r2 - s2
        norm_n_sq = d1 ** 2 + d2 ** 2

        c_k = s - x0 - v0 * t_n - 0.5 * a0 * t_n ** 2
        h_k = d1 * c_k[0] - s2 * c_k[1]
        R_k_l = 2 * norm_n_sq * c_k[1] * (c_k[1] * r2 + h_k)
        R_k_r = -2 * d2 * (c_k[1] * r2 + h_k) ** 2

        return (R_k_l + R_k_r) / norm_n_sq ** 2

    def isotropic_derivative_function_over_all_times(self, v0, *args):
        """Sum of absolute isotropic derivatives across all time points."""
        t, r, s, x0, a0 = args
        R_all = torch.zeros(1, dtype=torch.float64, device=self.device)
        for n, t_n in enumerate(t):
            R_all += torch.abs(
                self.isotropic_derivative_function(v0, t_n, r[n], s, x0, a0)
            )
        return R_all

    # ==================================================================
    # Utilities
    # ==================================================================

    def _to_device(self, obj):
        """Move tensor or nested structure of tensors to the correct device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, list):
            return [self._to_device(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._to_device(value) for key, value in obj.items()}
        return obj
