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
        Number of random multi-start trials for omega search (hard cap when
        ``omega_sup_threshold`` is active; fixed count otherwise).
    omega_sup_threshold : float, optional
        Stop the joint multi-start loop early when the supremum projection
        error ``max_{t,r} |sim(t,r) − obs(t,r)|`` drops below this value.
        If None (default), runs exactly ``n_omega_inits`` trials.
    omega_max_trials : int, optional
        Hard upper bound on trials when ``omega_sup_threshold`` is set.
        Defaults to ``n_omega_inits`` when None.
    """

    def __init__(self, d, N, sources, receivers, x0s, a0s,
                 omega_min, omega_max, device=None, output_dir=None,
                 N_traj_trials=None, n_omega_inits=None,
                 omega_sup_threshold=None, omega_max_trials=None):
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
        self.omega_sup_threshold = omega_sup_threshold
        self.omega_max_trials = omega_max_trials

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
            omega_sup_threshold=cfg.reconstruction.omega_sup_threshold,
            omega_max_trials=cfg.reconstruction.omega_max_trials,
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

        # Stage 1.5: ω initialization via model-fit grid search
        soln_dict = self._stage_omega_initialization(soln_dict)

        # Stage 1.5b: α initialization via non-negative least squares
        soln_dict = self._stage_alpha_initialization(soln_dict)

        # Stage 2: Multi-start joint optimization (ω + morphology)
        soln_dict, best_sup_err = self._stage_multistart_joint(soln_dict, warm_start=True)

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

        N_traj_trials = self.N_traj_trials or max(20, 2 * self.N)
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
            plot_assignment_quality,
            plot_gmm_and_projections,
            plot_trajectory_fitting,
        )
        plot_trajectory_estimations(self, best_res)
        plot_raw_receiver_heights(self)
        plot_heights_by_assignment(self)
        plot_assignment_quality(self, best_res)
        plot_gmm_and_projections(
            self, best_res,
            theta_true=getattr(self, 'theta_true', None),
        )
        plot_trajectory_fitting(self, best_res)

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

    def _stage_omega_initialization(self, soln_dict):
        """
        Stage 1.5: Per-Gaussian ω initialisation via residual-sinogram grid
        search.

        For each Gaussian k and each rotation plane (i, j):
          1. Form the residual sinogram by subtracting all other Gaussians'
             estimated contributions from the full observed sinogram.
          2. Sweep a uniform grid of ω candidates over [ω_min, ω_max],
             evaluating the forward model for Gaussian k alone at each
             candidate and measuring ‖p_resid − p_k(ω)‖₂.
          3. Set ω̂_k to the candidate that minimises the residual norm.

        This approach is dimension-agnostic: in d dimensions each Gaussian
        carries C(d, 2) angular velocities (one per rotation-plane).  The
        search is run sequentially over rotation planes, keeping all other
        planes' omegas fixed while sweeping the current one.

        Parameters
        ----------
        soln_dict : dict
            Current solution dict (must contain x0s, v0s, a0s, U_skews,
            alphas, omegas after Stage 1).

        Returns
        -------
        dict
            Updated soln_dict with 'omegas' replaced by grid-search estimates.
        """
        import math

        n_planes = math.comb(self.d, 2)
        n_grid   = 200  # candidates per plane

        print(f"\n{'='*50}")
        print("Stage 1.5: Residual-sinogram ω grid search...")
        print(f"  {n_planes} rotation plane(s), {n_grid} candidates each")

        theta_true = getattr(self, 'theta_true', None)
        if theta_true is not None and 'omegas' in theta_true:
            print("  True omegas:")
            for k, omega_true_k in enumerate(theta_true['omegas']):
                omega_true_str = ', '.join(f'{w.item():.4f}' for w in omega_true_k.flatten())
                print(f"    Gaussian {k}: ω_true = [{omega_true_str}] Hz")

        print(f"{'='*50}")

        proj_obs = self.proj_data          # (T, R) — full observed sinogram
        t        = self.t

        omega_candidates = torch.linspace(
            self.omega_min, self.omega_max, n_grid,
            dtype=torch.float64, device=self.device,
        )

        for k in range(self.N):
            # --- Residual: observed minus all other Gaussians ---
            bg_dict = {key: list(vals) for key, vals in soln_dict.items()}
            bg_dict['alphas'] = [
                torch.zeros(1, dtype=torch.float64, device=self.device)
                if j == k else soln_dict['alphas'][j].clone()
                for j in range(self.N)
            ]
            with torch.no_grad():
                proj_resid = (
                    proj_obs - self.process_projections(
                        self.generate_projections(t, bg_dict)
                    )
                )

            # --- Per-plane 1-D grid search ---
            # Work on a (n_planes,) tensor throughout to avoid shape mismatches.
            omega_k = soln_dict['omegas'][k].clone()   # shape (n_planes,)

            for plane_idx in range(n_planes):
                best_loss = float('inf')
                best_val  = omega_k[plane_idx].clone()

                for omega_val in omega_candidates:
                    # Build a clean (n_planes,) test omega for Gaussian k
                    test_omega_k = omega_k.clone()
                    test_omega_k[plane_idx] = omega_val

                    # Shallow-copy soln_dict; replace omegas list and alphas list
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
                        best_val  = omega_val.clone()

                omega_k[plane_idx] = best_val

            soln_dict['omegas'][k] = omega_k

            omega_str = ', '.join(f'{w.item():.4f}' for w in soln_dict['omegas'][k])
            if theta_true is not None and 'omegas' in theta_true:
                omega_true_k = theta_true['omegas'][k]
                omega_true_str = ', '.join(f'{w.item():.4f}' for w in omega_true_k.flatten())
                print(f"  Gaussian {k}: ω_est = [{omega_str}] Hz  |  ω_true = [{omega_true_str}] Hz")
            else:
                print(f"  Gaussian {k}: ω = [{omega_str}] Hz")

        print(f"{'='*50}")
        return soln_dict

    def _stage_alpha_initialization(self, soln_dict):
        """
        Stage 1.5b: Initialise attenuation coefficients via non-negative
        least squares (NNLS).

        With trajectories, shapes, and ω fixed from Stages 1 and 1.5, the
        forward model is linear in {α_k}:

            p_obs(t_i, r_j) ≈ Σ_k  α_k · φ_k(t_i, r_j)

        where φ_k is the unit-α projection of Gaussian k.  The optimal
        attenuation vector is therefore the solution to:

            min_{α ≥ 0}  ‖Φ α − p_obs‖₂²

        which is solved in closed form by SciPy's NNLS routine.  This
        replaces the random initialisation in [10, 15] with an estimate
        that is consistent with the observed data given the current shapes
        and trajectories.

        Parameters
        ----------
        soln_dict : dict
            Current solution dict with v0s, x0s, a0s, omegas, U_skews
            already set (output of Stage 1.5).

        Returns
        -------
        dict
            Updated soln_dict with 'alphas' replaced by NNLS estimates.
        """
        print(f"\n{'='*50}")
        print("Stage 1.5b: NNLS alpha initialisation...")
        print(f"{'='*50}")

        t_obs = self.t[self.peak_data.observable_indices]
        p_obs = self.proj_data[self.peak_data.observable_indices]   # (T_obs, R)
        T_obs, R = p_obs.shape

        # Build basis matrix Phi[:, k] = unit-alpha projection of Gaussian k
        Phi = torch.zeros(
            T_obs * R, self.N, dtype=torch.float64, device=self.device
        )
        with torch.no_grad():
            for k in range(self.N):
                unit_dict = {
                    'alphas': [
                        torch.ones(1, dtype=torch.float64, device=self.device)
                        if kk == k
                        else torch.zeros(1, dtype=torch.float64, device=self.device)
                        for kk in range(self.N)
                    ],
                    'U_skews': soln_dict['U_skews'],
                    'omegas':  soln_dict['omegas'],
                    'x0s':     soln_dict['x0s'],
                    'v0s':     soln_dict['v0s'],
                    'a0s':     soln_dict['a0s'],
                }
                proj_k = self.generate_projections(t_obs, unit_dict)
                Phi[:, k] = self.process_projections(proj_k).reshape(-1)

        # Guard: if the forward model produced non-finite values (e.g. degenerate
        # U_skew), skip alpha init and keep current values.
        if not torch.isfinite(Phi).all():
            print("  WARNING: non-finite values in basis matrix; skipping alpha init.")
            return soln_dict

        # Pure-PyTorch least-squares + non-negative clamp.
        # Using torch.linalg.lstsq avoids any scipy/BLAS threading conflict
        # with PyTorch's own LAPACK context (which caused segfaults on macOS).
        p_vec_t = p_obs.reshape(-1, 1)          # (T_obs*R, 1)
        result = torch.linalg.lstsq(Phi, p_vec_t, driver='gelsd')
        alpha_hat = result.solution.squeeze(1).clamp(min=0.0)  # enforce α ≥ 0
        residual = torch.norm(Phi @ alpha_hat - p_obs.reshape(-1)).item()

        soln_dict['alphas'] = [
            alpha_hat[k].reshape(1).detach().clone()
            for k in range(self.N)
        ]

        theta_true = getattr(self, 'theta_true', None)
        if theta_true is not None and 'alphas' in theta_true:
            alpha_true_str = ', '.join(
                f'{theta_true["alphas"][k].item():.3f}' for k in range(self.N)
            )
            alpha_est_str = ', '.join(f'{alpha_hat[k].item():.3f}' for k in range(self.N))
            print(f"  α_est  = [{alpha_est_str}]")
            print(f"  α_true = [{alpha_true_str}]")
        else:
            print(f"  α = {[f'{alpha_hat[k].item():.3f}' for k in range(self.N)]}")
        print(f"  NNLS residual ‖Φα − p_obs‖₂ = {residual:.4e}")
        print(f"{'='*50}")
        return soln_dict

    def _sup_projection_error(self, result_dict):
        """
        Compute the supremum projection error for a candidate solution.

        Returns ``max_{t,r} |sim_proj(t,r) - obs_proj(t,r)|`` evaluated at
        the observable time indices, using the current ``theta_fixed`` context.

        Parameters
        ----------
        result_dict : dict
            Parameter dictionary returned by :meth:`construct_soln_dict`.

        Returns
        -------
        float
            Supremum absolute projection error.
        """
        with torch.no_grad():
            sim_projs = self.generate_projections(self.t_observable, result_dict)
            sim_proc  = self.process_projections(sim_projs)
            obs_proc  = self.proj_data[self.peak_data.observable_indices]
            return torch.max(torch.abs(sim_proc - obs_proc)).item()

    def _stage_multistart_joint(self, soln_dict, warm_start=False):
        """Stage 2: Multi-start joint optimization of rotation + morphology.

        Parameters
        ----------
        soln_dict : dict
            Current solution (must contain omegas, alphas, U_skews, x0s,
            v0s, a0s).
        warm_start : bool, optional
            If True the first trial uses the omegas already stored in
            ``soln_dict`` (e.g. from model-fit initialization) instead of a
            random draw.  Subsequent trials (if any) are still randomized.

        Stopping behaviour
        ------------------
        * If ``self.omega_sup_threshold`` is set: trials continue until the
          supremum projection error ``max_{t,r} |sim - obs|`` drops below the
          threshold, up to a hard cap of ``self.omega_max_trials`` (defaults to
          ``self.n_omega_inits`` when unset).  The best result found so far is
          kept at each step, so the loop always returns something useful.
        * Otherwise: runs exactly ``self.n_omega_inits`` trials (fixed-count
          behaviour).
        """
        print(f"\n{'='*50}")
        print("Starting multi-start joint optimization...")
        print(f"{'='*50}")

        use_threshold = self.omega_sup_threshold is not None
        n_fixed = self.n_omega_inits or 5

        if use_threshold:
            cap = self.omega_max_trials or n_fixed
            print(f"Threshold mode: sup error < {self.omega_sup_threshold:.3e}, "
                  f"cap = {cap} trials")
        else:
            cap = n_fixed
            print(f"Fixed mode: {cap} trials")

        initial_alphas  = [a.clone().detach() for a in soln_dict['alphas']]
        initial_U_skews = [U.clone().detach() for U in soln_dict['U_skews']]
        omega_min = self.omega_min - 0.01
        omega_max = self.omega_max + 0.01

        all_losses, all_results, all_sup_errors = [], [], []
        threshold_met = False

        for trial_idx in range(cap):
            if warm_start and trial_idx == 0:
                initial_omegas = [
                    omega.clone().detach() for omega in soln_dict['omegas']
                ]
            else:
                initial_omegas = [
                    torch.tensor(
                        [np.random.uniform(omega_min, omega_max)],
                        dtype=torch.float64, device=self.device,
                    )
                    for _ in range(self.N)
                ]

            test_dict = {
                'alphas':  [a.clone().requires_grad_(True)
                            for a in initial_alphas],
                'U_skews': [U.clone().requires_grad_(True)
                            for U in initial_U_skews],
                'omegas':  [omega.requires_grad_(True)
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

            result_dict  = self.construct_soln_dict(res)
            final_loss   = res.fun.item()
            sup_err      = self._sup_projection_error(result_dict)

            all_losses.append(final_loss)
            all_results.append(result_dict)
            all_sup_errors.append(sup_err)

            final_omegas = [omega.item() for omega in result_dict['omegas']]
            sup_str = f", sup err = {sup_err:.3e}" if use_threshold else ""
            print(f"  Trial {trial_idx + 1}/{cap}: "
                  f"loss = {final_loss:.6e}"
                  f"{sup_str}, "
                  f"ω = {[f'{w:.3f}' for w in final_omegas]}")

            if use_threshold and sup_err < self.omega_sup_threshold:
                print(f"  ✓ Threshold met at trial {trial_idx + 1} "
                      f"(sup err {sup_err:.3e} < {self.omega_sup_threshold:.3e})")
                threshold_met = True
                break

        best_trial_idx = int(np.argmin(all_sup_errors))
        best_result    = all_results[best_trial_idx]
        best_loss      = all_losses[best_trial_idx]
        best_sup_err   = all_sup_errors[best_trial_idx]

        if use_threshold and not threshold_met:
            print(f"\n  ⚠ Threshold not met after {cap} trials "
                  f"(best sup err = {best_sup_err:.3e})")

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
        print(f"Best sup error: {best_sup_err:.3e}")
        print(f"Best loss: {best_loss:.6e}")
        print(f"Best ω: "
              f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")
        print(f"{'='*50}")

        return soln_dict, best_sup_err

    # ==================================================================
    # Optimization helpers
    # ==================================================================

    def _eval_joint_loss(self, soln_dict):
        """Evaluate the joint projection loss at current parameters."""
        self.theta_fixed = {
            'x0s': [x0.clone() for x0 in soln_dict['x0s']],
            'v0s': [v0.clone() for v0 in soln_dict['v0s']],
            'a0s': [a0.clone() for a0 in soln_dict['a0s']],
        }
        test_dict = {
            'alphas':  [a.clone().requires_grad_(False) for a in soln_dict['alphas']],
            'U_skews': [U.clone().requires_grad_(False) for U in soln_dict['U_skews']],
            'omegas':  [w.clone().requires_grad_(False) for w in soln_dict['omegas']],
            'x0s': soln_dict['x0s'],
            'v0s': soln_dict['v0s'],
            'a0s': soln_dict['a0s'],
        }
        theta_tensor = self.map_from_dict_to_tensor(test_dict, mode='joint')
        with torch.no_grad():
            return self._loss_joint(theta_tensor).item()


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

    def _fine_grid_search_omega(self, soln_dict, current_sup_error,
                                omega_range=3.0, omega_step=0.1):
        """
        Fine grid search around current omega estimate.

        Evaluates the supremum projection error on a grid ``±omega_range`` Hz
        with ``omega_step`` spacing, keeping morphology (alpha, U_skew) fixed.

        Parameters
        ----------
        soln_dict : dict
            Current solution.
        current_sup_error : float
            Current supremum projection error (baseline).
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
                sup_errors.append(self._sup_projection_error(test_dict))

            min_idx = np.argmin(sup_errors)
            min_sup_err = sup_errors[min_idx]
            best_omega_k = omega_candidates[min_idx]

            if min_sup_err < best_sup_err:
                improvement = best_sup_err - min_sup_err
                print(f"  Gaussian {k}: ω {omega_current:.4f} → "
                      f"{best_omega_k:.4f} Hz "
                      f"(Δsup = {improvement:.6e})")
                best_omegas[k] = torch.tensor(
                    [best_omega_k], dtype=torch.float64, device=self.device,
                )
                best_sup_err = min_sup_err
            else:
                print(f"  Gaussian {k}: No improvement found "
                      f"(keeping ω = {omega_current:.4f} Hz)")

        soln_dict['omegas'] = [
            omega.clone().detach() for omega in best_omegas
        ]

        if best_sup_err < current_sup_error:
            improvement = current_sup_error - best_sup_err
            print(f"\n  ✓ Grid search improved sup error: {current_sup_error:.6e} → "
                  f"{best_sup_err:.6e} (Δ = {improvement:.6e})")
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

        Huber loss between observed and simulated projections.
        """
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

                # Omega (if not fixed) — sigmoid reparametrization
                # Store z = logit((ω − ω_min) / (ω_max − ω_min)) so the
                # unconstrained z ∈ ℝ maps to ω ∈ (ω_min, ω_max) always.
                theta_fixed_keys = list(
                    getattr(self, 'theta_fixed', {}).keys()
                )
                if ('omegas' in theta_dict
                        and 'omegas' not in theta_fixed_keys):
                    omega_k = theta_dict["omegas"][k].clone()
                    omega_range = self.omega_max - self.omega_min
                    # Clamp to open interval before logit to avoid ±inf
                    p = torch.clamp(
                        (omega_k - self.omega_min) / omega_range,
                        min=1e-6, max=1.0 - 1e-6,
                    )
                    z_omega = torch.log(p / (1.0 - p))  # logit
                    row_parts.append(z_omega.reshape(-1))

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
                    U_skew[diag_mask], min=-4, max=4,
                )
                U_skew_final = U_skew.clone()
                U_skew_final[diag_mask] = torch.exp(diag_clamped)
                U_skews.append(U_skew_final)
                idx += n_U_params

                # Omega (if present in tensor) — inverse sigmoid to recover ω
                if len(row_k) > idx:
                    z_omega = row_k[idx]
                    omega = (
                        self.omega_min
                        + (self.omega_max - self.omega_min)
                        * torch.sigmoid(z_omega)
                    )
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