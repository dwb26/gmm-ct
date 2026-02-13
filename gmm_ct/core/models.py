import torch
import torch.nn as nn
from torchmin import minimize
from .optimizer import NewtonRaphsonLBFGS
from ..estimation.peak_analysis import PeakData
from ..config.defaults import GRAVITATIONAL_ACCELERATION
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import numpy as np
from dtaidistance import dtw

class GMM_reco():
    def __init__(self, d, N, sources, receivers, x0s, a0s, omega_min, omega_max, device=None, output_dir=None, N_traj_trials=None, n_omega_inits=None):
        """
        Initialize the GMM reconstruction model.
        
        Parameters:
        - d: Dimensionality 
        - N: Number of Gaussians
        - sources: Source locations
        - receivers: Receiver locations
        - x0s: Initial positions for all N Gaussians (assumed known)
        - a0s: Accelerations for all N Gaussians (assumed known)
        - omega_min: Minimum angular velocity for initialization
        - omega_max: Maximum angular velocity for initialization
        - device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
        - output_dir: Directory for saving plots and animations (default: 'plots/')
        - n_omega_inits: Number of random multi-start trials for omega search 
                              (None = auto, uses max(15, 3*K))
        """
        self.d = d
        self.N = N
        
        # Known physical parameters
        self.x0s = x0s
        self.a0s = a0s
        self.omega_min = omega_min
        self.omega_max = omega_max
        
        # Omega search configuration
        self.N_traj_trials = N_traj_trials
        self.n_omega_inits = n_omega_inits
        self.use_fft_omega = True  # Enable FFT-based omega estimation by default
        
        # Device management
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Output directory for plots
        if output_dir is None:
            self.output_dir = Path('plots')
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Precompute constants used in projections
        self.sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=self.device))
        
        # Move sources and receivers to device
        if isinstance(sources, list):
            self.sources = [s.to(self.device) if hasattr(s, 'to') else torch.tensor(s, device=self.device) for s in sources]
        else:
            self.sources = sources.to(self.device) if hasattr(sources, 'to') else torch.tensor(sources, device=self.device)
            
        if isinstance(receivers, list):
            self.receivers = [[r.to(self.device) if hasattr(r, 'to') else torch.tensor(r, device=self.device) for r in rec_list] for rec_list in receivers]
        else:
            self.receivers = receivers.to(self.device) if hasattr(receivers, 'to') else torch.tensor(receivers, device=self.device)
            
        self.n_sources = len(sources)        
        # Defensive check for receivers structure
        if isinstance(receivers, list) and len(receivers) > 0:
            self.n_rcvrs = len(receivers[0])        # In the case of multiple sources this will be an array (per source)
        else:
            # Handle case where receivers might not be a list of lists
            self.n_rcvrs = len(receivers) if hasattr(receivers, '__len__') else 1
        
        # Plotting font sizes (can be customized)
        self.plot_label_fontsize = 20
        self.plot_title_fontsize = 22
        self.plot_tick_fontsize = 16
        self.plot_legend_fontsize = 16


    def fit(self, proj_data, t):
        """
        Fits the parameters of the Gaussian mixture model.
        
        Parameters:
        - proj_data: The observed projections (measurements).
        - t: The time vector.
        
        Returns:
        - The complete output of the minimizer routine.
        """
        
        self.t = t.to(self.device) if hasattr(t, 'to') else torch.tensor(t, device=self.device)
        self.proj_data = self.process_projections(self._to_device(proj_data))
        
        
        '---------------------------'
        # 1. Trajectory optimization
        '---------------------------'
        print(f"\n{'='*50}")
        print(f"\n{'='*50}")
        print("Starting trajectory optimization...")
        print(f"{'='*50}")
        
        errors = []; results = []
        
        if self.N_traj_trials is None:
            N_traj_trials = max(10, 2 * self.N)
        else:
            N_traj_trials = self.N_traj_trials
        
        print(f"Running {N_traj_trials} trajectory multi-start trials")
        for n_trial in range(N_traj_trials):
            
            print(f"\nTrial {n_trial + 1}/{N_traj_trials}")
            
            self.theta_dict_init = self.initialize_parameters(t, proj_data)
            [v0_k.requires_grad_(True) for v0_k in self.theta_dict_init['v0s']]
            
            theta_tensor_init = self.map_from_dict_to_tensor(self.theta_dict_init, mode='trajectory')
            res_trial = minimize(self._loss_trajectory, x0=theta_tensor_init, method='l-bfgs', 
                                 tol=1e-8, options={'gtol': 1e-8, 'max_iter': 1500, 'disp': True})
            
            errors.append(res_trial.fun)
            results.append(res_trial)
        
        # Select the best initialization based on the lowest error
        best_res = results[np.argmin(np.array(errors))]
        soln_dict = self.construct_soln_dict(best_res)
        
        # Check if trajectory optimization succeeded
        if 'v0s' not in soln_dict:
            raise RuntimeError(f"Trajectory optimization failed - no v0s in result. Got keys: {list(soln_dict.keys())}")
        
        soln_dict["v0s"] = [v0_k.clone().detach() for v0_k in soln_dict["v0s"]]
        
        # Plot to test the trajectory fit
        self.plot_trajectory_estimations(best_res)
        self.plot_raw_receiver_heights()
        self.plot_heights_by_assignment()
        soln_dict = self.refine_initial_velocities_via_newton_raphson(soln_dict, best_res)
        
        soln_dict["omegas"] = [omega.clone().detach() for omega in self.theta_dict_init["omegas"]]
        soln_dict["alphas"] = [alpha.clone().detach() for alpha in self.theta_dict_init["alphas"]]
        
        # Initialize anisotropic U_skews aligned with velocity direction
        # This is essential for omega estimation - rotating isotropic Gaussians produces no signal
        print(f"\nInitializing anisotropic Gaussians aligned with velocity...")
        soln_dict["U_skews"] = self.initialize_anisotropic_U_skews(soln_dict["v0s"])
        print("✓ Anisotropic U_skews initialized.")
        
        
        '---------------------------------------'
        # 2. Multi-start joint optimization (rotation + morphology)
        '---------------------------------------'
        print(f"\n{'='*50}")
        print("Starting multi-start joint optimization...")
        print(f"{'='*50}")
        
        if self.n_omega_inits is None:
            n_random_starts = 5
        else:
            n_random_starts = self.n_omega_inits
        
        print(f"Running {n_random_starts} random omega initializations")
        
        # Save the initial parameters before multi-start
        initial_alphas = [alpha.clone().detach() for alpha in soln_dict['alphas']]
        initial_U_skews = [U.clone().detach() for U in soln_dict['U_skews']]
        omega_min, omega_max = self.omega_min - 0.01, self.omega_max + 0.01
        
        # Track all results
        all_losses = []
        all_results = []
        
        for trial_idx in range(n_random_starts):
            
            # Initialize each Gaussian with random omega (broad exploration)
            initial_omegas = [torch.tensor([np.random.uniform(omega_min, omega_max)], dtype=torch.float64, device=self.device) 
                            for _ in range(self.N)]
            
            # Create test dictionary with random omega, enable gradients
            test_dict = {
                'alphas': [alpha.clone() for alpha in initial_alphas],
                'U_skews': [U.clone() for U in initial_U_skews],
                'omegas': initial_omegas,
                'x0s': soln_dict['x0s'],
                'v0s': soln_dict['v0s'],
                'a0s': soln_dict['a0s']
            }
            test_dict['alphas'] = [alpha.requires_grad_(True) for alpha in test_dict['alphas']]
            test_dict['U_skews'] = [U.requires_grad_(True) for U in test_dict['U_skews']]
            test_dict['omegas'] = [omega.requires_grad_(True) for omega in test_dict['omegas']]
            
            # Store fixed trajectory parameters
            self.theta_fixed = {
                'x0s': [x0.clone() for x0 in soln_dict['x0s']],
                'v0s': [v0.clone() for v0 in soln_dict['v0s']],
                'a0s': [a0.clone() for a0 in soln_dict['a0s']]
            }
            
            # Optimize from this initialization
            theta_tensor = self.map_from_dict_to_tensor(test_dict, mode='joint')
            res = minimize(self._loss_joint, x0=theta_tensor, method='l-bfgs',
                          tol=1e-10, options={'gtol': 1e-10, 'max_iter': 1000, 'disp': False})
            
            result_dict = self.construct_soln_dict(res)
            final_loss = res.fun.item()
            all_losses.append(final_loss)
            all_results.append(result_dict)
            
            final_omegas = [omega.item() for omega in result_dict['omegas']]
            print(f"  Trial {trial_idx + 1}/{n_random_starts}: loss = {final_loss:.6e}, ω = {[f'{w:.3f}' for w in final_omegas]}")
        
        # Find the best result across all trials
        best_trial_idx = np.argmin(all_losses)
        best_result = all_results[best_trial_idx]
        best_loss = all_losses[best_trial_idx]
        
        soln_dict['alphas'] = [alpha.clone().detach() for alpha in best_result['alphas']]
        soln_dict['omegas'] = [omega.clone().detach() for omega in best_result['omegas']]
        soln_dict['U_skews'] = [U.clone().detach() for U in best_result['U_skews']]
        
        print(f"\n{'='*50}")
        print(f"Multi-start complete! Best trial: {best_trial_idx + 1}")
        print(f"Best loss: {best_loss:.6e}")
        print(f"Best ω: {[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")
        print(f"{'='*50}")
        
        
        '---------------------------------------'
        # 3. Fine grid search for marginal omega improvements
        '---------------------------------------'
        print(f"\n{'='*50}")
        print("Fine grid search around multi-start solution (±3 Hz, 0.1 Hz steps)...")
        print(f"{'='*50}")
        
        soln_dict = self._fine_grid_search_omega(soln_dict, best_loss, omega_range=3.0, omega_step=0.1)
        
        
        '---------------------------------------'
        # 4. Final joint refinement
        '---------------------------------------'
        print(f"\n{'='*50}")
        print("Final joint refinement...")
        print(f"{'='*50}")
        
        soln_dict = self._optimize_joint(soln_dict, max_iter=200)
        
        print(f"\n{'='*50}")
        print("Optimization complete!")
        print(f"{'='*50}")
        print(f"Final ω: {[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")
        
        return soln_dict
    
    
    def _optimize_joint(self, soln_dict, max_iter=300):
        """
        Joint optimization of omega, U_skew, alpha, and optionally v0.
        
        This optimizes parameters simultaneously using gradient descent
        on the projection loss. x0 and a0 remain fixed.
        
        Parameters
        ----------
        soln_dict : dict
            Current solution with all parameters
        max_iter : int
            Maximum L-BFGS iterations
        optimize_v0 : bool
            If True, also optimize v0 (useful when Phase 1 fails)
        
        Returns
        -------
        dict
            Updated solution with refined parameters
        """
        print("\n  Optimizing omega, U_skew, alpha jointly...")
        
        # Enable gradients on ALL parameters to optimize
        soln_dict["alphas"] = [alpha_k.requires_grad_(True) for alpha_k in soln_dict['alphas']]
        soln_dict["U_skews"] = [U_k.requires_grad_(True) for U_k in soln_dict['U_skews']]
        soln_dict["omegas"] = [omega_k.requires_grad_(True) for omega_k in soln_dict['omegas']]
        
        # Note: v0 may or may not be included in optimization depending on multi-start settings
        # Check if v0 already has gradients enabled (from multi-start)
        # Store ONLY truly fixed parameters (x0 and a0)
        self.theta_fixed = {
            'x0s': [x0.clone() for x0 in soln_dict['x0s']],
            'a0s': [a0.clone() for a0 in soln_dict['a0s']]
        }
        # Add v0 to fixed if not being optimized (check if gradients enabled)
        if not soln_dict['v0s'][0].requires_grad:
            self.theta_fixed['v0s'] = [v0.clone() for v0 in soln_dict['v0s']]
        
        # Convert to tensor for optimizer
        theta_tensor = self.map_from_dict_to_tensor(soln_dict, mode='joint')
        
        print(f"  Optimizing {theta_tensor.numel()} parameters (including omega)...")
        
        # Optimize using L-BFGS
        res = minimize(self._loss_joint, x0=theta_tensor, method='l-bfgs',
                      tol=1e-8, options={'gtol': 1e-8, 'max_iter': max_iter, 'disp': False})
        
        # Extract optimized parameters
        result_dict = self.construct_soln_dict(res)
        
        # Update all optimized parameters (alpha, U_skew, AND omega)
        soln_dict['alphas'] = [alpha.clone().detach() for alpha in result_dict['alphas']]
        soln_dict['U_skews'] = [U.clone().detach() for U in result_dict['U_skews']]
        soln_dict['omegas'] = [omega.clone().detach() for omega in result_dict['omegas']]
        
        # Report improvement
        final_loss = res.fun.item()
        n_iters = res.nit
        print(f"  Joint optimization: loss = {final_loss:.6e} ({n_iters} iterations)")
        print(f"  Refined ω: {[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")
        
        return soln_dict
    
    
    def _fine_grid_search_omega(self, soln_dict, current_loss, omega_range=3.0, omega_step=0.1):
        """
        Fine grid search around current omega estimate.
        
        Evaluates projection loss on a fine grid around the current omega,
        keeping morphology (alpha, U_skew) fixed. This provides marginal
        improvements after multi-start has found a good solution.
        
        Parameters
        ----------
        soln_dict : dict
            Current solution with all parameters
        current_loss : float
            Current projection loss
        omega_range : float
            Search range ±omega_range Hz around current omega
        omega_step : float
            Grid spacing in Hz
        
        Returns
        -------
        dict
            Updated solution with refined omega (if improvement found)
        """
        print(f"\n  Searching ±{omega_range} Hz with {omega_step} Hz steps...")
        
        # Store fixed parameters
        self.theta_fixed = {
            'x0s': soln_dict['x0s'],
            'v0s': soln_dict['v0s'],
            'a0s': soln_dict['a0s']
        }
        
        best_loss = current_loss
        best_omegas = [omega.clone() for omega in soln_dict['omegas']]
        
        for k in range(self.N):
            omega_current = soln_dict['omegas'][k].item()
            omega_min = omega_current - omega_range
            omega_max = omega_current + omega_range
            n_points = int((omega_max - omega_min) / omega_step) + 1
            
            omega_candidates = np.linspace(omega_min, omega_max, n_points)
            
            losses = []
            for omega_test in omega_candidates:
                # Create test dict with modified omega
                test_dict = {
                    'alphas': [alpha.clone().requires_grad_(False) for alpha in soln_dict['alphas']],
                    'U_skews': [U.clone().requires_grad_(False) for U in soln_dict['U_skews']],
                    'omegas': [omega.clone().requires_grad_(False) for omega in soln_dict['omegas']],
                    'x0s': soln_dict['x0s'],
                    'v0s': soln_dict['v0s'],
                    'a0s': soln_dict['a0s']
                }
                # Modify omega for Gaussian k
                test_dict['omegas'][k] = torch.tensor([omega_test], dtype=torch.float64, device=self.device)
                
                # Compute loss
                theta_tensor = self.map_from_dict_to_tensor(test_dict, mode='joint')
                loss = self._loss_joint(theta_tensor).item()
                losses.append(loss)
            
            # Find best omega for this Gaussian
            min_idx = np.argmin(losses)
            min_loss = losses[min_idx]
            best_omega_k = omega_candidates[min_idx]
            
            if min_loss < best_loss:
                improvement = best_loss - min_loss
                print(f"  Gaussian {k}: ω {omega_current:.4f} → {best_omega_k:.4f} Hz (Δloss = {improvement:.6e})")
                best_omegas[k] = torch.tensor([best_omega_k], dtype=torch.float64, device=self.device)
                best_loss = min_loss
            else:
                print(f"  Gaussian {k}: No improvement found (keeping ω = {omega_current:.4f} Hz)")
        
        # Update solution with best omegas
        soln_dict['omegas'] = [omega.clone().detach() for omega in best_omegas]
        
        if best_loss < current_loss:
            improvement = current_loss - best_loss
            print(f"\n  ✓ Grid search improved loss: {current_loss:.6e} → {best_loss:.6e} (Δ = {improvement:.6e})")
        else:
            print(f"\n  Grid search: No improvement found")
        
        return soln_dict


    def map_from_dict_to_tensor(self, theta_dict, mode='trajectory'):
        """
        Converts parameter dictionary to flattened tensor for optimization.
        
        The tensor format depends on which parameters are being optimized:
        - 'trajectory': Only v0 parameters (with log transform on v0[0])
        - 'joint': Alpha, U_skew, and omega parameters (logged)
        - 'joint_with_v0': v0, alpha, U_skew, and omega parameters
        
        Parameters
        ----------
        theta_dict : dict
            Parameter dictionary with keys like 'v0s', 'alphas', etc.
        mode : str, {'trajectory', 'joint', 'joint_with_v0'}
            Which optimization phase - determines which parameters to include
        
        Returns
        -------
        torch.Tensor
            Flattened parameter tensor for optimizer
        """
        d, N = self.d, self.N
        tensor_rows = []

        if mode == "trajectory":
            
            '-----------------------------'
            # Specify the fixed parameters
            '-----------------------------'
            # Fixed parameters: don't detach and ensure they have gradients to maintain computational graph
            self.theta_fixed = {'alphas': [alpha.clone().requires_grad_(True) for alpha in theta_dict['alphas']],
                                'U_skews': [U.clone().requires_grad_(True) for U in theta_dict['U_skews']],
                                'omegas': [omega.clone().requires_grad_(True) for omega in theta_dict['omegas']],
                                'x0s': [x0.clone().requires_grad_(True) for x0 in theta_dict['x0s']],
                                'a0s': [a0.clone().requires_grad_(True) for a0 in theta_dict['a0s']]}            
            
            '--------------------------'
            # Unpack the dict variables
            '--------------------------'
            for k in range(N):
                v0_k = theta_dict['v0s'][k]
                v0_k_0 = torch.log(torch.abs(v0_k[0]) + 1e-8)  # Log transform with small offset for stability
                v0_k_1 = v0_k[1]
                v0_k_modified = torch.stack([v0_k_0, v0_k_1])
                tensor_rows.append(v0_k_modified)
                
        elif mode == "joint" or mode == "joint_with_v0":
            
            '-----------------------------'
            # Specify the fixed parameters
            '-----------------------------'
            # Fixed parameters: don't detach otherwise the computational graph is broken
            # Only set theta_fixed if it doesn't already exist (avoid overwriting)
            if not hasattr(self, 'theta_fixed') or self.theta_fixed is None:
                self.theta_fixed = {'x0s': [x0.clone() for x0 in theta_dict['x0s']], 
                                    'a0s': [a0.clone() for a0 in theta_dict['a0s']]}
                # For standard joint, v0 is fixed; for joint_with_v0, it's optimized
                if mode == "joint":
                    self.theta_fixed['v0s'] = [v0.clone() for v0 in theta_dict['v0s']]
            
            
            '--------------------------'
            # Unpack the dict variables
            '--------------------------'
            for k in range(N):
                
                # Build tensor row for this Gaussian
                row_parts = []
                
                # If optimizing v0 (joint_with_v0 mode), include it first
                if mode == "joint_with_v0":
                    v0_k = theta_dict['v0s'][k]
                    v0_k_0_logged = torch.log(torch.abs(v0_k[0]) + 1e-8)  
                    v0_k_1 = v0_k[1]
                    # Reshape to ensure consistent 1D tensors
                    row_parts.append(v0_k_0_logged.reshape(-1))
                    row_parts.append(v0_k_1.reshape(-1))
            
                # Always include alpha and U_skew
                alpha_k = theta_dict["alphas"][k].clone()
                alpha_k_logged = torch.log(alpha_k).reshape(-1)

                U_skew_copy = theta_dict["U_skews"][k].clone()
                
                # Clamp diagonal to prevent log(0) or log(negative) which cause -inf or NaN
                EPS = 1e-8
                diag_clamped = torch.clamp(torch.diagonal(U_skew_copy), min=EPS)
                diag_logged = torch.log(diag_clamped)
                
                # Use non-in-place operations to preserve gradients
                U_skew_no_diag = U_skew_copy - torch.diag(torch.diagonal(U_skew_copy))      # Clear diagonal
                U_skew_with_logged_diag = U_skew_no_diag + torch.diag(diag_logged)          # Add logged diagonal
                
                # Extract upper triangular elements
                triu_idx = torch.triu_indices(d, d, device=U_skew_copy.device)
                U_skew_vals = U_skew_with_logged_diag[triu_idx[0], triu_idx[1]].reshape(-1)
                
                # Append alpha and U_skew
                row_parts.append(alpha_k_logged)
                row_parts.append(U_skew_vals)
                
                # Include omega if it's not in the fixed parameters
                theta_fixed_dict = getattr(self, 'theta_fixed', {})
                theta_fixed_keys = list(theta_fixed_dict.keys())
                include_omega = 'omegas' in theta_dict and 'omegas' not in theta_fixed_keys
                
                if include_omega:
                    omega_k = theta_dict["omegas"][k].clone().reshape(-1)
                    row_parts.append(omega_k)
                
                # Concatenate all parts for this Gaussian
                combined_k = torch.cat(row_parts)
                tensor_rows.append(combined_k)
                    
                
        # Stack tensor rows - handle single row case for geo optimization
        if len(tensor_rows) == 1:
            stacked_tensor = tensor_rows[0]
        else:
            stacked_tensor = torch.stack(tensor_rows)
        return stacked_tensor

    
    def map_from_tensor_to_dict(self, theta_tensor, mode='trajectory'):
        """
        Converts flattened parameter tensor back to dictionary format.
        
        Inverse of map_from_dict_to_tensor - reconstructs parameter dictionary
        from optimizer's flattened tensor representation.
        
        Parameters
        ----------
        theta_tensor : torch.Tensor
            Flattened parameter tensor from optimizer
        mode : str, {'trajectory', 'joint', 'joint_with_v0'}
            Which optimization phase - determines which parameters to extract
        
        Returns
        -------
        dict
            Parameter dictionary with keys like 'v0s', 'alphas', etc.
        """
        d, N = self.d, self.N
        
        # Create the dictionary and load the parameters to be estimated
        theta_dict = {}
        if mode == "trajectory":
            v0s = []
            if self.N == 1:
                theta_tensor = theta_tensor.squeeze(0)
                exp_first_component = torch.exp(theta_tensor[0])  # Exponential transformation for the first component
                second_component = theta_tensor[1]                # No transformation for the second component
                v0_modified = torch.stack([exp_first_component, second_component])
                v0s.append(v0_modified)
            else:
                for k in range(self.N):
                    exp_first_component = torch.exp(theta_tensor[k, 0])  # Exponential transformation for the first component
                    second_component = theta_tensor[k, 1]                # No transformation for the second component
                    v0_modified = torch.stack([exp_first_component, second_component])
                    v0s.append(v0_modified)
            theta_dict['v0s'] = v0s
            
        elif mode == "joint" or mode == "joint_with_v0":
            alphas = []
            U_skews = []
            omegas = []
            v0s = []  # Only used in joint_with_v0 mode
            
            # Calculate number of U_skew parameters (upper triangular elements)
            n_U_params = d * (d + 1) // 2
            
            # Correctly handle both single and multiple Gaussian cases
            if N > 1:
                rows = [theta_tensor[k] for k in range(N)]
            else:
                rows = [theta_tensor]

            for row_k in rows:
                idx = 0
                
                # Extract v0 if in joint_with_v0 mode (first 2 elements)
                if mode == "joint_with_v0":
                    v0_logged_0 = row_k[idx]
                    v0_1 = row_k[idx + 1]
                    v0_0 = torch.exp(v0_logged_0)
                    v0_k = torch.stack([v0_0, v0_1])
                    v0s.append(v0_k)
                    idx += 2
                
                # Alpha (next element)
                alpha_logged = row_k[idx]
                # BUG FIX: Clamp alpha before exponentiating to prevent overflow
                alpha_logged_clamped = torch.clamp(alpha_logged, min=-5, max=5)
                alpha = torch.exp(alpha_logged_clamped)
                idx += 1
                
                # U_skew (next n_U_params elements)
                U_skew_vals = row_k[idx : idx + n_U_params]
                if U_skew_vals.numel() != n_U_params:
                    raise ValueError(f"Expected {n_U_params} U_skew values, got {U_skew_vals.numel()}. "
                                   f"Row has {row_k.numel()} elements, idx={idx}, mode={mode}")
                U_skew = torch.zeros((d, d), dtype=theta_tensor.dtype, device=theta_tensor.device)
                triu_indices = torch.triu_indices(d, d)
                U_skew[triu_indices[0], triu_indices[1]] = U_skew_vals

                # Exponentiate the diagonal elements while preserving gradients
                diag_mask = torch.eye(d, dtype=torch.bool, device=theta_tensor.device)
                diag_elements = U_skew[diag_mask]
                
                # BUG FIX: Clamp diagonal values BEFORE exponentiating to prevent infinity
                diag_elements_clamped = torch.clamp(diag_elements, min=-5, max=10)
                diag_exp = torch.exp(diag_elements_clamped)
                
                # Create matrix with exponentialized diagonal and preserve gradients
                U_skew_final = U_skew.clone()
                U_skew_final[diag_mask] = diag_exp                
                idx += n_U_params
                
                # Omega (next element) - only if present in tensor
                # Check if omega is in the tensor by seeing if there are enough elements
                if len(row_k) > idx:
                    omega = row_k[idx]
                    omegas.append(omega.unsqueeze(0) if omega.dim() == 0 else omega)

                # Append alpha and U_skew for this Gaussian
                alphas.append(alpha.unsqueeze(0))
                U_skews.append(U_skew_final)
                
            theta_dict['alphas'] = alphas
            theta_dict['U_skews'] = U_skews
            
            # Only add omegas if they were extracted from the tensor
            if len(omegas) > 0:
                theta_dict['omegas'] = omegas
                
            # Only add v0s if in joint_with_v0 mode
            if mode == "joint_with_v0" and len(v0s) > 0:
                theta_dict['v0s'] = v0s
                
        return theta_dict


    def _loss_trajectory(self, theta_tensor):
        """
        Loss function for trajectory optimization (Phase 1).
        
        Minimizes L2 distance between predicted and observed peak heights
        using Hungarian algorithm for optimal assignment.
        
        Parameters
        ----------
        theta_tensor : torch.Tensor
            Flattened v0 parameter tensor from optimizer
        
        Returns
        -------
        loss : torch.Tensor
            Scalar L2 loss (sum of distances across all Gaussians)
        """
        # Convert tensor to dictionary format
        theta_dict = self.map_from_tensor_to_dict(theta_tensor, mode='trajectory')
        
        # Compute predicted receiver positions
        self.t_observable = self.t[self.peak_data.observable_indices]
        r_maxs_list = self.map_velocities_to_maximising_receivers(theta_dict)
        
        # Perform optimal assignment using Hungarian algorithm
        self._assign_peaks_hungarian(r_maxs_list)
        
        # Compute loss: sum of distances between predicted and assigned heights
        loss = self._compute_trajectory_loss(r_maxs_list)
        
        return loss
    
    
    def _loss_joint(self, theta_tensor):
        """
        Loss function for joint rotation + morphology optimization (Phase 2).
        
        Minimizes SmoothL1 loss between observed and simulated projections.
        Optimizes alpha, U_skew, omega, and optionally v0 parameters.
        
        Parameters
        ----------
        theta_tensor : torch.Tensor
            Flattened parameter tensor (may include v0s, alphas, U_skews, omegas)
        
        Returns
        -------
        loss : torch.Tensor
            Scalar SmoothL1 loss between projections
        """
        # Direct projection comparison
        loss_func = nn.SmoothL1Loss(beta=0.3)
        
        # Determine mode based on whether v0 is in theta_fixed
        has_v0_fixed = 'v0s' in self.theta_fixed
        mode = 'joint' if has_v0_fixed else 'joint_with_v0'
        
        theta_dict = self.map_from_tensor_to_dict(theta_tensor, mode=mode)
        
        # Add fixed parameters (x0s, a0s, and possibly v0s/omegas)
        for key, value in self.theta_fixed.items():
            if key not in theta_dict:
                theta_dict[key] = value
        
        # Generate simulated projections
        sim_projs = self.generate_projections(self.t_observable, theta_dict)
        sim_projs_processed = self.process_projections(sim_projs)
        
        # Compare to observed projections
        proj_data_observable = self.proj_data[self.peak_data.observable_indices]
        loss = loss_func(proj_data_observable, sim_projs_processed)
        
        return loss
    
    
    def _assign_peaks_hungarian(self, r_maxs_list):
        """
        Assign detected peaks to trajectories using Hungarian algorithm.
        
        At each time point, finds the optimal assignment that minimizes
        total distance between observed and predicted heights.
        
        Parameters
        ----------
        r_maxs_list : list of torch.Tensor
            Predicted receiver positions, shape (n_times, 2)
        
        Side Effects
        ------------
        Populates self.assigned_curve_data with temporary assignments
        (used during trajectory optimization, overwritten later)
        """
        self.assigned_curve_data = [[] for _ in range(self.N)]
        heights_dict = self.peak_data.get_heights_dict_non_empty()
        
        for time_idx, time_val in enumerate(self.t_observable):
            observed_heights = heights_dict[time_val.item()]
            
            # Build cost matrix: distance from each observed height to each predicted trajectory
            dist_matrix = torch.zeros(
                len(observed_heights), 
                self.N, 
                dtype=torch.float64, 
                device=self.device
            )
            
            for height_idx, height in enumerate(observed_heights):
                for gaussian_idx in range(self.N):
                    predicted_height = r_maxs_list[gaussian_idx][time_idx, 1]
                    distance = torch.abs(predicted_height - height)
                    
                    # Handle invalid predictions
                    if torch.isnan(distance) or torch.isinf(distance):
                        dist_matrix[height_idx, gaussian_idx] = 1e10
                    else:
                        dist_matrix[height_idx, gaussian_idx] = distance.item()
            
            # Solve assignment problem
            row_indices, col_indices = linear_sum_assignment(dist_matrix.cpu().numpy())
            
            # Store assignments
            for height_idx, gaussian_idx in zip(row_indices, col_indices):
                self.assigned_curve_data[gaussian_idx].append(
                    (time_idx, observed_heights[height_idx])
                )
    
    
    def _compute_trajectory_loss(self, r_maxs_list):
        """
        Compute L2 loss between predicted and assigned receiver heights.
        
        Parameters
        ----------
        r_maxs_list : list of torch.Tensor
            Predicted receiver positions for each Gaussian
        
        Returns
        -------
        loss : torch.Tensor
            Sum of L2 norms across all Gaussians
        """
        loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        for gaussian_idx in range(self.N):
            assignments_k = self.assigned_curve_data[gaussian_idx]
            
            if len(assignments_k) == 0:
                continue
            
            # Extract time indices and observed heights
            time_indices = [item[0] for item in assignments_k]
            observed_heights = torch.stack([item[1] for item in assignments_k])
            
            # Get predicted heights at those times
            predicted_heights = r_maxs_list[gaussian_idx][time_indices, 1]
            
            # Add L2 distance to loss
            loss += torch.norm(predicted_heights - observed_heights, p=2)
        
        return loss

    
    def generate_projections(self, t, theta_dict, loss_type=None):
        """
        Generates the X-ray projection data for the Gaussian mixture model.        
        Vectorized implementation that considers the whole GMM at each time
        
        Parameters:
        - t: Sampled time vector.
        - theta_true: A dictionary containing the true parameters of the model.
        
        Returns:
        - A list of projections for each source at each time point.
        """        
        # Compile each of the K rotation matrix and trajectory functions
        rot_mat_funcs = self.construct_rotation_matrix_funcs()
        traj_funcs = self.construct_trajectory_funcs()
        projs = [torch.zeros(len(t), self.n_rcvrs, dtype=torch.float64, device=self.device) for _ in range(self.n_sources)]
        
        # When optimisng, we need to collect all attached and detached parameters into one superficial tensor
        if loss_type is not None:
            
            # Create a complete theta dict without breaking computational graph
            complete_theta_dict = theta_dict.copy()
            
            # Add fixed parameters without overwriting gradient-enabled ones
            for key, value in self.theta_fixed.items():
                if key not in complete_theta_dict:
                    complete_theta_dict[key] = value
            theta_dict = complete_theta_dict

        # Generate the projections at each specifed time t
        for n_t, t_n in enumerate(t):
            
            # Compile the motion functions at time t for all k
            rot_mat_of_t = rot_mat_funcs(t_n, theta_dict)
            traj_of_t = traj_funcs(t_n, theta_dict)
            
            for n_s, s in enumerate(self.sources):
                
                # Vectorized receiver processing
                receivers_ns = self.receivers[n_s]
                r = torch.stack(receivers_ns)
                
                r_minus_s = r - s            
                r_minus_s_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True)
                
                # Iterate through the Gaussians
                for k in range(self.N):
                    
                    # Set the parameters specific to the kth Gaussian
                    alpha_k = theta_dict['alphas'][k].squeeze()  # Ensure scalar for broadcasting
                    U_k = theta_dict['U_skews'][k]
                    R_k_of_t = rot_mat_of_t[k]
                    mu_k_of_t = traj_of_t[k]
                    new_U_k = U_k @ R_k_of_t.mT
                    
                    # Compute terms once and reuse (avoid redundant matrix multiplications)
                    # Add epsilon to prevent division by zero
                    EPS = 1e-10
                    
                    # Precompute matrix products that are used multiple times
                    U_r_hat = new_U_k @ r_minus_s_hat.T  # Used for norm_term
                    U_r = new_U_k @ r_minus_s.T          # Used for rcvr_varying_term and divisor
                    U_traj = new_U_k @ (s - mu_k_of_t).unsqueeze(1)  # traj_varying_term
                    
                    # Compute projection terms
                    norm_term = torch.norm(U_r_hat, dim=0)
                    quotient_term = self.sqrt_pi * alpha_k / (norm_term + EPS)
                    
                    inner_prod_sq = torch.sum(U_r * U_traj, dim=0) ** 2
                    divisor = torch.norm(U_r, dim=0) ** 2 + EPS
                    subtractor = torch.norm(U_traj, dim=0) ** 2
                    
                    # Compute the projection at the individual receiver for the kth Gaussian at the jth time point and nth source
                    exp_arg = inner_prod_sq / divisor - subtractor
                    exp_term = torch.exp(exp_arg)
                    projs[n_s][n_t] += quotient_term * exp_term
        return projs



    '--------------------------------------------------------------------------------------------------------------------------'
    '----------------------------------------------- INITIALIZATION METHODS ---------------------------------------------------'
    '--------------------------------------------------------------------------------------------------------------------------'
    def initialize_parameters(self, t, proj_data):
        """
        Initialize all GMM parameters for optimization.
        
        Initialization strategy:
        1. Isotropic Gaussians (U_skew = scaled identity) - essential for trajectory fitting
        2. Peak detection + random v0 - provides starting point for trajectory optimization
        3. Omega at midpoint of range - simple starting guess (will be refined by optimization)
        4. Reasonable alpha values - typical attenuation coefficients
        
        The isotropic initialization is not a constraint - Phase 2 optimization will
        naturally discover anisotropy when fitting rotation + morphology.
        
        Parameters
        ----------
        t : torch.Tensor
            Time vector
        proj_data : torch.Tensor or list
            Observed projection data
        
        Returns
        -------
        dict
            Complete parameter dictionary with all initialized values
        """
        alphas = self.initialize_attenuation_coefficients()
        U_skews = self.initialize_U_skews()  # Isotropic (scaled identity)
        omegas = self.initialize_rotation_velocities()  # Midpoint guess
        v0s = self.initialize_initial_velocities(t, proj_data)  # Peak detection + random
        
        theta_dict_init = {
            'alphas': alphas,
            'U_skews': U_skews,
            'omegas': omegas,
            'x0s': self.x0s,
            'v0s': v0s,
            'a0s': self.a0s
        }
        return theta_dict_init
    
    
    def initialize_attenuation_coefficients(self):
        """
        Initialize attenuation coefficients (peak heights).
        
        Uses range [10, 15] based on typical experimental values.
        These will be refined during Phase 2 (joint optimization).
        
        Returns
        -------
        list of torch.Tensor
            Alpha values, one per Gaussian
        """
        alphas = []
        for _ in range(self.N):
            alpha_k = 10.0 + torch.rand(size=(1,), dtype=torch.float64, device=self.device) * 5.0
            alphas.append(alpha_k)
        return alphas
        
    
    def initialize_initial_velocities(self, t, proj_data):
        """
        Detects peaks in projection data and initializes velocity parameters.
        
        This method orchestrates three main tasks:
        1. Peak Detection: Find local maxima in projections over time
        2. Sequential Assignment: Initially assign peaks to Gaussians (bottom-to-top)
        3. Velocity Initialization: Return random v0 for optimization
        
        Parameters
        ----------
        t : torch.Tensor
            Time vector for observations
        proj_data : torch.Tensor or list
            Observed projection data [n_times, n_receivers]
        
        Returns
        -------
        v0s : list of torch.Tensor
            Random initial velocities for each Gaussian
        
        Side Effects
        ------------
        Creates and populates self.peak_data with all detection and assignment info
        """
        # Initialize peak data storage
        self.peak_data = PeakData(self.N, self.device)
        
        # Extract projection data for first source
        proj_data_array = proj_data[0] if isinstance(proj_data, list) else proj_data
        receivers = self.receivers[0]
        
        # Detect peaks at each time point
        print(f"\nDetecting peaks in {len(t)} time points...")
        self._detect_all_peaks(proj_data_array, receivers, t)
        
        # Finalize detection data (convert lists to tensors)
        self.peak_data.finalize_detections()
        
        # Create aliases for backward compatibility with plotting methods
        self._create_legacy_aliases()
        
        # Print detection summary
        self.peak_data.summary()
        
        # Initialize velocities with random values
        return self._create_random_initial_velocities()
    
    
    def _detect_all_peaks(self, proj_data, receivers, t):
        """
        Detect peaks across all time points using sliding 3-point window.
        
        Scans from bottom to top (high receiver index to low), assigning
        the first N peaks found at each time to Gaussians 0, 1, ..., N-1.
        
        Parameters
        ----------
        proj_data : torch.Tensor, shape (n_times, n_receivers)
            Projection data
        receivers : list of torch.Tensor
            Receiver positions
        t : torch.Tensor
            Time vector
        """
        for time_idx in range(len(t)):
            detected_heights = self._detect_peaks_at_single_time(
                proj_data[time_idx], 
                receivers, 
                t[time_idx], 
                time_idx
            )
            
            # Store all heights detected at this time
            self.peak_data.add_time_detections(t[time_idx].item(), detected_heights)
    
    
    def _detect_peaks_at_single_time(self, projection, receivers, time_val, time_idx):
        """
        Detect peaks at a single time point using 3-point sliding window.
        
        A peak is detected when the center value is greater than both neighbors.
        
        Parameters
        ----------
        projection : torch.Tensor, shape (n_receivers,)
            Projection values at all receivers
        receivers : list of torch.Tensor
            Receiver positions
        time_val : float or torch.Tensor
            Current time value
        time_idx : int
            Index into time array
        
        Returns
        -------
        detected_heights : list of float
            Receiver heights where peaks were detected
        """
        detected_heights = []
        gaussian_idx = 0
        
        # Scan from bottom to top (need 3 points for peak detection)
        for offset in range(self.n_rcvrs - 2):
            # Map to receiver indices (bottom=high index, top=low index)
            idx_center = self.n_rcvrs - 2 - offset
            idx_lower = idx_center + 1
            idx_upper = idx_center - 1
            
            # Check for local maximum: left < center > right
            if (projection[idx_lower] < projection[idx_center] and 
                projection[idx_center] > projection[idx_upper]):
                
                # Record this peak
                receiver_pos = receivers[idx_center]
                receiver_height = receiver_pos[1]
                
                self.peak_data.add_peak_detection(
                    time_idx, time_val, idx_center, receiver_pos,
                    projection[idx_center], gaussian_idx
                )
                
                detected_heights.append(receiver_height)
                gaussian_idx += 1
                
                # Stop if we've found peaks for all Gaussians
                if gaussian_idx >= self.N:
                    break
        
        return detected_heights
    
    
    def _create_legacy_aliases(self):
        """
        Create backward-compatible aliases for plotting methods.
        
        This maintains compatibility with existing plotting functions that
        expect the old scattered attribute names.
        """
        # Aliases for initial sequential detections
        self.t_obs_by_cluster = self.peak_data.times
        self.maximising_rcvrs = self.peak_data.receiver_positions
        self.maximising_inds = self.peak_data.receiver_indices
        self.peak_values = self.peak_data.peak_values
        self.observable_indices = self.peak_data.observable_indices
        
        # Aliases for height data
        self.time_rcvr_heights_dict_non_empty = self.peak_data.get_heights_dict_non_empty()
        self.sorted_list_of_heights_over_time = self.peak_data.get_heights_sorted_by_time()
    
    
    def _create_random_initial_velocities(self):
        """
        Create random initial velocity estimates for each Gaussian.
        
        Initializes around [1, 1] with Gaussian noise (std=1.5).
        
        Returns
        -------
        v0s : list of torch.Tensor
            Initial velocity tensors with gradients enabled
        """
        v0s = []
        for _ in range(self.N):
            v0 = torch.tensor([1.0, 1.0], dtype=torch.float64, device=self.device)
            v0 = v0 + 1.5 * torch.randn(2, dtype=torch.float64, device=self.device)
            v0.requires_grad_(True)
            v0s.append(v0)
        return v0s


    def initialize_U_skews(self):
        """
        Initialize covariance structure matrices as ISOTROPIC (scaled identity).
        
        Isotropy is essential for trajectory optimization (Phase 1) because:
        - Isotropic Gaussians have circular symmetry
        - Peak positions depend only on trajectory, not rotation
        - This decouples trajectory estimation from morphology
        
        Phase 2 optimization will naturally discover anisotropy when fitting
        the full projection data with rotation.
        
        Returns
        -------
        list of torch.Tensor
            Isotropic U_skew matrices (scaled identity), one per Gaussian
        """
        U_skews = [25.0 * torch.eye(self.d, dtype=torch.float64, device=self.device) 
                   for _ in range(self.N)]
        return U_skews
    
    
    def initialize_rotation_velocities(self):
        """
        Initialize rotation velocities to midpoint of valid range.
        
        NOTE: This is just a placeholder! Phase 1 (trajectory optimization) does NOT use omega.
        After Phase 1 completes, estimate_omega_from_trajectory() will replace these values
        with DTW-based estimates derived from the optimized trajectories and their assigned peaks.
        
        Returns
        -------
        list of torch.Tensor
            Omega placeholders (midpoint of range), replaced after trajectory optimization
        """
        omega_mean = 0.5 * (self.omega_min + self.omega_max)
        omegas = [omega_mean * torch.ones(size=(1,), dtype=torch.float64, device=self.device) 
                  for _ in range(self.N)]
        return omegas
    
    
    def initialize_anisotropic_U_skews(self, v0s):
        """
        Initialize anisotropic U_skew matrices aligned with velocity direction.
        
        This creates elongated Gaussians where the major axis is aligned with the
        velocity vector. This anisotropy is ESSENTIAL for omega estimation via DTW:
        - Isotropic (circular) Gaussians show no peak modulation under rotation
        - Anisotropic (elongated) Gaussians produce rotation-dependent peak patterns
        - DTW can then match these patterns to estimate omega
        
        Strategy:
        - Major axis (along velocity): scale factor ~30 (elongated)
        - Minor axis (perpendicular): scale factor ~15 (compressed)
        - Ratio of 2:1 provides clear rotation signal
        
        Parameters
        ----------
        v0s : list of torch.Tensor
            Optimized initial velocities from Phase 1, shape (2,) each
        
        Returns
        -------
        list of torch.Tensor
            Anisotropic U_skew matrices, one per Gaussian, shape (2, 2)
        """
        U_skews = []
        
        for k in range(self.N):
            v0_k = v0s[k]
            
            # Normalize velocity to get direction
            v_norm = torch.norm(v0_k)
            if v_norm < 1e-6:
                # If velocity is near zero, use arbitrary direction
                v_hat = torch.tensor([1.0, 0.0], dtype=torch.float64, device=self.device)
            else:
                v_hat = v0_k / v_norm
            
            # Perpendicular direction (rotate 90 degrees)
            v_perp = torch.tensor([-v_hat[1], v_hat[0]], dtype=torch.float64, device=self.device)
            
            # Create basis matrix: columns are precision space directions
            # For elongation ALONG velocity: velocity direction needs SMALLER precision (larger covariance)
            # Precision = U^T @ U, so smaller U values → larger covariance in that direction
            # Major axis (velocity direction): smaller precision scale (~15) → larger covariance
            # Minor axis (perpendicular): larger precision scale (~30) → smaller covariance
            major_direction_scale = 15.0  # Small precision → large covariance → major axis
            minor_direction_scale = 30.0  # Large precision → small covariance → minor axis
            
            # U_skew columns: [velocity_direction, perpendicular_direction]
            U_skew_k = torch.stack([v_hat * major_direction_scale, v_perp * minor_direction_scale], dim=1)
            
            U_skews.append(U_skew_k)
            
            print(f"  Gaussian {k}: Elongated along velocity direction (covariance ratio {(minor_direction_scale/major_direction_scale)**2:.1f}:1)")
        
        return U_skews


    def isotropic_derivative_function(self, v0, *args):
        """Computes the identity derivative function.

        Args:
            v0 (torch.Tensor): The initial velocity.

        Returns:
            torch.Tensor: The value of the identity derivative function.
        """
        t_n, r, s, x0, a0 = args

        # Compute the derivative function value
        r1, r2 = r[0], r[1]
        s1, s2 = s[0], s[1]
        d1, d2 = r1 - s1, r2 - s2
        norm_n_sq = d1**2 + d2**2
        
        c_k = s - x0 - v0 * t_n - 0.5 * a0 * t_n**2
        h_k = d1 * c_k[0] - s2 * c_k[1]
        R_k_l = 2 * norm_n_sq * c_k[1] * (c_k[1] * r2 + h_k)
        R_k_r = -2 * d2 * (c_k[1] * r2 + h_k) ** 2
        return (R_k_l + R_k_r) / norm_n_sq ** 2
    

    def isotropic_derivative_function_over_all_times(self, v0, *args):
        """Computes the identity derivative function over all times.

        Args:
            v0 (torch.Tensor): The initial velocity.

        Returns:
            torch.Tensor: The value of the identity derivative function.
        """
        t, r, s, x0, a0 = args
        R_all = torch.zeros(1, dtype=torch.float64, device=self.device)
        for n, t_n in enumerate(t):
            R_all += torch.abs(self.isotropic_derivative_function(v0, t_n, r[n], s, x0, a0))
        return R_all
    
    
    
    '--------------------------------------------------------------------------------------------------------------------------'
    '-------------------------------------------------- HELPER FUNCTIONS ------------------------------------------------------'
    '--------------------------------------------------------------------------------------------------------------------------'
    def process_projections(self, projections):
        if self.n_sources == 1:
            return projections[0]
        return torch.cat([proj for proj in projections], dim=0)

        
    def construct_rotation_matrix_funcs(self):
        """
        Constructs rotation matrices for the Gaussian mixture model.
        We index sequentially as rotations in axes (1, 2), ..., (1, d), (2, 3), ..., (2, d), ..., (d - 1, d)
        
        Parameters:
        - d: Dimensionality of the Gaussian mixture model.
        - K: Number of Gaussians in the GMM.
        - theta: General parameter tensor.
        
        Returns:
        - A list of K rotation matrix functions of time, each for the kth constituent Gaussian.
        """
        
        two_pi = 2 * torch.pi            
        def all_rot_mat_funcs(t, theta):
            rot_matrices = []
            for k in range(self.N):
                omegas_k = theta['omegas'][k]
                kth_component_rot_mats = []
                for n_rots, omega in enumerate(omegas_k):
                    i, j = torch.combinations(torch.arange(self.d, device=self.device), r=2)[n_rots]

                    rot_mat = torch.eye(self.d, dtype=torch.float64, device=self.device)
                    rot_mat[i, i] = torch.cos(two_pi * omega * t)
                    rot_mat[i, j] = -torch.sin(two_pi * omega * t)
                    rot_mat[j, i] = torch.sin(two_pi * omega * t)
                    rot_mat[j, j] = torch.cos(two_pi * omega * t)
                    
                    kth_component_rot_mats.append(rot_mat)
                
                kth_rot_mat = torch.eye(self.d, dtype=torch.float64, device=self.device)
                for rot_mat in kth_component_rot_mats:
                    kth_rot_mat = kth_rot_mat @ rot_mat
                rot_matrices.append(kth_rot_mat)
            return rot_matrices
        return all_rot_mat_funcs
    

    def construct_trajectory_funcs(self):
        """
        Constructs trajectory functions for the Gaussian mixture model.
        
        Parameters:
        - t: Sampled time vector.
        - theta: General parameter tensor.
        
        Returns:
        - A list of K trajectory functions of time, each for the kth constituent Gaussian.
        """
        
        def all_traj_funcs(t, theta):
            trajectories = []
            for k in range(self.N):
                x0 = theta['x0s'][k]
                v0 = theta['v0s'][k]
                a0 = theta['a0s'][k]
                
                # Handle both scalar and vector t
                if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
                    # Scalar time: simple computation
                    traj_k = x0 + v0 * t + 0.5 * a0 * t**2
                else:
                    # Vector time: reshape for broadcasting [N, 1] * [2] -> [N, 2]
                    t_reshaped = t.unsqueeze(1)  # [N] -> [N, 1]
                    traj_k = x0 + v0 * t_reshaped + 0.5 * a0 * t_reshaped**2
                trajectories.append(traj_k)
            return trajectories
        return all_traj_funcs


    def _to_device(self, obj):
        """Move tensor or nested structure of tensors to the correct device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, list):
            return [self._to_device(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._to_device(value) for key, value in obj.items()}
        else:
            return obj
    
    
    def map_velocities_to_maximising_receivers(self, theta_dict):
        """
        Maps the predicted trajectory parameters to maximizing receivers over time.
        
        Given trajectories μ_k(t) = x0_k + v0_k * t + 0.5 * a0_k * t^2 and a fixed 
        first receiver coordinate r_0, computes the receiver position r that maximizes
        the projection along each X-ray beam from source s.
        
        Uses the explicit formula derived from the orthogonality constraint:
            r = s + λ(s - c)
        where:
            λ = (r_0 - s_0) / (s_0 - c_0)
            c = μ_k(t) is the Gaussian center trajectory at time t
            
        This works for arbitrary dimension d >= 2.
        
        Returns:
            List of tensors, one per Gaussian, each of shape (n_times, d) containing
            the maximizing receiver positions over time.
        """
        
        r_maxs_list = []
        s = self.sources[0]
        r0 = self.receivers[0][0][0]  # Fixed first coordinate of receivers
        
        for k in range(self.N):
            # Time-independent parameters
            v0_k = theta_dict['v0s'][k]
            x0_k = self.theta_fixed['x0s'][k]
            a0_k = self.theta_fixed['a0s'][k]
            
            r_maxs_k = []
            for t_n in self.t_observable:
                # Compute trajectory center at time t_n
                c_k = x0_k + v0_k * t_n + 0.5 * a0_k * t_n**2
                
                # Compute scaling factor λ = (r_0 - s_0) / (s_0 - c_0)
                numerator = r0 - s[0]
                denominator = s[0] - c_k[0]
                
                # Prevent division by zero
                EPS = 1e-10
                denominator_safe = torch.where(torch.abs(denominator) < EPS,
                                              torch.sign(denominator) * EPS + (denominator == 0) * EPS,
                                              denominator)
                lambda_t = numerator / denominator_safe
                
                # Compute maximizing receiver: r = s + λ(s - c)
                r_max = s + lambda_t * (s - c_k)
                r_maxs_k.append(r_max)
            
            r_maxs_list.append(torch.stack(r_maxs_k))
        
        return r_maxs_list
    
    
    def construct_soln_dict(self, res):
        """
        Constructs the solution dictionary from the optimization result.
        Automatically determines mode based on what's in theta_fixed.
        """
        theta_tensor_init = res.x
        
        # Determine mode based on tensor size (most reliable indicator)
        # For d=2: trajectory has 2 params per Gaussian, morphology has 4, joint has 5
        # If v0 is being optimized jointly, we have 7 params per Gaussian (2 v0 + 1 alpha + 3 U_skew + 1 omega)
        tensor_size = theta_tensor_init.numel() if len(theta_tensor_init.shape) == 1 else theta_tensor_init.shape[0] * theta_tensor_init.shape[1]
        params_per_gaussian = tensor_size // self.N if self.N > 0 else tensor_size
        
        # Check if v0 is in theta_fixed or being optimized
        has_v0_fixed = hasattr(self, 'theta_fixed') and 'v0s' in self.theta_fixed
        
        if params_per_gaussian == 2:
            # Trajectory mode: 2 v0 components
            mode = 'trajectory'
        elif params_per_gaussian == 4 and hasattr(self, 'theta_fixed') and 'omegas' in self.theta_fixed:
            # Morphology mode: 1 alpha + 3 U_skew, omega is fixed
            mode = 'joint'
        elif params_per_gaussian == 7 or (params_per_gaussian >= 5 and not has_v0_fixed):
            # Joint + v0 mode: 2 v0 + 1 alpha + 3 U_skew + 1 omega = 7 params
            mode = 'joint_with_v0'
        elif params_per_gaussian >= 4:
            # Joint mode: alpha + U_skew + omega (v0 is fixed)
            mode = 'joint'
        else:
            raise ValueError(f"Cannot determine mode: {params_per_gaussian} params per Gaussian")
        
        soln_dict = self.map_from_tensor_to_dict(theta_tensor_init, mode=mode)
        
        # Add fixed parameters that aren't in soln_dict
        # Be careful not to overwrite optimized parameters
        for key, value in self.theta_fixed.items():
            if key not in soln_dict:
                soln_dict[key] = value.copy()
        return soln_dict
    
    
    def refine_initial_velocities_via_newton_raphson(self, soln_dict, res):
        """
        Refine initial velocities using Newton-Raphson method with optimal peak assignment.
        
        This performs two main tasks:
        1. Optimal Assignment: Assign detected peaks to predicted trajectories (closest match)
        2. Newton-Raphson Refinement: Optimize v0 to minimize deviation from assigned peaks
        
        Parameters
        ----------
        soln_dict : dict
            Current solution dictionary with v0s, x0s, a0s
        res : OptimizeResult
            Result from trajectory optimization
        
        Returns
        -------
        soln_dict : dict
            Updated solution with refined v0s
        
        Side Effects
        ------------
        Populates self.peak_data with optimal assignments (assigned_times, assigned_heights, assigned_values)
        """
        # Compute predicted receiver positions for current v0 estimates
        r_maxs_list = self.map_velocities_to_maximising_receivers(self.map_from_tensor_to_dict(res.x))
        
        # Perform optimal assignment of detected peaks to predicted trajectories
        self._assign_peaks_to_trajectories(r_maxs_list)
        
        # Update legacy aliases for plotting compatibility
        # Convert (times, heights) format to old [(time_idx, height)] format
        self.assigned_curve_data = []
        for gaussian_idx in range(self.N):
            times, heights = self.peak_data.get_assignment_data(gaussian_idx)
            # Convert back to old format for plotting compatibility
            data_k = []
            for time_val, height in zip(times, heights):
                # Find time index
                time_idx = torch.where(self.t_observable == time_val)[0]
                if len(time_idx) > 0:
                    data_k.append((time_idx[0].item(), torch.tensor(height, device=self.device)))
            self.assigned_curve_data.append(data_k)
        
        self.assigned_peak_values = self.peak_data.assigned_values
        
        # Visualize assignments
        self.plot_heights_by_assignment()
        
        # Refine v0s using Newton-Raphson on assigned data
        v0s_refined = self._newton_raphson_refinement(soln_dict)
        
        # Update solution dictionary
        soln_dict["v0s"] = [v0.clone().detach() for v0 in v0s_refined]
        return soln_dict
    
    
    def _assign_peaks_to_trajectories(self, r_maxs_list):
        """
        Assign detected peaks to predicted trajectories using nearest-neighbor matching.
        
        For each time point, assigns each detected peak height to the closest
        predicted trajectory curve based on vertical distance.
        
        Parameters
        ----------
        r_maxs_list : list of torch.Tensor
            Predicted receiver positions for each Gaussian, shape (n_times, 2)
        
        Side Effects
        ------------
        Populates self.peak_data with optimal assignments
        """
        # Get projection data for peak value extraction
        proj_data = self.proj_data if self.n_sources == 1 else self.proj_data
        
        # Process each time point
        for time_idx, detected_heights in enumerate(self.peak_data.get_heights_sorted_by_time()):
            # Match each detected height to closest predicted trajectory
            for height in detected_heights:
                # Compute distance to each Gaussian's predicted position
                distances = [
                    torch.abs(trajectory[time_idx, 1] - height).item() 
                    for trajectory in r_maxs_list
                ]
                gaussian_idx = np.argmin(distances)
                
                # Extract actual peak value from projection data
                receiver_heights = torch.tensor(
                    [r[1].item() for r in self.receivers[0]], 
                    dtype=torch.float64, 
                    device=self.device
                )
                receiver_idx = torch.argmin(torch.abs(receiver_heights - height)).item()
                peak_value = proj_data[time_idx, receiver_idx].item()
                
                # Store the optimal assignment
                time_val = self.t_observable[time_idx].item()
                self.peak_data.add_optimal_assignment(
                    gaussian_idx, time_val, height, peak_value
                )
    
    
    def _newton_raphson_refinement(self, soln_dict):
        """
        Refine v0 estimates using Newton-Raphson optimization.
        
        For each Gaussian, minimizes the sum of squared deviations between
        assigned receiver heights and predicted trajectory.
        
        Parameters
        ----------
        soln_dict : dict
            Current solution with x0s, v0s, a0s
        
        Returns
        -------
        v0s_refined : list of torch.Tensor
            Refined initial velocities with gradients enabled
        """
        v0s_refined = []
        r0 = self.receivers[0][0][0]  # Fixed x-coordinate of receiver line
        
        for gaussian_idx in range(self.N):
            # Extract assigned data for this Gaussian
            times, heights = self.peak_data.get_assignment_data(gaussian_idx)
            
            # Convert to tensors
            t_obs = torch.tensor(times, dtype=torch.float64, device=self.device)
            receivers = [
                torch.tensor([r0, h], dtype=torch.float64, device=self.device) 
                for h in heights
            ]
            
            # Get fixed parameters
            x0_k = soln_dict['x0s'][gaussian_idx]
            a0_k = soln_dict['a0s'][gaussian_idx]
            v0_k_init = soln_dict['v0s'][gaussian_idx]
            
            # Optimize v0 using Newton-Raphson
            v0_k_refined = NewtonRaphsonLBFGS(
                self.isotropic_derivative_function_over_all_times,
                v0_k_init,
                t_obs,
                receivers,
                self.sources[0],
                x0_k,
                a0_k
            )
            
            v0s_refined.append(v0_k_refined.requires_grad_(True))
        
        return v0s_refined
    
    
    def _generate_peak_pattern_for_omega(self, alpha, U_skew, omega, x0, v0, a0, times, gaussian_idx):
        """
        Generate predicted projection peak pattern for a given omega.
        
        This simulates what the projection peaks would look like if the Gaussian
        had a specific rotation rate omega, given its trajectory and morphology.
        
        Parameters
        ----------
        alpha : float
            Peak height coefficient
        U_skew : torch.Tensor, shape (d, d)
            Covariance structure matrix
        omega : float
            Angular velocity (Hz) to test
        x0, v0, a0 : torch.Tensor, shape (d,)
            Trajectory parameters
        times : torch.Tensor, shape (n_times,)
            Time points where peaks were observed
        gaussian_idx : int
            Index of Gaussian (for accessing correct sources/receivers)
        
        Returns
        -------
        peak_values : torch.Tensor, shape (n_times,)
            Predicted peak values at each time point
        """
        device = self.device
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
        
        peak_values = []
        
        # Get source and receiver line (use first source for all Gaussians)
        source = self.sources[0]
        receiver_line = self.receivers[0]
        
        for t_n in times:
            # Position at time t: μ(t) = x0 + v0*t + 0.5*a0*t²
            mu_t = x0 + v0 * t_n + 0.5 * a0 * t_n**2
            
            # Rotation matrix at time t: R(t) = R(2π·ω·t)
            angle = 2 * torch.pi * omega * t_n
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            R_t = torch.stack([
                torch.stack([cos_a, -sin_a]),
                torch.stack([sin_a, cos_a])
            ])
            
            # Rotated covariance structure: U(t) = U_skew @ R(t)^T
            U_rot = U_skew @ R_t.T
            
            # Compute projection at each receiver position
            projections = []
            for receiver in receiver_line:
                r_minus_s = receiver - source
                r_minus_s_hat = r_minus_s / torch.norm(r_minus_s)
                
                # Projection formula components
                U_r_hat = U_rot @ r_minus_s_hat
                U_r = U_rot @ r_minus_s
                U_mu = U_rot @ (source - mu_t)
                
                norm_term = torch.norm(U_r_hat)
                quotient = sqrt_pi * alpha / (norm_term + 1e-10)
                
                inner_prod = torch.dot(U_r.squeeze(), U_mu)
                inner_prod_sq = inner_prod ** 2
                divisor = torch.norm(U_r) ** 2 + 1e-10
                subtractor = torch.norm(U_mu) ** 2
                
                exp_arg = inner_prod_sq / divisor - subtractor
                proj = quotient * torch.exp(exp_arg)
                projections.append(proj)
            
            # Peak is maximum projection across receiver line
            peak_values.append(torch.max(torch.stack(projections)))
        
        return torch.stack(peak_values)
    
    
    
    '--------------------------------------------------------------------------------------------------------------------------'
    '------------------------------------------------- PLOTTING FUNCTIONS -----------------------------------------------------'
    '--------------------------------------------------------------------------------------------------------------------------'    
    def plot_trajectory_estimations(self, res):
        """
        Plots the estimated maximizing receiver heights over time for each Gaussian cluster.
        Saves the plot to the output directory.
        """
        r_maxs_list = self.map_velocities_to_maximising_receivers(self.map_from_tensor_to_dict(res.x))
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        times = self.t_observable
        min_rcvr_height = self.receivers[0][-1][1]; max_rcvr_height = self.receivers[0][0][1]
        for k in range(self.N):
            # Extract height component (index 1) from predicted receiver positions
            predicted_heights_k = r_maxs_list[k][:, 1]
            mask = (predicted_heights_k >= min_rcvr_height) & (predicted_heights_k <= max_rcvr_height) 

            times_k = self.t_observable[mask]
            predicted_heights_k = predicted_heights_k[mask]
            plt.plot(times_k, predicted_heights_k.cpu().detach().numpy(), label=f'Cluster {k}', lw=1)
                
            rcvr_heights = torch.zeros(len(self.maximising_rcvrs[k]), dtype=torch.float64, device=self.device)
            for i in range(len(self.maximising_rcvrs[k])):
                rcvr_heights[i] = self.maximising_rcvrs[k][i][1]
            ax.scatter(self.t_obs_by_cluster[k].cpu(), rcvr_heights.cpu(), label=k, s=10, color='black')
            
        plt.xlabel('Time', fontsize=self.plot_label_fontsize)
        plt.ylabel('Height', fontsize=self.plot_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_tick_fontsize)
        
        # Save the plot
        filename = self.output_dir / f'trajectory_estimations_K{self.N}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        
    def plot_heights_by_assignment(self, true_data=False):
        """
        Plots the receiver heights assigned to each Gaussian cluster over time.
        Saves the plot to the output directory.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for k, data_k in enumerate(self.assigned_curve_data):
            inds = [item[0] for item in data_k]
            heights = [item[1].item() for item in data_k]
            self.t_obs = self.t_observable[inds].cpu().numpy()
            ax.scatter(self.t_obs, heights, s=10)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_tick_fontsize)
        ax.set_xlabel('Time', fontsize=self.plot_label_fontsize)
        ax.set_ylabel('Assigned Receiver Heights', fontsize=self.plot_label_fontsize)
        if true_data:
            ax.set_ylabel('True Receiver Heights', fontsize=self.plot_title_fontsize)
        
        # Save the plot
        suffix = '_true_data' if true_data else ''
        filename = self.output_dir / f'heights_by_assignment_K{self.N}{suffix}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        
    def plot_raw_receiver_heights(self):
        """
        Plots the raw, unassigned receiver heights where peaks were detected.
        Saves the plot to the output directory.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Iterate through the dictionary of unassigned peak heights
        for time_val, heights in self.time_rcvr_heights_dict_non_empty.items():
            if heights:
                times = [time_val] * len(heights)
                height_vals = [h.item() for h in heights]
                ax.scatter(times, height_vals, s=10, color='black')

        ax.set_xlabel('Time', fontsize=self.plot_label_fontsize)
        ax.set_ylabel('Height', fontsize=self.plot_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=self.plot_tick_fontsize)
        ax.grid(True, alpha=0.3, linestyle='--')

        filename = self.output_dir / f'raw_receiver_heights_K{self.N}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()