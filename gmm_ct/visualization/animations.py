"""
    Plotting and animation codes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle
from ..core.reconstruction import GMM_reco
from torchmin import minimize

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sns.set_theme()


'--------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------- GMM_PLOTTING CLASS -------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'
class GMM_plotting(GMM_reco):
    """
    Plotting class that inherits from GMM_reco superclass.
    Contains all plotting and visualization methods for GMM reconstruction.
    """
    
    def __init__(self, d, K, sources, receivers, device=None):
        """
        Initialize the GMM plotting class.
        
        Parameters:
        - d: Dimensionality 
        - K: Number of Gaussians
        - sources: Source locations
        - receivers: Receiver locations  
        - device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(d, K, sources, receivers, device)
    
    
    def plot_raw_rcvr_heights_against_time(self, maximising_rcvrs, t_obs_by_cluster, min_rcvrs, max_rcvrs, t_min, t_max):
        fig, ax = plt.subplots(figsize=(12, 9))
        for k in range(self.K):
            if len(maximising_rcvrs[k]) > 0 and t_obs_by_cluster[k].numel() > 0:
                rcvr_heights = torch.zeros(len(maximising_rcvrs[k]), dtype=torch.float64, device=self.device)
                for i in range(len(maximising_rcvrs[k])):
                    rcvr_heights[i] = maximising_rcvrs[k][i][1]
                ax.scatter(t_obs_by_cluster[k].cpu(), rcvr_heights.cpu(), label=k, s=10)
        ax.hlines(min_rcvrs.cpu(), t_min, t_max, colors='k', linestyles='dashed', label='Min receiver height')
        ax.hlines(max_rcvrs.cpu(), t_min, t_max, colors='r', linestyles='dashed', label='Max receiver height')
        ax.set_xlabel("Time")
        ax.set_ylabel("Height")
        ax.set_title(f"Raw receiver heights for each Gaussian")
        ax.legend()
        plt.show()


    def data_we_base_the_fitting_on_plot(self, times_n_peaks, data_n_peaks_max, full_time_mesh, min_rcvrs, max_rcvrs, t_min, t_max, foo, theta_opt_all):
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in range(self.K):
            soln_curve_n = foo(full_time_mesh, *(theta_opt_all[3 * n : 3 * (n + 1)]))
            ax.scatter(times_n_peaks[n], data_n_peaks_max[n].cpu(), color='blue', s=10, label=f'Assigned to $\\rho_{{{n+1}}}$')
            ax.plot(full_time_mesh.cpu(), soln_curve_n.cpu(), color='orange', label=f'Fitted quadratic for $\\rho_{{{n+1}}}$')
        ax.hlines(min_rcvrs.cpu(), t_min, t_max, colors='k', linestyles='dashed', label='Min receiver height')
        ax.hlines(max_rcvrs.cpu(), t_min, t_max, colors='r', linestyles='dashed', label='Max receiver height')
        ax.set_xlabel("Time")
        ax.set_ylabel("Height")
        ax.set_title(f"Maximising receiver heights with 1 measurement")
        ax.legend()
        plt.show()
        
        
    def full_fit_plot(self, maximising_rcvrs, t_obs_by_cluster, full_time_mesh, min_rcvrs, max_rcvrs, t_min, t_max, foo, theta_opt_all):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in range(self.K):
            rcvr_heights = torch.zeros(len(maximising_rcvrs[k]), dtype=torch.float64, device=self.device)
            soln_curve_k = foo(full_time_mesh, *(theta_opt_all[3 * k : 3 * (k + 1)]))
            for i in range(len(maximising_rcvrs[k])):
                rcvr_heights[i] = maximising_rcvrs[k][i][1]
            ax.scatter(t_obs_by_cluster[k].cpu(), rcvr_heights.cpu(), label=f'Assigned to $\\rho_{{{k+1}}}$', s=10)
            ax.plot(full_time_mesh.cpu(), soln_curve_k.cpu(), label=f'Fitted quadratic for $\\rho_{{{k+1}}}$')
        ax.hlines(min_rcvrs.cpu(), t_min, t_max, colors='k', linestyles='dashed', label='Min receiver height')
        ax.hlines(max_rcvrs.cpu(), t_min, t_max, colors='r', linestyles='dashed', label='Max receiver height')
        ax.set_xlabel("Time")
        ax.set_ylabel("Height")
        ax.set_title(f"Fitted quadratic curves")
        ax.legend()
        plt.show()
        
        
    def assign_the_maximal_rcvrs_to_clusters(self, theta_opt, rcvrs, time_rcvr_heights_dict_non_empty, foo):
        maximising_rcvrs = [[] for _ in range(self.K)]
        t_obs_by_cluster = [[] for _ in range(self.K)]
        distance_from_curves_at_t = torch.zeros(self.K, dtype=torch.float64, device=self.device)
        for time, rcvr_list in time_rcvr_heights_dict_non_empty.items():
            for n_r, rcvr in enumerate(rcvr_list):                
                time_loc = torch.tensor([time], dtype=torch.float64, device=self.device)
                for k in range(self.K):
                    soln_curve_k = foo(time_loc, *(theta_opt[3 * k : 3 * (k + 1)]))
                    distance_from_curves_at_t[k] = torch.abs(soln_curve_k - rcvr)
                assigned_k = torch.argmin(distance_from_curves_at_t).item()
                maximising_rcvrs[assigned_k].append(torch.tensor([rcvrs[0][0], rcvr], dtype=torch.float64, device=self.device))
                t_obs_by_cluster[assigned_k].append(torch.tensor([time], dtype=torch.float64, device=self.device))
        for k in range(self.K):
            t_obs_by_cluster[k] = torch.tensor(t_obs_by_cluster[k], dtype=torch.float64, device=self.device)
        return t_obs_by_cluster, maximising_rcvrs
    
    
    def plot_peak_counts(self, sampled_times, peaks_per_time):
        fig, ax = plt.subplots(figsize=(8, 5))
        for time in sampled_times:
            peak_count = peaks_per_time[time]
            ax.scatter(time, peak_count, color='black', s=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Peaks Detected')
        ax.set_title('Number of Peaks Detected Over Time')
        plt.show()
        
        
    def plot_fitted_U_skews(self, soln_dict_n):
        
        rcvr_heights = torch.zeros(self.n_rcvrs, dtype=torch.float64, device=self.device)
        for i in range(self.n_rcvrs):
            rcvr_heights[i] = self.receivers[0][i][1]

        proj_n = self.generate_projections(self.t_n, soln_dict_n, loss_type=None)[0][0]
        true_proj_n = self.proj_data_n
        plt.figure(figsize=(10, 6))
        plt.plot(rcvr_heights.detach().numpy(), proj_n.detach().numpy(), 'b-', label='Model Projection', linewidth=2)
        plt.plot(rcvr_heights.detach().numpy(), true_proj_n.detach().numpy(), 'r--', label='Observed Projection', linewidth=2)
        # Handle omega formatting - could be tensor, list, or single value
        if hasattr(soln_dict_n['omegas'], 'item'):
            omega_val = soln_dict_n['omegas'].item()
        elif isinstance(soln_dict_n['omegas'], list):
            omega_val = soln_dict_n['omegas'][0] if len(soln_dict_n['omegas']) > 0 else 0.0
        else:
            omega_val = float(soln_dict_n['omegas'])
        
        # Ensure omega_val is a Python float, not a tensor
        if hasattr(omega_val, 'item'):
            omega_val = omega_val.item()
        omega_val = float(omega_val)
        
        # Handle U_skew formatting - could be tensor, list, or array
        if hasattr(soln_dict_n['U_skews'], 'detach'):
            U_skew_vals = soln_dict_n['U_skews'].detach().numpy()
        else:
            U_skew_vals = soln_dict_n['U_skews']
        
        # Convert U_skew_vals to a string representation for display
        if hasattr(U_skew_vals, 'tolist'):
            U_skew_str = str(U_skew_vals.tolist())
        else:
            U_skew_str = str(U_skew_vals)
        
        plt.title(f"Time {self.t_n.item():.3f}, omega_k = {omega_val:.6f}, U_k = {U_skew_str}")
        plt.xlabel('Receiver Height', fontsize=20, fontweight='bold')
        plt.ylabel('Projection Value', fontsize=20, fontweight='bold')
        plt.legend(fontsize=14, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tick_params(labelsize=16)
        plt.show()
        
    
    def omega_loss_plot(self, omega_vals, losses_by_omegas_k, omega_true, k):
        plt.plot(omega_vals, losses_by_omegas_k, label=f"Loss for Gaussian {k}")
        plt.axvline(x=omega_true, color='r', linestyle='--', label='True Omega')
        plt.xlabel(r"$\omega$")
        plt.ylabel("Loss")
        plt.title("Loss vs Omega")
        plt.legend()
        plt.show()
        
        
    def compute_U_skews_based_on_minimizing_omegas(self, omega_solns, saved_theta_dict):
        """Computes all U_skews jointly based on the minimizing omega values.
        """
        # Clear the current_gaussian_index to optimize all U_skews jointly
        if hasattr(self, 'current_gaussian_index'):
            delattr(self, 'current_gaussian_index')
        
        loc_list = []
        for k in range(self.K):
            omega_k = omega_solns[k]
            loc_list.append(omega_k)
        saved_theta_dict['omegas'] = loc_list.copy()
        theta_tensor_init = self.map_from_dict_to_tensor(saved_theta_dict) # Now includes all U_skews
        
        # Optimize for all U_skews jointly with fixed omegas
        res = minimize(self.loss_functions, x0=theta_tensor_init, method='l-bfgs', tol=1e-16, options={'gtol': 1e-16})
        soln_dict = self.map_from_tensor_to_dict(res.x)
        for key, value in self.theta_fixed.items():
            soln_dict[key] = value
        return soln_dict


    def plot_projection_wrt_local_omega_minimum(self, omega_solns, saved_theta_dict):
        soln_dict = self.compute_U_skews_based_on_minimizing_omegas(omega_solns, saved_theta_dict)        
        for n in self.observable_indices:
            self.t_n = self.t[n].unsqueeze(0)
            self.proj_data_n = self.proj_data[n]
            self.plot_fitted_U_skews(soln_dict)
        
        
    def debug_velocity_plot_projections(self, t, theta_dict, proj_data, t_min, t_max):
        """
        Debug plotting function for projections comparison.
        
        Parameters:
        - t: Time vector
        - theta_dict: Parameter dictionary containing v0s and other parameters
        - proj_data: Observed projection data
        - t_min: Minimum time value
        - t_max: Maximum time value
        """
        rcvr_heights = torch.zeros(self.n_rcvrs, dtype=torch.float64, device=self.device)
        for i in range(self.n_rcvrs):
            rcvr_heights[i] = self.receivers[0][i][1]
        for n, t_n in enumerate(t):                
            if t_n < t_min or t_n > t_max:
                pass
            else:
                # Plot the respective projections
                proj_k_n = self.generate_projections([t_n], theta_dict, loss_type=None)[0][0]
                plt.figure(figsize=(8, 5))
                plt.plot(rcvr_heights.detach().cpu(), proj_data[n].detach().cpu(), label='Observed Projection')
                plt.plot(rcvr_heights.detach().cpu(), proj_k_n.detach().numpy(), label='Model Projection')
                plt.title(f"Projection at time {t_n.item()}")
                plt.legend()
                plt.show()
                
                
    def plot_trajectory_estimations(self, r2_maxs_list, assigned_curve_data):
        plt.figure()
        ax = plt.gca()
        for k in range(self.K):
            data_k = assigned_curve_data[k]
            inds = [item[0] for item in data_k]
            times_k = [self.t[self.observable_indices[ind]] for ind in inds]
            sampled_r2_maxs_tensor_k = r2_maxs_list[k][inds]
            plt.plot(times_k, sampled_r2_maxs_tensor_k.cpu().detach().numpy(), label=f'Cluster {k}', lw=1)

            rcvr_heights = torch.zeros(len(self.maximising_rcvrs[k]), dtype=torch.float64, device=self.device)
            for i in range(len(self.maximising_rcvrs[k])):
                rcvr_heights[i] = self.maximising_rcvrs[k][i][1]
            ax.scatter(self.t_obs_by_cluster[k].cpu(), rcvr_heights.cpu(), label=k, s=10, color='black')
        plt.xlabel('Time')
        plt.ylabel('Maximizing Receiver')
        plt.legend()
        plt.show()
        
        
    def plot_trajectory_estimations_alt(self, res):
        r2_maxs_list = self.map_velocities_to_maximising_receivers(self.map_from_tensor_to_dict(res.x))
        # r2_maxs_list = self.map_velocities_to_maximising_receivers(self.map_from_tensor_to_dict(res))
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        times = self.t_observable
        min_rcvr_height = self.receivers[0][-1][1]; max_rcvr_height = self.receivers[0][0][1]
        for k in range(self.K):
            sampled_r2_maxs_tensor_k = r2_maxs_list[k]
            mask = (sampled_r2_maxs_tensor_k >= min_rcvr_height) & (sampled_r2_maxs_tensor_k <= max_rcvr_height)
            sampled_r2_maxs_tensor_k = sampled_r2_maxs_tensor_k[mask]
            times = self.t_observable[mask]
            plt.plot(times, sampled_r2_maxs_tensor_k.cpu().detach().numpy(), label=f'Cluster {k}', lw=1)
                
            rcvr_heights = torch.zeros(len(self.maximising_rcvrs[k]), dtype=torch.float64, device=self.device)
            for i in range(len(self.maximising_rcvrs[k])):
                rcvr_heights[i] = self.maximising_rcvrs[k][i][1]
            ax.scatter(self.t_obs_by_cluster[k].cpu(), rcvr_heights.cpu(), label=k, s=10, color='black')
            
        plt.xlabel('Time')
        plt.ylabel('Maximizing Receiver')
        # plt.legend()
        plt.show()


    def plot_heights_by_assignment(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for k, data_k in enumerate(self.assigned_curve_data):
            inds = [item[0] for item in data_k]
            heights = [item[1].item() for item in data_k]
            self.t_obs = self.t_observable[inds].cpu().numpy()
            ax.scatter(self.t_obs, heights, s=10)
        plt.show()


'--------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

'--------------------------------------------------------------------------------------------------------------------------'
'-------------------------------------------------- MUTUAL FUNCTIONS ---------------------------------------------------==-'
'--------------------------------------------------------------------------------------------------------------------------'
def _setup_figure_and_axes():
    """Setup the matplotlib figure and axes."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)    
    return fig, ax


def _compute_trajectories(theta_true, K, d, t_anim):
    """Compute trajectories for all Gaussians using kinematic equations."""
    trajectories = {}
    trajectories_yz = {}
    trajectories_xy = {}
    
    for k in range(K):
        # Extract initial conditions
        x0 = theta_true['x0s'][k]  # Initial position
        v0 = theta_true['v0s'][k]  # Initial velocity  
        a0 = theta_true['a0s'][k]  # Acceleration
        
        # Kinematic equations: x(t) = x0 + v0*t + 0.5*a0*t^2
        traj_x = x0[0] + v0[0] * t_anim + 0.5 * a0[0] * t_anim**2
        traj_y = x0[1] + v0[1] * t_anim + 0.5 * a0[1] * t_anim**2
        
        if d == 3:
            traj_z = x0[2] + v0[2] * t_anim + 0.5 * a0[2] * t_anim**2
            trajectories[k] = torch.stack([traj_x, traj_y, traj_z], dim=1)
            trajectories_yz[k] = torch.stack([traj_y, traj_z], dim=1)
            trajectories_xy[k] = torch.stack([traj_x, traj_y], dim=1)
        else:
            trajectories[k] = torch.stack([traj_x, traj_y], dim=1)
    
    if d == 2:
        return trajectories
    elif d == 3:
        return trajectories, trajectories_yz, trajectories_xy
    
    
def _plot_trajectories(ax, trajectories, frame, colors, d, style='true'):
    """Plot trajectory lines up to the current frame."""
    for k, traj in trajectories.items():
        color = colors[k]
        current_traj = traj[:frame+1]  # Up to current frame
        
        # Set line style based on whether this is true or estimated
        linestyle = '--' if style == 'estimated' else '-'
        alpha = 0.4 if style == 'estimated' else 0.6
        
        if len(current_traj) > 1:  # Need at least 2 points for a line
            # Handle tensors that may or may not require gradients
            def tensor_to_numpy(tensor):
                return tensor.detach().numpy() if tensor.requires_grad else tensor.numpy()
            
            if d == 3:
                ax.plot(tensor_to_numpy(current_traj[:, 0]), 
                       tensor_to_numpy(current_traj[:, 1]), 
                       tensor_to_numpy(current_traj[:, 2]), 
                       linestyle, color=color, alpha=alpha, linewidth=2)
            else:
                ax.plot(tensor_to_numpy(current_traj[:, 0]), 
                       tensor_to_numpy(current_traj[:, 1]), 
                       linestyle, color=color, alpha=alpha, linewidth=2)


def _get_gaussian_parameters(theta_true, k):
    """Extract and validate Gaussian parameters for the k-th Gaussian."""
    if isinstance(theta_true['alphas'], torch.Tensor):
        if len(theta_true['alphas'].shape) > 1:
            alpha_k = theta_true['alphas'][k]
        elif len(theta_true['alphas'].shape) == 1 and theta_true['alphas'].shape[0] > k:
            alpha_k = theta_true['alphas'][k]
        else:
            alpha_k = theta_true['alphas']
    else:
        alpha_k = theta_true['alphas'][k]
    return alpha_k


def _compute_precision_matrix(theta_true, k, current_time, sources, rcvrs, d, K):
    """Compute the precision matrix for the k-th Gaussian at current time."""
    # Get rotation matrices
    # Extract x0s and a0s from theta_true (they are known physical parameters)
    x0s = theta_true['x0s']
    a0s = theta_true['a0s']
    GMM = GMM_reco(d, K, sources, rcvrs, x0s, a0s, omega_min=0.0, omega_max=10.0)
    rot_mat_funcs = GMM.construct_rotation_matrix_funcs()
    rot_mat_of_t = rot_mat_funcs(current_time, theta_true)
    kth_rot_mat_of_t = rot_mat_of_t[k]
    
    # Precision matrix calculation with proper tensor-to-numpy conversion
    U_k = theta_true["U_skews"][k]
    U_k_np = U_k.detach().numpy() if isinstance(U_k, torch.Tensor) else U_k
    kth_rot_mat_of_t_np = (kth_rot_mat_of_t.detach().numpy() 
                          if isinstance(kth_rot_mat_of_t, torch.Tensor) 
                          else kth_rot_mat_of_t)
    
    U_kR_kT = U_k_np @ (kth_rot_mat_of_t_np.T)
    precision_mat = (U_kR_kT.T) @ U_kR_kT
    
    return precision_mat


def _plot_acquisition_geometry_in_animation(ax, sources, receivers, d, view_type=None):
    """
    Plot acquisition geometry (sources, receivers, and connecting lines) for animation frames.
    
    Parameters:
    - ax: The axes object to plot on.
    - sources: List of source tensors.
    - receivers: List of receiver tensors.
    - d: Dimensionality of the application (2 or 3).
    - view_type: For 3D, specify "yz" or "xy" for 2D projections, None for full 3D.
    """
    markersize = 2
    
    if d == 2:
        # 2D plotting
        for n_s, source in enumerate(sources):
            ax.plot(source[0].item(), source[1].item(), 'ro', markersize=markersize+3, alpha=0.7)
            for n_r, rcvr in enumerate(receivers[n_s]):
                ax.plot(rcvr[0].item(), rcvr[1].item(), 'bo', markersize=markersize, alpha=0.7)
                ax.plot([source[0].item(), rcvr[0].item()], 
                       [source[1].item(), rcvr[1].item()], 
                       color='gold', linewidth=0.5, alpha=0.6, zorder=1)
    
    elif d == 3:
        if view_type == "yz":
            # YZ projection for 3D data
            for n_s, source in enumerate(sources):
                ax.plot(source[1].item(), source[2].item(), 'ro', markersize=markersize+3, alpha=0.7)
                for n_r, rcvr in enumerate(receivers[n_s]):
                    ax.plot(rcvr[1].item(), rcvr[2].item(), 'bo', markersize=markersize, alpha=0.7)
                    ax.plot([source[1].item(), rcvr[1].item()], 
                           [source[2].item(), rcvr[2].item()], 
                           color='gold', linewidth=0.5, alpha=0.6, zorder=1)
        
        elif view_type == "xy":
            # XY projection for 3D data
            for n_s, source in enumerate(sources):
                ax.plot(source[0].item(), source[1].item(), 'ro', markersize=markersize+3, alpha=0.7)
                for n_r, rcvr in enumerate(receivers[n_s]):
                    ax.plot(rcvr[0].item(), rcvr[1].item(), 'bo', markersize=markersize, alpha=0.7)
                    ax.plot([source[0].item(), rcvr[0].item()], 
                           [source[1].item(), rcvr[1].item()], 
                           color='gold', linewidth=0.5, alpha=0.6, zorder=1)
        
        else:
            # Full 3D plotting (if ever needed for 3D axis)
            for n_s, source in enumerate(sources):
                ax.scatter(source[0].item(), source[1].item(), source[2].item(), 
                          c='r', marker='o', s=markersize+30, alpha=0.7)
                for n_r, rcvr in enumerate(receivers[n_s]):
                    ax.scatter(rcvr[0].item(), rcvr[1].item(), rcvr[2].item(), 
                              c='b', marker='o', s=markersize, alpha=0.7)
                    ax.plot([source[0].item(), rcvr[0].item()], 
                           [source[1].item(), rcvr[1].item()], 
                           [source[2].item(), rcvr[2].item()], 
                           color='gold', linewidth=0.5, alpha=0.6, zorder=1)





'--------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------------- ESTIMATION ANIMATIONS ----------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'                
def _plot_gaussians_along_trajectory(ax, trajectories, theta, t, K, sources, rcvrs, d, colors, style='true', view_type=None):
    """
    Plot Gaussian ellipses at multiple time points along the entire trajectory.
    
    Parameters:
    - ax: The axes object to plot on
    - trajectories: List of trajectory tensors for each Gaussian
    - theta: Parameter dictionary containing Gaussian parameters
    - t: Array of time points
    - K: Number of Gaussians
    - sources, rcvrs: Source and receiver geometries
    - d: Dimensionality
    - colors: Color array for each Gaussian
    - style: 'true' or 'estimated' for styling
    - view_type: For 3D, specify "yz" or "xy" projections
    """
    # Sample time points to avoid overcrowding
    n_time_samples = min(10, len(t))  # Maximum 10 ellipses per trajectory
    time_indices = np.linspace(0, len(t)-1, n_time_samples, dtype=int)
    
    for k in range(K):
        for i, time_idx in enumerate(time_indices):
            current_pos = trajectories[k][time_idx]
            center = np.array([current_pos[0].item(), current_pos[1].item()])
            
            # Get Gaussian parameters
            alpha_k = _get_gaussian_parameters(theta, k)
            precision_mat = _compute_precision_matrix(theta, k, t[time_idx], 
                                                    sources, rcvrs, d, K)
            
            # Convert precision matrix to covariance matrix
            try:
                covariance_mat = np.linalg.inv(precision_mat)
                if view_type == "yz" and d == 3:
                    covariance_mat = covariance_mat[1:3, 1:3]
                elif view_type == "xy" and d == 3:
                    covariance_mat = covariance_mat[0:2, 0:2]
                
                # Compute eigenvalues and eigenvectors for ellipse parameters
                eigenvals, eigenvecs = np.linalg.eigh(covariance_mat)
                eigenvals = np.abs(eigenvals)  # Ensure positive
                
                # Use only the outermost confidence level for trajectory visualization
                chi2_val = 4.0  # ~2σ equivalent
                
                # Ellipse dimensions
                width = 2 * np.sqrt(chi2_val * eigenvals[0])
                height = 2 * np.sqrt(chi2_val * eigenvals[1])
                
                # Rotation angle in degrees
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Alpha scaling based on time position (earlier times more faded)
                time_factor = (i + 1) / n_time_samples  # 0.125 to 1.0
                alpha_scale = alpha_k.item() if isinstance(alpha_k, torch.Tensor) else alpha_k
                ellipse_alpha = min(0.6, max(0.1, alpha_scale * time_factor * 0.4))
                
                # Adjust style based on whether this is true or estimated
                if style == 'estimated':
                    ellipse_alpha *= 0.8    # Make estimated ellipses more transparent
                    edge_style = '--'       # Dashed edges for estimates
                    line_width = 0.8
                else:
                    edge_style = '-'        # Solid edges for true
                    line_width = 1.0
                
                # Only label the final ellipse in the trajectory
                if i == len(time_indices) - 1:
                    if d == 2:
                        label_prefix = 'Est G' if style == 'estimated' else 'G'
                        label = f'{label_prefix}{k+1}: '
                        rots_per_second = theta['omegas'][k].item()
                        label = label + f'{rots_per_second:.2f} rots/s'
                    elif d == 3:
                        label_prefix = 'Est G' if style == 'estimated' else 'G'
                        label = f'{label_prefix}{k+1}: '
                        if view_type == "yz":
                            yz_rots_per_second = theta['omegas'][k][2].item()
                            label = label + f"{yz_rots_per_second:.2f} rots/s"
                        elif view_type == "xy":
                            xy_rots_per_second = theta['omegas'][k][0].item()
                            label = label + f"{xy_rots_per_second:.2f} rots/s"
                else:
                    label = None
                    
                # Create and add ellipse
                ellipse = Ellipse(center, width, height, angle=angle, 
                                facecolor=colors[k], edgecolor='black',
                                alpha=ellipse_alpha, linewidth=line_width,
                                linestyle=edge_style, label=label)
                ax.add_patch(ellipse)
                
            except (np.linalg.LinAlgError, IndexError, KeyError):
                # Fallback for ill-conditioned matrices or parameter issues
                radius = 0.05 * np.sqrt(alpha_k.item() if isinstance(alpha_k, torch.Tensor) else alpha_k)
                alpha_fallback = 0.3 if style == 'estimated' else 0.4
                circle = Circle(center, radius, facecolor=colors[k], 
                              edgecolor='black', alpha=alpha_fallback,
                              label=f'Gaussian {k+1}' if i == len(time_indices) - 1 else None)
                ax.add_patch(circle)
    

def animate_GMM_evolution(theta_hat, theta_true, d, K, sources, rcvrs, t, 
                         show_trajectory=True, show_gaussians=True, show_acquisition_geometry=True,
                         frames_per_iteration=10, pause_frames=5):
    """
    Creates an animation showing the evolution of Gaussian mixture model parameters through optimization iterations.
    
    Parameters:
    - theta_hat: List of parameter dictionaries from optimization iterations.
    - theta_true: Dictionary containing the true parameters of the GMM.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - show_trajectory: Whether to show the trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian visualization (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    - frames_per_iteration: Number of frames to show each iteration (default: 10).
    - pause_frames: Number of frames to pause at the end of each iteration (default: 5).
    
    Returns:
    - anim: The animation object.
    
    Notes:
    - Shows the evolution of Gaussian parameters through optimization iterations
    - Each iteration is displayed for multiple frames with a pause between iterations
    - True parameters are shown in solid lines, current estimate in dashed lines
    - Final estimate is highlighted differently
    """
    
    if not theta_hat or len(theta_hat) == 0:
        raise ValueError("theta_hat must contain at least one parameter dictionary")
    
    '-------------'
    # Setup figure
    '-------------'
    if d == 2:
        fig, ax = _setup_figure_and_axes()
        
        # Calculate bounds
        x_min = sources[0][0].item() - 0.5
        all_rcvr_x_coords = [rcvr[0].item() for rcvr in rcvrs[0]]
        x_max = max(all_rcvr_x_coords) + 0.5
        
        all_rcvr_y_coords = [rcvr[1].item() for rcvr in rcvrs[0]]
        y_max = max(all_rcvr_y_coords) + 0.5
        y_min = min(all_rcvr_y_coords) - 0.5
        
    elif d == 3:
        fig_yz, ax_yz = _setup_figure_and_axes()
        fig_xy, ax_xy = _setup_figure_and_axes()
    else:
        raise ValueError("Only 2D and 3D are currently supported")
    
    
    '-----------------------------------------------------'
    # Compute all trajectories for each parameter estimate
    '-----------------------------------------------------'
    trajectories_evolution = []
    for theta in theta_hat:
        if d == 2:
            traj = _compute_trajectories(theta, K, d, t)
            trajectories_evolution.append(traj)
        elif d == 3:
            traj, traj_yz, traj_xy = _compute_trajectories(theta, K, d, t)
            trajectories_evolution.append((traj, traj_yz, traj_xy))
    
    '----------------------------'
    # Compute the true trajectory
    '----------------------------'
    if d == 2:
        true_trajectories = _compute_trajectories(theta_true, K, d, t)
    elif d == 3:
        true_trajectories, true_trajectories_yz, true_trajectories_xy = _compute_trajectories(theta_true, K, d, t)
    
    # Setup colors
    colors = cm.rainbow(np.linspace(0, 1, K))
    # colors = ["blue", "red", "green"]
    
    # Calculate total frames
    n_iterations = len(theta_hat)
    total_frames = n_iterations * (frames_per_iteration + pause_frames)
    
    def animate(frame):
        """Animation function called for each frame."""
        # Determine which iteration we're showing
        iteration_idx = frame // (frames_per_iteration + pause_frames)
        iteration_idx = min(iteration_idx, n_iterations - 1)  # Clamp to valid range
        
        # Determine if we're in pause phase
        frame_in_iteration = frame % (frames_per_iteration + pause_frames)
        is_paused = frame_in_iteration >= frames_per_iteration
        
        current_theta = theta_hat[iteration_idx]
        # print(f"Animating frame {frame+1}/{total_frames}, Iteration {iteration_idx + 1}/{n_iterations}")
        
        if d == 2:
            ax.clear()
            
            '------------------'
            # Plot trajectories
            '------------------'
            # Plot true trajectories (faded)
            if show_trajectory:
                _plot_trajectories(ax, true_trajectories, len(t) - 1, colors, d, style='true')
                for line in ax.lines:
                    line.set_alpha(0.3)
            
            # Plot current iteration trajectory
            if show_trajectory and iteration_idx < len(trajectories_evolution):
                current_trajectories = trajectories_evolution[iteration_idx]
                _plot_trajectories(ax, current_trajectories, len(t) - 1, colors, d, style='estimated')
            
            '---------------'
            # Plot Gaussians
            '---------------'
            if show_gaussians:
                # True Gaussians along trajectory (faded)
                _plot_gaussians_along_trajectory(ax, true_trajectories, theta_true, t, K, sources, rcvrs, d, colors, style='true')
                for patch in ax.patches:
                    patch.set_alpha(0.15)
                
                # Current iteration estimated Gaussians along trajectory
                if iteration_idx < len(trajectories_evolution):
                    current_trajectories = trajectories_evolution[iteration_idx]
                    _plot_gaussians_along_trajectory(ax, current_trajectories, current_theta, t, K, sources, rcvrs, d, colors, style='estimated')
            
            # Plot acquisition geometry
            if show_acquisition_geometry:
                _plot_acquisition_geometry_in_animation(ax, sources, rcvrs, d)
            
            # Update plot aesthetics
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add legend to distinguish between true and estimated parameters
            if show_gaussians and iteration_idx < len(trajectories_evolution):
                
                from matplotlib.patches import Patch
                legend_elements = [
                    # Patch(edgecolor='black', alpha=0.3, linestyle='-', label='True Gaussians'),
                    # Patch(edgecolor='black', label='True Gaussians'),
                    # Patch(facecolor=colors[k], edgecolor='black', alpha=0.3, linestyle='-', label='True Gaussians'),
                # legend_elements = [
                ]
                if show_trajectory:
                    from matplotlib.lines import Line2D
                    legend_elements.extend([
                        Line2D([0], [0], color='gray', alpha=0.5, linestyle='-', label='True traj', lw=2),
                        Line2D([0], [0], color='gray', alpha=0.8, linestyle='--', label='Est. traj', lw=2)
                    ])
                
                # ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.8, fancybox=True, shadow=True)
            
            # Add iteration information
            is_final = iteration_idx == n_iterations - 1
            status = "FINAL" if is_final else f"Iteration {iteration_idx + 1}/{n_iterations}"
            if is_paused:
                status += " (PAUSED)"
            
            ax.set_title(f'GMM Evolution - {status}', fontsize=22, fontweight='bold', pad=15)
            ax.set_xlabel('X Position', fontsize=20, fontweight='bold')
            ax.set_ylabel('Y Position', fontsize=20, fontweight='bold')
            ax.tick_params(labelsize=16)
            
        return []
    
    # Create animation
    if d == 2:
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=200, repeat=True, blit=False)
        return anim
    elif d == 3:
        anim_yz = FuncAnimation(fig_yz, animate, frames=total_frames, interval=200, repeat=True, blit=False)
        anim_xy = FuncAnimation(fig_xy, animate, frames=total_frames, interval=200, repeat=True, blit=False)
        return anim_yz, anim_xy


def save_GMM_evolution_animation(theta_hat, theta_true, d, K, sources, rcvrs, t,
                                filename='gmm_evolution.mp4', show_trajectory=True, show_gaussians=True,
                                show_acquisition_geometry=True, frames_per_iteration=10, pause_frames=5, fps=18):
    """
    Creates and saves an animation showing the evolution of GMM parameters through optimization iterations.
    
    Parameters:
    - theta_hat: List of parameter dictionaries from optimization iterations.
    - theta_true: Dictionary containing the true parameters of the GMM.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - filename: Output filename for the animation (default: 'gmm_evolution.mp4').
    - show_trajectory: Whether to show the trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian ellipses/spheres (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    - frames_per_iteration: Number of frames to show each iteration (default: 10).
    - pause_frames: Number of frames to pause at the end of each iteration (default: 5).
    - fps: Frames per second for the saved animation (default: 18).
    """
    if d == 2:
        anim = animate_GMM_evolution(theta_hat, theta_true, d, K, sources, rcvrs, t,
                                   show_trajectory=show_trajectory, show_gaussians=show_gaussians,
                                   show_acquisition_geometry=show_acquisition_geometry,
                                   frames_per_iteration=frames_per_iteration, pause_frames=pause_frames)
        # Save animation
        if filename.endswith('.gif'):
            anim.save(filename, writer='pillow', fps=fps)
        elif filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=fps)
        else:
            # Default to gif
            anim.save(filename + '.gif', writer='pillow', fps=fps)
        
        print(f"Evolution animation saved as {filename}")
        print(f"- Shows {len(theta_hat)} optimization iterations")
        print("- True parameters shown faded in background")
        print("- Current iteration parameters highlighted")
        return anim
    elif d == 3:
        anim_yz, anim_xy = animate_GMM_evolution(theta_hat, theta_true, d, K, sources, rcvrs, t,
                                               show_trajectory=show_trajectory, show_gaussians=show_gaussians,
                                               show_acquisition_geometry=show_acquisition_geometry,
                                               frames_per_iteration=frames_per_iteration, pause_frames=pause_frames)
        # Save animations
        if filename.endswith('.gif'):
            anim_yz.save(filename.replace('.gif', '_yz.gif'), writer='pillow', fps=fps)
            anim_xy.save(filename.replace('.gif', '_xy.gif'), writer='pillow', fps=fps)
        elif filename.endswith('.mp4'):
            anim_yz.save(filename.replace('.mp4', '_yz.mp4'), writer='ffmpeg', fps=fps)
            anim_xy.save(filename.replace('.mp4', '_xy.mp4'), writer='ffmpeg', fps=fps)
        else:
            # Default to gif
            anim_yz.save(filename + '_yz.gif', writer='pillow', fps=fps)
            anim_xy.save(filename + '_xy.gif', writer='pillow', fps=fps)
        
        base_name = filename.replace('.gif', '').replace('.mp4', '')
        print(f"Evolution animations saved as {base_name}_yz and {base_name}_xy")
        print(f"- Shows {len(theta_hat)} optimization iterations")
        print("- True parameters shown faded in background")
        print("- Current iteration parameters highlighted")
        return anim_yz, anim_xy
    



'--------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------------ ORIGINAL MOTION CODE ----------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'
def _plot_gaussian_ellipses(ax, trajectories, frame, theta_true, 
                          current_time, K, sources, rcvrs, d, colors, view_type=None, style='true'):
    """Plot Gaussian ellipses for all Gaussians at the current frame."""
    
    for k in range(K):
        current_pos = trajectories[k][frame]
        center = np.array([current_pos[0].item(), current_pos[1].item()])
        
        # Get Gaussian parameters
        alpha_k = _get_gaussian_parameters(theta_true, k)
        precision_mat = _compute_precision_matrix(theta_true, k, current_time, 
                                                sources, rcvrs, d, K)
        
        # Convert precision matrix to covariance matrix
        try:
            # Ensure precision matrix is well-conditioned
            covariance_mat = np.linalg.inv(precision_mat)
            if view_type == "yz":
                covariance_mat = covariance_mat[1:3, 1:3]
            elif view_type == "xy":
                covariance_mat = covariance_mat[0:2, 0:2]
            
            # Compute eigenvalues and eigenvectors for ellipse parameters
            eigenvals, eigenvecs = np.linalg.eigh(covariance_mat)
            
            # Ensure eigenvalues are positive (numerical stability)
            eigenvals = np.abs(eigenvals)
            
            # Calculate ellipse parameters for different confidence levels
            confidence_levels = [0.39, 0.86, 0.99]  # ~1σ, 2σ, 3σ equivalent
            chi2_vals = [1.0, 4.0, 9.0]  # Chi-squared values for 2D
            
            for i, (conf, chi2) in enumerate(zip(confidence_levels, chi2_vals)):
                # Ellipse dimensions (semi-major and semi-minor axes)
                width = 2 * np.sqrt(chi2 * eigenvals[0])
                height = 2 * np.sqrt(chi2 * eigenvals[1])
                
                # Rotation angle in degrees
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Alpha scaling based on confidence level and Gaussian weight
                alpha_scale = alpha_k.item() if isinstance(alpha_k, torch.Tensor) else alpha_k
                ellipse_alpha = min(0.8, max(0.1, alpha_scale * (1.0 - i * 0.2)))
                
                # Adjust style based on whether this is true or estimated
                if style == 'estimated':
                    ellipse_alpha *= 0.6  # Make estimated ellipses more transparent
                    edge_style = '--'      # Dashed edges for estimates
                    line_width = 1.0 - i * 0.2
                else:
                    edge_style = '-'       # Solid edges for true
                    line_width = 1.5 - i * 0.3
                
                # Label for the ellipse
                if d == 2:
                    if i == 0:
                        label_prefix = r'Est $\rho_' if style == 'estimated' else r'$\rho_'
                        label = f'{label_prefix}{{{k+1}}}$: '
                        rots_per_second = theta_true['omegas'][k].item()
                        label = label + f'{rots_per_second:.2f} rots/s'
                    else:
                        label = None
                elif d == 3:
                    if i == 0:
                        label_prefix = r'Est $\rho_' if style == 'estimated' else r'$\rho_'
                        label = f'{label_prefix}{{{k+1}}}$: '
                        if view_type == "yz":
                            yz_rots_per_second = theta_true['omegas'][k][2].item()
                            label = label + f"{yz_rots_per_second:.2f} rots/s"
                        elif view_type == "xy":
                            xy_rots_per_second = theta_true['omegas'][k][0].item()
                            label = label + f"{xy_rots_per_second:.2f} rots/s"
                    else:
                        label = None
                    
                # Create and add ellipse
                ellipse = Ellipse(center, width, height, angle=angle, 
                                facecolor=colors[k], edgecolor='black',
                                alpha=ellipse_alpha, linewidth=line_width,
                                linestyle=edge_style, label=label)
                ax.add_patch(ellipse)
                
        except np.linalg.LinAlgError:
            # Fallback for ill-conditioned matrices: plot simple circles
            radius = 0.1 * np.sqrt(alpha_k.item() if isinstance(alpha_k, torch.Tensor) else alpha_k)
            circle = Circle(center, radius, facecolor=colors[k], 
                              edgecolor='black', alpha=0.6,
                              label=f'Gaussian {k+1}')
            ax.add_patch(circle)


def _plot_projection_data(ax, projs_by_source, t, frame, sources, receivers, d, style='true'):
    """Plot projection data for the current frame."""
    if d == 2:
        for n_s, source in enumerate(sources):
            projs = projs_by_source[n_s]
            max_proj_value = projs.max().item()
            receiver_heights = np.array([rcvr[1].item() for rcvr in receivers[n_s]])[::-1]
            min_height, max_height = receiver_heights[0], receiver_heights[-1]
            tau = -min_height - max_height
            
            # Set line style based on whether this is true or estimated
            if style == 'estimated':
                line_style = '--'
                line_color = 'red'
                alpha = 0.7
                line_width = 1.5
            else:
                line_style = '-'
                line_color = 'black'
                alpha = 1.0
                line_width = 2
            
            # Handle tensors that may or may not require gradients
            def tensor_to_numpy(tensor):
                return tensor.detach().numpy() if tensor.requires_grad else tensor.numpy()
            
            ax.plot(tensor_to_numpy(projs[frame]), -receiver_heights - tau, 
                   linestyle=line_style, color=line_color, alpha=alpha, linewidth=line_width)
            ax.xaxis.set_inverted(True)
            
        ax.set_xlim(0, max_proj_value * 1.05)  # Add small margin
        title_prefix = 'Est. ' if style == 'estimated' else ''
        ax.set_title(f'{title_prefix}Projection at Time {t[frame]:.2f}s')
        ax.set_ylabel('')
        ax.set_xlabel('Projection Value')
        
        # Y-limits will be synchronized in the main animate function
    elif d == 3:  # Should be a 2D projection
        cmap = 'plasma' if style == 'estimated' else 'viridis'
        proj_data = projs_by_source[0][frame]
        proj_values = proj_data.detach().numpy() if proj_data.requires_grad else proj_data.numpy()
        ax.scatter(sources[0][0].item(), sources[0][1].item(), c=proj_values, cmap=cmap)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        title_prefix = 'Est. ' if style == 'estimated' else ''
        ax.set_title(f'{title_prefix}Projection data (t = {t[frame]:.2f}s')


def _update_plot_aesthetics(ax, trajectories, K, d, current_time, view_type=None, force_equal_aspect=False):
    """Update plot labels, limits, and aesthetics."""
    # Calculate bounds for axis limits
    all_trajs = torch.cat([trajectories[k] for k in range(K)], dim=0)
    margin = 0.1 * (all_trajs.max() - all_trajs.min())
    
    if d == 3:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(all_trajs[:, 0].min() - margin, all_trajs[:, 0].max() + margin)
        ax.set_ylim(all_trajs[:, 1].min() - margin, all_trajs[:, 1].max() + margin)
        ax.set_zlim(all_trajs[:, 2].min() - margin, all_trajs[:, 2].max() + margin)
    elif view_type == "yz":
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Height (m)')
        ax.set_xlim(all_trajs[:, 0].min() - margin, all_trajs[:, 0].max() + margin)
        ax.set_ylim(all_trajs[:, 1].min() - margin, all_trajs[:, 1].max() + margin)
        if force_equal_aspect:
            ax.set_aspect('equal')
    elif view_type == "xy":
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_xlim(all_trajs[:, 0].min() - margin, all_trajs[:, 0].max() + margin)
        ax.set_ylim(all_trajs[:, 1].min() - margin, all_trajs[:, 1].max() + margin)
        if force_equal_aspect:
            ax.set_aspect('equal')
    else:
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Height (m)')
        ax.set_xlim(all_trajs[:, 0].min() - margin, all_trajs[:, 0].max() + margin)
        ax.set_ylim(all_trajs[:, 1].min() - margin, all_trajs[:, 1].max() + margin)
        # Don't set equal aspect in animation to prevent subplot size mismatches
        if force_equal_aspect:
            ax.set_aspect('equal')
    
    if d == 2 and view_type is None:
        ax.set_title(f'Animated {d}-dim {K}-GMM in Motion (t = {current_time:.2f}s)')
    elif d == 2 and view_type == "yz":
        ax.set_title(f'2-dim animated slice of {3}-dim {K}-GMM in Motion (t = {current_time:.2f}s)\n(Side YZ view)')
    elif d == 2 and view_type == "xy":
        ax.set_title(f'2-dim animated slice of {3}-dim {K}-GMM in Motion (t = {current_time:.2f}s)\n(Bird\'s eye XY view)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.5)
    
    
def animate_GMM_motion(theta_true, d, K, sources, rcvrs, t, projs_by_source, 
                       title="GMM Animation", show_trajectory=True, show_gaussians=True, show_acquisition_geometry=True, plot_projection_data=True,
                       theta_hat=None, sim_projs=None, show_estimates=False):
    """
    Creates an animation of the Gaussian mixture model in motion.
    
    Parameters:
    - theta_true: Dictionary containing the true parameters of the GMM.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - projs_by_source: True projection data for visualization.
    - title: Animation title (default: "GMM Animation").
    - show_trajectory: Whether to show the trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian visualization (default: True).
                     For 2D: confidence ellipses showing covariance structure
                     For 3D: confidence ellipsoids/spheres showing 3D covariance
    - show_acquisition_geometry: Whether to show acquisition geometry (sources, receivers, connecting lines) (default: True).
    - plot_projection_data: Whether to show projection data plots alongside the GMM visualization (default: True).
    - theta_hat: List of estimated parameter dictionaries from optimization iterations (default: None).
    - sim_projs: List of simulated projections corresponding to theta_hat estimates (default: None).
    - show_estimates: Whether to overlay estimated parameters and projections on the animation (default: False).
    
    Returns:
    - anim: The animation object.
    
    Notes:
    - 2D visualization uses mathematically precise ellipses with multiple confidence levels
    - 3D visualization uses ellipsoids or spheres to represent Gaussian shapes
    - Both approaches provide intuitive understanding of Gaussian shape and orientation
    - Acquisition geometry shows the experimental setup with sources, receivers, and ray paths
    - When show_estimates=True, estimated Gaussians are overlaid in a different style (dashed lines, different colors)
    """
    
    '---------------------------------------'
    # 1. SETUP: Time points and trajectories
    '---------------------------------------'
    n_frames = len(t)
    if d == 2:
        trajectories = _compute_trajectories(theta_true, K, d, t)
        # Compute estimated trajectories if provided
        if show_estimates and theta_hat is not None:
            # Use the final estimate for overlay
            final_theta_hat = theta_hat[-1] if isinstance(theta_hat, list) else theta_hat
            trajectories_hat = _compute_trajectories(final_theta_hat, K, d, t)
    elif d == 3:
        trajectories, trajectories_yz, trajectories_xy = _compute_trajectories(theta_true, K, d, t)
        if show_estimates and theta_hat is not None:
            final_theta_hat = theta_hat[-1] if isinstance(theta_hat, list) else theta_hat
            trajectories_hat, trajectories_hat_yz, trajectories_hat_xy = _compute_trajectories(final_theta_hat, K, d, t)
    
    
    '---------------------------------------'
    # 2. SETUP: Figure and plotting elements
    '---------------------------------------'
    if d == 2:
        # Standard layout - overlay estimates on the same plot if provided
        fig, ax = _setup_figure_and_axes()
        if plot_projection_data:
            fig, (ax, proj_ax) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            fig.subplots_adjust(wspace=0.1)
        
        # Set overall figure title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        x_min = sources[0][0].item() - 0.5
        all_rcvr_x_coords = [rcvr[0].item() for rcvr in rcvrs[0]]
        x_max = max(all_rcvr_x_coords) + 0.5
        
        all_rcvr_y_coords = [rcvr[1].item() for rcvr in rcvrs[0]]
        y_max = max(all_rcvr_y_coords) + 0.5
        y_min = min(all_rcvr_y_coords) - 0.5
        # y_min, y_max = -3, 4  # Fixed y-limits for consistency
    elif d == 3:
        fig_yz, ax_yz = _setup_figure_and_axes()
        fig_xy, ax_xy = _setup_figure_and_axes()
        
        # Set overall figure titles for both views
        fig_yz.suptitle(f"{title} - Side View (YZ)", fontsize=14, fontweight='bold')
        fig_xy.suptitle(f"{title} - Top View (XY)", fontsize=14, fontweight='bold')
        
    colors = cm.rainbow(np.linspace(0, 1, K))
    
    
    '--------------------------------------------'
    # 3. ANIMATION: Create the animation function
    '--------------------------------------------'
    def animate(frame):
        """Animation function called for each frame."""
        current_time = t[frame]
        if d == 2:
            ax.clear()
            if plot_projection_data:
                proj_ax.clear()
        else:
            ax_yz.clear()
            ax_xy.clear()
        
        # Plot trajectories if requested
        if show_trajectory:
            if d == 2:
                _plot_trajectories(ax, trajectories, frame, colors, d)
                # Plot estimated trajectories if available
                if show_estimates and theta_hat is not None and 'trajectories_hat' in locals():
                    _plot_trajectories(ax, trajectories_hat, frame, colors, d, style='estimated')
            elif d == 3: # We're running for 2D trajectories but twice when it's 3D
                _plot_trajectories(ax_yz, trajectories_yz, frame, colors, 2)
                _plot_trajectories(ax_xy, trajectories_xy, frame, colors, 2)
                # Plot estimated trajectories if available
                if show_estimates and theta_hat is not None and 'trajectories_hat_yz' in locals():
                    _plot_trajectories(ax_yz, trajectories_hat_yz, frame, colors, 2, style='estimated')
                    _plot_trajectories(ax_xy, trajectories_hat_xy, frame, colors, 2, style='estimated')
        
        # Plot Gaussian visualization if requested  
        if show_gaussians:
            if d == 2:
                _plot_gaussian_ellipses(ax, trajectories, frame, 
                                      theta_true, current_time, K, sources, rcvrs, d, colors)
                # Plot estimated Gaussians if available
                if show_estimates and theta_hat is not None:
                    final_theta_hat = theta_hat[-1] if isinstance(theta_hat, list) else theta_hat
                    _plot_gaussian_ellipses(ax, trajectories_hat, frame, 
                                          final_theta_hat, current_time, K, sources, rcvrs, d, colors, style='estimated')
            elif d == 3:
                # The other 3D function was here
                _plot_gaussian_ellipses(ax_yz, trajectories_yz, frame, 
                                      theta_true, current_time, K, sources, rcvrs, d, colors, view_type="yz")
                _plot_gaussian_ellipses(ax_xy, trajectories_xy, frame, 
                                      theta_true, current_time, K, sources, rcvrs, d, colors, view_type="xy")
                # Plot estimated Gaussians if available
                if show_estimates and theta_hat is not None:
                    final_theta_hat = theta_hat[-1] if isinstance(theta_hat, list) else theta_hat
                    _plot_gaussian_ellipses(ax_yz, trajectories_hat_yz, frame, 
                                          final_theta_hat, current_time, K, sources, rcvrs, d, colors, view_type="yz", style='estimated')
                    _plot_gaussian_ellipses(ax_xy, trajectories_hat_xy, frame, 
                                          final_theta_hat, current_time, K, sources, rcvrs, d, colors, view_type="xy", style='estimated')
        
        # Plot acquisition geometry if requested
        if show_acquisition_geometry:
            if d == 2:
                _plot_acquisition_geometry_in_animation(ax, sources, rcvrs, d)
            elif d == 3:
                _plot_acquisition_geometry_in_animation(ax_yz, sources, rcvrs, d, view_type="yz")
                _plot_acquisition_geometry_in_animation(ax_xy, sources, rcvrs, d, view_type="xy")
                
        if plot_projection_data:
            _plot_projection_data(proj_ax, projs_by_source, t, frame, sources, rcvrs, d)
            # Plot estimated projections if available
            if show_estimates and sim_projs is not None:
                # Use the final projection or show evolution of projections
                final_proj = sim_projs[-1] if isinstance(sim_projs, list) else sim_projs
                _plot_projection_data(proj_ax, final_proj, t, frame, sources, rcvrs, d, style='estimated')
                        
        
        # Update plot aesthetics
        if d == 2:
            _update_plot_aesthetics(ax, trajectories, K, d, current_time)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Ensure both subplots have the same y-limits and consistent sizing
            if plot_projection_data:
                proj_ax.set_ylim(y_min, y_max)
                plt.setp(proj_ax.get_yticklabels(), visible=False)  # Hide y-tick labels for projection plot
        elif d == 3:
            _update_plot_aesthetics(ax_yz, trajectories_yz, K, 2, current_time, view_type="yz", force_equal_aspect=True)
            _update_plot_aesthetics(ax_xy, trajectories_xy, K, 2, current_time, view_type="xy", force_equal_aspect=True)

        return []  # Return empty list since we're using ax.clear()
    
    
    '----------------------------'
    # 4. CREATE: Animation object
    '----------------------------'
    if d == 2:
        return FuncAnimation(fig, animate, frames=n_frames, interval=250, repeat=True)
    elif d == 3:
        anim_yz = FuncAnimation(fig_yz, animate, frames=n_frames, interval=250, repeat=True)
        anim_xy = FuncAnimation(fig_xy, animate, frames=n_frames, interval=250, repeat=True)
        return anim_yz, anim_xy


def save_GMM_animation(theta_true, d, K, sources, rcvrs, t, projs_by_source, 
                       filename='gmm_animation', title="GMM Animation", show_trajectory=True, show_gaussians=True, 
                       show_acquisition_geometry=True, plot_projection_data=True, fps=18,
                       theta_hat=None, sim_projs=None, show_estimates=False):
    """
    Creates and saves an animation of the Gaussian mixture model in motion.
    
    Parameters:
    - theta_true: Dictionary containing the true parameters of the GMM.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - projs_by_source: True projection data for visualization.
    - filename: Output filename for the animation (without extension, default: 'gmm_animation').
    - title: Animation title (default: "GMM Animation").
    - show_trajectory: Whether to show the trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian ellipses/spheres (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    - plot_projection_data: Whether to show projection data plots (default: True).
    - fps: Frames per second for the saved animation (default: 18).
    - theta_hat: List of estimated parameter dictionaries from optimization iterations (default: None).
    - sim_projs: List of simulated projections corresponding to theta_hat estimates (default: None).
    - show_estimates: Whether to overlay estimated parameters and projections on the animation (default: False).
    """
    if d == 2:
        anim = animate_GMM_motion(theta_true, d, K, sources, rcvrs, t, projs_by_source, 
                                 title=title, show_trajectory=show_trajectory, show_gaussians=show_gaussians,
                                 show_acquisition_geometry=show_acquisition_geometry, 
                                 plot_projection_data=plot_projection_data,
                                 theta_hat=theta_hat, sim_projs=sim_projs, show_estimates=show_estimates)
        # Save animation
        if filename.endswith('.gif'):
            anim.save(filename, writer='pillow', fps=fps)
        elif filename.endswith('.mp4'):
            anim.save(filename, writer='ffmpeg', fps=fps)
        else:
            # Default to gif
            anim.save(filename + '.gif', writer='pillow', fps=fps)
    
        print(f"Animation saved as {filename}")
        if show_estimates and theta_hat is not None:
            print("- Animation includes both true and estimated GMM parameters")
            print("- True parameters shown with solid lines")
            print("- Estimated parameters shown with dashed lines")
        return anim
    elif d == 3:
        anim_yz, anim_xy = animate_GMM_motion(theta_true, d, K, sources, rcvrs, t, projs_by_source, 
                                             title=title, show_trajectory=show_trajectory, show_gaussians=show_gaussians,
                                             show_acquisition_geometry=show_acquisition_geometry,
                                             plot_projection_data=plot_projection_data,
                                             theta_hat=theta_hat, sim_projs=sim_projs, show_estimates=show_estimates)
        # Save animations
        if filename.endswith('.gif'):
            anim_yz.save(filename.replace('.gif', '_yz.gif'), writer='pillow', fps=fps)
            anim_xy.save(filename.replace('.gif', '_xy.gif'), writer='pillow', fps=fps)
        elif filename.endswith('.mp4'):
            anim_yz.save(filename.replace('.mp4', '_yz.mp4'), writer='ffmpeg', fps=fps)
            anim_xy.save(filename.replace('.mp4', '_xy.mp4'), writer='ffmpeg', fps=fps)
        else:
            # Default to gif
            anim_yz.save(filename + '_yz.gif', writer='pillow', fps=fps)
            anim_xy.save(filename + '_xy.gif', writer='pillow', fps=fps)
        
        base_name = filename.replace('.gif', '').replace('.mp4', '')
        print(f"Animations saved as {base_name}_yz and {base_name}_xy")
        if show_estimates and theta_hat is not None:
            print("- Animations include both true and estimated GMM parameters")
            print("- True parameters shown with solid lines")
            print("- Estimated parameters shown with dashed lines")
        return anim_yz, anim_xy


def animate_projection_comparison(proj_data, sim_projs, t, sources, receivers, d, 
                                  title="Projection Comparison Animation", interval=100, figsize=(12, 6)):
    """
    Create an animation that shows all projections in sim_projs from start to finish 
    as one continuous animation in time.
    
    Parameters:
    - proj_data: True projection data (list of tensors for each source)
    - sim_projs: Simulated/approximate projection data (list of optimization iterations, 
                 each containing a list of tensors for each source)  
    - t: Time vector
    - sources: List of source positions
    - receivers: List of receiver positions for each source
    - d: Dimensionality (2 or 3)
    - title: Animation title (default: "Projection Comparison Animation")
    - interval: Animation interval in milliseconds (default: 100)
    - figsize: Figure size (default: (12, 6))
    
    Returns:
    - FuncAnimation object
    """
    
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy safely handling gradients."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor
    
    # Convert tensors to numpy for plotting
    proj_data_np = [tensor_to_numpy(proj) for proj in proj_data]
    sim_projs_np = [[tensor_to_numpy(proj) for proj in iteration_projs] for iteration_projs in sim_projs]
    t_np = tensor_to_numpy(t)
    
    # Setup figure with side-by-side subplots
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    # fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Calculate total number of frames: iterations × time_frames
    n_iterations = len(sim_projs_np)
    n_time_frames = len(t_np)
    total_frames = n_iterations * n_time_frames
    
    if d == 2:
        # Get receiver positions for y-axis
        receiver_heights = np.array([rcvr[1].item() for rcvr in receivers[0]])[::-1]
        min_height, max_height = receiver_heights[0], receiver_heights[-1]
        receiver_heights = np.array([rcvr[1].item() for rcvr in receivers[0]])
        tau = -min_height - max_height
        # y_positions = -receiver_heights - tau
        y_positions = receiver_heights
        
        # Calculate global limits for consistent scaling across all time frames and iterations
        all_values = []
        all_values.extend([proj_data_np[0].flatten()])
        for sim_proj_iter in sim_projs_np:
            all_values.extend([sim_proj_iter[0].flatten()])
        
        all_proj_values = np.concatenate(all_values)
        max_proj_value = np.max(all_proj_values)
        min_proj_value = np.min(all_proj_values)
        x_margin = (max_proj_value - min_proj_value) * 0.05
        
        def animate(frame):
            # Calculate which iteration and time frame we're in
            iteration_idx = frame // n_time_frames
            # Normal time progression: left to right as time increases
            time_frame_idx = frame % n_time_frames
            
            # Clear both subplots
            ax1.clear()
            
            # Plot true projections (left subplot) - always the same
            true_proj_frame = proj_data_np[0][time_frame_idx]
            
            # Ensure true_proj_frame is 1D with correct length
            if true_proj_frame.ndim > 1:
                true_proj_frame = true_proj_frame.flatten()
            if len(true_proj_frame) != len(y_positions):
                print(f"Warning: true_proj_frame shape {true_proj_frame.shape} doesn't match y_positions shape {y_positions.shape}")
                return
                
            # Sort receiver positions from lowest to highest for plotting
            # This ensures lowest receiver value is on the left, highest on the right
            sorted_indices = np.argsort(-y_positions)  # Sort ascending
            sorted_y_positions = y_positions[sorted_indices]
            sorted_true_proj = true_proj_frame[sorted_indices]
            
            # Plot with sorted data - x-axis represents sorted receiver positions
            ax1.plot(sorted_y_positions, sorted_true_proj, 
                    linestyle='-', color='blue', linewidth=2.5, 
                    label='True Projections', alpha=0.8)
            
            # Set axis limits based on sorted receiver positions
            ax1.set_xlim(sorted_y_positions.min() - 0.1, sorted_y_positions.max() + 0.1)
            ax1.set_ylim(min_proj_value - x_margin, max_proj_value + x_margin)
            ax1.set_xlabel('Receiver Position (Low → High)')
            ax1.set_ylabel('Projection Value')
            ax1.grid(True, alpha=0.3)
            
            # Plot maximum value as black circle
            # Use time_frame_idx since max arrays are sized for time frames, not iterations
            # if time_frame_idx < len(max_best_traj_projs_args_np) and time_frame_idx < len(true_maxs_k_np):
            #     # Get the receiver index and corresponding position
            #     max_receiver_idx = max_best_traj_projs_args_np[time_frame_idx]
            #     max_value = true_maxs_k_np[time_frame_idx]
                
            #     # Get the corresponding receiver position (need to sort it the same way)
            #     max_receiver_pos = y_positions[max_receiver_idx]
                
            #     # Plot the black circle
            #     ax1.scatter(max_receiver_pos, max_value, 
            #                color='black', s=100, marker='o', 
            #                label='Maximum', zorder=5)
            
            if iteration_idx < len(sim_projs_np):
                sim_data = sim_projs_np[iteration_idx][0]  # First (and typically only) source
                
                if sim_data.ndim == 2:
                    sim_proj_frame = sim_data[time_frame_idx]
                elif sim_data.ndim == 1:
                    sim_proj_frame = sim_data
                else:
                    print(f"Warning: Unexpected sim_data dimensions: {sim_data.shape}")
                    return
                
                # Ensure sim_proj_frame is 1D with correct length
                if sim_proj_frame.ndim > 1:
                    sim_proj_frame = sim_proj_frame.flatten()
                if hasattr(sim_proj_frame, '__len__') and len(sim_proj_frame) != len(y_positions):
                    print(f"Warning: sim_proj_frame shape {sim_proj_frame.shape} doesn't match y_positions shape {y_positions.shape}")
                    return
                    
                # Sort simulated projection data to match receiver position ordering
                sorted_sim_proj = sim_proj_frame[sorted_indices]
                    
                # Plot simulated projections with same sorted receiver positions
                ax1.plot(sorted_y_positions, sorted_sim_proj,
                        linestyle='--', color='red', linewidth=2.5,
                        label=f'Iteration {iteration_idx}', alpha=0.8)
            
            # Plot maximum value of simulated projections as green circle
            # Try different indexing strategies based on array structure
            combined_frame = iteration_idx * n_time_frames + time_frame_idx
            
            # First try combined frame indexing
            # if combined_frame < len(max_best_traj_projs_args_np) and combined_frame < len(max_best_traj_projs_np):
            #     sim_max_receiver_idx = max_best_traj_projs_args_np[combined_frame]
            #     sim_max_value = max_best_traj_projs_np[combined_frame]
            #     sim_max_receiver_pos = y_positions[sim_max_receiver_idx]
            #     ax1.scatter(sim_max_receiver_pos, sim_max_value, 
            #                color='green', s=100, marker='o', 
            #                label='Sim Maximum', zorder=5)
            # # If that fails, try iteration-based indexing (assuming arrays are structured per iteration)
            # elif iteration_idx < len(max_best_traj_projs_args_np) and time_frame_idx < len(max_best_traj_projs_args_np[0]) if len(max_best_traj_projs_args_np.shape) > 1 else False:
            #     sim_max_receiver_idx = max_best_traj_projs_args_np[iteration_idx][time_frame_idx]
            #     sim_max_value = max_best_traj_projs_np[iteration_idx][time_frame_idx]
            #     sim_max_receiver_pos = y_positions[sim_max_receiver_idx]
            #     ax1.scatter(sim_max_receiver_pos, sim_max_value, 
            #                color='green', s=100, marker='o', 
            #                label='Sim Maximum', zorder=5)
            # # If both fail, try just time frame indexing (same max for all iterations)
            # elif time_frame_idx < len(max_best_traj_projs_args_np) and time_frame_idx < len(max_best_traj_projs_np):
            #     sim_max_receiver_idx = max_best_traj_projs_args_np[time_frame_idx]
            #     sim_max_value = max_best_traj_projs_np[time_frame_idx]
            #     sim_max_receiver_pos = y_positions[sim_max_receiver_idx]
            #     ax1.scatter(sim_max_receiver_pos, sim_max_value, 
            #                color='green', s=100, marker='o', 
            #                label='Sim Maximum', zorder=5)
            
            # Set title and legend after all plotting is done
            ax1.set_title(f'Iteration {iteration_idx}\nTime: {t_np[time_frame_idx]:.3f}s', fontweight='bold')
            ax1.legend()
            
    elif d == 3:
        # For 3D case, plot as 2D projection data
        # Calculate global limits for consistent scaling
        all_values = []
        all_values.extend([proj_data_np[0].flatten()])
        for sim_proj_iter in sim_projs_np:
            all_values.extend([sim_proj_iter[0].flatten()])
        
        all_proj_values = np.concatenate(all_values)
        vmin, vmax = np.min(all_proj_values), np.max(all_proj_values)
        
        def animate(frame):
            # Calculate which iteration and time frame we're in
            iteration_idx = frame // n_time_frames
            # Normal time progression: left to right as time increases
            time_frame_idx = frame % n_time_frames
            
            # Clear both subplots
            ax1.clear()
            ax2.clear()
            
            # Plot true projections (left subplot) - always the same
            true_data = proj_data_np[0][time_frame_idx].reshape(-1, 1)
            im1 = ax1.imshow(true_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
            ax1.set_title(f'True Projections\nTime: {t_np[time_frame_idx]:.3f}s', fontweight='bold')
            ax1.set_xlabel('Detector X')
            ax1.set_ylabel('Detector Y')
            
            # Plot simulated projections (right subplot) - changes with iteration
            if iteration_idx < len(sim_projs_np):
                sim_data = sim_projs_np[iteration_idx][0][time_frame_idx].reshape(-1, 1)
                im2 = ax2.imshow(sim_data, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
                ax2.set_title(f'Iteration {iteration_idx}\nTime: {t_np[time_frame_idx]:.3f}s', fontweight='bold')
                ax2.set_xlabel('Detector X')
                ax2.set_ylabel('Detector Y')
    
    else:
        raise ValueError(f"Unsupported dimensionality: {d}")
    
    # Create animation - animates through all iterations sequentially, each showing full time evolution
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=interval, repeat=True)
    
    plt.tight_layout()
    return anim


def save_projection_comparison_animation(proj_data, sim_projs, t, sources, receivers, d, 
                                         filename="projection_comparison", 
                                         title="Projection Comparison Animation",
                                         fps=10, interval=100, figsize=(12, 6)):
    """
    Create and save an animation comparing true and simulated projections.
    
    Parameters:
    - proj_data: True projection data
    - sim_projs: Simulated projection data
    - t: Time vector
    - sources: List of source positions
    - receivers: List of receiver positions
    - d: Dimensionality
    - filename: Output filename (without extension)
    - title: Animation title
    - fps: Frames per second for saved animation
    - interval: Animation interval in milliseconds
    - figsize: Figure size
    
    Returns:
    - FuncAnimation object
    """
    anim = animate_projection_comparison(proj_data, sim_projs, t, sources, receivers, d, title=title, interval=interval, figsize=figsize)
    
    # Save animation
    if filename.endswith('.gif'):
        anim.save(filename, writer='pillow', fps=fps)
    elif filename.endswith('.mp4'):
        anim.save(filename, writer='ffmpeg', fps=fps)
    else:
        # Default to gif
        anim.save(filename + '.gif', writer='pillow', fps=fps)
    
    print(f"📽️  Projection comparison animation saved as {filename}")
    print(f"   - Continuous animation: All optimization iterations from start to finish")
    print(f"   - Left: True projections, Right: Current iteration projections")
    print(f"   - Total frames: {len(sim_projs) * len(t)} ({len(sim_projs)} iterations × {len(t)} time points)")
    print(f"   - FPS: {fps}")
    
    return anim


'--------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------- COMBINED GMM AND PROJECTION ANIMATION ------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'
def animate_GMM_with_projection_comparison(theta_true, d, K, sources, rcvrs, t, proj_data, sim_projs,
                                          title="GMM Animation with Projection Comparison",
                                          show_trajectory=True, show_gaussians=True, 
                                          show_acquisition_geometry=True,
                                          theta_hat=None, show_estimates=False):
    """
    Creates a combined animation showing GMM motion on top and projection comparison below,
    synchronized in time.
    
    Layout:
    - Top row: GMM animation (left) and projection data (right) from animate_GMM_motion
    - Bottom row: Projection comparison spanning full width
    
    Parameters:
    - theta_true: Dictionary containing the true parameters of the GMM.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - proj_data: True projection data for visualization.
    - sim_projs: List of simulated projections from optimization iterations.
    - title: Overall animation title (default: "GMM Animation with Projection Comparison").
    - show_trajectory: Whether to show the trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian ellipses (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    - theta_hat: List of estimated parameter dictionaries from optimization iterations (default: None).
    - show_estimates: Whether to overlay estimated parameters on the animation (default: False).
    
    Returns:
    - anim: The animation object.
    
    Notes:
    - Only supports 2D case currently
    - All three subplots are synchronized to the same time frame
    - Bottom projection comparison shows evolution through optimization iterations
    """
    
    if d != 2:
        raise NotImplementedError("Combined animation currently only supports 2D case")
    
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy safely handling gradients."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor
    
    '---------------------------------------'
    # 1. SETUP: Compute trajectories
    '---------------------------------------'
    n_frames = len(t)
    trajectories = _compute_trajectories(theta_true, K, d, t)
    
    # Compute estimated trajectories if provided
    if show_estimates and theta_hat is not None:
        final_theta_hat = theta_hat[-1] if isinstance(theta_hat, list) else theta_hat
        trajectories_hat = _compute_trajectories(final_theta_hat, K, d, t)
    
    '---------------------------------------'
    # 2. SETUP: Figure with 3 subplots
    '---------------------------------------'
    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.2)
    
    # Top row: GMM animation (left) and projection data (right)
    ax_gmm = fig.add_subplot(gs[0, 0])
    ax_proj_data = fig.add_subplot(gs[0, 1], sharey=ax_gmm)
    
    # Bottom row: Projection comparison (spans full width)
    ax_proj_comp = fig.add_subplot(gs[1, :])
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Calculate axis limits
    x_min = sources[0][0].item() - 0.5
    all_rcvr_x_coords = [rcvr[0].item() for rcvr in rcvrs[0]]
    x_max = max(all_rcvr_x_coords) + 0.5
    
    all_rcvr_y_coords = [rcvr[1].item() for rcvr in rcvrs[0]]
    y_max = max(all_rcvr_y_coords) + 0.5
    y_min = min(all_rcvr_y_coords) - 0.5
    
    colors = cm.rainbow(np.linspace(0, 1, K))
    
    '---------------------------------------'
    # 3. SETUP: Projection comparison data
    '---------------------------------------'
    proj_data_np = [tensor_to_numpy(proj) for proj in proj_data]
    sim_projs_np = [[tensor_to_numpy(proj) for proj in iteration_projs] for iteration_projs in sim_projs]
    t_np = tensor_to_numpy(t)
    
    n_iterations = len(sim_projs_np)
    n_time_frames = len(t_np)
    
    # Get receiver positions for projection comparison
    receiver_heights = np.array([rcvr[1].item() for rcvr in rcvrs[0]])
    min_height, max_height = receiver_heights[0], receiver_heights[-1]
    y_positions = receiver_heights
    
    # Calculate global limits for projection comparison
    all_values = []
    all_values.extend([proj_data_np[0].flatten()])
    for sim_proj_iter in sim_projs_np:
        all_values.extend([sim_proj_iter[0].flatten()])
    
    all_proj_values = np.concatenate(all_values)
    max_proj_value = np.max(all_proj_values)
    min_proj_value = np.min(all_proj_values)
    x_margin = (max_proj_value - min_proj_value) * 0.05
    
    '--------------------------------------------'
    # 4. ANIMATION: Create the animation function
    '--------------------------------------------'
    def animate(frame):
        """Animation function called for each frame."""
        current_time = t[frame]
        
        # Clear all axes
        ax_gmm.clear()
        ax_proj_data.clear()
        ax_proj_comp.clear()
        
        '--------------------------------'
        # TOP LEFT: GMM Animation
        '--------------------------------'
        # Plot trajectories if requested
        if show_trajectory:
            _plot_trajectories(ax_gmm, trajectories, frame, colors, d)
            # Plot estimated trajectories if available
            if show_estimates and theta_hat is not None and 'trajectories_hat' in locals():
                _plot_trajectories(ax_gmm, trajectories_hat, frame, colors, d, style='estimated')
        
        # Plot Gaussian visualization if requested  
        if show_gaussians:
            _plot_gaussian_ellipses(ax_gmm, trajectories, frame, 
                                  theta_true, current_time, K, sources, rcvrs, d, colors)
            # Plot estimated Gaussians if available
            if show_estimates and theta_hat is not None:
                final_theta_hat = theta_hat[-1] if isinstance(theta_hat, list) else theta_hat
                _plot_gaussian_ellipses(ax_gmm, trajectories_hat, frame, 
                                      final_theta_hat, current_time, K, sources, rcvrs, d, colors, style='estimated')
        
        # Plot acquisition geometry if requested
        if show_acquisition_geometry:
            _plot_acquisition_geometry_in_animation(ax_gmm, sources, rcvrs, d)
        
        # Update plot aesthetics
        _update_plot_aesthetics(ax_gmm, trajectories, K, d, current_time)
        ax_gmm.set_xlim(x_min, x_max)
        ax_gmm.set_ylim(y_min, y_max)
        
        '--------------------------------'
        # TOP RIGHT: Projection Data
        '--------------------------------'
        _plot_projection_data(ax_proj_data, proj_data, t, frame, sources, rcvrs, d)
        # Plot estimated projections if available
        if show_estimates and sim_projs is not None:
            final_proj = sim_projs[-1] if isinstance(sim_projs, list) else sim_projs
            _plot_projection_data(ax_proj_data, final_proj, t, frame, sources, rcvrs, d, style='estimated')
        
        # Ensure both top subplots have the same y-limits
        ax_proj_data.set_ylim(y_min, y_max)
        plt.setp(ax_proj_data.get_yticklabels(), visible=False)
        
        '--------------------------------'
        # BOTTOM: Projection Comparison
        '--------------------------------'
        # Calculate which iteration we're showing (cycle through iterations as time progresses)
        iteration_idx = min(frame * n_iterations // n_time_frames, n_iterations - 1)
        
        # Plot true projections
        true_proj_frame = proj_data_np[0][frame]
        
        if true_proj_frame.ndim > 1:
            true_proj_frame = true_proj_frame.flatten()
        
        # Sort receiver positions from lowest to highest for plotting
        sorted_indices = np.argsort(-y_positions)
        sorted_y_positions = y_positions[sorted_indices]
        sorted_true_proj = true_proj_frame[sorted_indices]
        
        ax_proj_comp.plot(sorted_y_positions, sorted_true_proj, 
                linestyle='-', color='blue', linewidth=2.5, 
                label='True Projections', alpha=0.8)
        
        # Plot simulated projections
        if iteration_idx < len(sim_projs_np):
            sim_data = sim_projs_np[iteration_idx][0]
            
            if sim_data.ndim == 2:
                sim_proj_frame = sim_data[frame]
            elif sim_data.ndim == 1:
                sim_proj_frame = sim_data
            else:
                return []
            
            if sim_proj_frame.ndim > 1:
                sim_proj_frame = sim_proj_frame.flatten()
                
            sorted_sim_proj = sim_proj_frame[sorted_indices]
                
            ax_proj_comp.plot(sorted_y_positions, sorted_sim_proj,
                    linestyle='--', color='red', linewidth=2.5,
                    label=f'Iteration {iteration_idx}', alpha=0.8)
        
        # Set axis limits and labels
        ax_proj_comp.set_xlim(sorted_y_positions.min() - 0.1, sorted_y_positions.max() + 0.1)
        ax_proj_comp.set_ylim(min_proj_value - x_margin, max_proj_value + x_margin)
        ax_proj_comp.set_xlabel('Receiver Position (Low → High)', fontsize=12)
        ax_proj_comp.set_ylabel('Projection Value', fontsize=12)
        ax_proj_comp.set_title(f'Projection Comparison - Iteration {iteration_idx}, Time: {t_np[frame]:.3f}s', fontweight='bold')
        ax_proj_comp.grid(True, alpha=0.3)
        ax_proj_comp.legend()
        
        return []
    
    '----------------------------'
    # 5. CREATE: Animation object
    '----------------------------'
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=250, repeat=True)
    
    return anim


def save_GMM_with_projection_comparison(theta_true, d, K, sources, rcvrs, t, proj_data, sim_projs,
                                       filename='gmm_combined_animation',
                                       title="GMM Animation with Projection Comparison",
                                       show_trajectory=True, show_gaussians=True, 
                                       show_acquisition_geometry=True, fps=18,
                                       theta_hat=None, show_estimates=False):
    """
    Creates and saves a combined animation showing GMM motion on top and projection comparison below.
    
    Parameters:
    - theta_true: Dictionary containing the true parameters of the GMM.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - proj_data: True projection data for visualization.
    - sim_projs: List of simulated projections from optimization iterations.
    - filename: Output filename for the animation (without extension, default: 'gmm_combined_animation').
    - title: Overall animation title (default: "GMM Animation with Projection Comparison").
    - show_trajectory: Whether to show the trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian ellipses (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    - fps: Frames per second for the saved animation (default: 18).
    - theta_hat: List of estimated parameter dictionaries from optimization iterations (default: None).
    - show_estimates: Whether to overlay estimated parameters on the animation (default: False).
    """
    anim = animate_GMM_with_projection_comparison(theta_true, d, K, sources, rcvrs, t, proj_data, sim_projs,
                                                 title=title, show_trajectory=show_trajectory, 
                                                 show_gaussians=show_gaussians,
                                                 show_acquisition_geometry=show_acquisition_geometry,
                                                 theta_hat=theta_hat, show_estimates=show_estimates)
    
    # Save animation
    if filename.endswith('.gif'):
        anim.save(filename, writer='pillow', fps=fps)
    elif filename.endswith('.mp4'):
        anim.save(filename, writer='ffmpeg', fps=fps)
    else:
        # Default to gif
        anim.save(filename + '.gif', writer='pillow', fps=fps)
    
    print(f"📽️  Combined GMM and projection animation saved as {filename}")
    print(f"   - Top row: GMM motion (left) and projection data (right)")
    print(f"   - Bottom row: Projection comparison across iterations")
    print(f"   - All synchronized to the same time frames")
    if show_estimates and theta_hat is not None:
        print("   - Animation includes both true and estimated GMM parameters")
        print("   - True parameters shown with solid lines, estimated with dashed lines")
    
    return anim


'--------------------------------------------------------------------------------------------------------------------------'
'--------------------------------- OPTIMIZATION STAGES ANIMATION ----------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'
def animate_optimization_stages(theta_true, theta_init, theta_after_traj, theta_final,
                                d, K, sources, rcvrs, t, proj_data, 
                                proj_init, proj_after_traj, proj_final,
                                title="GMM Optimization Progress",
                                show_trajectory=True, show_gaussians=True,
                                show_acquisition_geometry=True):
    """
    Creates an animation showing GMM on the left and three projection comparison stages on the right.
    
    Layout:
    - Left column (tall): GMM animation showing true and final estimated parameters
    - Right column (3 rows):
      * Top: Initial projections (before trajectory optimization)
      * Middle: After trajectory optimization (before rotation/structure optimization)  
      * Bottom: Final projections (after all optimization)
    
    Parameters:
    - theta_true: Dictionary containing the true parameters of the GMM.
    - theta_init: Initial parameter estimates (before trajectory optimization).
    - theta_after_traj: Parameters after trajectory optimization.
    - theta_final: Final optimized parameters.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - proj_data: True projection data.
    - proj_init: Initial estimated projections.
    - proj_after_traj: Projections after trajectory optimization.
    - proj_final: Final optimized projections.
    - title: Overall animation title (default: "GMM Optimization Progress").
    - show_trajectory: Whether to show trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian ellipses (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    
    Returns:
    - anim: The animation object.
    
    Notes:
    - Only supports 2D case currently
    - All subplots are synchronized to the same time frame
    - Shows progression through optimization stages
    """
    
    if d != 2:
        raise NotImplementedError("Optimization stages animation currently only supports 2D case")
    
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy safely handling gradients."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        return tensor
    
    '---------------------------------------'
    # 1. SETUP: Compute trajectories for all stages
    '---------------------------------------'
    n_frames = len(t)
    trajectories_true = _compute_trajectories(theta_true, K, d, t)
    trajectories_init = _compute_trajectories(theta_init, K, d, t)
    trajectories_after_traj = _compute_trajectories(theta_after_traj, K, d, t)
    trajectories_final = _compute_trajectories(theta_final, K, d, t)
    
    '---------------------------------------'
    # 2. SETUP: Figure with custom layout
    '---------------------------------------'
    # Create figure with gridspec: 1 column for GMM, 1 column for 3 projection plots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.25)
    
    # Left column: GMM animation (spans all 3 rows)
    ax_gmm = fig.add_subplot(gs[:, 0])
    
    # Right column: Three projection comparison plots
    ax_proj_init = fig.add_subplot(gs[0, 1])
    ax_proj_traj = fig.add_subplot(gs[1, 1])
    ax_proj_final = fig.add_subplot(gs[2, 1])
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Calculate axis limits for GMM plot
    x_min = sources[0][0].item() - 0.5
    all_rcvr_x_coords = [rcvr[0].item() for rcvr in rcvrs[0]]
    x_max = max(all_rcvr_x_coords) + 0.5
    
    all_rcvr_y_coords = [rcvr[1].item() for rcvr in rcvrs[0]]
    y_max = max(all_rcvr_y_coords) + 0.5
    y_min = min(all_rcvr_y_coords) - 0.5
    
    colors = cm.rainbow(np.linspace(0, 1, K))
    
    '---------------------------------------'
    # 3. SETUP: Projection data
    '---------------------------------------'
    # Convert all projection data to numpy, handling both list and tensor cases
    def convert_proj_to_numpy(proj):
        """Convert projection data to numpy, handling lists and tensors recursively."""
        if isinstance(proj, torch.Tensor):
            return tensor_to_numpy(proj)
        elif isinstance(proj, list):
            # Recursively convert all elements
            return [convert_proj_to_numpy(p) for p in proj]
        elif isinstance(proj, np.ndarray):
            return proj
        else:
            # Try to convert whatever it is
            try:
                return np.array(proj)
            except:
                return proj
    
    proj_data_np = convert_proj_to_numpy(proj_data)
    proj_init_np = convert_proj_to_numpy(proj_init)
    proj_after_traj_np = convert_proj_to_numpy(proj_after_traj)
    proj_final_np = convert_proj_to_numpy(proj_final)
    t_np = tensor_to_numpy(t)
    
    # Get receiver positions
    receiver_heights = np.array([rcvr[1].item() for rcvr in rcvrs[0]])
    y_positions = receiver_heights
    
    # Calculate global limits for all projection plots
    # Handle both list-of-arrays and single array cases
    def safe_flatten(proj_np):
        """Safely flatten projection data regardless of structure."""
        # First, ensure it's numpy (handle any remaining tensors)
        if isinstance(proj_np, torch.Tensor):
            proj_np = tensor_to_numpy(proj_np)
        
        if isinstance(proj_np, list) and len(proj_np) > 0:
            first_elem = proj_np[0]
            # Handle tensor in list
            if isinstance(first_elem, torch.Tensor):
                first_elem = tensor_to_numpy(first_elem)
            if isinstance(first_elem, np.ndarray):
                return first_elem.flatten()
            else:
                return np.array(first_elem).flatten()
        elif isinstance(proj_np, np.ndarray):
            return proj_np.flatten()
        else:
            # Fallback - shouldn't happen after convert_proj_to_numpy
            return np.array(proj_np).flatten()
    
    all_proj_values = np.concatenate([
        safe_flatten(proj_data_np),
        safe_flatten(proj_init_np),
        safe_flatten(proj_after_traj_np),
        safe_flatten(proj_final_np)
    ])
    max_proj_value = np.max(all_proj_values)
    min_proj_value = np.min(all_proj_values)
    x_margin = (max_proj_value - min_proj_value) * 0.05
    
    '--------------------------------------------'
    # 4. ANIMATION: Create the animation function
    '--------------------------------------------'
    def animate(frame):
        """Animation function called for each frame."""
        current_time = t[frame]
        
        # Clear all axes
        ax_gmm.clear()
        ax_proj_init.clear()
        ax_proj_traj.clear()
        ax_proj_final.clear()
        
        '--------------------------------'
        # LEFT: GMM Animation (showing true parameters only)
        '--------------------------------'
        # Plot trajectories if requested (only true trajectories)
        if show_trajectory:
            _plot_trajectories(ax_gmm, trajectories_true, frame, colors, d, style='true')
        
        # Plot Gaussian visualization if requested (only true Gaussians)
        if show_gaussians:
            _plot_gaussian_ellipses(ax_gmm, trajectories_true, frame, 
                                  theta_true, current_time, K, sources, rcvrs, d, colors, style='true')
        
        # Plot acquisition geometry if requested
        if show_acquisition_geometry:
            _plot_acquisition_geometry_in_animation(ax_gmm, sources, rcvrs, d)
        
        # Update plot aesthetics
        _update_plot_aesthetics(ax_gmm, trajectories_true, K, d, current_time)
        ax_gmm.set_xlim(x_min, x_max)
        ax_gmm.set_ylim(y_min, y_max)
        ax_gmm.set_title('True GMM Animation', fontsize=12, fontweight='bold')
        
        '--------------------------------'
        # Helper function for projection plots
        '--------------------------------'
        def plot_projection_comparison(ax, proj_true, proj_est, stage_title):
            """Plot true vs estimated projections for a given stage."""
            # Get data for current frame - handle different data structures
            # proj_true and proj_est are results after convert_proj_to_numpy
            
            # Handle true projection data
            if isinstance(proj_true, list) and len(proj_true) > 0:
                if isinstance(proj_true[0], np.ndarray) and proj_true[0].ndim >= 2:
                    # Structure: [sources][time][receivers]
                    true_frame = proj_true[0][frame]
                elif isinstance(proj_true[0], np.ndarray):
                    # Structure might be [time][receivers]
                    true_frame = proj_true[frame] if frame < len(proj_true) else proj_true[0]
                else:
                    # Fallback
                    true_frame = np.array(proj_true[frame] if frame < len(proj_true) else proj_true[0])
            elif isinstance(proj_true, np.ndarray):
                # Single array case
                if proj_true.ndim >= 2:
                    true_frame = proj_true[frame] if frame < len(proj_true) else proj_true[0]
                else:
                    true_frame = proj_true
            else:
                raise ValueError(f"Unexpected proj_true structure: {type(proj_true)}, length: {len(proj_true) if hasattr(proj_true, '__len__') else 'N/A'}")
            
            # Handle estimated projection data (same logic)
            if isinstance(proj_est, list) and len(proj_est) > 0:
                if isinstance(proj_est[0], np.ndarray) and proj_est[0].ndim >= 2:
                    # Structure: [sources][time][receivers]
                    est_frame = proj_est[0][frame]
                elif isinstance(proj_est[0], np.ndarray):
                    # Structure might be [time][receivers]
                    est_frame = proj_est[frame] if frame < len(proj_est) else proj_est[0]
                else:
                    # Fallback
                    est_frame = np.array(proj_est[frame] if frame < len(proj_est) else proj_est[0])
            elif isinstance(proj_est, np.ndarray):
                # Single array case
                if proj_est.ndim >= 2:
                    est_frame = proj_est[frame] if frame < len(proj_est) else proj_est[0]
                else:
                    est_frame = proj_est
            else:
                raise ValueError(f"Unexpected proj_est structure: {type(proj_est)}, length: {len(proj_est) if hasattr(proj_est, '__len__') else 'N/A'}")
            
            # Ensure frames are 1D
            if hasattr(true_frame, 'ndim') and true_frame.ndim > 1:
                true_frame = true_frame.flatten()
            if hasattr(est_frame, 'ndim') and est_frame.ndim > 1:
                est_frame = est_frame.flatten()
            
            # Sort receiver positions
            sorted_indices = np.argsort(-y_positions)
            sorted_y_positions = y_positions[sorted_indices]
            sorted_true = true_frame[sorted_indices]
            sorted_est = est_frame[sorted_indices]
            
            # Plot
            ax.plot(sorted_y_positions, sorted_true, 
                   linestyle='-', color='blue', linewidth=2.5, 
                   label='True', alpha=0.8)
            ax.plot(sorted_y_positions, sorted_est,
                   linestyle='--', color='red', linewidth=2.5,
                   label='Estimated', alpha=0.8)
            
            # Set limits and labels
            ax.set_xlim(sorted_y_positions.min() - 0.1, sorted_y_positions.max() + 0.1)
            ax.set_ylim(min_proj_value - x_margin, max_proj_value + x_margin)
            ax.set_xlabel('Receiver Position', fontsize=10)
            ax.set_ylabel('Projection Value', fontsize=10)
            ax.set_title(f'{stage_title}\nTime: {t_np[frame]:.3f}s', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        '--------------------------------'
        # RIGHT TOP: Initial Projections
        '--------------------------------'
        plot_projection_comparison(ax_proj_init, proj_data_np, proj_init_np, 
                                  '1. Initial (Before Trajectory Opt)')
        
        '--------------------------------'
        # RIGHT MIDDLE: After Trajectory Opt
        '--------------------------------'
        plot_projection_comparison(ax_proj_traj, proj_data_np, proj_after_traj_np,
                                  '2. After Trajectory Opt')
        
        '--------------------------------'
        # RIGHT BOTTOM: Final Projections
        '--------------------------------'
        plot_projection_comparison(ax_proj_final, proj_data_np, proj_final_np,
                                  '3. Final (After All Optimization)')
        
        return []
    
    '----------------------------'
    # 5. CREATE: Animation object
    '----------------------------'
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=250, repeat=True)
    
    return anim


def save_optimization_stages_animation(theta_true, theta_init, theta_after_traj, theta_final,
                                      d, K, sources, rcvrs, t, proj_data,
                                      proj_init, proj_after_traj, proj_final,
                                      filename='optimization_stages',
                                      title="GMM Optimization Progress",
                                      show_trajectory=True, show_gaussians=True,
                                      show_acquisition_geometry=True, fps=18):
    """
    Creates and saves an animation showing optimization progress across stages.
    
    Parameters:
    - theta_true: Dictionary containing the true parameters of the GMM.
    - theta_init: Initial parameter estimates (before trajectory optimization).
    - theta_after_traj: Parameters after trajectory optimization.
    - theta_final: Final optimized parameters.
    - d: Dimensionality of the application (2 or 3).
    - K: Number of Gaussians in the GMM.
    - sources, rcvrs: Source and receiver geometries for the GMM.
    - t: Array of time points for animation frames.
    - proj_data: True projection data.
    - proj_init: Initial estimated projections.
    - proj_after_traj: Projections after trajectory optimization.
    - proj_final: Final optimized projections.
    - filename: Output filename (without extension, default: 'optimization_stages').
    - title: Overall animation title (default: "GMM Optimization Progress").
    - show_trajectory: Whether to show trajectory trails (default: True).
    - show_gaussians: Whether to show Gaussian ellipses (default: True).
    - show_acquisition_geometry: Whether to show acquisition geometry (default: True).
    - fps: Frames per second for the saved animation (default: 18).
    
    Returns:
    - anim: The animation object.
    """
    anim = animate_optimization_stages(theta_true, theta_init, theta_after_traj, theta_final,
                                      d, K, sources, rcvrs, t, proj_data,
                                      proj_init, proj_after_traj, proj_final,
                                      title=title, show_trajectory=show_trajectory,
                                      show_gaussians=show_gaussians,
                                      show_acquisition_geometry=show_acquisition_geometry)
    
    # Save animation
    if filename.endswith('.gif'):
        anim.save(filename, writer='pillow', fps=fps)
    elif filename.endswith('.mp4'):
        anim.save(filename, writer='ffmpeg', fps=fps)
    else:
        # Default to gif
        anim.save(filename + '.gif', writer='pillow', fps=fps)
    
    print(f"📽️  Optimization stages animation saved as {filename}")
    print(f"   - Left: GMM with true (solid) and final estimate (dashed)")
    print(f"   - Right top: Initial projections (before trajectory opt)")
    print(f"   - Right middle: After trajectory optimization")
    print(f"   - Right bottom: Final projections (after all optimization)")
    print(f"   - All synchronized to the same time frames")
    
    return anim


# Test for derivative plot
# import matplotlib.pyplot as plt
# s = sources[0]
# x0_k, v0_k, a0_k = theta_true['x0s'][0], theta_true['v0s'][0], theta_true['a0s'][0]
# sample_ind = N_projs // 3
# sample_time = t[sample_ind]
# c_k = s - x0_k - v0_k * sample_time - 0.5 * a0_k * sample_time**2
# s1, s2 = s[0], s[1]
# df_dr2 = torch.empty(n_rcvrs, dtype=torch.float64, device=device)
# rcvr_heights = torch.zeros(n_rcvrs, dtype=torch.float64, device=device)
# for n_r, r in enumerate(rcvrs[0]):
#     r1, r2 = r[0], r[1]
#     h_k = (r1 - s1) * c_k[0] - s2 * c_k[1]
#     R_k_l = 2 * ((r1 - s1)**2 + (r2 - s2)**2) * c_k[1] * (c_k[1] * r2 + h_k)
#     R_k_r = -2 * (r2 - s2) * (c_k[1] * r2 + h_k) ** 2
#     denom = ((r1 - s1) ** 2 + (r2 - s2) ** 2) ** 2
#     R = (R_k_l + R_k_r) / denom
#     f = proj_data[0][sample_ind][n_r]
#     df_dr2[n_r] = R * f
#     rcvr_heights[n_r] = r2
# fig, ax = plt.subplots()
# ax.plot(rcvr_heights.cpu(), df_dr2.cpu())
# ax.set_title("Derivative of projection data at time {:.2f}s".format(sample_time.item()))
# ax.set_xlabel("Receiver position (m)")
# ax.set_ylabel("Derivative of projection data")
# plt.show()


# for n, proj_n in enumerate(proj_data[0]):
#     t_n = t[n]
#     plt.figure(figsize=(8, 5))
#     plt.plot(rcvr_heights.cpu(), proj_n.cpu(), label="True projection", color='black')
#     proj_k_n = GMM.generate_projections([t_n], theta_hat[-1], loss_type=None)[0][0]
#     plt.plot(rcvr_heights.cpu(), proj_k_n.detach().numpy(), label='Model Projection')
#     plt.title(f"Projection at time {t[n].item()} for v0 {theta_hat[0]["v0s"][0].cpu().detach().numpy()}")
#     plt.legend()
#     plt.show()