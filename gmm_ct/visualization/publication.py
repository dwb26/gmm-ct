"""
Publication-quality plotting functions for journal manuscripts.
Creates high-resolution, publication-ready figures with consistent styling.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch
from scipy.optimize import linear_sum_assignment
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

# Publication-quality settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['grid.alpha'] = 0.3

# High-resolution output
DPI = 300
FIGSIZE_SINGLE = (3.5, 2.8)  # Single column width (~3.5 inches)
FIGSIZE_DOUBLE = (7.0, 3.5)  # Double column width (~7 inches)
FIGSIZE_TALL = (7.0, 5.0)    # Double column, taller

# Color schemes for consistency
COLORS_TRUE = '#2E86AB'      # Blue for true values
COLORS_EST = '#A23B72'       # Purple/magenta for estimates
COLORS_INIT = '#F18F01'      # Orange for initial values
COLORS_ERROR = '#C73E1D'     # Red for errors
COLORS_GRAY = '#6C757D'      # Gray for auxiliary info


def match_estimated_to_true_gaussians(theta_true, theta_est, K):
    """
    Match each estimated Gaussian to the closest true Gaussian based on parameter distance.
    
    Uses a weighted combination of:
    - Initial position (x0) distance
    - Initial velocity (v0) distance  
    - Initial acceleration (a0) distance
    
    Returns a list of indices where matching_indices[k_est] = k_true
    """
    import numpy as np
    
    # Compute pairwise distances between estimated and true Gaussians
    cost_matrix = np.zeros((K, K))
    
    for k_est in range(K):
        for k_true in range(K):
            # Extract parameters
            x0_est = theta_est['x0s'][k_est].detach().cpu().numpy()
            x0_true = theta_true['x0s'][k_true].detach().cpu().numpy()
            
            v0_est = theta_est['v0s'][k_est].detach().cpu().numpy()
            v0_true = theta_true['v0s'][k_true].detach().cpu().numpy()
            
            a0_est = theta_est['a0s'][k_est].detach().cpu().numpy()
            a0_true = theta_true['a0s'][k_true].detach().cpu().numpy()
            
            # Compute weighted distance (position is most important)
            dist_x0 = np.linalg.norm(x0_est - x0_true)
            dist_v0 = np.linalg.norm(v0_est - v0_true)
            dist_a0 = np.linalg.norm(a0_est - a0_true)
            
            # Weight: position > velocity > acceleration
            cost_matrix[k_est, k_true] = 10.0 * dist_x0 + 3.0 * dist_v0 + 1.0 * dist_a0
    
    # Use Hungarian algorithm to find optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping: matching_indices[k_est] = k_true
    matching_indices = col_ind.tolist()
    
    return matching_indices


def reorder_theta_to_match_true(theta_true, theta_est, K):
    """
    Reorder estimated Gaussian parameters to match true Gaussians.
    
    This function should be called once after optimization to ensure all
    subsequent plotting functions display correctly matched Gaussians.
    
    Parameters:
    - theta_true: True GMM parameters
    - theta_est: Estimated GMM parameters
    - K: Number of Gaussians
    
    Returns:
    - theta_est_reordered: Reordered estimated parameters
    - matching_indices: The matching indices used (for reference)
    """
    matching_indices = match_estimated_to_true_gaussians(theta_true, theta_est, K)
    
    # matching_indices[k_est] = k_true means Est[k_est] matches True[k_true]
    # We want to create new_est where new_est[k_true] = Est[k_est]
    # So we need to invert the mapping
    inverse_matching = [0] * K
    for k_est, k_true in enumerate(matching_indices):
        inverse_matching[k_true] = k_est
    
    theta_est_reordered = {
        'x0s': [theta_est['x0s'][inverse_matching[k]] for k in range(K)],
        'v0s': [theta_est['v0s'][inverse_matching[k]] for k in range(K)],
        'a0s': [theta_est['a0s'][inverse_matching[k]] for k in range(K)],
        'omegas': [theta_est['omegas'][inverse_matching[k]] for k in range(K)],
        'alphas': [theta_est['alphas'][inverse_matching[k]] for k in range(K)],
        'U_skews': [theta_est['U_skews'][inverse_matching[k]] for k in range(K)]
    }
    
    return theta_est_reordered, matching_indices


def tensor_to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array safely."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return np.array([tensor_to_numpy(t) for t in tensor])
    return np.array(tensor)


def save_figure(fig, filename, dpi=DPI, bbox_inches='tight'):
    """Save figure with publication settings."""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {filename}")



'--------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------- FIGURE 2: INDIVIDUAL GAUSSIAN RECONSTRUCTION -------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def plot_individual_gaussian_reconstruction(theta_true, theta_est, K, d, gaussian_indices=None,
                                           filename=None, resolution=256):
    """
    Figure 2: Individual Gaussian reconstruction accuracy comparison.
    
    Shows fixed-body Gaussian shapes (alpha, U_skew) centered at origin.
    Creates a 3×3 grid showing:
    - Row 1: Ground truth for 3 selected Gaussians (at origin)
    - Row 2: Reconstructed versions of those Gaussians (at origin)
    - Row 3: Absolute difference |True - Reconstructed|
    
    Parameters:
    - sources: Source positions
    - receivers: Receiver positions
    - theta_true: True GMM parameters
    - theta_est: Estimated GMM parameters
    - t: Time vector (not used, kept for API consistency)
    - K: Number of Gaussians
    - d: Dimensionality (must be 2)
    - gaussian_indices: List of 3 Gaussian indices to show (default: auto-select most diverse)
    - filename: Output filename
    - resolution: Image resolution in pixels (default: 256)
    
    Returns:
    - fig: Matplotlib figure object
    """
    # from ..core.models import GMM_reco
    import numpy as np
    
    if d != 2:
        raise NotImplementedError("Image reconstruction comparison currently only supports 2D")
    
    # If K > 5, select the 5 worst approximations based on image reconstruction error
    if K > 5:
        print(f"K={K} > 5, selecting the 5 worst Gaussian approximations for plotting...")
        errors = []
        device = theta_true['alphas'][0].device

        for k in range(K):
            # Extract parameters for this Gaussian
            alpha_true_k = theta_true['alphas'][k]
            U_true_k = theta_true['U_skews'][k]
            alpha_est_k = theta_est['alphas'][k]
            U_est_k = theta_est['U_skews'][k]

            # Align estimated U_skew to true U_skew orientation for a fair comparison
            try:
                U_true_np = U_true_k.cpu().numpy()
                U_est_np = U_est_k.cpu().numpy()
                precision_true = U_true_np.T @ U_true_np
                precision_est = U_est_np.T @ U_est_np
                cov_true = np.linalg.inv(precision_true)
                cov_est = np.linalg.inv(precision_est)
                _, eigvecs_true = np.linalg.eigh(cov_true)
                _, eigvecs_est = np.linalg.eigh(cov_est)
                R_align = eigvecs_true @ eigvecs_est.T
                U_est_aligned_np = U_est_np @ R_align.T
                U_est_aligned = torch.tensor(U_est_aligned_np, dtype=U_est_k.dtype, device=U_est_k.device)
            except:
                U_est_aligned = U_est_k # Fallback

            # Create a grid to evaluate the Gaussian shape
            extent_size = 1.0 # A standard extent for error calculation
            x_coords = np.linspace(-extent_size, extent_size, resolution)
            y_coords = np.linspace(-extent_size, extent_size, resolution)
            X, Y = np.meshgrid(x_coords, y_coords)
            pixel_positions = torch.stack([torch.tensor(X.ravel(), dtype=torch.float64, device=device), 
                                           torch.tensor(Y.ravel(), dtype=torch.float64, device=device)], dim=1)
            mu_origin = torch.zeros(d, dtype=torch.float64, device=device)

            # Evaluate true Gaussian
            quadratic_form_true = torch.sum((U_true_k @ (pixel_positions - mu_origin).T).T ** 2, dim=1)
            gaussian_true = alpha_true_k * torch.exp(-0.5 * quadratic_form_true)

            # Evaluate estimated Gaussian
            quadratic_form_est = torch.sum((U_est_aligned @ (pixel_positions - mu_origin).T).T ** 2, dim=1)
            gaussian_est = alpha_est_k * torch.exp(-0.5 * quadratic_form_est)

            # Calculate Mean Absolute Error
            mae = torch.mean(torch.abs(gaussian_true - gaussian_est)).item()
            errors.append(mae)

        # Get the indices of the 5 Gaussians with the highest error
        worst_indices = np.argsort(errors)[-5:]
        gaussian_indices = sorted(worst_indices)
        print(f"Plotting worst 5 Gaussians (by MAE): {gaussian_indices}")
        K = 5 # We are now plotting 5 Gaussians

    # Select 3 most diverse Gaussians if not specified
    if gaussian_indices is None:
        # Compute diversity based on geometric features (alpha and U_skew structure)
        alphas = theta_true['alphas']
        U_skews = theta_true['U_skews']
        
        # Extract features: alpha value and U_skew matrix elements (upper triangular)
        features_list = []
        for k in range(K):
            alpha_k = alphas[k].item() if hasattr(alphas[k], 'item') else alphas[k]
            U_k = U_skews[k]
            
            # Get upper triangular elements (including diagonal)
            triu_indices = torch.triu_indices(d, d)
            U_elements = U_k[triu_indices[0], triu_indices[1]].cpu().numpy()
            
            # Combine alpha and U_skew elements
            feature_k = np.concatenate([[alpha_k], U_elements])
            features_list.append(feature_k)
        
        features = np.array(features_list)  # (K, feature_dim)
        
        # Normalize features for fair distance calculation
        features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Find 3 most diverse: start with extremes, add most distant
        distances = np.linalg.norm(features_normalized[:, None] - features_normalized[None, :], axis=2)
        
        # Pick the pair with maximum distance
        i, j = np.unravel_index(distances.argmax(), distances.shape)
        selected = [i, j]
        
        # Add the point most distant from both
        min_dist_to_selected = np.minimum(distances[i], distances[j])
        k = np.argmax(min_dist_to_selected)
        selected.append(k)
        
        gaussian_indices = sorted(selected)
        print(f"Auto-selected most diverse Gaussians (by geometry): {gaussian_indices}")
        print(f"  Alphas: {[alphas[idx].item() if hasattr(alphas[idx], 'item') else alphas[idx] for idx in gaussian_indices]}")
    
    # Create figure with 3 columns (3 Gaussians) and 3 rows (True, Reconstructed, Difference)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, K, hspace=0.1, wspace=0.05, 
                          left=0.08, right=0.9, bottom=0.08, top=0.92)
    
    device = theta_true['alphas'][0].device
    
    # Storage for images
    images_true = []
    images_est = []
    images_diff = []
    extents = []
    
    # Determine a common extent for all plots to ensure same size
    max_extent_size = 0
    for k in gaussian_indices:
        U_np = theta_true['U_skews'][k].cpu().numpy()
        precision = U_np.T @ U_np
        try:
            covariance = np.linalg.inv(precision)
            eigenvals = np.linalg.eigvalsh(covariance)
            max_extent_size = max(max_extent_size, 3.0 * np.sqrt(eigenvals.max()))
        except:
            max_extent_size = max(max_extent_size, 1.0)
    common_extent = (-max_extent_size, max_extent_size, -max_extent_size, max_extent_size)

    print(f"\nReconstructing individual Gaussians centered at origin...")
    
    # Reconstruct each selected Gaussian
    for k in gaussian_indices:
        print(f"  Gaussian ρ_{k+1}...")
        
        # Extract parameters for this Gaussian
        alpha_true_k = theta_true['alphas'][k]
        U_true_k = theta_true['U_skews'][k]
        alpha_est_k = theta_est['alphas'][k]
        U_est_k = theta_est['U_skews'][k]
        
        # Align estimated U_skew to true U_skew orientation
        # Compute covariances and extract principal directions
        U_true_np = U_true_k.cpu().numpy()
        U_est_np = U_est_k.cpu().numpy()
        
        precision_true = U_true_np.T @ U_true_np
        precision_est = U_est_np.T @ U_est_np
        
        try:
            cov_true = np.linalg.inv(precision_true)
            cov_est = np.linalg.inv(precision_est)
            
            # Get eigenvectors (principal directions)
            _, eigvecs_true = np.linalg.eigh(cov_true)
            _, eigvecs_est = np.linalg.eigh(cov_est)
            
            # Compute rotation from estimated to true orientation
            # R_align rotates estimated eigenvectors to align with true eigenvectors
            R_align = eigvecs_true @ eigvecs_est.T
            
            # Apply rotation to estimated U_skew
            U_est_aligned_np = U_est_np @ R_align.T
            U_est_aligned = torch.tensor(U_est_aligned_np, dtype=U_est_k.dtype, device=U_est_k.device)
        except:
            # Fallback: use original if alignment fails
            U_est_aligned = U_est_k
        
        extent_size = max_extent_size
        # Create grid centered at origin
        x_coords = np.linspace(-extent_size, extent_size, resolution)
        y_coords = np.linspace(-extent_size, extent_size, resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        X_flat = torch.tensor(X.ravel(), dtype=torch.float64, device=device)
        Y_flat = torch.tensor(Y.ravel(), dtype=torch.float64, device=device)
        pixel_positions = torch.stack([X_flat, Y_flat], dim=1)  # (N_pixels, 2)
        
        # Evaluate true Gaussian at origin (no rotation, no translation)
        mu_origin = torch.zeros(d, dtype=torch.float64, device=device)
        deviations_true = pixel_positions - mu_origin
        U_deviations_true = (U_true_k @ deviations_true.T).T
        quadratic_form_true = torch.sum(U_deviations_true ** 2, dim=1)
        gaussian_true = alpha_true_k * torch.exp(-0.5 * quadratic_form_true)
        img_true = gaussian_true.reshape(resolution, resolution).cpu().numpy()
        
        # Evaluate estimated Gaussian at origin (with aligned orientation)
        deviations_est = pixel_positions - mu_origin
        U_deviations_est = (U_est_aligned @ deviations_est.T).T
        quadratic_form_est = torch.sum(U_deviations_est ** 2, dim=1)
        gaussian_est = alpha_est_k * torch.exp(-0.5 * quadratic_form_est)
        img_est = gaussian_est.reshape(resolution, resolution).cpu().numpy()
        
        # Compute difference
        # Compute absolute error and plot on log10 scale
        img_diff = np.abs(img_true - img_est)
        # Avoid log10(0) by setting a small minimum value
        img_diff_log = np.log10(np.clip(img_diff, 1e-8, None))
        
        images_true.append(img_true)
        images_est.append(img_est)
        images_diff.append(img_diff_log)
        extents.append(common_extent)
    
    # Determine consistent color scales
    vmin_gauss = 0
    vmax_gauss = max([img.max() for img in images_true + images_est])
    # For log difference plot, set a dynamic range, e.g., from -8 to max value
    vmin_diff = -8.0 
    vmax_diff = max([vmin_diff + 1e-3] + [img.max() for img in images_diff]) # Ensure vmax > vmin
    
    print(f"Gaussian value range: [0.00, {vmax_gauss:.2f}]")
    print(f"Log10 Difference value range: [{vmin_diff:.2f}, {vmax_diff:.2f}]")
    
    # Store base axes for sharing
    ax_row1_base, ax_row2_base, ax_row3_base = None, None, None

    # Plot grid
    for col_idx, (k, img_true, img_est, img_diff, extent) in enumerate(zip(
            gaussian_indices, images_true, images_est, images_diff, extents)):
        
        # Row 1: Ground Truth
        if ax_row1_base is None:
            ax_true = fig.add_subplot(gs[0, col_idx])
            ax_row1_base = ax_true
        else:
            ax_true = fig.add_subplot(gs[0, col_idx], sharey=ax_row1_base)

        ax_true.grid(False)
        im_true = ax_true.imshow(img_true, extent=extent, origin='lower', cmap='viridis', 
                                 vmin=vmin_gauss, vmax=vmax_gauss, aspect='equal')
        # Add contours
        X_contour = np.linspace(extent[0], extent[1], img_true.shape[1])
        Y_contour = np.linspace(extent[2], extent[3], img_true.shape[0])
        ax_true.contour(X_contour, Y_contour, img_true, levels=3, colors='black', linewidths=0.7, alpha=0.4)
        if col_idx == 0:
            ax_true.set_ylabel('Ground Truth\n\nHeight', fontweight='bold', fontsize=18)
        else:
            ax_true.tick_params(axis='y', labelleft=False)
        ax_true.set_title(f'$\\rho_{{{k+1}}}$', fontweight='bold', fontsize=20, pad=12)
        ax_true.tick_params(axis='both', which='major', labelsize=15, labelbottom=False)
        
        # Row 2: Reconstructed
        if ax_row2_base is None:
            ax_est = fig.add_subplot(gs[1, col_idx])
            ax_row2_base = ax_est
        else:
            ax_est = fig.add_subplot(gs[1, col_idx], sharey=ax_row2_base)
        ax_est.grid(False)
        im_est = ax_est.imshow(img_est, extent=extent, origin='lower', cmap='viridis', vmin=vmin_gauss, vmax=vmax_gauss, aspect='equal')
        # Add contours
        X_contour = np.linspace(extent[0], extent[1], img_est.shape[1])
        Y_contour = np.linspace(extent[2], extent[3], img_est.shape[0])
        ax_est.contour(X_contour, Y_contour, img_est, levels=3, colors='black', linewidths=0.7, alpha=0.4)
        if col_idx == 0:
            ax_est.set_ylabel('Reconstructed\n\nHeight', fontweight='bold', fontsize=18)
        else:
            ax_est.tick_params(axis='y', labelleft=False)
        ax_est.set_title(f'$\\widehat{{\\rho}}_{{{k+1}}}$', fontweight='bold', fontsize=20, pad=12)
        ax_est.tick_params(axis='both', which='major', labelsize=15, labelbottom=False)
        
        # Row 3: Log10 Absolute Error
        if ax_row3_base is None:
            ax_diff = fig.add_subplot(gs[2, col_idx])
            ax_row3_base = ax_diff
        else:
            ax_diff = fig.add_subplot(gs[2, col_idx], sharey=ax_row3_base)
        ax_diff.grid(False)
        im_diff = ax_diff.imshow(img_diff, extent=extent, origin='lower', cmap='inferno', vmin=vmin_diff, vmax=vmax_diff, aspect='equal')
        # Add contours
        X_contour = np.linspace(extent[0], extent[1], img_diff.shape[1])
        Y_contour = np.linspace(extent[2], extent[3], img_diff.shape[0])
        ax_diff.contour(X_contour, Y_contour, img_diff, levels=3, 
                       colors='white', linewidths=0.7, alpha=0.4)
        if col_idx == 0:
            ax_diff.set_ylabel('Log$_{10}$ Error\n\nHeight', fontweight='bold', fontsize=18)
        else:
            ax_diff.tick_params(axis='y', labelleft=False)
        ax_diff.set_title(f'$\\log_{{10}}|\\rho_{{{k+1}}} - \\widehat{{\\rho}}_{{{k+1}}}|$', fontweight='bold', fontsize=20, pad=12)
        ax_diff.set_xlabel('Depth (m)', fontweight='bold', fontsize=18)
        ax_diff.tick_params(axis='both', which='major', labelsize=15)
        
    # Add colorbars in a separate column to maintain subplot sizes
    # Create axes for the colorbars on the right
    cax_true = fig.add_axes([0.91, ax_row1_base.get_position().y0, 0.015, ax_row1_base.get_position().height])
    cbar_true = fig.colorbar(im_true, cax=cax_true)
    cbar_true.set_label('Attenuation', fontweight='bold', fontsize=16)
    cbar_true.ax.tick_params(labelsize=14)

    cax_est = fig.add_axes([0.91, ax_row2_base.get_position().y0, 0.015, ax_row2_base.get_position().height])
    cbar_est = fig.colorbar(im_est, cax=cax_est)
    cbar_est.set_label('Attenuation', fontweight='bold', fontsize=16)
    cbar_est.ax.tick_params(labelsize=14)

    cax_diff = fig.add_axes([0.91, ax_row3_base.get_position().y0, 0.015, ax_row3_base.get_position().height])
    cbar_diff = fig.colorbar(im_diff, cax=cax_diff)
    cbar_diff.set_label('Log$_{10}$ Error', fontweight='bold', fontsize=16)
    cbar_diff.ax.tick_params(labelsize=14)
    
    # Overall title
    fig.suptitle(f'Gaussian Reconstruction', 
                 fontweight='bold', fontsize=22, y=0.98)
    
    if filename:
        save_figure(fig, filename)
        print(f"\n✓ Figure 2 (Individual Gaussian Reconstruction) saved: {filename}")
    
    return fig


'--------------------------------------------------------------------------------------------------------------------------'
'---------------------------------- FIGURE 3: TEMPORAL GMM COMPARISON (TRUE VS ESTIMATED) ---------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def plot_temporal_gmm_comparison(sources, receivers, theta_true, theta_est, 
                                 t, K, d, time_indices=None, n_times=4,
                                 filename=None, show_trajectories=True,
                                 spatial_bounds=None, title='Reconstructed',
                                 title_fontsize=20, label_fontsize=18, tick_fontsize=16):
    """
    Figure 3: Symmetric comparison of ground truth and estimated GMMs over time.
    
    Creates a symmetric 3-column layout for each time snapshot:
    - Left: True GMM (source on left, receivers on right)
    - Center: Overlaid projections (true solid blue, estimated dashed red)
    - Right: Estimated GMM (mirrored - receivers on left, source on right)
    
    This symmetric layout allows direct visual comparison with projections in between.
    
    Parameters:
    - sources: Source positions
    - receivers: Receiver positions  
    - theta_true: True GMM parameters
    - theta_est: Estimated GMM parameters
    - t: Time vector
    - K: Number of Gaussians
    - d: Dimensionality (must be 2)
    - time_indices: Specific time indices to show (default: auto-select evenly spaced)
    - n_times: Number of time points to show if time_indices not provided (default: 4)
    - filename: Output filename
    - show_trajectories: Whether to show trajectory paths (default: True)
    - spatial_bounds: Custom spatial bounds [xmin, xmax, ymin, ymax] (default: auto)
    
    Returns:
    - fig: Matplotlib figure object
    """
    from ..core.models import GMM_reco
    import numpy as np
    
    if d != 2:
        raise NotImplementedError("Temporal GMM comparison currently only supports 2D")
    
    # Select time indices
    if time_indices is None:
        # Evenly spaced times across the duration
        time_indices = np.linspace(0, len(t) - 1, n_times, dtype=int)
    else:
        n_times = len(time_indices)
    
    selected_times = t[time_indices]
    
    print(f"\nCreating Figure 3: Temporal GMM Comparison (Symmetric Layout)...")
    print(f"Time points: {[f'{t_val.item():.3f}s' for t_val in selected_times]}")
    
    # Note: theta_est should already be reordered to match theta_true via reorder_theta_to_match_true()
    # If not pre-processed, do the matching here
    try:
        # Check if parameters are tensors (original) or already matched
        if hasattr(theta_est['x0s'][0], 'device'):
            # Assume already matched if we got here without error
            theta_est_reordered = theta_est
    except:
        # Fall back to matching if needed
        matching_indices = match_estimated_to_true_gaussians(theta_true, theta_est, K)
        print(f"Gaussian matching (est → true): {matching_indices}")
        theta_est_reordered = {
            'x0s': [theta_est['x0s'][i] for i in matching_indices],
            'v0s': [theta_est['v0s'][i] for i in matching_indices],
            'a0s': [theta_est['a0s'][i] for i in matching_indices],
            'omegas': [theta_est['omegas'][i] for i in matching_indices],
            'alphas': [theta_est['alphas'][i] for i in matching_indices],
            'U_skews': [theta_est['U_skews'][i] for i in matching_indices]
        }
    
    # Determine spatial bounds if not provided
    if spatial_bounds is None:
        # Calculate bounds from trajectories
        all_positions = []
        for k in range(K):
            x0_true = theta_true['x0s'][k].detach().cpu().numpy()
            v0_true = theta_true['v0s'][k].detach().cpu().numpy()
            a0_true = theta_true['a0s'][k].detach().cpu().numpy()
            
            for t_val in t:
                t_np = t_val.item()
                # Include acceleration in position calculation
                pos = x0_true + v0_true * t_np + 0.5 * a0_true * t_np**2
                all_positions.append(pos)
        
        all_positions = np.array(all_positions)
        margin = 0.3
        xmin, xmax = all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
        ymin, ymax = all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
        spatial_bounds = [xmin, xmax, ymin, ymax]
    
    # Generate projections at selected time points
    GMM_true_obj = GMM_reco(d, K, sources, receivers, 
                            theta_true['x0s'], theta_true['a0s'], omega_min=0.0, omega_max=10.0,
                            device=theta_true['x0s'][0].device)
    GMM_est_obj = GMM_reco(d, K, sources, receivers,
                           theta_est_reordered['x0s'], theta_est_reordered['a0s'], omega_min=0.0, omega_max=10.0,
                           device=theta_est_reordered['x0s'][0].device)
    
    proj_true = GMM_true_obj.generate_projections(t, theta_true)
    proj_est = GMM_est_obj.generate_projections(t, theta_est_reordered)
    
    # Create figure - N rows (one per time), 3 columns (true GMM, projections, estimated GMM)
    fig = plt.figure(figsize=(18, 5 * n_times), dpi=DPI)
    
    # Color map for Gaussians - use rainbow like the animation
    colors = plt.cm.rainbow(np.linspace(0, 1, K))
    
    # Get receiver positions for projection plotting
    receiver_heights = np.array([rcvr[1].item() for rcvr in receivers[0]])
    gs = GridSpec(n_times, 3, figure=fig, hspace=0.15, wspace=0.12, width_ratios=[1.2, 0.8, 1.2])
    
    # Plot each time point
    for row_idx, (t_idx, t_val) in enumerate(zip(time_indices, selected_times)):
        t_val_scalar = t_val.item()
        
        # LEFT COLUMN: True GMM (normal orientation)
        ax_left = fig.add_subplot(gs[row_idx, 0])
        
        if show_trajectories:
            plot_trajectories_single(ax_left, theta_true, t, K, colors, mirror=False)
        
        plot_gmm_snapshot(ax_left, theta_true, t_val_scalar, K, d, colors,
                         is_true=True, show_centroids=True)
        
        plot_acquisition_geometry(ax_left, sources, receivers, d, mirror=False)
        
        # Title (only column name on first row)
        if row_idx == 0:
            ax_left.set_title('Ground Truth', fontweight='bold', fontsize=title_fontsize, pad=12)
            # Add legend with Gaussian labels (vertical stack)
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            legend_elements = [Patch(facecolor=colors[k], edgecolor='black', 
                                    label=f'$\\rho_{{{k+1}}}$') for k in range(K)]
            # Add acquisition geometry legend items
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='r', 
                       markersize=8, alpha=0.7, label='Source'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', 
                       markersize=6, alpha=0.5, label='Receivers'),
                Line2D([0], [0], color='gold', linewidth=2, alpha=0.1, label='Rays')
            ])
            ax_left.legend(handles=legend_elements, loc='upper left', 
                          fontsize=14, framealpha=0.9, ncol=1, 
                          handlelength=1.5, handletextpad=0.5, borderpad=0.4)
        
        # Y-label shows time on all rows
        ax_left.set_ylabel(f't = {t_val_scalar:.2f} s\nHeight', 
                          fontweight='bold', fontsize=label_fontsize)
        if row_idx == n_times - 1:  # Only on bottom row
            ax_left.set_xlabel('Depth', fontweight='bold', fontsize=label_fontsize)
            ax_left.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        else:
            ax_left.tick_params(axis='y', which='major', labelsize=tick_fontsize)
            ax_left.tick_params(axis='x', which='major', labelbottom=False)
        # Use source/receiver geometry to set x-limits with margin
        source_x = sources[0][0].item()
        receiver_x = receivers[0][0][0].item()  # First receiver x-coordinate
        x_margin = 0.5
        ax_left.set_xlim(source_x - x_margin, receiver_x + x_margin)
        # Set y-limits based on receiver height range with margin
        receiver_heights_arr = np.array([rcvr[1].item() for rcvr in receivers[0]])
        y_min = receiver_heights_arr.min() - 0.1
        y_max = receiver_heights_arr.max() + 0.1
        ax_left.set_ylim(y_min, y_max)
        ax_left.grid(True, alpha=0.3, linestyle='--')
        ax_left.set_facecolor('#f8f9fa')
        
        # CENTER COLUMN: Overlaid projections
        ax_center = fig.add_subplot(gs[row_idx, 1])
        
        # Extract projection data at this time index
        proj_true_t = proj_true[0][t_idx].detach().cpu().numpy()
        proj_est_t = proj_est[0][t_idx].detach().cpu().numpy()
        
        # Sort by receiver position (from bottom to top)
        sorted_indices = np.argsort(receiver_heights)
        sorted_heights = receiver_heights[sorted_indices]
        sorted_proj_true = proj_true_t[sorted_indices]
        sorted_proj_est = proj_est_t[sorted_indices]
        
        # Plot projections
        ax_center.plot(sorted_proj_true, sorted_heights,
                      linestyle='-', color='blue', linewidth=2.5,
                      label='True', alpha=0.8)
        ax_center.plot(sorted_proj_est, sorted_heights,
                      linestyle='--', color='red', linewidth=2.5,
                      label='Estimated', alpha=0.8)
        
        # Title (no time on center/right columns)
        if row_idx == 0:
            ax_center.set_title('Projections', fontweight='bold', fontsize=title_fontsize, pad=12)
            ax_center.legend(fontsize=14, loc='best', framealpha=0.9)
        
        if row_idx == n_times - 1:  # Only on bottom row
            ax_center.set_xlabel('Intensity', fontweight='bold', fontsize=label_fontsize)
            ax_center.tick_params(axis='x', which='major', labelsize=tick_fontsize)
        else:
            ax_center.tick_params(axis='x', which='major', labelbottom=False)
        ax_center.tick_params(axis='y', which='major', labelleft=False)
        ax_center.grid(True, alpha=0.3, linestyle='--')
        ax_center.set_facecolor('#ffffff')
        
        # RIGHT COLUMN: Estimated GMM (mirrored orientation)
        ax_right = fig.add_subplot(gs[row_idx, 2])
        
        if show_trajectories:
            plot_trajectories_single(ax_right, theta_est_reordered, t, K, colors, mirror=True)
        
        plot_gmm_snapshot(ax_right, theta_est_reordered, t_val_scalar, K, d, colors,
                         is_true=False, show_centroids=True, mirror=True)
        
        plot_acquisition_geometry(ax_right, sources, receivers, d, mirror=True)
        
        # Title (no time on center/right columns)
        if row_idx == 0:
            ax_right.set_title(f'{title}', fontweight='bold', fontsize=title_fontsize, pad=12)
            # Add legend with estimated Gaussian labels (vertical stack)
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[k], edgecolor='black', 
                                    label=f'$\\widehat{{\\rho}}_{{{k+1}}}$') for k in range(K)]
            ax_right.legend(handles=legend_elements, loc='upper right', 
                          fontsize=14, framealpha=0.9, ncol=1, 
                          handlelength=1.3, handletextpad=0.4, borderpad=0.3)
        
        if row_idx == n_times - 1:  # Only on bottom row
            ax_right.set_xlabel('Depth', fontweight='bold', fontsize=label_fontsize)
            ax_right.tick_params(axis='x', which='major', labelsize=tick_fontsize)
        else:
            ax_right.tick_params(axis='x', which='major', labelbottom=False)
        ax_right.tick_params(axis='y', which='major', labelleft=False)
        # Mirror the x-axis limits (symmetric reflection about y-axis)
        source_x = sources[0][0].item()
        receiver_x = receivers[0][0][0].item()
        x_margin = 0.5
        # For mirror: negate the x-limits from left plot
        ax_right.set_xlim(-receiver_x - x_margin, -source_x + x_margin)
        # Set y-limits based on receiver height range with margin
        receiver_heights_arr = np.array([rcvr[1].item() for rcvr in receivers[0]])
        y_min = receiver_heights_arr.min() - 0.1
        y_max = receiver_heights_arr.max() + 0.1
        ax_right.set_ylim(y_min, y_max)
        ax_right.grid(True, alpha=0.3, linestyle='--')
        ax_right.set_facecolor('#f8f9fa')
        
        # Negate x-tick labels to show positive values on right side
        if row_idx == n_times - 1:  # Only on bottom row where labels are shown
            xticks = ax_right.get_xticks()
            ax_right.set_xticks(xticks)
            ax_right.set_xticklabels([f'{int(-x)}' for x in xticks])
    
    # Overall title
    fig.suptitle(f'Temporal Evolution: Ground Truth ← Projections → {title}',
                 fontweight='bold', fontsize=title_fontsize + 2, y=0.995)
    
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
        print(f"\n✓ Figure 3 (Temporal GMM Comparison) saved: {filename}")
    
    return fig


def animate_temporal_gmm_comparison(sources, receivers, theta_true, theta_est, 
                                     t, K, d, filename=None, fps=10,
                                     show_trajectories=True,
                                     title='Reconstructed', 
                                     title_fontsize=20, label_fontsize=18, tick_fontsize=16):
    """
    Create an animation showing temporal evolution of ground truth and estimated GMMs.
    
    Similar to plot_temporal_gmm_comparison but animates through all time points.
    
    Parameters:
    - sources: Source positions
    - receivers: Receiver positions  
    - theta_true: True GMM parameters
    - theta_est: Estimated GMM parameters
    - t: Time vector
    - K: Number of Gaussians
    - d: Dimensionality (must be 2)
    - filename: Output filename (should end with .mp4 or .gif)
    - fps: Frames per second (default: 10)
    - show_trajectories: Whether to show trajectory paths (default: True)
    
    Returns:
    - anim: Matplotlib animation object
    """
    from matplotlib.animation import FuncAnimation
    from ..core.models import GMM_reco
    import numpy as np
    
    if d != 2:
        raise NotImplementedError("Temporal GMM animation currently only supports 2D")
    
    # Note: theta_est should already be reordered to match theta_true via reorder_theta_to_match_true()
    # Use as-is (assume pre-processed)
    theta_est_reordered = theta_est
    
    # Generate projections
    GMM_true_obj = GMM_reco(d, K, sources, receivers, 
                            theta_true['x0s'], theta_true['a0s'], omega_min=0.0, omega_max=10.0,
                            device=theta_true['x0s'][0].device)
    GMM_est_obj = GMM_reco(d, K, sources, receivers,
                           theta_est_reordered['x0s'], theta_est_reordered['a0s'], omega_min=0.0, omega_max=10.0,
                           device=theta_est_reordered['x0s'][0].device)
    
    proj_true = GMM_true_obj.generate_projections(t, theta_true)
    proj_est = GMM_est_obj.generate_projections(t, theta_est_reordered)
    
    receiver_heights = np.array([rcvr[1].item() for rcvr in receivers[0]])
    
    # Compute global x-limits for projections (across all time points)
    proj_true_np = proj_true[0].detach().cpu().numpy()
    proj_est_np = proj_est[0].detach().cpu().numpy()
    proj_min = min(proj_true_np.min(), proj_est_np.min())
    proj_max = max(proj_true_np.max(), proj_est_np.max())
    proj_margin = (proj_max - proj_min) * 0.05
    
    # Create figure with 1 row, 3 columns
    fig = plt.figure(figsize=(18, 6), dpi=DPI)
    gs = GridSpec(1, 3, figure=fig, hspace=0.15, wspace=0.12, width_ratios=[1.2, 0.8, 1.2],
                  top=0.85, bottom=0.12)  # Adjust top margin to make room for time display
    
    # Color map
    colors = plt.cm.rainbow(np.linspace(0, 1, K))
    
    # Create subplots
    ax_left = fig.add_subplot(gs[0, 0])
    ax_center = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    
    # Setup axes limits and labels
    source_x = sources[0][0].item()
    receiver_x = receivers[0][0][0].item()
    x_margin = 0.5
    receiver_heights_arr = np.array([rcvr[1].item() for rcvr in receivers[0]])
    y_min = receiver_heights_arr.min() - 0.1
    y_max = receiver_heights_arr.max() + 0.1
    
    # Left column setup
    ax_left.set_xlim(source_x - x_margin, receiver_x + x_margin)
    ax_left.set_ylim(y_min, y_max)
    ax_left.set_xlabel('Depth (m)', fontweight='bold', fontsize=label_fontsize)
    ax_left.set_ylabel('Height (m)', fontweight='bold', fontsize=label_fontsize)
    ax_left.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax_left.grid(True, alpha=0.3, linestyle='--')
    ax_left.set_facecolor('#f8f9fa')
    ax_left.set_title('Ground Truth', fontweight='bold', fontsize=title_fontsize, pad=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], edgecolor='black', 
                            label=f'$\\rho_{{{k+1}}}$') for k in range(K)]
    ax_left.legend(handles=legend_elements, loc='upper left', 
                  fontsize=14, framealpha=0.9, ncol=1)
    
    # Center column setup
    ax_center.set_xlim(proj_min - proj_margin, proj_max + proj_margin)
    ax_center.set_ylim(y_min, y_max)
    ax_center.set_xlabel('Intensity', fontweight='bold', fontsize=label_fontsize)
    ax_center.tick_params(axis='x', which='major', labelsize=tick_fontsize)
    ax_center.tick_params(axis='y', which='major', labelleft=False)
    ax_center.grid(True, alpha=0.3, linestyle='--')
    ax_center.set_facecolor('#ffffff')
    ax_center.set_title('Projections', fontweight='bold', fontsize=title_fontsize, pad=12)
    
    # Right column setup (mirrored x-axis)
    ax_right.set_xlim(-receiver_x - x_margin, -source_x + x_margin)
    ax_right.set_ylim(y_min, y_max)
    ax_right.set_xlabel('Depth (m)', fontweight='bold', fontsize=label_fontsize)
    ax_right.tick_params(axis='x', which='major', labelsize=tick_fontsize)
    ax_right.tick_params(axis='y', which='major', labelleft=False)
    ax_right.grid(True, alpha=0.3, linestyle='--')
    ax_right.set_facecolor('#f8f9fa')
    ax_right.set_title(title, fontweight='bold', fontsize=title_fontsize, pad=12)
    
    # Negate x-tick labels to show positive values on right side
    xticks = ax_right.get_xticks()
    ax_right.set_xticklabels([f'{int(-x)}' for x in xticks])
    
    # Plot trajectories (static background)
    if show_trajectories:
        plot_trajectories_single(ax_left, theta_true, t, K, colors, mirror=False)
        plot_trajectories_single(ax_right, theta_est_reordered, t, K, colors, mirror=True)
    
    # Plot acquisition geometry (static)
    plot_acquisition_geometry(ax_left, sources, receivers, d, mirror=False)
    plot_acquisition_geometry(ax_right, sources, receivers, d, mirror=True)
    
    # Initialize containers for dynamic elements
    left_artists = []
    center_artists = []
    right_artists = []
    
    # Add time counter as suptitle (more reliable for animations)
    time_text = fig.suptitle('t = 0.000 s', fontsize=title_fontsize + 2, fontweight='bold',
                             y=0.98, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                              edgecolor='black', linewidth=2))

    def init():
        time_text.set_text('t = 0.000 s')  # Show initial time so we know it's working
        return [time_text]
    
    def update(frame):
        # Clear previous dynamic elements
        for artist in left_artists + center_artists + right_artists:
            artist.remove()
        left_artists.clear()
        center_artists.clear()
        right_artists.clear()
        
        t_val = t[frame].item()
        time_text.set_text(f't = {t_val:.3f} s')
        
        # Plot GMMs at current time
        plot_gmm_snapshot_animated(ax_left, theta_true, t_val, K, d, colors, 
                                   left_artists, is_true=True, mirror=False)
        plot_gmm_snapshot_animated(ax_right, theta_est_reordered, t_val, K, d, colors,
                                   right_artists, is_true=False, mirror=True)
        
        # Plot projections at current time
        proj_true_t = proj_true[0][frame].detach().cpu().numpy()
        proj_est_t = proj_est[0][frame].detach().cpu().numpy()
        
        sorted_indices = np.argsort(receiver_heights)
        sorted_heights = receiver_heights[sorted_indices]
        sorted_proj_true = proj_true_t[sorted_indices]
        sorted_proj_est = proj_est_t[sorted_indices]
        
        line_true, = ax_center.plot(sorted_proj_true, sorted_heights,
                                     linestyle='-', color='blue', linewidth=2.5,
                                     label='True' if frame == 0 else '', alpha=0.8)
        line_est, = ax_center.plot(sorted_proj_est, sorted_heights,
                                    linestyle='--', color='red', linewidth=2.5,
                                    label='Estimated' if frame == 0 else '', alpha=0.8)
        
        center_artists.extend([line_true, line_est])
        
        if frame == 0:
            ax_center.legend(fontsize=12, loc='upper right', framealpha=0.9)
        
        return left_artists + center_artists + right_artists + [time_text]
    
    # Create animation (blit=False to ensure suptitle is updated)
    anim = FuncAnimation(fig, update, init_func=init, frames=len(t),
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save animation
    if filename:
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=DPI//2)
        print(f"✓ Animation saved: {filename}")
    
    return anim


def plot_gmm_snapshot_animated(ax, theta, t_val, K, d, colors, artists, 
                                is_true=True, mirror=False):
    """
    Plot GMM snapshot and add artists to the provided list for animation.
    """
    import numpy as np
    
    chi2_vals = [1.0, 4.0, 9.0]
    linestyle = '-'
    marker = 'o'
    marker_size = 8
    zorder_ellipse_base = 10
    zorder_centroid = 20
    
    for k in range(K):
        x0 = theta['x0s'][k].detach().cpu().numpy()
        v0 = theta['v0s'][k].detach().cpu().numpy()
        a0 = theta['a0s'][k].detach().cpu().numpy()
        omega = theta['omegas'][k].item()
        alpha = theta['alphas'][k].item()
        U_skew = theta['U_skews'][k].detach().cpu().numpy()
        
        mu = x0 + v0 * t_val + 0.5 * a0 * t_val**2
        
        # Compute rotation matrix R(t) = R(2π·ω·t)
        theta_t = 2 * np.pi * omega * t_val
        
        R = np.array([[np.cos(theta_t), -np.sin(theta_t)],
                     [np.sin(theta_t), np.cos(theta_t)]])
        precision = U_skew.T @ U_skew
        covariance = np.linalg.inv(precision)
        Sigma_rotated = R @ covariance @ R.T
        
        # Mirror if requested (y-axis reflection: negate x and reflect covariance)
        if mirror:
            mu[0] = -mu[0]
            # Reflection matrix: M = [[-1, 0], [0, 1]]
            M = np.array([[-1, 0], [0, 1]])
            Sigma_rotated = M @ Sigma_rotated @ M.T
        
        for i, chi2 in enumerate(chi2_vals):
            Sigma_scaled = chi2 * Sigma_rotated
            alpha_ellipse = min(0.8, max(0.1, alpha * (1.0 - i * 0.2)))
            facecolor = colors[k]
            edgecolor = 'black'
            linewidth = 1.5 - i * 0.3
            
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma_scaled)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)
            
            ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                            facecolor=facecolor, edgecolor=edgecolor,
                            alpha=alpha_ellipse, linewidth=linewidth, linestyle=linestyle,
                            zorder=zorder_ellipse_base + i)
            ax.add_patch(ellipse)
            artists.append(ellipse)


def plot_acquisition_geometry(ax, sources, receivers, d, mirror=False):
    """
    Plot acquisition geometry (sources, receivers, and connecting lines).
    
    Parameters:
    - ax: Matplotlib axis
    - sources: Source positions
    - receivers: Receiver positions
    - d: Dimensionality (must be 2)
    - mirror: If True, flip x-coordinates for mirrored view
    """
    if d != 2:
        raise NotImplementedError("Acquisition geometry plotting only supports 2D")
    
    markersize = 6
    
    for n_s, source in enumerate(sources):
        src_x = -source[0].item() if mirror else source[0].item()
        src_y = source[1].item()
        
        # Plot source as red circle
        ax.plot(src_x, src_y, 'ro', 
               markersize=markersize+2, alpha=0.7, label='Source' if n_s == 0 else None,
               zorder=50)
        
        for n_r, rcvr in enumerate(receivers[n_s]):
            rcvr_x = -rcvr[0].item() if mirror else rcvr[0].item()
            rcvr_y = rcvr[1].item()
            
            # Plot receiver as blue circle
            ax.plot(rcvr_x, rcvr_y, 'bo', 
                   markersize=markersize, alpha=0.5,
                   label='Receivers' if n_s == 0 and n_r == 0 else None,
                   zorder=50)
            
            # Plot line connecting source to receiver
            ax.plot([src_x, rcvr_x], [src_y, rcvr_y], 
                   color='gold', linewidth=0.3, alpha=0.1, 
                   label='Rays' if n_s == 0 and n_r == 0 else None,
                   zorder=5)


def plot_trajectories_single(ax, theta, t, K, colors, mirror=False):
    """
    Plot trajectory paths for a single GMM (either true or estimated).
    
    Parameters:
    - ax: Matplotlib axis
    - theta: Parameter dictionary
    - t: Time vector
    - K: Number of Gaussians
    - colors: Color array
    - mirror: If True, flip x-coordinates for mirrored view
    """
    import numpy as np
    
    for k in range(K):
        x0 = theta['x0s'][k].detach().cpu().numpy()
        v0 = theta['v0s'][k].detach().cpu().numpy()
        a0 = theta['a0s'][k].detach().cpu().numpy()
        
        trajectory = []
        for t_i in t:
            t_np = t_i.item()
            # Translation with acceleration: x0 + v0*t + 0.5*a0*t^2
            pos = x0 + v0 * t_np + 0.5 * a0 * t_np**2
            if mirror:
                pos[0] = -pos[0]  # Flip x-coordinate
            trajectory.append(pos)
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=colors[k], alpha=0.4, linewidth=1.5, 
               linestyle='-', zorder=1)


def plot_trajectories(ax, theta_true, theta_est, t, K, colors):
    """
    Plot trajectory paths for both true and estimated GMMs.
    """
    import numpy as np
    
    # True trajectories - thin solid lines
    for k in range(K):
        x0 = theta_true['x0s'][k].detach().cpu().numpy()
        v0 = theta_true['v0s'][k].detach().cpu().numpy()
        a0 = theta_true['a0s'][k].detach().cpu().numpy()
        
        trajectory = []
        for t_i in t:
            t_np = t_i.item()
            # Translation with acceleration: x0 + v0*t + 0.5*a0*t^2
            pos = x0 + v0 * t_np + 0.5 * a0 * t_np**2
            trajectory.append(pos)
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=colors[k], alpha=0.2, linewidth=2.5, 
               linestyle='-', zorder=1)
    
    # Estimated trajectories - thin dashed lines
    for k in range(K):
        x0 = theta_est['x0s'][k].detach().cpu().numpy()
        v0 = theta_est['v0s'][k].detach().cpu().numpy()
        a0 = theta_est['a0s'][k].detach().cpu().numpy()
        
        trajectory = []
        for t_i in t:
            t_np = t_i.item()
            # Translation with acceleration: x0 + v0*t + 0.5*a0*t^2
            pos = x0 + v0 * t_np + 0.5 * a0 * t_np**2
            trajectory.append(pos)
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               color=colors[k], alpha=0.15, linewidth=2.5, 
               linestyle='--', zorder=1)


def plot_gmm_snapshot(ax, theta, t_val, K, d, colors, is_true=True, show_centroids=True, mirror=False):
    """
    Plot GMM configuration at a specific time with styling for true vs estimated.
    Uses 3 confidence levels (1σ, 2σ, 3σ) to match animation style.
    
    Parameters:
    - ax: Matplotlib axis
    - theta: Parameter dictionary
    - t_val: Time value (scalar)
    - K: Number of Gaussians
    - d: Dimensionality
    - colors: Color array for Gaussians
    - is_true: Whether this is ground truth (affects styling)
    - show_centroids: Whether to mark centroids
    - mirror: If True, flip x-coordinates for mirrored view
    """
    import numpy as np
    
    # Confidence levels (1σ, 2σ, 3σ equivalent in 2D)
    chi2_vals = [1.0, 4.0, 9.0]  # Chi-squared values for 2D Gaussian
    
    # Styling based on true vs estimated
    if is_true:
        # Ground truth: filled ellipses with solid black edges
        linestyle = '-'
        marker = 'o'
        marker_size = 8
        zorder_ellipse_base = 10
        zorder_centroid = 20
    else:
        # Estimated: same style as true (solid edges, no dashes)
        linestyle = '-'
        marker = 'x'
        marker_size = 10
        zorder_ellipse_base = 15
        zorder_centroid = 25
    
    # Plot Gaussians at the current time
    for k in range(K):
        x0 = theta['x0s'][k].detach().cpu().numpy()
        v0 = theta['v0s'][k].detach().cpu().numpy()
        a0 = theta['a0s'][k].detach().cpu().numpy()
        omega = theta['omegas'][k].item()
        alpha = theta['alphas'][k].item()
        U_skew = theta['U_skews'][k].detach().cpu().numpy()
        
        # Compute position at time t_val: x0 + v0*t + 0.5*a0*t^2
        # Center moves with acceleration, but doesn't rotate
        mu = x0 + v0 * t_val + 0.5 * a0 * t_val**2
        
        # Compute covariance rotated by 2π·ω·t (rotation of the Gaussian shape)
        # The Gaussian rotates in place around its center
        # FIXED: Use 2π·ω·t to match rotation model in generate_projections
        theta_t = 2 * np.pi * omega * t_val
        R = np.array([[np.cos(theta_t), -np.sin(theta_t)],
                     [np.sin(theta_t), np.cos(theta_t)]])
        precision = U_skew.T @ U_skew
        covariance = np.linalg.inv(precision)
        Sigma_rotated = R @ covariance @ R.T
        
        # Mirror if requested (y-axis reflection: negate x and reflect covariance)
        if mirror:
            mu[0] = -mu[0]
            # Reflection matrix: M = [[-1, 0], [0, 1]]
            M = np.array([[-1, 0], [0, 1]])
            Sigma_rotated = M @ Sigma_rotated @ M.T
        
        # Plot 3 confidence level ellipses (like animation)
        for i, chi2 in enumerate(chi2_vals):
            # Scale covariance for this confidence level
            Sigma_scaled = chi2 * Sigma_rotated
            
            # Compute transparency (decreases with confidence level)
            if is_true:
                # Ground truth: filled with decreasing opacity
                alpha_ellipse = min(0.8, max(0.1, alpha * (1.0 - i * 0.2)))
                facecolor = colors[k]
                edgecolor = 'black'
                linewidth = 1.5 - i * 0.3
            else:
                # Estimated: same style as true (filled with same colors and edges)
                alpha_ellipse = min(0.8, max(0.1, alpha * (1.0 - i * 0.2)))
                facecolor = colors[k]
                edgecolor = 'black'
                linewidth = 1.5 - i * 0.3
            
            plot_gaussian_ellipse(ax, mu, Sigma_scaled, 
                                facecolor=facecolor,
                                edgecolor=edgecolor,
                                alpha=alpha_ellipse,
                                linewidth=linewidth,
                                linestyle=linestyle,
                                label=None,
                                zorder=zorder_ellipse_base + i)
        
        # Mark centroid (only for true Gaussians)
        if show_centroids and is_true:
            marker_color = colors[k]
            ax.plot(mu[0], mu[1], marker=marker, color=marker_color, 
                   markersize=marker_size, markeredgewidth=2, zorder=zorder_centroid)


def plot_gaussian_ellipse(ax, mu, Sigma, facecolor='blue', edgecolor=None,
                          alpha=0.3, linewidth=2, linestyle='-', label=None, zorder=10):
    """
    Plot a 2D Gaussian ellipse.
    
    Parameters:
    - ax: Matplotlib axis
    - mu: Mean position [x, y]
    - Sigma: 2×2 covariance matrix (pre-scaled for desired confidence level)
    - facecolor: Fill color (or 'none')
    - edgecolor: Edge color (default: same as fill)
    - alpha: Transparency
    - linewidth: Edge width
    - linestyle: Edge style ('-', '--', etc.)
    - label: Legend label
    - zorder: Drawing order
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Ellipse dimensions from scaled covariance
    width, height = 2 * np.sqrt(eigenvalues)
    
    if edgecolor is None:
        edgecolor = facecolor
    
    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                     facecolor=facecolor, edgecolor=edgecolor,
                     alpha=alpha, linewidth=linewidth, linestyle=linestyle,
                     label=label, zorder=zorder)
    ax.add_patch(ellipse)


'--------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------- FIGURE 4: PARAMETER RECOVERY COMPARISON ------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def plot_parameter_recovery(theta_true, theta_est, K, filename=None, 
                            title="Parameter Recovery"):
    """
    Figure 2: Comprehensive parameter comparison showing true vs estimated values.
    
    Creates subplots for:
    - Rotation velocities (ω)
    - Attenuation coefficients (α)
    - Initial velocities (v0)
    - Covariance structure (U_skew diagonal elements)
    
    Parameters:
    - theta_true: True parameter dictionary
    - theta_est: Estimated parameter dictionary
    - K: Number of Gaussians
    - filename: Output filename
    - title: Overall figure title
    """
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # Extract parameters
    omegas_true = tensor_to_numpy([theta_true['omegas'][k].item() for k in range(K)])
    omegas_est = tensor_to_numpy([theta_est['omegas'][k].item() for k in range(K)])
    
    alphas_true = tensor_to_numpy([theta_true['alphas'][k].item() for k in range(K)])
    alphas_est = tensor_to_numpy([theta_est['alphas'][k].item() for k in range(K)])
    
    v0s_true = tensor_to_numpy(torch.stack(theta_true['v0s']))
    v0s_est = tensor_to_numpy(torch.stack(theta_est['v0s']))
    
    # 1. Rotation velocities (ω)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(K)
    width = 0.35
    ax1.bar(x - width/2, omegas_true, width, label='True', color=COLORS_TRUE, edgecolor='black', linewidth=0.8)
    ax1.bar(x + width/2, omegas_est, width, label='Estimated', color=COLORS_EST, edgecolor='black', linewidth=0.8)
    ax1.set_xlabel('Gaussian Index', fontweight='bold')
    ax1.set_ylabel(r'$\omega$ (rot/s)', fontweight='bold')
    ax1.set_title('Rotation Velocities', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'G{k+1}' for k in range(K)])
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Attenuation coefficients (α)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - width/2, alphas_true, width, label='True', color=COLORS_TRUE, edgecolor='black', linewidth=0.8)
    ax2.bar(x + width/2, alphas_est, width, label='Estimated', color=COLORS_EST, edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Gaussian Index', fontweight='bold')
    ax2.set_ylabel('$\\alpha$', fontweight='bold')
    ax2.set_title('Attenuation Coefficients', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'G{k+1}' for k in range(K)])
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Initial velocities (v0 components)
    ax3 = fig.add_subplot(gs[1, :])
    x_pos = np.arange(K * 2)  # 2 components per Gaussian
    colors_comp1 = [COLORS_TRUE if i % 2 == 0 else COLORS_EST for i in range(K * 2)]
    
    # Interleave true and estimated for each component
    v0_values = np.zeros(K * 2)
    v0_labels = []
    for k in range(K):
        v0_values[2*k] = v0s_true[k, 0]
        v0_values[2*k + 1] = v0s_est[k, 0]
        v0_labels.extend([rf'G{k+1}$_{{\mathrm{{true}}}}$', rf'G{k+1}$_{{\mathrm{{est}}}}$'])
    
    # v0_x components
    ax3.barh(x_pos, v0_values, height=0.35, color=colors_comp1, edgecolor='black', linewidth=0.6)
    ax3.set_yticks(x_pos)
    ax3.set_yticklabels(v0_labels, fontsize=8)
    ax3.set_xlabel('$v_{0,x}$ (m/s)', fontweight='bold')
    ax3.set_title('Initial Velocity ($x$-component)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # 4. Initial velocities (v0_y components)
    ax4 = fig.add_subplot(gs[2, :])
    v0_values_y = np.zeros(K * 2)
    for k in range(K):
        v0_values_y[2*k] = v0s_true[k, 1]
        v0_values_y[2*k + 1] = v0s_est[k, 1]
    
    ax4.barh(x_pos, v0_values_y, height=0.35, color=colors_comp1, edgecolor='black', linewidth=0.6)
    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(v0_labels, fontsize=8)
    ax4.set_xlabel('$v_{0,y}$ (m/s)', fontweight='bold')
    ax4.set_title('Initial Velocity ($y$-component)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    fig.suptitle(title, fontweight='bold', fontsize=14, y=0.98)
    
    if filename:
        save_figure(fig, filename)
    
    return fig


'--------------------------------------------------------------------------------------------------------------------------'
'------------------------------------------ FIGURE 3: ERROR ANALYSIS ------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def plot_error_analysis(theta_true, theta_est, K, filename=None,
                        title="Parameter Recovery Errors"):
    """
    Figure 3: Error analysis showing relative errors for all parameters.
    
    Parameters:
    - theta_true: True parameter dictionary
    - theta_est: Estimated parameter dictionary
    - K: Number of Gaussians
    - filename: Output filename
    - title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE)
    
    # Calculate relative errors
    omegas_true = tensor_to_numpy([theta_true['omegas'][k].item() for k in range(K)])
    omegas_est = tensor_to_numpy([theta_est['omegas'][k].item() for k in range(K)])
    omega_errors = 100 * np.abs(omegas_est - omegas_true) / np.abs(omegas_true)
    
    alphas_true = tensor_to_numpy([theta_true['alphas'][k].item() for k in range(K)])
    alphas_est = tensor_to_numpy([theta_est['alphas'][k].item() for k in range(K)])
    alpha_errors = 100 * np.abs(alphas_est - alphas_true) / np.abs(alphas_true)
    
    v0s_true = tensor_to_numpy(torch.stack(theta_true['v0s']))
    v0s_est = tensor_to_numpy(torch.stack(theta_est['v0s']))
    v0_errors_x = 100 * np.abs(v0s_est[:, 0] - v0s_true[:, 0]) / np.abs(v0s_true[:, 0])
    v0_errors_y = 100 * np.abs(v0s_est[:, 1] - v0s_true[:, 1]) / np.abs(v0s_true[:, 1])
    
    x = np.arange(K)
    
    # 1. Omega errors
    axes[0, 0].bar(x, omega_errors, color=COLORS_ERROR, edgecolor='black', linewidth=0.8)
    axes[0, 0].set_xlabel('Gaussian Index', fontweight='bold')
    axes[0, 0].set_ylabel('Relative Error (%)', fontweight='bold')
    axes[0, 0].set_title('Rotation Velocity Error', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'G{k+1}' for k in range(K)])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Alpha errors
    axes[0, 1].bar(x, alpha_errors, color=COLORS_ERROR, edgecolor='black', linewidth=0.8)
    axes[0, 1].set_xlabel('Gaussian Index', fontweight='bold')
    axes[0, 1].set_ylabel('Relative Error (%)', fontweight='bold')
    axes[0, 1].set_title('Attenuation Coefficient Error', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'G{k+1}' for k in range(K)])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. v0_x errors
    axes[1, 0].bar(x, v0_errors_x, color=COLORS_ERROR, edgecolor='black', linewidth=0.8)
    axes[1, 0].set_xlabel('Gaussian Index', fontweight='bold')
    axes[1, 0].set_ylabel('Relative Error (%)', fontweight='bold')
    axes[1, 0].set_title('Initial Velocity ($v_{0,x}$) Error', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'G{k+1}' for k in range(K)])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. v0_y errors
    axes[1, 1].bar(x, v0_errors_y, color=COLORS_ERROR, edgecolor='black', linewidth=0.8)
    axes[1, 1].set_xlabel('Gaussian Index', fontweight='bold')
    axes[1, 1].set_ylabel('Relative Error (%)', fontweight='bold')
    axes[1, 1].set_title('Initial Velocity ($v_{0,y}$) Error', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'G{k+1}' for k in range(K)])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title, fontweight='bold', fontsize=13, y=0.995)
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    return fig


'--------------------------------------------------------------------------------------------------------------------------'
'----------------------------------------- FIGURE 4: SINOGRAM COMPARISON --------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def plot_sinogram_comparison(proj_data, proj_est, t, receivers, filename=None,
                             title="Projection Data Comparison"):
    """
    Figure 4: Sinogram-style visualization comparing true and estimated projection data.
    
    Parameters:
    - proj_data: True projection data [N_sources x N_time x N_receivers]
    - proj_est: Estimated projection data [N_sources x N_time x N_receivers]
    - t: Time vector
    - receivers: Receiver positions
    - filename: Output filename
    - title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    # Convert to numpy
    proj_true_np = tensor_to_numpy(proj_data[0])  # First source
    proj_est_np = tensor_to_numpy(proj_est[0])
    t_np = tensor_to_numpy(t)
    
    # Receiver heights for y-axis
    receiver_heights = tensor_to_numpy(torch.tensor([r[1].item() for r in receivers[0]]))
    
    # Transpose to get time on x-axis, receivers on y-axis
    proj_true_T = proj_true_np.T  # Shape: [N_receivers x N_time]
    proj_est_T = proj_est_np.T
    residual_T = np.abs(proj_true_T - proj_est_T)
    
    # Shared colorbar limits
    vmin = min(proj_true_T.min(), proj_est_T.min())
    vmax = max(proj_true_T.max(), proj_est_T.max())
    
    # 1. True projections
    im1 = axes[0].imshow(proj_true_T, aspect='auto', cmap='viridis', 
                        extent=[t_np[0], t_np[-1], receiver_heights[-1], receiver_heights[0]],
                        vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_xlabel('Time (s)', fontweight='bold')
    axes[0].set_ylabel('Receiver Height (m)', fontweight='bold')
    axes[0].set_title('True Projections', fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity', fontweight='bold')
    
    # 2. Estimated projections
    im2 = axes[1].imshow(proj_est_T, aspect='auto', cmap='viridis',
                        extent=[t_np[0], t_np[-1], receiver_heights[-1], receiver_heights[0]],
                        vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_xlabel('Time (s)', fontweight='bold')
    axes[1].set_ylabel('Receiver Height (m)', fontweight='bold')
    axes[1].set_title('Estimated Projections', fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Intensity', fontweight='bold')
    
    # 3. Residual (absolute difference)
    im3 = axes[2].imshow(residual_T, aspect='auto', cmap='Reds',
                        extent=[t_np[0], t_np[-1], receiver_heights[-1], receiver_heights[0]],
                        interpolation='bilinear')
    axes[2].set_xlabel('Time (s)', fontweight='bold')
    axes[2].set_ylabel('Receiver Height (m)', fontweight='bold')
    axes[2].set_title('Absolute Residual', fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('|Difference|', fontweight='bold')
    
    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.00)
    plt.tight_layout()
    
    if filename:
        save_figure(fig, filename)
    
    return fig


'--------------------------------------------------------------------------------------------------------------------------'
'-------------------------------------- FIGURE 5: TRAJECTORY COMPARISON ---------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def plot_trajectory_comparison(theta_true, theta_est, t, K, sources, receivers, d,
                                filename=None, title="Trajectory Reconstruction"):
    """
    Figure 5: Spatial visualization showing true and estimated Gaussian trajectories.
    
    Parameters:
    - theta_true: True parameter dictionary
    - theta_est: Estimated parameter dictionary  
    - t: Time vector
    - K: Number of Gaussians
    - sources: Source positions
    - receivers: Receiver positions
    - d: Dimensionality
    - filename: Output filename
    - title: Plot title
    """
    from ..core.models import GMM_reco
    
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    
    # Get x0s and a0s (assumed known)
    x0s = theta_true['x0s']
    a0s = theta_true['a0s']
    
    # Compute trajectories
    GMM = GMM_reco(d, K, sources, receivers, x0s, a0s, omega_min=0.0, omega_max=10.0)
    
    # True trajectories
    traj_funcs_true = GMM.construct_trajectory_funcs()
    colors = cm.rainbow(np.linspace(0, 1, K))
    
    for k in range(K):
        traj_true = []
        traj_est = []
        for t_n in t:
            # True trajectory
            pos_true = traj_funcs_true(t_n, theta_true)[k]
            traj_true.append(tensor_to_numpy(pos_true))
            
            # Estimated trajectory  
            pos_est = traj_funcs_true(t_n, theta_est)[k]
            traj_est.append(tensor_to_numpy(pos_est))
        
        traj_true = np.array(traj_true)
        traj_est = np.array(traj_est)
        
        # Plot trajectories
        ax.plot(traj_true[:, 0], traj_true[:, 1], '-', color=colors[k], linewidth=2,
               label=f'G{k+1} (True)' if K <= 5 else (f'G{k+1}' if k < 2 else None), alpha=0.8)
        ax.plot(traj_est[:, 0], traj_est[:, 1], '--', color=colors[k], linewidth=2,
               label=f'G{k+1} (Est)' if K <= 5 else (f'G{k+1}' if k < 2 else None), alpha=0.8)
        
        # Mark start and end points
        ax.plot(traj_true[0, 0], traj_true[0, 1], 'o', color=colors[k], markersize=8,
               markeredgewidth=1.5, markeredgecolor='black')
        ax.plot(traj_true[-1, 0], traj_true[-1, 1], 's', color=colors[k], markersize=8,
               markeredgewidth=1.5, markeredgecolor='black')
    
    # Plot acquisition geometry (lighter)
    src = tensor_to_numpy(sources[0])
    ax.plot(src[0], src[1], 'r*', markersize=12, label='Source', 
           markeredgewidth=0.5, markeredgecolor='black', zorder=10)
    
    rcvrs = tensor_to_numpy(torch.stack(receivers[0]))
    ax.plot(rcvrs[:, 0], rcvrs[:, 1], 'ko', markersize=2, alpha=0.3, label='Receivers')
    
    ax.set_xlabel('$x$ (m)', fontweight='bold')
    ax.set_ylabel('$y$ (m)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(loc='best', framealpha=0.9, ncol=2 if K > 3 else 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    if filename:
        save_figure(fig, filename)
    
    return fig, ax


'--------------------------------------------------------------------------------------------------------------------------'
'--------------------------------------- COMBINED PUBLICATION FIGURE ------------------------------------------------------'
'--------------------------------------------------------------------------------------------------------------------------'

def create_publication_figure(theta_true, theta_est, proj_data, proj_est, t, K, 
                              sources, receivers, d, output_dir, prefix=""):
    """
    Create all publication-quality figures at once.
    
    Parameters:
    - theta_true: True parameter dictionary
    - theta_est: Estimated parameter dictionary
    - proj_data: True projection data
    - proj_est: Estimated projection data
    - t: Time vector
    - K: Number of Gaussians
    - sources: Source positions
    - receivers: Receiver positions
    - d: Dimensionality
    - output_dir: Output directory for saving figures
    - prefix: Optional prefix for filenames
    
    Returns:
    - Dictionary of figure objects
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    print("\n" + "="*70)
    print("Creating Publication-Quality Figures")
    print("="*70)
    
    # Figure 1: Acquisition geometry
    print("\n[1/5] Acquisition Geometry...")
    fig1 = plot_figure1_experimental_setup(
        sources, receivers, theta_true, t, K, d,
        filename=output_dir / f"{prefix}fig1_experimental_setup.pdf",
        time_snapshots=[0.62, 0.8]  # Two time points to show evolution
    )
    figures['geometry'] = fig1
    
    # Figure 2: Parameter recovery
    print("[2/5] Parameter Recovery...")
    fig2 = plot_parameter_recovery(
        theta_true, theta_est, K,
        filename=output_dir / f"{prefix}fig2_parameter_recovery.pdf",
        title="Parameter Recovery Comparison"
    )
    figures['parameters'] = fig2
    
    # Figure 3: Error analysis
    print("[3/5] Error Analysis...")
    fig3 = plot_error_analysis(
        theta_true, theta_est, K,
        filename=output_dir / f"{prefix}fig3_error_analysis.pdf",
        title="Parameter Recovery Errors"
    )
    figures['errors'] = fig3
    
    # Figure 4: Sinogram comparison
    print("[4/5] Sinogram Comparison...")
    fig4 = plot_sinogram_comparison(
        proj_data, proj_est, t, receivers,
        filename=output_dir / f"{prefix}fig4_sinogram_comparison.pdf",
        title="Projection Data Comparison"
    )
    figures['sinogram'] = fig4
    
    # Figure 5: Trajectory comparison
    print("[5/5] Trajectory Comparison...")
    fig5, _ = plot_trajectory_comparison(
        theta_true, theta_est, t, K, sources, receivers, d,
        filename=output_dir / f"{prefix}fig5_trajectory_comparison.pdf",
        title="Trajectory Reconstruction"
    )
    figures['trajectories'] = fig5
    
    print("\n" + "="*70)
    print("✓ All publication figures created successfully!")
    print(f"✓ Saved to: {output_dir}")
    print("="*70 + "\n")
    
    return figures
