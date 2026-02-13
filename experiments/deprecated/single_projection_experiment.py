"""
    DEPRECATED — This file uses old import paths (sys.path hack to nonexistent src/ directory)
    and is retained for reference only. It will not run without modification.

    Single Projection Recovery Experiment
    
    Test if we can recover a STATIC single Gaussian from a single projection measurement.
    The Gaussian does not move or rotate - we only recover amplitude (alpha) and shape (U).
    Uses "best of 3" strategy: runs 3 independent trials and reports the best result.
"""

import sys
from pathlib import Path
from datetime import datetime

GRAVITATIONAL_ACCELERATION = 9.81  # m/s^2

# Add src directory to path so we can import gmm_ct
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from gmm_ct import GMM_reco, construct_receivers, set_random_seeds, export_parameters
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from torchmin import minimize


def compute_static_parameter_errors(theta_true, theta_est, N):
    """
    Compute relative L2 errors for static Gaussian parameters (alpha and U only).
    
    Parameters:
        theta_true: True parameters dictionary
        theta_est: Estimated parameters dictionary
        N: Number of Gaussians
    
    Returns:
        dict: Relative errors for alpha and U_skew
        float: Total combined error
    """
    errors = {}
    
    # Alphas (N scalar values)
    alphas_true = torch.stack([theta_true['alphas'][k] for k in range(N)])
    alphas_est = torch.stack([theta_est['alphas'][k] for k in range(N)])
    errors['alphas'] = (torch.norm(alphas_true - alphas_est) / torch.norm(alphas_true)).item()
    
    # U_skews (N matrices of dimension d x d)
    U_skews_true = torch.stack([theta_true['U_skews'][k].flatten() for k in range(N)])
    U_skews_est = torch.stack([theta_est['U_skews'][k].flatten() for k in range(N)])
    errors['U_skews'] = (torch.norm(U_skews_true - U_skews_est) / torch.norm(U_skews_true)).item()
    
    # Compute total error
    total_error = np.sqrt(errors['alphas']**2 + errors['U_skews']**2)
    
    return errors, total_error


def compute_projection_error(proj_true, proj_est):
    """
    Compute relative L2 error for projections.
    
    Parameters:
        proj_true: True projections (list of tensors)
        proj_est: Estimated projections (list of tensors)
    
    Returns:
        float: Relative projection error
    """
    proj_true_flat = torch.cat([p.flatten() for p in proj_true])
    proj_est_flat = torch.cat([p.flatten() for p in proj_est])
    return (torch.norm(proj_true_flat - proj_est_flat) / torch.norm(proj_true_flat)).item()


def visualize_static_gaussian_recovery(sources, rcvrs, theta_true, theta_est, output_dir, device):
    """
    Create comprehensive visualization of the static Gaussian recovery experiment.
    
    Parameters:
        sources: Source positions
        rcvrs: Receiver positions
        theta_true: True parameters
        theta_est: Estimated parameters
        output_dir: Directory to save plots
        device: Torch device
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    
    # Extract parameters
    x0 = theta_true['x0s'][0].cpu().numpy()
    alpha_true = theta_true['alphas'][0].item()
    alpha_est = theta_est['alphas'][0].item()
    U_true = theta_true['U_skews'][0].cpu().numpy()
    U_est = theta_est['U_skews'][0].cpu().numpy()
    
    source = sources[0].cpu().numpy()
    
    # Handle receivers - could be list of lists or tensor
    if isinstance(rcvrs, list) and len(rcvrs) > 0:
        if isinstance(rcvrs[0], list):
            receivers_array = torch.stack(rcvrs[0]).cpu().numpy()
        else:
            receivers_array = torch.stack(rcvrs).cpu().numpy()
    else:
        receivers_array = rcvrs.cpu().numpy()
    
    # Compute covariance matrices for ellipse visualization
    # Sigma = U @ U.T
    Sigma_true = U_true @ U_true.T
    Sigma_est = U_est @ U_est.T
    
    # Eigenvalues and eigenvectors for ellipse parameters
    eigvals_true, eigvecs_true = np.linalg.eigh(Sigma_true)
    eigvals_est, eigvecs_est = np.linalg.eigh(Sigma_est)
    
    # Ellipse parameters (2-sigma contour)
    angle_true = np.degrees(np.arctan2(eigvecs_true[1, 1], eigvecs_true[0, 1]))
    angle_est = np.degrees(np.arctan2(eigvecs_est[1, 1], eigvecs_est[0, 1]))
    width_true, height_true = 2 * 2 * np.sqrt(eigvals_true)  # 2-sigma
    width_est, height_est = 2 * 2 * np.sqrt(eigvals_est)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # ============================================
    # Subplot 1: Geometry - Equipment and Gaussian
    # ============================================
    ax1 = plt.subplot(131)
    
    # Plot source
    ax1.plot(source[0], source[1], 'r*', markersize=20, label='Source', zorder=5)
    
    # Plot receivers
    ax1.plot(receivers_array[:, 0], receivers_array[:, 1], 'b.', markersize=3, 
             label=f'Receivers (n={len(receivers_array)})', alpha=0.6)
    
    # Plot Gaussian center
    ax1.plot(x0[0], x0[1], 'ko', markersize=10, label='Gaussian center', zorder=5)
    
    # Plot true Gaussian ellipse
    ellipse_true = Ellipse(x0, width_true, height_true, angle=angle_true,
                          facecolor='green', alpha=0.3, edgecolor='green', 
                          linewidth=2, label='True Gaussian (2σ)')
    ax1.add_patch(ellipse_true)
    
    # Plot estimated Gaussian ellipse
    ellipse_est = Ellipse(x0, width_est, height_est, angle=angle_est,
                         facecolor='none', edgecolor='orange', linestyle='--',
                         linewidth=2, label='Estimated Gaussian (2σ)')
    ax1.add_patch(ellipse_est)
    
    ax1.set_xlabel('x₁ (m)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('x₂ (m)', fontsize=20, fontweight='bold')
    ax1.set_title('Experimental Setup', fontsize=22, fontweight='bold', pad=15)
    ax1.legend(fontsize=14, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axis('equal')
    ax1.tick_params(labelsize=16)
    
    # ============================================
    # Subplot 2: Projection Comparison
    # ============================================
    ax2 = plt.subplot(132)
    
    # Generate projections
    t_zero = torch.tensor([0.0], dtype=torch.float64, device=device)
    
    GMM_true = GMM_reco(2, 1, sources, [receivers_array],
                       [theta_true['x0s'][0]],
                       [torch.zeros(2, dtype=torch.float64, device=device)],
                       -24.0, -20.0, device=device)
    proj_true = GMM_true.generate_projections(t_zero, theta_true)[0].cpu().numpy().flatten()
    
    GMM_est = GMM_reco(2, 1, sources, [receivers_array],
                      [theta_est['x0s'][0]],
                      [torch.zeros(2, dtype=torch.float64, device=device)],
                      -24.0, -20.0, device=device)
    proj_est = GMM_est.generate_projections(t_zero, theta_est)[0].cpu().numpy().flatten()
    
    receiver_heights = receivers_array[:, 1]
    
    ax2.plot(proj_true, receiver_heights, 'g-', linewidth=2, label='True projection', alpha=0.8)
    ax2.plot(proj_est, receiver_heights, 'orange', linestyle='--', linewidth=2, 
             label='Estimated projection', alpha=0.8)
    
    ax2.set_xlabel('Projection intensity', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Receiver height (m)', fontsize=20, fontweight='bold')
    ax2.set_title('Projection Comparison', fontsize=22, fontweight='bold', pad=15)
    ax2.legend(fontsize=14, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=16)
    
    # ============================================
    # Subplot 3: Parameter Comparison Table
    # ============================================
    ax3 = plt.subplot(133)
    ax3.axis('off')
    
    # Create comparison table
    table_data = [
        ['Parameter', 'True Value', 'Estimated', 'Rel. Error'],
        ['', '', '', ''],
        ['α (amplitude)', f'{alpha_true:.6f}', f'{alpha_est:.6f}', 
         f'{abs(alpha_true - alpha_est)/abs(alpha_true):.2%}'],
        ['', '', '', ''],
        ['U₁₁', f'{U_true[0,0]:.6f}', f'{U_est[0,0]:.6f}', 
         f'{abs(U_true[0,0] - U_est[0,0])/abs(U_true[0,0]):.2%}'],
        ['U₁₂', f'{U_true[0,1]:.6f}', f'{U_est[0,1]:.6f}', 
         f'{abs(U_true[0,1] - U_est[0,1])/max(abs(U_true[0,1]), 1e-6):.2%}'],
        ['U₂₁', f'{U_true[1,0]:.6f}', f'{U_est[1,0]:.6f}', 
         f'{abs(U_true[1,0] - U_est[1,0])/max(abs(U_true[1,0]), 1e-6):.2%}'],
        ['U₂₂', f'{U_true[1,1]:.6f}', f'{U_est[1,1]:.6f}', 
         f'{abs(U_true[1,1] - U_est[1,1])/abs(U_true[1,1]):.2%}'],
        ['', '', '', ''],
        ['x₀ (position)', f'[{x0[0]:.2f}, {x0[1]:.2f}]', 'Fixed (known)', '—'],
    ]
    
    table = ax3.table(cellText=table_data, cellLoc='center', 
                     bbox=[0.1, 0.1, 0.85, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style separator rows
    for row_idx in [1, 3, 8]:
        for i in range(4):
            table[(row_idx, i)].set_facecolor('#E8E8E8')
    
    ax3.set_title('Parameter Recovery', fontsize=22, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'static_gaussian_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'static_gaussian_visualization.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: static_gaussian_visualization.pdf/png")
    plt.close()
    
    # ============================================
    # Additional plot: Residual (error in projection)
    # ============================================
    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    residual = proj_true - proj_est
    ax.plot(residual, receiver_heights, 'r-', linewidth=2, label='Residual (True - Est)')
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Projection residual', fontsize=20, fontweight='bold')
    ax.set_ylabel('Receiver height (m)', fontsize=20, fontweight='bold')
    ax.set_title('Projection Fit Residual', fontsize=22, fontweight='bold', pad=15)
    ax.legend(fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'projection_residual.pdf', dpi=300, bbox_inches='tight')
    print(f"   Saved: projection_residual.pdf")
    plt.close()


def optimize_static_gaussian(GMM, proj_target, theta_init, max_iter=500):
    """
    Optimize only alpha and U_skew for a static Gaussian (no motion, no rotation).
    
    Parameters:
        GMM: GMM_reco instance
        proj_target: Target projection to match
        theta_init: Initial parameter guess
        max_iter: Maximum optimization iterations
        
    Returns:
        dict: Optimized parameters
    """
    # Extract optimizable parameters
    alpha = theta_init['alphas'][0].clone().requires_grad_(True)
    U_skew = theta_init['U_skews'][0].clone().requires_grad_(True)
    
    # Fixed parameters
    x0 = theta_init['x0s'][0]
    
    def loss_fn(params_flat):
        """Loss function: L2 distance between simulated and target projection."""
        # Unpack parameters
        alpha_current = params_flat[0:1]
        U_skew_flat = params_flat[1:5].reshape(2, 2)
        
        # Build theta dict for this configuration
        theta_dict = {
            'alphas': [alpha_current],
            'U_skews': [U_skew_flat],
            'x0s': [x0],
            'v0s': [torch.zeros(2, dtype=torch.float64, device=GMM.device)],  # Static: v0=0
            'a0s': [torch.zeros(2, dtype=torch.float64, device=GMM.device)],  # Static: a0=0
            'omegas': [torch.zeros(1, dtype=torch.float64, device=GMM.device)]  # Static: ω=0
        }
        
        # Generate projection
        proj_sim = GMM.generate_projections(torch.tensor([0.0], device=GMM.device), theta_dict)
        
        # Compute L2 loss
        proj_sim_flat = proj_sim[0].flatten()
        proj_target_flat = proj_target[0].flatten()
        loss = torch.norm(proj_sim_flat - proj_target_flat)
        
        return loss
    
    # Pack initial parameters into flat tensor
    params_init = torch.cat([alpha.flatten(), U_skew.flatten()])
    
    # Optimize
    result = minimize(loss_fn, x0=params_init, method='l-bfgs',
                     tol=1e-10, options={'gtol': 1e-10, 'max_iter': max_iter, 'disp': False})
    
    # Unpack optimized parameters
    params_opt = result.x
    alpha_opt = params_opt[0:1]
    U_skew_opt = params_opt[1:5].reshape(2, 2)
    
    # Build solution dictionary
    soln_dict = {
        'alphas': [alpha_opt.detach()],
        'U_skews': [U_skew_opt.detach()],
        'x0s': [x0],
        'v0s': [torch.zeros(2, dtype=torch.float64, device=GMM.device)],
        'a0s': [torch.zeros(2, dtype=torch.float64, device=GMM.device)],
        'omegas': [torch.zeros(1, dtype=torch.float64, device=GMM.device)]
    }
    
    return soln_dict, result.fun.item()


def run_single_trial(N, d, sources, rcvrs, theta_true, trial_num, device, experiment_dir):
    """
    Run a single reconstruction trial with one projection of a STATIC Gaussian.
    
    Parameters:
        N: Number of Gaussians (should be 1 for static case)
        d: Dimensionality
        sources: Source positions
        rcvrs: Receiver positions
        theta_true: True parameters
        trial_num: Trial identifier
        device: Torch device
        experiment_dir: Output directory
        
    Returns:
        tuple: (parameter_errors, total_error, projection_error, solution_dict)
    """
    # Create GMM for true parameters
    GMM_true = GMM_reco(d, N, sources, rcvrs, 
                       [theta_true['x0s'][0]], 
                       [torch.zeros(d, dtype=torch.float64, device=device)],  # a0=0 for static
                       -24.0, -20.0,  # omega range (not used for static)
                       device=device, output_dir=experiment_dir)
    
    # Generate single projection at t=0 (static, so time doesn't matter)
    t_zero = torch.tensor([0.0], dtype=torch.float64, device=device)
    proj_data = GMM_true.generate_projections(t_zero, theta_true)
    
    print(f"\n  Trial {trial_num}: Optimizing static Gaussian (α and U only)...")
    
    # Create GMM for reconstruction
    GMM = GMM_reco(d, N, sources, rcvrs,
                   [theta_true['x0s'][0]],  # x0 is known/fixed
                   [torch.zeros(d, dtype=torch.float64, device=device)],
                   -24.0, -20.0,
                   device=device, output_dir=experiment_dir)
    
    # Random initialization for alpha and U_skew
    alpha_init = torch.rand(1, dtype=torch.float64, device=device) * 2.0  # Random in [0, 2]
    U_skew_init = torch.randn(d, d, dtype=torch.float64, device=device) * 0.1  # Small random
    
    theta_init = {
        'alphas': [alpha_init],
        'U_skews': [U_skew_init],
        'x0s': [theta_true['x0s'][0]],
        'v0s': [torch.zeros(d, dtype=torch.float64, device=device)],
        'a0s': [torch.zeros(d, dtype=torch.float64, device=device)],
        'omegas': [torch.zeros(1, dtype=torch.float64, device=device)]
    }
    
    # Optimize
    soln_dict, final_loss = optimize_static_gaussian(GMM, proj_data, theta_init)
    
    # Compute errors (only alpha and U_skew matter)
    param_errors, total_error = compute_static_parameter_errors(theta_true, soln_dict, N)
    
    # Projection error
    proj_final = GMM.generate_projections(t_zero, soln_dict)
    proj_error = compute_projection_error(proj_data, proj_final)
    
    print(f"    Alpha error: {param_errors['alphas']:.4e}")
    print(f"    U_skew error: {param_errors['U_skews']:.4e}")
    print(f"    Total parameter error: {total_error:.4e}")
    print(f"    Projection error (loss): {proj_error:.4e}")
    
    return param_errors, total_error, proj_error, soln_dict


def main():
    start_time = time()
    
    # Experiment configuration
    starting_seed = 90
    RANDOM_SEED = starting_seed
    set_random_seeds(RANDOM_SEED)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    print(f"🚀 Random seed: {RANDOM_SEED}")
    
    # Hyperparameters
    d = 2
    N = 1  # Single STATIC Gaussian
    
    # Create experiment-specific output directory
    project_root = Path(__file__).parent.parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = project_root / 'plots' / f"{timestamp}_static_gaussian_seed{RANDOM_SEED}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {experiment_dir}")
    
    # Source/receiver setup
    sources = [torch.tensor([-1, -1], dtype=torch.float64, device=device)]
    n_rcvrs = 128
    x1 = sources[0][0].item() + 5.  # Receiver depth
    x2_min = sources[0][1].item() - 2.  # Min receiver height
    x2_max = sources[0][1].item() + 2.  # Max receiver height
    rcvrs = construct_receivers(device, (n_rcvrs, x1, x2_min, x2_max))
    
    # Generate true parameters for a STATIC Gaussian
    # Position Gaussian halfway between source and receivers (depth), and centered vertically
    x0_depth = (sources[0][0].item() + x1) / 2  # Halfway in x1 (depth)
    x0_height = (x2_min + x2_max) / 2  # Centered in x2 (height)
    x0 = torch.tensor([x0_depth, x0_height], dtype=torch.float64, device=device)
    alpha_true = torch.tensor([1.5], dtype=torch.float64, device=device)
    U_skew_true = torch.tensor([[0.1, -0.05], [0.05, 0.15]], dtype=torch.float64, device=device)
    
    theta_true = {
        'alphas': [alpha_true],
        'U_skews': [U_skew_true],
        'x0s': [x0],
        'v0s': [torch.zeros(2, dtype=torch.float64, device=device)],
        'a0s': [torch.zeros(2, dtype=torch.float64, device=device)],
        'omegas': [torch.zeros(1, dtype=torch.float64, device=device)]
    }
    
    print("\n" + "="*70)
    print("STATIC GAUSSIAN RECOVERY - BEST OF 3 TRIALS")
    print("="*70)
    print(f"Configuration: N={N} static Gaussian, single projection per trial")
    print(f"Parameters to recover: α (amplitude) and U (shape matrix)")
    print(f"\nTrue parameters:")
    print(f"  α = {alpha_true.item():.4f}")
    print(f"  U = [[{U_skew_true[0,0]:.4f}, {U_skew_true[0,1]:.4f}],")
    print(f"       [{U_skew_true[1,0]:.4f}, {U_skew_true[1,1]:.4f}]]")
    print(f"  x0 = [{x0[0]:.4f}, {x0[1]:.4f}] (fixed/known)")
    
    # Run 3 trials with different random initializations
    trials_results = []
    for trial_num in range(1, 4):
        param_errors, total_error, proj_error, soln_dict = run_single_trial(
            N, d, sources, rcvrs, theta_true, trial_num, device, experiment_dir
        )
        
        trials_results.append({
            'trial_num': trial_num,
            'param_errors': param_errors,
            'total_error': total_error,
            'proj_error': proj_error,
            'solution': soln_dict
        })
    
    # Find best trial (lowest total parameter error)
    best_trial = min(trials_results, key=lambda x: x['total_error'])
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for result in trials_results:
        marker = " ⭐ BEST" if result['trial_num'] == best_trial['trial_num'] else ""
        print(f"\nTrial {result['trial_num']}:{marker}")
        print(f"  Total parameter error: {result['total_error']:.4e}")
        print(f"  Projection error: {result['proj_error']:.4e}")
        print(f"  Individual parameter errors:")
        for param_name, err in result['param_errors'].items():
            print(f"    {param_name:12s}: {err:.4e}")
        
        # Show recovered values
        alpha_est = result['solution']['alphas'][0].item()
        U_est = result['solution']['U_skews'][0]
        print(f"  Recovered α: {alpha_est:.4f} (true: {alpha_true.item():.4f})")
        print(f"  Recovered U: [[{U_est[0,0]:.4f}, {U_est[0,1]:.4f}],")
        print(f"                [{U_est[1,0]:.4f}, {U_est[1,1]:.4f}]]")
    
    print("\n" + "="*70)
    print(f"⭐ BEST TRIAL: #{best_trial['trial_num']}")
    print(f"   Total error: {best_trial['total_error']:.4e}")
    print(f"   Projection error: {best_trial['proj_error']:.4e}")
    
    alpha_best = best_trial['solution']['alphas'][0].item()
    U_best = best_trial['solution']['U_skews'][0]
    print(f"\n   Best recovered α: {alpha_best:.6f}")
    print(f"   True α:          {alpha_true.item():.6f}")
    print(f"   Relative error:  {best_trial['param_errors']['alphas']:.2%}")
    print("="*70)
    
    # Export results
    export_parameters(theta_true, experiment_dir / "true_parameters.md", 
                     title="Ground Truth Parameters (Static Gaussian)")
    export_parameters(best_trial['solution'], experiment_dir / "best_estimated_parameters.md", 
                     title=f"Best Estimated Parameters (Trial {best_trial['trial_num']})", 
                     theta_true=theta_true)
    
    # Visualize the setup and results
    print(f"\n📊 Generating visualizations...")
    visualize_static_gaussian_recovery(
        sources, rcvrs, theta_true, best_trial['solution'], 
        experiment_dir, device
    )
    
    print(f"\n✅ Experiment complete!")
    print(f"   Total time: {time() - start_time:.2f} seconds")
    print(f"   Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
