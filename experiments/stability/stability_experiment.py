"""
Stability Experiment: Evaluate GMM_reco accuracy as a function of N (number of Gaussians).

This experiment runs multiple simulations for different values of N, measuring:
1. Parameter space accuracy (how well we recover alpha, mu, U, omega)
2. Projection space accuracy (how well reconstructed projections match true projections)

Results are saved to CSV and visualized with publication-quality plots.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from gmm_ct import GMM_reco, generate_true_param, construct_receivers, set_random_seeds
from gmm_ct.visualization.publication import reorder_theta_to_match_true
from gmm_ct.config.defaults import GRAVITATIONAL_ACCELERATION


def compute_parameter_accuracy_local(theta_true, theta_est, N, device):
    """
    Compute relative accuracy in parameter space as a percentage using LOCAL averaging.
    
    Computes per-Gaussian accuracy then averages across all Gaussians.
    This ensures each Gaussian contributes equally to the final metric.
    
    Returns:
        float: Parameter accuracy percentage (100% = perfect match)
    """
    gaussian_accuracies = []
    
    for k in range(N):
        # Collect all parameters for this Gaussian into one vector
        params_true = []
        params_est = []
        
        # Alpha
        params_true.append(theta_true['alphas'][k].flatten())
        params_est.append(theta_est['alphas'][k].flatten())
        
        # x0
        params_true.append(theta_true['x0s'][k].flatten())
        params_est.append(theta_est['x0s'][k].flatten())
        
        # v0
        params_true.append(theta_true['v0s'][k].flatten())
        params_est.append(theta_est['v0s'][k].flatten())
        
        # U_skew (flatten matrix)
        params_true.append(theta_true['U_skews'][k].flatten())
        params_est.append(theta_est['U_skews'][k].flatten())
        
        # omega
        params_true.append(theta_true['omegas'][k].flatten())
        params_est.append(theta_est['omegas'][k].flatten())
        
        # Concatenate all parameters for Gaussian k
        theta_k_true = torch.cat(params_true)
        theta_k_est = torch.cat(params_est)
        
        # Compute relative L2 error for this Gaussian
        norm_true = torch.norm(theta_k_true).item()
        if norm_true > 1e-10:
            relative_error_k = torch.norm(theta_k_true - theta_k_est).item() / norm_true
            accuracy_k = 100.0 * (1.0 - relative_error_k)
            # Clip to [0, 100] for this Gaussian
            accuracy_k = np.clip(accuracy_k, 0.0, 100.0)
            gaussian_accuracies.append(accuracy_k)
        else:
            # Edge case: if parameters are near zero, consider perfect match
            gaussian_accuracies.append(100.0)
    
    # Average accuracy across all Gaussians
    mean_accuracy = np.mean(gaussian_accuracies)
    
    return mean_accuracy


def compute_parameter_accuracy_global(theta_true, theta_est, N, device):
    """
    Compute relative accuracy in parameter space as a percentage using GLOBAL concatenation.
    
    Concatenates all N Gaussians' parameters into one large vector, then computes
    a single relative L2 error. This gives more weight to Gaussians with larger
    parameter magnitudes.
    
    Returns:
        float: Parameter accuracy percentage (100% = perfect match)
    """
    # Collect all parameters for ALL Gaussians into one vector
    all_params_true = []
    all_params_est = []
    
    for k in range(N):
        # Alpha
        all_params_true.append(theta_true['alphas'][k].flatten())
        all_params_est.append(theta_est['alphas'][k].flatten())
        
        # x0
        all_params_true.append(theta_true['x0s'][k].flatten())
        all_params_est.append(theta_est['x0s'][k].flatten())
        
        # v0
        all_params_true.append(theta_true['v0s'][k].flatten())
        all_params_est.append(theta_est['v0s'][k].flatten())
        
        # U_skew (flatten matrix)
        all_params_true.append(theta_true['U_skews'][k].flatten())
        all_params_est.append(theta_est['U_skews'][k].flatten())
        
        # omega
        all_params_true.append(theta_true['omegas'][k].flatten())
        all_params_est.append(theta_est['omegas'][k].flatten())
    
    # Concatenate all parameters from all Gaussians
    theta_all_true = torch.cat(all_params_true)
    theta_all_est = torch.cat(all_params_est)
    
    # Compute single relative L2 error for all parameters
    norm_true = torch.norm(theta_all_true).item()
    if norm_true > 1e-10:
        relative_error = torch.norm(theta_all_true - theta_all_est).item() / norm_true
        accuracy = 100.0 * (1.0 - relative_error)
        accuracy = np.clip(accuracy, 0.0, 100.0)
    else:
        # Edge case: if parameters are near zero, consider perfect match
        accuracy = 100.0
    
    return accuracy


def compute_projection_accuracy_l2(proj_true, proj_est):
    """
    Compute relative accuracy in projection space as a percentage using L2 norm.
    
    Parameters:
        proj_true: True projections (list of tensors)
        proj_est: Estimated projections (list of tensors)
    
    Returns:
        float: Projection accuracy percentage (100% = perfect match)
    """
    # Concatenate all sources' projections
    proj_true_flat = torch.cat([p.flatten() for p in proj_true])
    proj_est_flat = torch.cat([p.flatten() for p in proj_est])
    
    # Compute relative L2 error
    norm_true = torch.norm(proj_true_flat, p=2).item()
    if norm_true < 1e-10:
        return 100.0  # Edge case: if true projections are zero
    
    relative_error = torch.norm(proj_true_flat - proj_est_flat, p=2).item() / norm_true
    accuracy_percent = 100.0 * (1.0 - relative_error)
    
    return accuracy_percent


def compute_projection_accuracy_l1(proj_true, proj_est):
    """
    Compute relative accuracy in projection space as a percentage using L1 norm.
    
    L1 norm is less sensitive to large outliers compared to L2, giving a metric
    that may better reflect average pointwise accuracy.
    
    Parameters:
        proj_true: True projections (list of tensors)
        proj_est: Estimated projections (list of tensors)
    
    Returns:
        float: Projection accuracy percentage (100% = perfect match)
    """
    # Concatenate all sources' projections
    proj_true_flat = torch.cat([p.flatten() for p in proj_true])
    proj_est_flat = torch.cat([p.flatten() for p in proj_est])
    
    # Compute relative L1 error
    norm_true = torch.norm(proj_true_flat, p=1).item()
    if norm_true < 1e-10:
        return 100.0  # Edge case: if true projections are zero
    
    relative_error = torch.norm(proj_true_flat - proj_est_flat, p=1).item() / norm_true
    accuracy_percent = 100.0 * (1.0 - relative_error)
    
    return accuracy_percent


def run_single_experiment(N, seed, d, N_projs, t, sources, rcvrs, omega_min, omega_max, 
                          i_loc, v_loc, a_loc, device, output_dir, save_animations=True):
    """
    Run a single GMM reconstruction experiment.
    
    Returns:
        tuple: (param_accuracy_local, param_accuracy_global, proj_accuracy_l2, proj_accuracy_l1, computation_time)
    """
    from gmm_ct.visualization.publication import animate_temporal_gmm_comparison
    
    set_random_seeds(seed)
    
    # Generate true parameters
    theta_true = generate_true_param(d, N, i_loc, v_loc, a_loc, omega_min, omega_max, device=device)
    
    x0s = theta_true['x0s']
    a0s = theta_true['a0s']
    
    # Generate true projections
    GMM_true = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=output_dir)
    proj_true = GMM_true.generate_projections(t, theta_true)
    
    # Fit model
    start_time = time()
    GMM = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=output_dir)
    theta_est = GMM.fit(proj_true, t)
    computation_time = time() - start_time
    
    # Reorder to match true Gaussians
    theta_est, _ = reorder_theta_to_match_true(theta_true, theta_est, N)
    
    # Generate estimated projections
    proj_est = GMM.generate_projections(t, theta_est)
    
    # Compute accuracies (both local and global averaging methods)
    param_accuracy_local = compute_parameter_accuracy_local(theta_true, theta_est, N, device)
    param_accuracy_global = compute_parameter_accuracy_global(theta_true, theta_est, N, device)
    proj_accuracy_l2 = compute_projection_accuracy_l2(proj_true, proj_est)
    proj_accuracy_l1 = compute_projection_accuracy_l1(proj_true, proj_est)
    
    # Save animation for debugging if requested
    if save_animations:
        animation_filename = output_dir / f"animation_N{N}_seed{seed}.mp4"
        print(f"    📹 Saving animation to: {animation_filename}")
        try:
            animate_temporal_gmm_comparison(sources, rcvrs, theta_true, theta_est, 
                                           t, N, d, filename=animation_filename)
            print(f"    ✓ Animation saved successfully")
        except Exception as e:
            print(f"    ⚠️  Animation failed: {e}")
            import traceback
            traceback.print_exc()
    
    return param_accuracy_local, param_accuracy_global, proj_accuracy_l2, proj_accuracy_l1, computation_time


def run_stability_experiment(N_values, N_simulations_per_N, base_seed=100, save_animations=True):
    """
    Run the full stability experiment across multiple N values and random seeds.
    
    Parameters:
        N_values: List of N values to test (e.g., [3, 5, 7, 10, 15])
        N_simulations_per_N: Number of random seeds per N value
        base_seed: Starting seed value
        save_animations: Whether to save MP4 animations for each experiment (for debugging)
    
    Returns:
        pd.DataFrame: Results with columns [N, seed, param_accuracy, proj_accuracy, time]
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    print(f"📊 Running stability experiment:")
    print(f"   N values: {N_values}")
    print(f"   Simulations per N: {N_simulations_per_N}")
    print(f"   Total experiments: {len(N_values) * N_simulations_per_N}")
    print(f"   Save animations: {save_animations}")
    
    # Fixed experiment parameters
    d = 2
    N_projs = 2**6 + 1
    t = torch.linspace(0., 2.0, N_projs, dtype=torch.float64, device=device)
    
    # Source/receiver configuration
    sources = [torch.tensor([-1, -1], dtype=torch.float64, device=device)]
    n_rcvrs = 128
    x1 = sources[0][0].item() + 5.
    x2_min = sources[0][1].item() - 2.
    x2_max = sources[0][1].item() + 2.
    rcvrs = construct_receivers(device, (n_rcvrs, x1, x2_min, x2_max))
    
    # Physical parameters
    i_loc = torch.tensor([1., 1.], dtype=torch.float64, device=device)
    v_loc = torch.tensor([.75, .5], dtype=torch.float64, device=device)
    a_loc = torch.tensor([0., -GRAVITATIONAL_ACCELERATION], dtype=torch.float64, device=device)
    omega_min = -24.0
    omega_max = omega_min + 4.0
    
    # Create output directory
    project_root = Path(__file__).parent.parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = project_root / 'plots' / f"stability_experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = []
    total_experiments = len(N_values) * N_simulations_per_N
    experiment_count = 0
    
    for N in N_values:
        print(f"\n{'='*60}")
        print(f"Testing N = {N} Gaussians")
        print(f"{'='*60}")
        
        for sim_idx in range(N_simulations_per_N):
            experiment_count += 1
            seed = base_seed + sim_idx
            
            print(f"  [{experiment_count}/{total_experiments}] N={N}, seed={seed}...", end=" ", flush=True)
            
            try:
                param_acc_local, param_acc_global, proj_acc_l2, proj_acc_l1, comp_time = run_single_experiment(
                    N, seed, d, N_projs, t, sources, rcvrs, 
                    omega_min, omega_max, i_loc, v_loc, a_loc, 
                    device, experiment_dir, save_animations=save_animations
                )
                
                results.append({
                    'N': N,
                    'seed': seed,
                    'param_accuracy_local': param_acc_local,
                    'param_accuracy_global': param_acc_global,
                    'proj_accuracy_l2': proj_acc_l2,
                    'proj_accuracy_l1': proj_acc_l1,
                    'computation_time': comp_time
                })
                
                print(f"✓ Param(L): {param_acc_local:.2f}%, Param(G): {param_acc_global:.2f}%, Proj(L2): {proj_acc_l2:.2f}%, Proj(L1): {proj_acc_l1:.2f}%, Time: {comp_time:.2f}s")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append({
                    'N': N,
                    'seed': seed,
                    'param_accuracy_local': np.nan,
                    'param_accuracy_global': np.nan,
                    'proj_accuracy_l2': np.nan,
                    'proj_accuracy_l1': np.nan,
                    'computation_time': np.nan
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = experiment_dir / "stability_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Print summary of saved files
    if save_animations:
        animation_files = list(experiment_dir.glob("animation_*.mp4"))
        print(f"📹 Animations saved: {len(animation_files)} files")
        if animation_files:
            print(f"   Location: {experiment_dir}")
            for anim in sorted(animation_files):
                print(f"   - {anim.name}")
    
    return df, experiment_dir


def plot_stability_results(df, output_dir):
    """
    Create publication-quality box plots with mean overlays for parameter and projection accuracy.
    
    Parameters:
        df: DataFrame with columns [N, seed, param_accuracy_local, param_accuracy_global, 
                                    proj_accuracy_l2, proj_accuracy_l1, computation_time]
        output_dir: Directory to save plots
    """
    # Get unique N values
    N_values = sorted(df['N'].unique())
    
    # Prepare data for box plots (filter out NaNs)
    param_local_data = []
    param_global_data = []
    proj_l2_data = []
    proj_l1_data = []
    param_local_means = []
    param_global_means = []
    proj_l2_means = []
    proj_l1_means = []
    
    for N in N_values:
        data_N = df[df['N'] == N]
        
        # Filter out NaN values for box plots
        param_local_values = data_N['param_accuracy_local'].dropna().values
        param_global_values = data_N['param_accuracy_global'].dropna().values
        proj_l2_values = data_N['proj_accuracy_l2'].dropna().values
        proj_l1_values = data_N['proj_accuracy_l1'].dropna().values
        
        param_local_data.append(param_local_values)
        param_global_data.append(param_global_values)
        proj_l2_data.append(proj_l2_values)
        proj_l1_data.append(proj_l1_values)
        
        # Compute means (pandas mean already ignores NaN)
        param_local_means.append(data_N['param_accuracy_local'].mean())
        param_global_means.append(data_N['param_accuracy_global'].mean())
        proj_l2_means.append(data_N['proj_accuracy_l2'].mean())
        proj_l1_means.append(data_N['proj_accuracy_l1'].mean())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============================================
    # Subplot 1: Parameter Space Accuracy
    # ============================================
    
    # Box plots for local averaging (primary, more visible)
    bp1 = ax1.boxplot(param_local_data, positions=N_values, widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.6, edgecolor='blue', linewidth=1.5),
                      medianprops=dict(color='darkblue', linewidth=2),
                      whiskerprops=dict(color='blue', linewidth=1.5),
                      capprops=dict(color='blue', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='blue', markersize=6, 
                                     markeredgecolor='darkblue', alpha=0.5))
    
    # Overlay mean lines for both methods
    ax1.plot(N_values, param_local_means, 'ro-', linewidth=3, markersize=10, 
            label='Local avg.', zorder=10, markeredgewidth=2, markeredgecolor='darkred')
    ax1.plot(N_values, param_global_means, 'go--', linewidth=3, markersize=10, 
            label='Global avg.', zorder=10, markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax1.set_xlabel('Number of Gaussians (N)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Parameter space accuracy (%)', fontsize=20, fontweight='bold')
    ax1.set_title('Parameter space accuracy vs N', fontsize=22, fontweight='bold', pad=15)
    ax1.set_xticks(N_values)
    ax1.set_xticklabels([int(n) for n in N_values])
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-5, 105)
    
    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.6, label='Local accuracy distribution (box plot)'),
        Line2D([0], [0], color='red', marker='o', linewidth=3, markersize=10, 
               markeredgewidth=2, markeredgecolor='darkred', label='Mean local accuracy'),
        Line2D([0], [0], color='green', marker='o', linestyle='--', linewidth=3, markersize=10,
               markeredgewidth=2, markeredgecolor='darkgreen', label='Mean global accuracy')
    ]
    ax1.legend(handles=legend_elements, fontsize=14, loc='lower left', framealpha=0.9)
    
    # ============================================
    # Subplot 2: Projection Space Accuracy (L1 norm only)
    # ============================================
    
    # Box plots with lighter color (L1 norm)
    bp2 = ax2.boxplot(proj_l1_data, positions=N_values, widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.6, edgecolor='red', linewidth=1.5),
                      medianprops=dict(color='darkred', linewidth=2),
                      whiskerprops=dict(color='red', linewidth=1.5),
                      capprops=dict(color='red', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                     markeredgecolor='darkred', alpha=0.5))
    
    # Overlay mean for L1 norm
    ax2.plot(N_values, proj_l1_means, 'go-', linewidth=3, markersize=10,
            label='Mean accuracy', zorder=10, markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax2.set_xlabel('Number of Gaussians (N)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Projection space accuracy (%)', fontsize=20, fontweight='bold')
    ax2.set_title('Projection space accuracy vs N (L1 norm)', fontsize=22, fontweight='bold', pad=15)
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([int(n) for n in N_values])
    ax2.tick_params(labelsize=16)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-5, 105)
    
    # Custom legend
    legend_elements = [
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.6, label='Accuracy distribution (box plot)'),
        Line2D([0], [0], color='green', marker='o', linewidth=3, markersize=10,
               markeredgewidth=2, markeredgecolor='darkgreen', label='Mean accuracy')
    ]
    ax2.legend(handles=legend_elements, fontsize=14, loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'stability_boxplot_with_means.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    output_path_png = output_dir / 'stability_boxplot_with_means.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path_png}")
    
    plt.close()
    
    # ============================================
    # Create summary statistics table
    # ============================================
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    print(f"{'N':<4} {'Local Mean':<12} {'Local Std':<12} {'Global Mean':<13} {'Global Std':<13} {'Proj(L2) Mean':<15} {'Proj(L2) Std':<15} {'Proj(L1) Mean':<15} {'Proj(L1) Std':<15} {'Valid':<10}")
    print("-"*120)
    
    for N in N_values:
        data_N = df[df['N'] == N]
        param_local_mean = data_N['param_accuracy_local'].mean()
        param_local_std = data_N['param_accuracy_local'].std()
        param_global_mean = data_N['param_accuracy_global'].mean()
        param_global_std = data_N['param_accuracy_global'].std()
        proj_l2_mean = data_N['proj_accuracy_l2'].mean()
        proj_l2_std = data_N['proj_accuracy_l2'].std()
        proj_l1_mean = data_N['proj_accuracy_l1'].mean()
        proj_l1_std = data_N['proj_accuracy_l1'].std()
        n_total = len(data_N)
        n_valid_local = data_N['param_accuracy_local'].notna().sum()
        n_valid_global = data_N['param_accuracy_global'].notna().sum()
        n_valid_proj_l2 = data_N['proj_accuracy_l2'].notna().sum()
        n_valid_proj_l1 = data_N['proj_accuracy_l1'].notna().sum()
        
        print(f"{int(N):<4} {param_local_mean:>11.2f}% {param_local_std:>11.2f}% "
              f"{param_global_mean:>12.2f}% {param_global_std:>12.2f}% "
              f"{proj_l2_mean:>14.2f}% {proj_l2_std:>14.2f}% "
              f"{proj_l1_mean:>14.2f}% {proj_l1_std:>14.2f}% {n_valid_local}/{n_total}")
    
    print("="*120)


