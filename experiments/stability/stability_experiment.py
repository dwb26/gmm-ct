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


def compute_param_rel_error_local(theta_true, theta_est, N, device):
    """
    Compute mean per-Gaussian relative L2 error in parameter space.

    Each Gaussian contributes equally: rel_err_k = ||θ_k_est - θ_k_true||_2 / ||θ_k_true||_2.
    Returns the mean over k.  Small values (→ 0) indicate better reconstruction.

    Returns:
        float: Mean relative L2 error across Gaussians.
    """
    rel_errors = []

    for k in range(N):
        params_true, params_est = [], []

        for key in ('alphas', 'x0s', 'v0s', 'U_skews', 'omegas'):
            params_true.append(theta_true[key][k].flatten())
            params_est.append(theta_est[key][k].flatten())

        theta_k_true = torch.cat(params_true)
        theta_k_est  = torch.cat(params_est)

        norm_true = torch.norm(theta_k_true).item()
        if norm_true > 1e-10:
            rel_errors.append(torch.norm(theta_k_true - theta_k_est).item() / norm_true)
        else:
            rel_errors.append(0.0)

    return float(np.mean(rel_errors))


def compute_param_rel_error_global(theta_true, theta_est, N, device):
    """
    Compute relative L2 error over the concatenated all-Gaussian parameter vector.

    Gives more weight to Gaussians with larger parameter magnitudes.

    Returns:
        float: Global relative L2 error.
    """
    all_params_true = []
    all_params_est  = []
    
    for k in range(N):
        for key in ('alphas', 'x0s', 'v0s', 'U_skews', 'omegas'):
            all_params_true.append(theta_true[key][k].flatten())
            all_params_est.append(theta_est[key][k].flatten())

    theta_all_true = torch.cat(all_params_true)
    theta_all_est  = torch.cat(all_params_est)

    norm_true = torch.norm(theta_all_true).item()
    if norm_true > 1e-10:
        return (torch.norm(theta_all_true - theta_all_est).item() / norm_true)
    return 0.0


def compute_proj_rel_error(proj_true, proj_est):
    """
    Compute relative L2 error in projection space.

    Returns:
        float: ||p_est - p_true||_2 / ||p_true||_2  (small → better)
    """
    proj_true_flat = torch.cat([p.flatten() for p in proj_true])
    proj_est_flat  = torch.cat([p.flatten() for p in proj_est])

    norm_true = torch.norm(proj_true_flat).item()
    if norm_true < 1e-10:
        return 0.0
    return (torch.norm(proj_true_flat - proj_est_flat).item() / norm_true)


def run_single_experiment(N, seed, d, N_projs, t, sources, rcvrs, omega_min, omega_max, 
                          i_loc, v_loc, a_loc, device, output_dir, save_animations=True):
    """
    Run a single GMM reconstruction experiment.
    
    Returns:
        tuple: (param_relerr_local, param_relerr_global, proj_relerr, computation_time)
    """
    from gmm_ct.visualization.publication import animate_temporal_gmm_comparison
    
    set_random_seeds(seed)

    # Sampling interval for Nyquist aliasing check in generate_true_param
    dt = (t[-1] - t[0]).item() / (len(t) - 1)

    # Generate true parameters
    theta_true = generate_true_param(d, N, i_loc, v_loc, a_loc, omega_min, omega_max,
                                     device=device, sampling_dt=dt)
    
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
    
    # Compute relative errors (smaller = better)
    param_relerr_local  = compute_param_rel_error_local(theta_true, theta_est, N, device)
    param_relerr_global = compute_param_rel_error_global(theta_true, theta_est, N, device)
    proj_relerr         = compute_proj_rel_error(proj_true, proj_est)
    
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
    
    return param_relerr_local, param_relerr_global, proj_relerr, computation_time


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
    
    # Fixed experiment parameters (match current simulate.yaml defaults)
    d = 2
    N_projs = 150
    t = torch.linspace(0., 1.5, N_projs, dtype=torch.float64, device=device)
    
    # Source/receiver configuration
    sources = [torch.tensor([-1, -1], dtype=torch.float64, device=device)]
    n_rcvrs = 128
    x1 = sources[0][0].item() + 5.
    x2_min = sources[0][1].item() - 2.
    x2_max = sources[0][1].item() + 2.
    rcvrs = construct_receivers(device, (n_rcvrs, x1, x2_min, x2_max))
    
    # Physical parameters (match current simulate.yaml defaults)
    i_loc = torch.tensor([1., 1.], dtype=torch.float64, device=device)
    v_loc = torch.tensor([.75, .5], dtype=torch.float64, device=device)
    a_loc = torch.tensor([0., -GRAVITATIONAL_ACCELERATION], dtype=torch.float64, device=device)
    omega_min = 2.0
    omega_max = 6.0
    
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
                param_relerr_local, param_relerr_global, proj_relerr, comp_time = run_single_experiment(
                    N, seed, d, N_projs, t, sources, rcvrs,
                    omega_min, omega_max, i_loc, v_loc, a_loc,
                    device, experiment_dir, save_animations=save_animations
                )

                results.append({
                    'N': N,
                    'seed': seed,
                    'param_relerr_local':  param_relerr_local,
                    'param_relerr_global': param_relerr_global,
                    'proj_relerr':         proj_relerr,
                    'computation_time':    comp_time
                })

                print(f"✓ Param err (local): {param_relerr_local:.4f}, "
                      f"Param err (global): {param_relerr_global:.4f}, "
                      f"Proj err: {proj_relerr:.4f}, "
                      f"Time: {comp_time:.2f}s")

            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append({
                    'N': N,
                    'seed': seed,
                    'param_relerr_local':  np.nan,
                    'param_relerr_global': np.nan,
                    'proj_relerr':         np.nan,
                    'computation_time':    np.nan
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
    Create publication-quality box plots with mean overlays for parameter and projection
    relative L2 error vs N.  Lower values indicate better reconstruction.

    Accepts DataFrames with either the new columns (param_relerr_local, param_relerr_global,
    proj_relerr) or the legacy accuracy columns (param_accuracy_local, param_accuracy_global,
    proj_accuracy_l2), converting the latter via rel_err = 1 - accuracy/100.

    Parameters:
        df: DataFrame produced by run_stability_experiment (or loaded from CSV).
        output_dir: pathlib.Path — directory where plots are saved.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # --- resolve column names (new vs legacy) ---
    if 'param_relerr_local' in df.columns:
        col_local  = 'param_relerr_local'
        col_global = 'param_relerr_global'
        col_proj   = 'proj_relerr'
    else:
        # Legacy accuracy columns → convert
        df = df.copy()
        df['param_relerr_local']  = 1.0 - df['param_accuracy_local']  / 100.0
        df['param_relerr_global'] = 1.0 - df['param_accuracy_global'] / 100.0
        df['proj_relerr']         = 1.0 - df['proj_accuracy_l2']      / 100.0
        col_local  = 'param_relerr_local'
        col_global = 'param_relerr_global'
        col_proj   = 'proj_relerr'

    N_values = sorted(df['N'].unique())

    param_local_data, param_global_data, proj_data = [], [], []
    param_local_means, param_global_means, proj_means = [], [], []

    def geom_mean(s):
        """Geometric mean of a Series, ignoring NaNs and non-positive values."""
        s = s.dropna()
        s = s[s > 0]
        return float(np.exp(np.log(s).mean())) if len(s) > 0 else np.nan

    for N in N_values:
        data_N = df[df['N'] == N]
        param_local_data.append(data_N[col_local].dropna().values)
        param_global_data.append(data_N[col_global].dropna().values)
        proj_data.append(data_N[col_proj].dropna().values)
        param_local_means.append(geom_mean(data_N[col_local]))
        param_global_means.append(geom_mean(data_N[col_global]))
        proj_means.append(geom_mean(data_N[col_proj]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # ------------------------------------------------------------------ #
    # Subplot 1: Parameter-space relative error
    # ------------------------------------------------------------------ #
    bp1 = ax1.boxplot(
        param_local_data, positions=N_values, widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.6, edgecolor='blue', linewidth=1.5),
        medianprops=dict(color='darkblue', linewidth=2),
        whiskerprops=dict(color='blue', linewidth=1.5),
        capprops=dict(color='blue', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='blue', markersize=6,
                        markeredgecolor='darkblue', alpha=0.5),
    )
    ax1.plot(N_values, param_local_means, 'ro-', linewidth=3, markersize=10,
             zorder=10, markeredgewidth=2, markeredgecolor='darkred', label='Geom. mean (local avg.)')
    # ax1.plot(N_values, param_global_means, 'go--', linewidth=3, markersize=10,
            #  zorder=10, markeredgewidth=2, markeredgecolor='darkgreen', label='Geom. mean (global avg.)')

    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Gaussians (N)', fontsize=20, fontweight='bold')
    ax1.set_ylabel(r'Relative $\ell_2$ error  $\|\hat{\theta}-\theta^*\|/\|\theta^*\|$',
                   fontsize=16, fontweight='bold')
    ax1.set_title('Parameter-space error vs N', fontsize=22, fontweight='bold', pad=15)
    ax1.set_xticks(N_values)
    ax1.set_xticklabels([int(n) for n in N_values])
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')

    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.6, label='Error distribution (box plot)'),
        Line2D([0], [0], color='red', marker='o', linewidth=3, markersize=10,
               markeredgewidth=2, markeredgecolor='darkred', label='Geom. mean error (local avg.)'),
        # Line2D([0], [0], color='green', marker='o', linestyle='--', linewidth=3, markersize=10,
            #    markeredgewidth=2, markeredgecolor='darkgreen', label='Geom. mean error (global avg.)'),
    ]
    ax1.legend(handles=legend_elements, fontsize=13, loc='upper left', framealpha=0.9)

    # ------------------------------------------------------------------ #
    # Subplot 2: Projection-space relative error
    # ------------------------------------------------------------------ #
    bp2 = ax2.boxplot(
        proj_data, positions=N_values, widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor='lightcoral', alpha=0.6, edgecolor='red', linewidth=1.5),
        medianprops=dict(color='darkred', linewidth=2),
        whiskerprops=dict(color='red', linewidth=1.5),
        capprops=dict(color='red', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=6,
                        markeredgecolor='darkred', alpha=0.5),
    )
    ax2.plot(N_values, proj_means, 'go-', linewidth=3, markersize=10,
             zorder=10, markeredgewidth=2, markeredgecolor='darkgreen', label='Geom. mean error')

    ax2.set_xlabel('Number of Gaussians (N)', fontsize=20, fontweight='bold')
    ax2.set_title('Projection-space error vs N', fontsize=22, fontweight='bold', pad=15)
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([int(n) for n in N_values])
    ax2.tick_params(labelsize=16, labelleft=False)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')

    legend_elements = [
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.6, label='Error distribution (box plot)'),
        Line2D([0], [0], color='green', marker='o', linewidth=3, markersize=10,
               markeredgewidth=2, markeredgecolor='darkgreen', label='Geom. mean error'),
    ]
    ax2.legend(handles=legend_elements, fontsize=13, loc='upper left', framealpha=0.9)

    plt.tight_layout()

    for fname in ('stability_boxplot_with_means.pdf', 'stability_boxplot_with_means.png'):
        out = output_dir / fname
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print("\n" + "="*80)
    print("SUMMARY  (relative L2 error — lower is better; mean = geometric mean)")
    print("="*80)
    print(f"{'N':<5} {'Param(local) geom.mean':<24} {'Param(local) std':<20} "
          f"{'Proj geom.mean':<16} {'Proj std':<15} {'Valid':<6}")
    print("-"*80)
    for N in N_values:
        data_N = df[df['N'] == N]
        n_valid = data_N[col_local].notna().sum()
        n_total = len(data_N)
        print(f"{int(N):<5} {geom_mean(data_N[col_local]):>22.5f}   {data_N[col_local].std():>18.5f}   "
              f"{geom_mean(data_N[col_proj]):>14.5f}   {data_N[col_proj].std():>13.5f}   "
              f"{n_valid}/{n_total}")
    print("="*80)


