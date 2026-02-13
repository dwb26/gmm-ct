#!/usr/bin/env python3
"""
Compare two stability experiments by overlaying the mean accuracies from an older
experiment onto the box plots from a newer experiment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime


def compare_stability_experiments(old_csv_path, new_csv_path, output_dir=None):
    """
    Load two stability experiment results and create comparison plots.
    
    Parameters:
        old_csv_path: Path to older experiment's CSV file
        new_csv_path: Path to newer experiment's CSV file
        output_dir: Directory to save comparison plots (creates new folder if None)
    """
    # Load data
    print(f"Loading older experiment: {old_csv_path}")
    df_old = pd.read_csv(old_csv_path)
    print(f"  N values: {sorted(df_old['N'].unique())}")
    print(f"  Total trials: {len(df_old)}")
    
    print(f"\nLoading newer experiment: {new_csv_path}")
    df_new = pd.read_csv(new_csv_path)
    print(f"  N values: {sorted(df_new['N'].unique())}")
    print(f"  Total trials: {len(df_new)}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(new_csv_path).parent.parent / f"stability_comparison_{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Get N values
    N_values_old = sorted(df_old['N'].unique())
    N_values_new = sorted(df_new['N'].unique())
    
    # Use all N values from new experiment for box plots
    # Only use N values up to 7 from old experiment for overlay
    N_values_old_overlay = [n for n in N_values_old if n <= 7]
    
    print(f"\nNew experiment N values (for box plots): {N_values_new}")
    print(f"Old experiment N values (for overlay): {N_values_old_overlay}")
    
    # Compute means from old experiment (only up to N=7)
    old_param_means = []
    
    for N in N_values_old_overlay:
        data_N_old = df_old[df_old['N'] == N]
        old_param_means.append(data_N_old['param_accuracy'].mean())
    
    # Prepare box plot data from new experiment (all N values)
    new_param_data = []
    new_proj_data = []
    new_param_means = []
    new_proj_means = []
    
    for N in N_values_new:
        data_N_new = df_new[df_new['N'] == N]
        # Filter out NaN values for box plots
        param_values = data_N_new['param_accuracy'].dropna().values
        proj_values = data_N_new['proj_accuracy'].dropna().values
        
        new_param_data.append(param_values)
        new_proj_data.append(proj_values)
        new_param_means.append(data_N_new['param_accuracy'].mean())
        new_proj_means.append(data_N_new['proj_accuracy'].mean())
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============================================
    # Subplot 1: Parameter Space Accuracy
    # ============================================
    
    # Box plots from new experiment
    bp1 = ax1.boxplot(new_param_data, positions=N_values_new, widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.6, edgecolor='blue', linewidth=1.5),
                      medianprops=dict(color='darkblue', linewidth=2),
                      whiskerprops=dict(color='blue', linewidth=1.5),
                      capprops=dict(color='blue', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='blue', markersize=6, 
                                     markeredgecolor='darkblue', alpha=0.5))
    
    # Overlay new experiment means
    ax1.plot(N_values_new, new_param_means, 'ro-', linewidth=3, markersize=10, 
            label='Loc. avg.', zorder=10, markeredgewidth=2, markeredgecolor='darkred')
    
    # Overlay old experiment means (only up to N=7)
    ax1.plot(N_values_old_overlay, old_param_means, 'go--', linewidth=3, markersize=10, 
            label='Glob. avg.', zorder=10, markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax1.set_xlabel('Number of Gaussians (N)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Parameter Space Accuracy (%)', fontsize=20, fontweight='bold')
    ax1.set_title('Parameter Space Accuracy Comparison', fontsize=22, fontweight='bold', pad=15)
    ax1.set_xticks(N_values_new)
    ax1.set_xticklabels([int(n) for n in N_values_new])
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-5, 105)
    
    # Custom legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.6, label='Distribution (box plot)'),
        Line2D([0], [0], color='red', marker='o', linewidth=3, markersize=10, 
               markeredgewidth=2, markeredgecolor='darkred', label='Loc. avg.'),
        Line2D([0], [0], color='green', marker='o', linestyle='--', linewidth=3, markersize=10,
               markeredgewidth=2, markeredgecolor='darkgreen', label='Glob. avg.')
    ]
    ax1.legend(handles=legend_elements, fontsize=14, loc='lower left', framealpha=0.9)
    
    # ============================================
    # Subplot 2: Projection Space Accuracy
    # ============================================
    
    # Box plots from new experiment
    bp2 = ax2.boxplot(new_proj_data, positions=N_values_new, widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.6, edgecolor='red', linewidth=1.5),
                      medianprops=dict(color='darkred', linewidth=2),
                      whiskerprops=dict(color='red', linewidth=1.5),
                      capprops=dict(color='red', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                     markeredgecolor='darkred', alpha=0.5))
    
    # Overlay new experiment means only (no old projection results)
    ax2.plot(N_values_new, new_proj_means, 'go-', linewidth=3, markersize=10,
            label='Mean accuracy', zorder=10, markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax2.set_xlabel('Number of Gaussians (N)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Projection Space Accuracy (%)', fontsize=20, fontweight='bold')
    ax2.set_title('Projection Space Accuracy', fontsize=22, fontweight='bold', pad=15)
    ax2.set_xticks(N_values_new)
    ax2.set_xticklabels([int(n) for n in N_values_new])
    ax2.tick_params(labelsize=16)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-5, 105)
    
    # Custom legend
    legend_elements = [
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.6, label='Distribution (box plot)'),
        Line2D([0], [0], color='green', marker='o', linewidth=3, markersize=10,
               markeredgewidth=2, markeredgecolor='darkgreen', label='Mean accuracy')
    ]
    ax2.legend(handles=legend_elements, fontsize=14, loc='lower left', framealpha=0.9)
    output_path_pdf = output_dir / 'stability_comparison.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path_pdf}")
    
    output_path_png = output_dir / 'stability_comparison.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path_png}")
    
    plt.close()
    
    # ============================================
    # Create comparison statistics table
    # ============================================
    print("\n" + "="*100)
    print("COMPARISON STATISTICS")
    print("="*100)
    print(f"{'N':<4} {'Glob. Param':<14} {'Loc. Param':<14} {'Δ Param':<12} {'Loc. Proj':<14}")
    print("-"*100)
    
    # Print statistics for N values that have old data
    for i, N in enumerate(N_values_old_overlay):
        idx_new = N_values_new.index(N)
        old_p = old_param_means[i]
        new_p = new_param_means[idx_new]
        delta_p = new_p - old_p
        new_proj = new_proj_means[idx_new]
        
        print(f"{int(N):<4} {old_p:>13.2f}% {new_p:>13.2f}% {delta_p:>+11.2f}% {new_proj:>13.2f}%")
    
    # Print statistics for N values that only have new data
    for N in N_values_new:
        if N not in N_values_old_overlay:
            idx_new = N_values_new.index(N)
            new_p = new_param_means[idx_new]
            new_proj = new_proj_means[idx_new]
            print(f"{int(N):<4} {'N/A':>13} {new_p:>13.2f}% {'N/A':>11} {new_proj:>13.2f}%")
    
    print("="*100)
    
    # Save comparison statistics to CSV
    comparison_data = []
    for i, N in enumerate(N_values_old_overlay):
        idx_new = N_values_new.index(N)
        comparison_data.append({
            'N': N,
            'glob_param_mean': old_param_means[i],
            'loc_param_mean': new_param_means[idx_new],
            'delta_param': new_param_means[idx_new] - old_param_means[i],
            'loc_proj_mean': new_proj_means[idx_new]
        })
    
    # Add new N values without old data
    for N in N_values_new:
        if N not in N_values_old_overlay:
            idx_new = N_values_new.index(N)
            comparison_data.append({
                'N': N,
                'glob_param_mean': np.nan,
                'loc_param_mean': new_param_means[idx_new],
                'delta_param': np.nan,
                'loc_proj_mean': new_proj_means[idx_new]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = output_dir / 'comparison_statistics.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n💾 Comparison statistics saved to: {comparison_csv}")
    
    return output_dir


if __name__ == "__main__":
    import sys
    
    # Default paths
    old_csv = "/Users/danburrows/Documents/Codes/GaussianMixtureCT/plots/stability_experiment_20251213_000411/stability_results.csv"
    new_csv = "/Users/danburrows/Documents/Codes/GaussianMixtureCT/plots/stability_experiment_20251215_201929/stability_results.csv"
    
    # Allow command line override
    if len(sys.argv) > 2:
        old_csv = sys.argv[1]
        new_csv = sys.argv[2]
    
    print("="*100)
    print("STABILITY EXPERIMENT COMPARISON")
    print("="*100)
    
    output_dir = compare_stability_experiments(old_csv, new_csv)
    
    print(f"\n{'='*100}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*100}")
    print(f"Results saved to: {output_dir}")
