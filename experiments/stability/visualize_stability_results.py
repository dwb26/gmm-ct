"""
    Visualize Stability Experiment Results from CSV
    
    Creates publication-quality box plots with mean overlays from saved CSV data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_stability_results(csv_path):
    """
    Load stability experiment results from CSV file.
    
    Parameters:
        csv_path: Path to stability_results.csv
        
    Returns:
        DataFrame with results
    """
    df = pd.read_csv(csv_path)
    # Drop rows with NaN values
    df = df.dropna()
    return df


def plot_stability_with_means(df, output_dir):
    """
    Create box plots with mean overlays for parameter and projection accuracy.
    
    Parameters:
        df: DataFrame with columns [N, seed, param_accuracy, proj_accuracy, computation_time]
        output_dir: Directory to save plots
    """
    # Get unique N values
    N_values = sorted(df['N'].unique())
    
    # Prepare data for box plots (filter out NaNs)
    param_data = []
    proj_data = []
    param_means = []
    proj_means = []
    
    for N in N_values:
        data_N = df[df['N'] == N]
        # Filter out NaN values for box plots
        param_values = data_N['param_accuracy'].dropna().values
        proj_values = data_N['proj_accuracy'].dropna().values
        
        param_data.append(param_values)
        proj_data.append(proj_values)
        # Use pandas mean which already ignores NaN
        param_means.append(data_N['param_accuracy'].mean())
        proj_means.append(data_N['proj_accuracy'].mean())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============================================
    # Subplot 1: Parameter Space Accuracy
    # ============================================
    
    # Box plots with lighter color
    bp1 = ax1.boxplot(param_data, positions=N_values, widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.6, edgecolor='blue', linewidth=1.5),
                      medianprops=dict(color='darkblue', linewidth=2),
                      whiskerprops=dict(color='blue', linewidth=1.5),
                      capprops=dict(color='blue', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='blue', markersize=6, 
                                     markeredgecolor='darkblue', alpha=0.5))
    
    # Overlay means with distinct styling
    ax1.plot(N_values, param_means, 'ro-', linewidth=3, markersize=10, 
            label='Mean accuracy', zorder=10, markeredgewidth=2, markeredgecolor='darkred')
    
    ax1.set_xlabel('Number of Gaussians (N)', fontsize=20)
    ax1.set_ylabel('Parameter Space Accuracy (%)', fontsize=20)
    ax1.set_title('Parameter Space Accuracy vs N', fontsize=22, fontweight='bold', pad=15)
    ax1.set_xticks(N_values)
    ax1.set_xticklabels([int(n) for n in N_values])
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-5, 105)
    
    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.6, label='Distribution (box plot)'),
        Line2D([0], [0], color='red', marker='o', linewidth=3, markersize=10, 
               markeredgewidth=2, markeredgecolor='darkred', label='Mean accuracy')
    ]
    ax1.legend(handles=legend_elements, fontsize=14, loc='lower left', framealpha=0.9)
    
    # ============================================
    # Subplot 2: Projection Space Accuracy
    # ============================================
    
    # Box plots with lighter color
    bp2 = ax2.boxplot(proj_data, positions=N_values, widths=0.6,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.6, edgecolor='red', linewidth=1.5),
                      medianprops=dict(color='darkred', linewidth=2),
                      whiskerprops=dict(color='red', linewidth=1.5),
                      capprops=dict(color='red', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                     markeredgecolor='darkred', alpha=0.5))
    
    # Overlay means with distinct styling
    ax2.plot(N_values, proj_means, 'go-', linewidth=3, markersize=10,
            label='Mean accuracy', zorder=10, markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax2.set_xlabel('Number of Gaussians (N)', fontsize=20)
    ax2.set_ylabel('Projection Space Accuracy (%)', fontsize=20)
    ax2.set_title('Projection Space Accuracy vs N', fontsize=22, fontweight='bold', pad=15)
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([int(n) for n in N_values])
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
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'stability_boxplot_with_means.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    output_path_png = Path(output_dir) / 'stability_boxplot_with_means.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path_png}")
    
    plt.close()
    
    # ============================================
    # Create summary statistics table
    # ============================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'N':<4} {'Param Mean':<12} {'Param Std':<12} {'Proj Mean':<12} {'Proj Std':<12} {'Valid/Total':<12}")
    print("-"*80)
    
    for N in N_values:
        data_N = df[df['N'] == N]
        param_mean = data_N['param_accuracy'].mean()
        param_std = data_N['param_accuracy'].std()
        proj_mean = data_N['proj_accuracy'].mean()
        proj_std = data_N['proj_accuracy'].std()
        n_total = len(data_N)
        n_valid_param = data_N['param_accuracy'].notna().sum()
        n_valid_proj = data_N['proj_accuracy'].notna().sum()
        
        print(f"{int(N):<4} {param_mean:>11.2f}% {param_std:>11.2f}% {proj_mean:>11.2f}% {proj_std:>11.2f}% {n_valid_param}/{n_total} | {n_valid_proj}/{n_total}")
    
    print("="*80)


def main():
    """Main function to process and visualize stability results."""
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_stability_results.py <path_to_stability_results.csv>")
        print("\nExample:")
        print("  python visualize_stability_results.py ../../../plots/stability_experiment_20251213_000411/stability_results.csv")
        return
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    print(f"📊 Loading stability results from: {csv_path}")
    
    # Load data
    df = load_stability_results(csv_path)
    
    print(f"✅ Loaded {len(df)} valid trials")
    print(f"   N values: {sorted(df['N'].unique())}")
    print(f"   Seeds: {df['seed'].min()} to {df['seed'].max()}")
    
    # Output directory is same as CSV location
    output_dir = csv_path.parent
    
    # Generate plots
    print(f"\n📈 Generating box plots with mean overlays...")
    plot_stability_with_means(df, output_dir)
    
    print(f"\n✅ Visualization complete!")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
