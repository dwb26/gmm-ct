#!/usr/bin/env python3
"""
Quick launcher for the stability experiment with customizable parameters.
"""

from stability_experiment import run_stability_experiment, plot_stability_results
from time import time

if __name__ == "__main__":
    
    # Which N values to test
    N_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # How many random seeds per N (higher = more robust statistics)
    N_simulations_per_N = 50
    
    # Starting random seed
    base_seed = 100
    
    # Save animations for debugging (creates MP4 for each experiment)
    save_animations = False  # Set to False to skip animations and save time
    
    # ================================
    
    print("="*70)
    print(" GMM Reconstruction Stability Experiment")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  N values to test: {N_values}")
    print(f"  Simulations per N: {N_simulations_per_N}")
    print(f"  Total experiments: {len(N_values) * N_simulations_per_N}")
    print(f"  Base seed: {base_seed}")
    print(f"  Save animations: {save_animations}")
    print("\nStarting experiment...\n")
    
    # Run
    start = time()
    df, output_dir = run_stability_experiment(N_values, N_simulations_per_N, base_seed, save_animations)
    
    # Plot
    plot_stability_results(df, output_dir)
    
    elapsed = time() - start
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"Results directory: {output_dir}")
    
    # Show where animations are if they were saved
    if save_animations:
        from pathlib import Path
        animations = list(Path(output_dir).glob("animation*.mp4"))
        if animations:
            print(f"\n📹 Animations: {len(animations)} files saved")
            print(f"   Location: {output_dir}")
            print(f"\n   To open directory:")
            print(f"   open {output_dir}")
    
    print(f"{'='*70}\n")
