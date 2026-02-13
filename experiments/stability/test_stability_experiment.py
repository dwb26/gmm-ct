#!/usr/bin/env python3
"""
Quick test of the stability experiment with minimal parameters.
Run this first to verify everything works before running the full experiment.
"""

from stability_experiment import run_stability_experiment, plot_stability_results
from time import time

if __name__ == "__main__":
    print("="*70)
    print(" QUICK TEST - Stability Experiment")
    print("="*70)
    print("\nRunning minimal test with N=[3, 5] and 2 simulations each...")
    print("Animations will be saved for debugging.")
    print("This should take ~1-2 minutes.\n")
    
    start = time()
    
    # Minimal test configuration
    df, output_dir = run_stability_experiment(
        N_values=[1, 2, 3],
        N_simulations_per_N=2,
        base_seed=42,
        save_animations=True  # Enable animations for debugging
    )
    
    # Generate plots
    plot_stability_results(df, output_dir)
    
    elapsed = time() - start
    
    print(f"\n{'='*70}")
    print(f"✅ TEST PASSED!")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Results directory: {output_dir}")
    
    # List animations
    from pathlib import Path
    animations = list(Path(output_dir).glob("animation*.mp4"))
    if animations:
        print(f"\n📹 Animations saved ({len(animations)} files):")
        for anim in sorted(animations):
            print(f"   {anim}")
        print(f"\nTo view animations, open the directory:")
        print(f"   open {output_dir}")
    
    print(f"\nIf this worked, you're ready to run the full experiment!")
    print(f"Edit run_stability_experiment.py to customize parameters.")
    print(f"{'='*70}\n")
