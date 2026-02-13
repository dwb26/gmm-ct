"""
Test spectral omega estimation across multiple omega values and motion scenarios.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tests.unit.test_spectral_omega_estimation import test_spectral_omega_estimation


def test_range_of_omegas():
    """Test spectral method across range of omega values."""
    
    omega_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    scenarios = [
        ('Fixed', False, [0.0, 0.0], [0.0, 0.0]),
        ('Slow motion', True, [2.0, 1.0], [0.0, -5.0]),
        ('Fast motion', True, [5.0, 2.0], [0.0, -9.81]),
    ]
    
    results_summary = {}
    
    for scenario_name, with_motion, v0, a0 in scenarios:
        print("\n" + "="*70)
        print(f"SCENARIO: {scenario_name}")
        print("="*70)
        
        results_summary[scenario_name] = {}
        
        for omega in omega_values:
            print(f"\n--- Testing ω = {omega} ---")
            
            # Choose duration to get ~2-3 rotations
            duration = 2.5 / omega
            
            results = test_spectral_omega_estimation(
                true_omega=omega,
                duration=duration,
                n_time_points=500,
                v0=v0,
                a0=a0,
                test_with_motion=with_motion
            )
            
            # Extract best result
            best_method = 'Detrend + Hann window'
            if results[best_method] is not None:
                error = results[best_method]['error']
                omega_est = results[best_method]['omega_est']
                
                results_summary[scenario_name][omega] = {
                    'error': error,
                    'omega_est': omega_est
                }
                
                print(f"✓ ω={omega:.1f}: Estimated={omega_est:.4f}, Error={error:.2f}%")
            else:
                results_summary[scenario_name][omega] = None
                print(f"✗ ω={omega:.1f}: Failed")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Estimated vs True
    ax1 = axes[0]
    for scenario_name in results_summary.keys():
        omegas = []
        omega_ests = []
        
        for omega in omega_values:
            if results_summary[scenario_name].get(omega) is not None:
                omegas.append(omega)
                omega_ests.append(results_summary[scenario_name][omega]['omega_est'])
        
        ax1.plot(omegas, omega_ests, 'o-', linewidth=2, markersize=8, label=scenario_name)
    
    # Perfect line
    omega_range = np.linspace(0, max(omega_values) + 0.5, 100)
    ax1.plot(omega_range, omega_range, 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
    
    ax1.set_xlabel('True ω')
    ax1.set_ylabel('Estimated ω')
    ax1.set_title('Spectral Method: Estimated vs True ω')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(0, max(omega_values) + 0.5)
    ax1.set_ylim(0, max(omega_values) + 0.5)
    
    # Plot 2: Relative error
    ax2 = axes[1]
    for scenario_name in results_summary.keys():
        omegas = []
        errors = []
        
        for omega in omega_values:
            if results_summary[scenario_name].get(omega) is not None:
                omegas.append(omega)
                errors.append(results_summary[scenario_name][omega]['error'])
        
        ax2.plot(omegas, errors, 'o-', linewidth=2, markersize=8, label=scenario_name)
    
    ax2.axhline(5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
    ax2.axhline(10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
    
    ax2.set_xlabel('True ω')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Spectral Method: Error across ω values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(omega_values) + 0.5)
    
    plt.tight_layout()
    output_file = 'test_output/spectral_method_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\nSummary figure saved to {output_file}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Scenario':<20} {'ω':<6} {'Estimated':<12} {'Error (%)':<10} {'Status'}")
    print("-"*70)
    
    for scenario_name in results_summary.keys():
        for omega in omega_values:
            if results_summary[scenario_name].get(omega) is not None:
                res = results_summary[scenario_name][omega]
                error = res['error']
                omega_est = res['omega_est']
                
                if error < 5:
                    status = "✓✓ Excellent"
                elif error < 10:
                    status = "✓ Good"
                else:
                    status = "⚠ Marginal"
                
                print(f"{scenario_name:<20} {omega:<6.1f} {omega_est:<12.4f} {error:<10.2f} {status}")
            else:
                print(f"{scenario_name:<20} {omega:<6.1f} {'FAILED':<12} {'-':<10} ✗")
    
    print("="*70)
    
    # Overall statistics
    all_errors = []
    for scenario_name in results_summary.keys():
        for omega in omega_values:
            if results_summary[scenario_name].get(omega) is not None:
                all_errors.append(results_summary[scenario_name][omega]['error'])
    
    if len(all_errors) > 0:
        print(f"\nOVERALL STATISTICS:")
        print(f"  Mean error: {np.mean(all_errors):.2f}%")
        print(f"  Median error: {np.median(all_errors):.2f}%")
        print(f"  Max error: {np.max(all_errors):.2f}%")
        print(f"  Success rate (< 5%): {100 * np.sum(np.array(all_errors) < 5) / len(all_errors):.1f}%")
        print(f"  Success rate (< 10%): {100 * np.sum(np.array(all_errors) < 10) / len(all_errors):.1f}%")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    mean_error = np.mean(all_errors)
    if mean_error < 5:
        print("\n✓✓ EXCELLENT: Spectral method is highly accurate and robust!")
        print("   Recommended as PRIMARY method for omega initialization.")
    elif mean_error < 10:
        print("\n✓ GOOD: Spectral method provides useful initialization")
        print("   Should significantly reduce multi-start requirements.")
    else:
        print("\n⚠ Spectral method shows promise but needs refinement")
    
    print("\nKey advantages:")
    print("  • Direct frequency extraction (no search required)")
    print("  • Robust to trajectory effects")
    print("  • Can handle multiple Gaussians (multiple peaks)")
    print("  • Computationally cheap")
    print("="*70)


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("#"*70)
    print("# COMPREHENSIVE SPECTRAL METHOD VALIDATION")
    print("#"*70)
    print("\nTesting across:")
    print("  • Multiple omega values: 0.5, 1.0, 1.5, 2.0, 2.5")
    print("  • Multiple motion scenarios: Fixed, Slow, Fast")
    print("#"*70)
    
    test_range_of_omegas()
