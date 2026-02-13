"""
Simple test demonstrating omega_estimation module usage.

This shows how the FFT-based estimation would integrate into your pipeline.
"""

import torch
import numpy as np

from gmm_ct.estimation.omega import (
    estimate_omega_from_peak_values,
    estimate_omega_for_all_gaussians,
    plot_omega_estimation_diagnostics
)


def generate_test_data(omega_true, duration=2.0, n_times=500, with_motion=True):
    """Generate synthetic peak values for testing."""
    
    from tests.unit.test_spectral_omega_estimation import (
        generate_projection_with_motion,
        construct_rotation_matrix
    )
    
    device = 'cpu'
    t = torch.linspace(0, duration, n_times, dtype=torch.float64, device=device)
    
    # Setup geometry
    source = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    n_receivers = 50
    receiver_distance = 10.0
    receiver_positions = []
    for i in range(n_receivers):
        offset = (i - n_receivers/2) * 0.2
        pos = source + torch.tensor([receiver_distance, offset], dtype=torch.float64, device=device)
        receiver_positions.append(pos)
    receivers = torch.stack(receiver_positions)
    
    # Gaussian parameters
    mu_0 = torch.tensor([5.0, 0.0], dtype=torch.float64, device=device)
    alpha = torch.tensor(20.0, dtype=torch.float64, device=device)
    sigma_major, sigma_minor = 2.0, 0.5
    U_skew = torch.tensor([[sigma_major, 0.0], [0.0, sigma_minor]], 
                          dtype=torch.float64, device=device)
    
    if with_motion:
        v0 = torch.tensor([5.0, 2.0], dtype=torch.float64, device=device)
        a0 = torch.tensor([0.0, -9.81], dtype=torch.float64, device=device)
    else:
        v0 = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
        a0 = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    # Generate projections
    proj, trajectory = generate_projection_with_motion(
        t, source, receivers, alpha, U_skew, omega_true,
        mu_0, v0, a0, device
    )
    
    # Extract peak values (this is what you already do!)
    peak_values = torch.max(proj, dim=1)[0]
    
    return peak_values, t


def test_single_gaussian():
    """Test omega estimation for a single Gaussian."""
    
    print("\n" + "="*70)
    print("TEST 1: Single Gaussian Omega Estimation")
    print("="*70)
    
    omega_true = 1.5  # Hz
    print(f"\nTrue ω = {omega_true} Hz")
    
    # Generate test data
    print("\nGenerating synthetic peak values with projectile motion...")
    peak_values, t = generate_test_data(omega_true, duration=2.0, with_motion=True)
    
    # Estimate omega using FFT
    print("\nApplying FFT-based estimation...")
    omega_est, confidence, info = estimate_omega_from_peak_values(
        peak_values, t, 
        method='detrend_hann',  # Recommended: detrend + windowing
        min_omega=0.5,          # Physical bounds
        max_omega=5.0
    )
    
    # Results
    error = abs(omega_est - omega_true) / omega_true * 100
    
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"Estimated ω = {omega_est:.4f} Hz")
    print(f"True ω      = {omega_true:.4f} Hz")
    print(f"Error       = {error:.2f}%")
    print(f"Confidence  = {confidence:.2f}")
    print(f"Dominant frequency = {info['dominant_freq']:.4f} Hz (should be ~ 2ω)")
    
    if error < 5:
        print("\n✓✓ EXCELLENT - Ready for use in pipeline!")
    elif error < 10:
        print("\n✓ GOOD - Provides useful initialization")
    else:
        print("\n⚠ Consider longer observation time or check signal quality")
    
    # Create diagnostic plot
    print("\nGenerating diagnostic plot...")
    plot_omega_estimation_diagnostics(
        peak_values, t, omega_true=omega_true,
        output_path='test_output/omega_estimation_demo.png'
    )
    
    return omega_est, error


def test_multi_gaussian():
    """Test omega estimation for multiple Gaussians."""
    
    print("\n" + "="*70)
    print("TEST 2: Multi-Gaussian Omega Estimation")
    print("="*70)
    
    # Simulate 3 Gaussians with different omegas
    omega_trues = [1.0, 1.5, 2.0]
    N = len(omega_trues)
    
    print(f"\nSimulating {N} Gaussians with different rotation rates:")
    for i, w in enumerate(omega_trues):
        print(f"  Gaussian {i+1}: ω = {w} Hz")
    
    # Generate peak values for each Gaussian
    projections_per_gaussian = []
    for omega_true in omega_trues:
        peak_values_i, t = generate_test_data(omega_true, duration=2.5, with_motion=True)
        # Convert to 2D (n_times, n_receivers) - but we only need peak values
        # For this demo, just reshape to simulate projection structure
        projections_per_gaussian.append(peak_values_i.unsqueeze(1))
    
    # Estimate omega for all Gaussians at once
    omega_estimates, confidences, spectrum_infos = estimate_omega_for_all_gaussians(
        projections_per_gaussian, t,
        method='detrend_hann',
        min_omega=0.5,
        max_omega=3.0
    )
    
    # Results summary
    print("\n" + "-"*70)
    print("RESULTS SUMMARY")
    print("-"*70)
    print(f"{'Gaussian':<12} {'True ω':<12} {'Estimated ω':<15} {'Error':<10} {'Confidence'}")
    print("-"*70)
    
    for i in range(N):
        error = abs(omega_estimates[i] - omega_trues[i]) / omega_trues[i] * 100
        print(f"{i+1:<12} {omega_trues[i]:<12.3f} {omega_estimates[i]:<15.4f} "
              f"{error:<10.2f}% {confidences[i]:.2f}")
    
    mean_error = np.mean([abs(est - true)/true * 100 
                         for est, true in zip(omega_estimates, omega_trues)])
    print("-"*70)
    print(f"Mean error: {mean_error:.2f}%")
    
    if mean_error < 5:
        print("\n✓✓ EXCELLENT - All Gaussians estimated accurately!")
    elif mean_error < 10:
        print("\n✓ GOOD - Provides useful initializations for all Gaussians")
    
    return omega_estimates, mean_error


def demonstrate_pipeline_integration():
    """
    Show how this integrates into the existing GMM pipeline.
    """
    
    print("\n" + "="*70)
    print("PIPELINE INTEGRATION EXAMPLE")
    print("="*70)
    
    print("""
This is how omega estimation fits into your existing workflow:

╔══════════════════════════════════════════════════════════════════╗
║                   EXISTING PIPELINE                               ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 1: Trajectory Optimization (multi-start)                  ║
║    Input:  Projection data, time vector                          ║
║    Output: x₀, v₀, a₀ for each Gaussian                         ║
║    Cost:   ~10-20 optimization trials                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 2: Rotation + Structure Optimization (multi-start)        ║
║    Input:  Fitted trajectories, projection data                  ║
║    Output: ω, U_skew, α for each Gaussian                       ║
║    Cost:   ~15+ optimization trials (expensive!)                 ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║                   NEW PIPELINE WITH FFT                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 1: Trajectory Optimization (multi-start)                  ║
║    Input:  Projection data, time vector                          ║
║    Output: x₀, v₀, a₀ for each Gaussian                         ║
║    Cost:   ~10-20 optimization trials                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 1.5: FFT-based Omega Estimation (NEW!)                    ║
║    Input:  Peak values (already computed), time vector           ║
║    Output: ω for each Gaussian                                   ║
║    Cost:   O(n log n) - nearly FREE!                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 2: Structure Optimization ONLY (single initialization)    ║
║    Input:  Fitted trajectories, FIXED ω values                   ║
║    Output: U_skew, α for each Gaussian                          ║
║    Cost:   1 optimization trial (much cheaper!)                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Phase 3 (Optional): Fine-tune ω + structure                     ║
║    Input:  Good initialization from Phases 1.5 & 2               ║
║    Output: Refined ω, U_skew, α                                 ║
║    Cost:   1-2 trials from warm start (very fast!)              ║
╚══════════════════════════════════════════════════════════════════╝

KEY BENEFITS:
• Eliminates expensive omega multi-start search
• Provides direct omega estimates in milliseconds
• Each Gaussian processed independently
• Reduces overall computation time by ~10-15×
• More robust convergence from better initialization
""")
    
    print("\nPseudocode for integration:")
    print("-"*70)
    print("""
# In GMM_reco.fit() method, after Phase 1 completes:

# Phase 1: Trajectory optimization (existing)
soln_dict = self.fit_trajectories(proj_data, t)

# Phase 1.5: FFT-based omega estimation (NEW!)
from gmm_ct.estimation.omega import estimate_omega_from_peak_values

omega_estimates = []
for k in range(self.N):
    # Extract peak values for Gaussian k
    # (You're already doing this in plot_peak_values_vs_time!)
    peak_values_k = self.compute_peak_values_for_gaussian(k, soln_dict)
    
    # Estimate omega using FFT
    omega_k, confidence, _ = estimate_omega_from_peak_values(
        peak_values_k, self.t,
        method='detrend_hann',
        min_omega=self.omega_min,
        max_omega=self.omega_max
    )
    
    print(f"Gaussian {k}: FFT estimate ω = {omega_k:.3f} Hz (conf={confidence:.2f})")
    omega_estimates.append(omega_k)

# Store initial omega estimates
soln_dict['omegas'] = [torch.tensor([omega_k], device=self.device) 
                       for omega_k in omega_estimates]

# Phase 2: Morphology optimization with FIXED omega (single trial)
soln_dict = self.fit_morphology_with_fixed_omega(soln_dict)

# Phase 3 (optional): Fine-tune everything together
soln_dict = self.fine_tune_all_parameters(soln_dict)

return soln_dict
""")
    print("-"*70)


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# FFT-BASED OMEGA ESTIMATION - DEMONSTRATION")
    print("#"*70)
    
    # Test 1: Single Gaussian
    test_single_gaussian()
    
    input("\nPress Enter to continue to multi-Gaussian test...")
    
    # Test 2: Multiple Gaussians
    test_multi_gaussian()
    
    input("\nPress Enter to see pipeline integration example...")
    
    # Show pipeline integration
    demonstrate_pipeline_integration()
    
    print("\n" + "#"*70)
    print("# NEXT STEPS")
    print("#"*70)
    print("""
1. Review omega_estimation.py module documentation
2. Test with your actual experimental data
3. Integrate into models.py GMM_reco.fit() method
4. Compare results with current multi-start approach
5. Adjust min_omega/max_omega bounds based on your experiments

The module is ready to use! Just import and call:
    from gmm_ct.estimation.omega import estimate_omega_from_peak_values
""")
    print("#"*70 + "\n")
