"""
Test DTW with MOVING Gaussian (projectile motion).

Critical test: Does DTW still work when the Gaussian is moving along a trajectory?
This is the actual scenario in your problem.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw


def generate_peak_pattern_with_trajectory(alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line):
    """
    Generate projection peak values for a MOVING, ROTATING anisotropic Gaussian.
    
    Parameters
    ----------
    alpha : float
        Peak height coefficient
    U_skew : torch.Tensor, shape (2, 2)
        Covariance structure matrix
    omega : float
        Angular velocity (Hz)
    x0 : torch.Tensor, shape (2,)
        Initial position
    v0 : torch.Tensor, shape (2,)
        Initial velocity
    a0 : torch.Tensor, shape (2,)
        Acceleration
    t : torch.Tensor, shape (n_times,)
        Time points
    source : torch.Tensor, shape (2,)
        X-ray source position
    receiver_line : list of torch.Tensor
        Receiver positions
    
    Returns
    -------
    peak_values : np.ndarray
        Peak projection values at each time point
    """
    device = t.device
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
    
    peak_values = []
    
    for t_n in t:
        # Trajectory: μ(t) = x0 + v0*t + 0.5*a0*t^2
        mu_t = x0 + v0 * t_n + 0.5 * a0 * t_n**2
        
        # Rotation matrix at time t
        angle = 2 * torch.pi * omega * t_n
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R_t = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], 
                          dtype=torch.float64, device=device)
        
        # Rotated covariance structure
        U_rot = U_skew @ R_t.T
        
        # Compute projection at each receiver
        projections = []
        for receiver in receiver_line:
            r_minus_s = receiver - source
            r_minus_s_hat = r_minus_s / torch.norm(r_minus_s)
            
            # Projection formula
            U_r_hat = U_rot @ r_minus_s_hat
            U_r = U_rot @ r_minus_s
            U_mu = U_rot @ (source - mu_t)  # Note: mu_t is time-dependent!
            
            norm_term = torch.norm(U_r_hat)
            quotient = sqrt_pi * alpha / (norm_term + 1e-10)
            
            inner_prod = torch.dot(U_r.squeeze(), U_mu)
            inner_prod_sq = inner_prod ** 2
            divisor = torch.norm(U_r) ** 2 + 1e-10
            subtractor = torch.norm(U_mu) ** 2
            
            exp_arg = inner_prod_sq / divisor - subtractor
            proj = quotient * torch.exp(exp_arg)
            projections.append(proj.item())
        
        # Extract peak value (max across receivers)
        peak_values.append(max(projections))
    
    return np.array(peak_values)


def test_dtw_with_trajectory():
    """
    Test DTW omega estimation with MOVING Gaussian.
    
    This is the real scenario: Gaussian moves AND rotates.
    """
    
    print("="*70)
    print("DTW WITH PROJECTILE MOTION TEST")
    print("="*70)
    
    device = 'cpu'
    
    # Setup geometry
    source = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    n_receivers = 40
    receiver_line = [
        torch.tensor([5.0, y], dtype=torch.float64, device=device)
        for y in np.linspace(-2.0, 2.0, n_receivers)
    ]
    
    # Time points (sparse like in real experiment)
    n_times = 20
    t = torch.linspace(0, 1, n_times, dtype=torch.float64, device=device)
    
    # True parameters
    alpha_true = 12.0
    true_omega = 1.85  # Hz
    
    # Anisotropic covariance: ellipse with axes ratio ~3:1
    U_true = torch.tensor([[30.0, 0.0], 
                           [0.0, 10.0]], dtype=torch.float64, device=device)
    
    # TRAJECTORY PARAMETERS (the new element!)
    x0_true = torch.tensor([1.0, 0.5], dtype=torch.float64, device=device)
    v0_true = torch.tensor([2.0, 1.7], dtype=torch.float64, device=device)
    a0_true = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    print(f"\nTrue parameters:")
    print(f"  Alpha: {alpha_true}")
    print(f"  Omega: {true_omega} Hz")
    print(f"  U_skew: Anisotropic (3:1 ratio)")
    print(f"  x0: {x0_true.numpy()}")
    print(f"  v0: {v0_true.numpy()}")
    print(f"  a0: {a0_true.numpy()}")
    print(f"  Time points: {n_times}")
    
    # Generate TRUE peak pattern (with trajectory motion)
    print(f"\nGenerating TRUE peak pattern with trajectory motion...")
    true_peaks = generate_peak_pattern_with_trajectory(
        alpha_true, U_true, true_omega, x0_true, v0_true, a0_true,
        t, source, receiver_line
    )
    
    print(f"  Peak range: [{true_peaks.min():.4f}, {true_peaks.max():.4f}]")
    print(f"  Peak variation: {true_peaks.std():.4f}")
    
    # Test range of candidate omegas
    omega_min, omega_max = 1.0, 2.5
    n_candidates = 100
    omegas_test = np.linspace(omega_min, omega_max, n_candidates)
    
    print(f"\nTesting {n_candidates} candidate omegas...")
    
    dtw_distances = []
    
    for omega_candidate in omegas_test:
        # Generate PREDICTED peak pattern for this omega
        # Use SAME trajectory parameters (in real code, these come from Phase 1)
        pred_peaks = generate_peak_pattern_with_trajectory(
            alpha_true, U_true, omega_candidate, 
            x0_true, v0_true, a0_true,  # Same trajectory!
            t, source, receiver_line
        )
        
        # Compute DTW distance
        distance = dtw.distance(true_peaks, pred_peaks)
        dtw_distances.append(distance)
    
    dtw_distances = np.array(dtw_distances)
    
    # Find minimum
    min_idx = np.argmin(dtw_distances)
    estimated_omega = omegas_test[min_idx]
    min_dtw = dtw_distances[min_idx]
    
    error = abs(estimated_omega - true_omega)
    error_pct = 100 * error / true_omega
    
    print(f"\nRESULTS:")
    print(f"  True omega:      {true_omega:.4f} Hz")
    print(f"  Estimated omega: {estimated_omega:.4f} Hz")
    print(f"  Error:           {error:.4f} Hz ({error_pct:.2f}%)")
    print(f"  Min DTW:         {min_dtw:.6f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: DTW distance vs omega
    ax1.plot(omegas_test, dtw_distances, 'b-', linewidth=2, label='DTW Distance')
    ax1.axvline(true_omega, color='r', linestyle='--', linewidth=2, 
               label=f'True ω = {true_omega:.3f} Hz')
    ax1.plot(estimated_omega, min_dtw, 'go', markersize=12, 
            label=f'Estimated ω = {estimated_omega:.3f} Hz')
    
    ax1.set_xlabel('Omega (Hz)', fontsize=14)
    ax1.set_ylabel('DTW Distance', fontsize=14)
    ax1.set_title(f'DTW with Trajectory Motion (Error: {error_pct:.2f}%)', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Compare peak patterns
    t_np = t.cpu().numpy()
    
    # Get predicted pattern at estimated omega
    pred_peaks_best = generate_peak_pattern_with_trajectory(
        alpha_true, U_true, estimated_omega,
        x0_true, v0_true, a0_true,
        t, source, receiver_line
    )
    
    # Also show a wrong omega for comparison
    wrong_omega = true_omega + 0.5
    pred_peaks_wrong = generate_peak_pattern_with_trajectory(
        alpha_true, U_true, wrong_omega,
        x0_true, v0_true, a0_true,
        t, source, receiver_line
    )
    
    ax2.plot(t_np, true_peaks, 'ro-', linewidth=2, markersize=8, 
            label=f'True (ω={true_omega:.3f} Hz)', alpha=0.7)
    ax2.plot(t_np, pred_peaks_best, 'g^--', linewidth=2, markersize=6,
            label=f'Estimated (ω={estimated_omega:.3f} Hz)', alpha=0.7)
    ax2.plot(t_np, pred_peaks_wrong, 'bs:', linewidth=1.5, markersize=5,
            label=f'Wrong (ω={wrong_omega:.3f} Hz)', alpha=0.5)
    
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Peak Value', fontsize=14)
    ax2.set_title('Peak Value Patterns (with Trajectory Motion)', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dtw_omega_with_trajectory.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: dtw_omega_with_trajectory.png")
    
    plt.show()
    
    # Compare to stationary case
    print(f"\n{'='*70}")
    print("COMPARISON TO STATIONARY GAUSSIAN")
    print(f"{'='*70}")
    
    # Test with stationary Gaussian (x0 at origin, v0 = 0, a0 = 0)
    x0_stat = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    v0_stat = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    a0_stat = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    true_peaks_stat = generate_peak_pattern_with_trajectory(
        alpha_true, U_true, true_omega, x0_stat, v0_stat, a0_stat,
        t, source, receiver_line
    )
    
    dtw_distances_stat = []
    for omega_candidate in omegas_test:
        pred_peaks_stat = generate_peak_pattern_with_trajectory(
            alpha_true, U_true, omega_candidate, 
            x0_stat, v0_stat, a0_stat,
            t, source, receiver_line
        )
        distance = dtw.distance(true_peaks_stat, pred_peaks_stat)
        dtw_distances_stat.append(distance)
    
    dtw_distances_stat = np.array(dtw_distances_stat)
    min_idx_stat = np.argmin(dtw_distances_stat)
    estimated_omega_stat = omegas_test[min_idx_stat]
    error_stat = abs(estimated_omega_stat - true_omega)
    
    print(f"\nStationary Gaussian:")
    print(f"  Estimated omega: {estimated_omega_stat:.4f} Hz")
    print(f"  Error: {error_stat:.4f} Hz ({100*error_stat/true_omega:.2f}%)")
    
    print(f"\nMoving Gaussian:")
    print(f"  Estimated omega: {estimated_omega:.4f} Hz")
    print(f"  Error: {error:.4f} Hz ({error_pct:.2f}%)")
    
    # Check if DTW successfully identified omega
    print(f"\n{'='*70}")
    if error_pct < 5.0:
        print("✓ SUCCESS: DTW works with trajectory motion (< 5% error)")
    elif error_pct < 10.0:
        print("⚠ PARTIAL: DTW struggles slightly with motion (5-10% error)")
    else:
        print("✗ FAILED: DTW doesn't work with trajectory motion (> 10% error)")
    print(f"{'='*70}")


if __name__ == '__main__':
    test_dtw_with_trajectory()
