"""
Spectral analysis for omega estimation using Fourier transform.

Key Insight: Even though trajectory distorts the TIMING of inflection points,
the rotation still creates OSCILLATIONS at a characteristic frequency.

The trajectory effects show up as low-frequency trends.
The rotation effects show up as peaks at frequency ω (and harmonics).

By analyzing the power spectrum, we can:
1. Get a robust initialization for ω without multi-start
2. Handle multiple Gaussians by identifying multiple frequency peaks
3. Be less sensitive to trajectory effects than timing-based methods

This could dramatically reduce the dimensionality problem!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend, windows
from scipy.fft import fft, fftfreq


def construct_rotation_matrix(t, omega, device='cpu'):
    """Construct 2D rotation matrix: θ(t) = 2π×ω×t"""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64, device=device)
    if not isinstance(omega, torch.Tensor):
        omega = torch.tensor(omega, dtype=torch.float64, device=device)
    
    two_pi = 2 * torch.pi
    angle = two_pi * omega * t
    
    rot_mat = torch.eye(2, dtype=torch.float64, device=device)
    rot_mat[0, 0] = torch.cos(angle)
    rot_mat[0, 1] = -torch.sin(angle)
    rot_mat[1, 0] = torch.sin(angle)
    rot_mat[1, 1] = torch.cos(angle)
    
    return rot_mat


def compute_trajectory(t, mu_0, v0, a0, device='cpu'):
    """Compute projectile motion: μ(t) = μ₀ + v₀*t + 0.5*a₀*t²"""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64, device=device)
    if not isinstance(mu_0, torch.Tensor):
        mu_0 = torch.tensor(mu_0, dtype=torch.float64, device=device)
    if not isinstance(v0, torch.Tensor):
        v0 = torch.tensor(v0, dtype=torch.float64, device=device)
    if not isinstance(a0, torch.Tensor):
        a0 = torch.tensor(a0, dtype=torch.float64, device=device)
    
    if t.dim() == 0:
        return mu_0 + v0 * t + 0.5 * a0 * t**2
    else:
        t_reshaped = t.unsqueeze(1)
        return mu_0 + v0 * t_reshaped + 0.5 * a0 * t_reshaped**2


def generate_projection_with_motion(t, source, receivers, alpha, U_skew, omega, 
                                    mu_0, v0, a0, device='cpu'):
    """Generate projection data for rotating + translating Gaussian."""
    n_times = len(t)
    n_receivers = len(receivers)
    
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
    EPS = 1e-10
    
    proj = torch.zeros(n_times, n_receivers, dtype=torch.float64, device=device)
    trajectory = torch.zeros(n_times, 2, dtype=torch.float64, device=device)
    
    # Convert inputs
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, dtype=torch.float64, device=device)
    if not isinstance(receivers, torch.Tensor):
        receivers = torch.tensor(receivers, dtype=torch.float64, device=device)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float64, device=device)
    if not isinstance(U_skew, torch.Tensor):
        U_skew = torch.tensor(U_skew, dtype=torch.float64, device=device)
    
    r_minus_s = receivers - source
    r_minus_s_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True)
    
    for n_t, t_n in enumerate(t):
        R_t = construct_rotation_matrix(t_n, omega, device)
        U_t = U_skew @ R_t.mT
        mu_t = compute_trajectory(t_n, mu_0, v0, a0, device)
        trajectory[n_t] = mu_t
        
        U_r_hat = U_t @ r_minus_s_hat.T
        U_r = U_t @ r_minus_s.T
        U_diff = U_t @ (source - mu_t).unsqueeze(1)
        
        norm_term = torch.norm(U_r_hat, dim=0)
        quotient_term = sqrt_pi * alpha / (norm_term + EPS)
        
        inner_prod_sq = torch.sum(U_r * U_diff, dim=0) ** 2
        divisor = torch.norm(U_r, dim=0) ** 2 + EPS
        subtractor = torch.norm(U_diff, dim=0) ** 2
        
        exp_arg = inner_prod_sq / divisor - subtractor
        exp_term = torch.exp(exp_arg)
        
        proj[n_t] = quotient_term * exp_term
    
    return proj, trajectory


def estimate_omega_from_spectrum(t, peak_values, expected_harmonics=4, 
                                 detrend_method='linear', window='hann'):
    """
    Estimate omega from the power spectrum of peak values.
    
    The rotation causes oscillations at frequency ω (and harmonics 2ω, 3ω, 4ω).
    The anisotropic Gaussian aligned/perpendicular creates a pattern with 
    fundamental at 4ω (quarter-period), but we can look for all harmonics.
    
    Parameters:
    - t: time array
    - peak_values: peak values over time
    - expected_harmonics: number of harmonics to look for
    - detrend_method: 'linear', 'constant', or None
    - window: window function for FFT ('hann', 'hamming', 'blackman', or None)
    
    Returns:
    - omega_estimate: estimated omega value
    - spectrum_info: dictionary with spectrum details
    """
    # Detrend to remove low-frequency trajectory effects
    if detrend_method:
        signal = detrend(peak_values, type=detrend_method)
    else:
        signal = peak_values - np.mean(peak_values)
    
    # Apply window to reduce spectral leakage
    if window:
        if window == 'hann':
            win = windows.hann(len(signal))
        elif window == 'hamming':
            win = windows.hamming(len(signal))
        elif window == 'blackman':
            win = windows.blackman(len(signal))
        else:
            win = np.ones(len(signal))
        signal = signal * win
    
    # Compute FFT
    n = len(signal)
    dt = (t[-1] - t[0]) / (n - 1)
    
    # Use zero-padding for better frequency resolution
    n_fft = 2 ** int(np.ceil(np.log2(4 * n)))  # Zero-pad to next power of 2
    
    spectrum = fft(signal, n=n_fft)
    freqs = fftfreq(n_fft, dt)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power = np.abs(spectrum[pos_mask]) ** 2
    
    # Find peaks in power spectrum
    # CRITICAL: Anisotropic ellipse has 180° symmetry!
    # When it rotates by π (half rotation), it looks identical.
    # Therefore, the FUNDAMENTAL frequency in peak values is 2ω, not ω.
    # 
    # The dominant peak should be at 2ω.
    # We need to DIVIDE by 2 to get ω!
    
    # Look for significant peaks
    # Normalize power
    power_norm = power / np.max(power)
    
    # Find peaks with prominence
    peaks, properties = find_peaks(power_norm, prominence=0.1, distance=5)
    
    if len(peaks) == 0:
        # No clear peaks found
        return None, {'freqs': freqs_pos, 'power': power, 'peaks': [], 'message': 'No peaks found'}
    
    # Sort peaks by power
    peak_powers = power[peaks]
    sorted_idx = np.argsort(peak_powers)[::-1]
    peaks_sorted = peaks[sorted_idx]
    
    # The dominant peak should be at 2ω (or possibly 2ω)
    # Try to identify which harmonic this is
    
    dominant_freq = freqs_pos[peaks_sorted[0]]
    
    # UPDATED STRATEGY: 
    # The ellipse has 180° symmetry, so dominant frequency is 2ω.
    # But there could be additional harmonics at 4ω, 6ω, etc.
    # 
    # Test hypotheses:
    # H1: dominant_freq = 2ω  (most likely)
    # H2: dominant_freq = 4ω  (if we see quarter-period structure)
    # H3: dominant_freq = ω   (if asymmetry exists)
    
    candidate_omegas = []
    
    # The dominant harmonic for ellipse is 2ω
    for k in [2, 4, 6, 1, 3]:  # Prioritize even harmonics
        omega_candidate = dominant_freq / k
        
        # Check if we see expected harmonics at multiples of omega_candidate
        # For ellipse with 180° symmetry, expect peaks at 2ω, 4ω, 6ω, ...
        expected_freqs = [omega_candidate * m for m in [2, 4, 6, 8]]
        
        # Count how many expected harmonics we actually see
        matches = 0
        for exp_freq in expected_freqs:
            # Find if there's a peak near this frequency
            # Allow 5% tolerance
            tolerance = 0.05 * exp_freq
            nearby = np.abs(freqs_pos[peaks_sorted[:10]] - exp_freq) < tolerance
            if np.any(nearby):
                matches += 1
        
        candidate_omegas.append({
            'omega': omega_candidate,
            'harmonic': k,
            'matches': matches,
            'score': matches / expected_harmonics
        })
    
    # Select best candidate
    best_candidate = max(candidate_omegas, key=lambda x: x['score'])
    omega_estimate = best_candidate['omega']
    
    return omega_estimate, {
        'freqs': freqs_pos,
        'power': power,
        'power_norm': power_norm,
        'peaks': peaks_sorted[:10],  # Top 10 peaks
        'dominant_freq': dominant_freq,
        'candidates': candidate_omegas,
        'best_candidate': best_candidate,
        'dt': dt,
        'n_fft': n_fft
    }


def test_spectral_omega_estimation(true_omega, duration, n_time_points=500,
                                   v0=[5.0, 2.0], a0=[0.0, -9.81],
                                   test_with_motion=True):
    """
    Test spectral analysis for omega estimation.
    """
    device = 'cpu'
    
    # Time array
    t = torch.linspace(0, duration, n_time_points, dtype=torch.float64, device=device)
    
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
    if test_with_motion:
        mu_0 = torch.tensor([5.0, 0.0], dtype=torch.float64, device=device)
        v0_tensor = torch.tensor(v0, dtype=torch.float64, device=device)
        a0_tensor = torch.tensor(a0, dtype=torch.float64, device=device)
    else:
        # Fixed position
        mu_0 = torch.tensor([5.0, 0.0], dtype=torch.float64, device=device)
        v0_tensor = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
        a0_tensor = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    alpha = torch.tensor(20.0, dtype=torch.float64, device=device)
    
    sigma_major = 2.0
    sigma_minor = 0.5
    U_skew = torch.tensor([[sigma_major, 0.0], [0.0, sigma_minor]], 
                          dtype=torch.float64, device=device)
    
    motion_str = "WITH motion" if test_with_motion else "FIXED position"
    
    print("\n" + "="*70)
    print(f"SPECTRAL OMEGA ESTIMATION TEST - {motion_str}")
    print("="*70)
    print(f"\nTest parameters:")
    print(f"  TRUE angular velocity ω = {true_omega}")
    print(f"  Duration = {duration} s")
    print(f"  Number of complete rotations = {true_omega * duration}")
    if test_with_motion:
        print(f"  Trajectory: v₀={v0}, a₀={a0}")
    print("="*70)
    
    # Generate projections
    print(f"\nGenerating projections {motion_str.lower()}...")
    proj, trajectory = generate_projection_with_motion(
        t, source, receivers, alpha, U_skew, true_omega, 
        mu_0, v0_tensor, a0_tensor, device
    )
    
    peak_values = torch.max(proj, dim=1)[0].cpu().numpy()
    t_np = t.cpu().numpy()
    
    # Estimate omega from spectrum
    print("\nPerforming spectral analysis...")
    
    methods = [
        ('Detrend + Hann window', 'linear', 'hann'),
        ('Detrend only', 'linear', None),
        ('Raw signal', None, None),
    ]
    
    results = {}
    
    for method_name, detrend_method, window in methods:
        print(f"\n--- {method_name} ---")
        
        omega_est, spec_info = estimate_omega_from_spectrum(
            t_np, peak_values, expected_harmonics=4,
            detrend_method=detrend_method, window=window
        )
        
        if omega_est is not None:
            error = abs(omega_est - true_omega) / true_omega * 100
            
            print(f"  Estimated ω = {omega_est:.6f}")
            print(f"  True ω = {true_omega:.6f}")
            print(f"  Relative error = {error:.2f}%")
            print(f"  Dominant frequency = {spec_info['dominant_freq']:.6f} Hz")
            print(f"  Best interpretation: {spec_info['best_candidate']['harmonic']}×ω harmonic")
            print(f"  Harmonic matching score: {spec_info['best_candidate']['score']:.2f}")
            
            if error < 1:
                print(f"  ✓✓ EXCELLENT - Error < 1%")
            elif error < 5:
                print(f"  ✓ GOOD - Error < 5%")
            elif error < 10:
                print(f"  ⚠ MARGINAL - Error < 10%")
            else:
                print(f"  ✗ POOR - Error > 10%")
            
            results[method_name] = {
                'omega_est': omega_est,
                'error': error,
                'spec_info': spec_info
            }
        else:
            print(f"  ✗ Failed to estimate omega")
            results[method_name] = None
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    # Plot 0: Trajectory (if with motion)
    if test_with_motion:
        ax0 = fig.add_subplot(gs[0, 0])
        traj_np = trajectory.cpu().numpy()
        ax0.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2)
        ax0.scatter(traj_np[0, 0], traj_np[0, 1], c='green', s=100, marker='o', label='Start')
        ax0.scatter(traj_np[-1, 0], traj_np[-1, 1], c='red', s=100, marker='X', label='End')
        ax0.scatter(source[0].cpu(), source[1].cpu(), c='orange', s=150, marker='*', label='Source')
        ax0.set_xlabel('x (m)')
        ax0.set_ylabel('y (m)')
        ax0.set_title('Trajectory')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        ax0.axis('equal')
    
    # Plot 1: Peak values over time
    ax1_pos = gs[0, 1] if test_with_motion else gs[0, :]
    ax1 = fig.add_subplot(ax1_pos)
    ax1.plot(t_np, peak_values, 'b-', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Peak Value')
    ax1.set_title(f'Peak Values Over Time (ω={true_omega})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-4: Power spectra for each method
    row_offset = 1
    for idx, (method_name, detrend_method, window) in enumerate(methods):
        ax = fig.add_subplot(gs[row_offset + idx, :])
        
        if results[method_name] is not None:
            res = results[method_name]
            spec_info = res['spec_info']
            
            # Plot power spectrum
            freqs = spec_info['freqs']
            power_norm = spec_info['power_norm']
            
            ax.plot(freqs, power_norm, 'b-', linewidth=1, alpha=0.7)
            ax.set_xlim(0, min(10 * true_omega, freqs[-1]))
            
            # Mark identified peaks
            peaks = spec_info['peaks']
            ax.scatter(freqs[peaks], power_norm[peaks], 
                      c='red', s=100, zorder=5, label='Detected peaks')
            
            # Mark expected harmonics of true omega (2ω, 4ω, 6ω due to symmetry)
            for k in [2, 4, 6, 8]:
                expected_freq = k * true_omega
                if expected_freq < freqs[-1]:
                    label = f'{k}ω' if k == 2 else None
                    ax.axvline(expected_freq, color='green', linestyle='--', 
                              alpha=0.5, linewidth=2, label=label)
                    ax.text(expected_freq, ax.get_ylim()[1] * 0.9, f'{k}ω', 
                           ha='center', fontsize=9, color='green')
            
            # Mark estimated omega harmonics (2ω, 4ω, 6ω)
            omega_est = res['omega_est']
            for k in [2, 4, 6, 8]:
                est_freq = k * omega_est
                if est_freq < freqs[-1]:
                    ax.axvline(est_freq, color='red', linestyle=':', 
                              alpha=0.3, linewidth=1.5)
            
            error_str = f"{res['error']:.2f}%"
            ax.set_title(f'{method_name} | ω_est = {omega_est:.4f} | Error = {error_str}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Normalized Power')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{method_name}\nFailed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    motion_tag = "motion" if test_with_motion else "fixed"
    output_file = f'test_output/spectral_omega_est_{motion_tag}_omega_{true_omega:.3f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\nFigure saved to {output_file}")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best_method = None
    best_error = float('inf')
    
    for method_name in results.keys():
        if results[method_name] is not None:
            error = results[method_name]['error']
            if error < best_error:
                best_error = error
                best_method = method_name
    
    if best_method:
        print(f"\n✓ BEST METHOD: {best_method}")
        print(f"  Error: {best_error:.2f}%")
        
        if best_error < 5:
            print(f"\n✓✓ SUCCESS: Spectral method provides excellent omega estimate!")
            print(f"   This can be used as initialization for further optimization.")
        elif best_error < 10:
            print(f"\n✓ GOOD: Spectral method provides useful initialization")
        else:
            print(f"\n⚠ Spectral method needs refinement")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# SPECTRAL ANALYSIS FOR OMEGA ESTIMATION")
    print("#"*70)
    print("\nKey Idea: Use FFT to find the oscillation frequency directly!")
    print("\nAdvantages:")
    print("  1. Robust to trajectory effects (they're low-frequency)")
    print("  2. No multi-start needed - direct estimate")
    print("  3. Can handle multiple Gaussians (multiple frequency peaks)")
    print("  4. Provides good initialization for refinement")
    print("#"*70)
    
    # Test 1: Fixed position (baseline)
    print("\n\n" + "="*70)
    print("TEST 1: FIXED POSITION (Baseline)")
    print("="*70)
    results_fixed = test_spectral_omega_estimation(
        true_omega=1.0,
        duration=2.0,
        n_time_points=500,
        test_with_motion=False
    )
    
    input("\nPress Enter to continue to test with motion...")
    
    # Test 2: With projectile motion
    print("\n\n" + "="*70)
    print("TEST 2: WITH PROJECTILE MOTION")
    print("="*70)
    results_motion = test_spectral_omega_estimation(
        true_omega=1.0,
        duration=2.0,
        n_time_points=500,
        v0=[5.0, 2.0],
        a0=[0.0, -9.81],
        test_with_motion=True
    )
    
    print("\n" + "#"*70)
    print("# RECOMMENDATION FOR YOUR PIPELINE")
    print("#"*70)
    print("\nAfter Phase 1 (trajectory fitting):")
    print("\n1. Extract peak values over time for each Gaussian")
    print("2. Apply FFT with detrending and windowing")
    print("3. Identify dominant frequency peaks")
    print("4. Use these as INITIALIZATIONS for omega (no multi-start needed!)")
    print("5. Refine with your forward model if needed")
    print("\nThis should dramatically reduce computational cost!")
    print("#"*70)
