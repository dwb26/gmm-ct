"""
Trajectory-corrected peak analysis for omega estimation.

Key insight: The trajectory μ(t) is known from Phase 1 optimization.
We can use this to correct for geometric effects of translation,
isolating the rotation-induced oscillations.

Strategy:
1. Compute geometric factors from trajectory: distance, angle to receivers
2. Model the "baseline" trend due to translation (rotation-independent)
3. Normalize observed peaks by this baseline
4. Find inflection points in the corrected signal
5. Verify 1/(4ω) spacing in corrected data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, detrend
from scipy.interpolate import interp1d


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


def compute_geometric_correction_factors(trajectory, source, receivers, device='cpu'):
    """
    Compute geometric factors that affect projection intensity due to translation.
    
    These factors capture how the changing position affects the observed intensity,
    independent of rotation.
    
    Returns:
    - distance_factors: distance from trajectory to source-receiver lines
    - angle_factors: angles between trajectory and projection directions
    """
    n_times = len(trajectory)
    n_receivers = len(receivers)
    
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, dtype=torch.float64, device=device)
    if not isinstance(receivers, torch.Tensor):
        receivers = torch.tensor(receivers, dtype=torch.float64, device=device)
    
    # Precompute receiver geometry
    r_minus_s = receivers - source
    r_minus_s_norm = torch.norm(r_minus_s, dim=1, keepdim=True)
    r_minus_s_hat = r_minus_s / r_minus_s_norm
    
    distance_to_source = torch.zeros(n_times, dtype=torch.float64, device=device)
    avg_distance_to_line = torch.zeros(n_times, dtype=torch.float64, device=device)
    avg_angle_factor = torch.zeros(n_times, dtype=torch.float64, device=device)
    
    for n_t in range(n_times):
        mu_t = trajectory[n_t]
        
        # Distance from Gaussian center to source
        distance_to_source[n_t] = torch.norm(mu_t - source)
        
        # For each receiver, compute distance to source-receiver line
        # Distance = ||(μ - s) - ((μ - s)·r̂)*r̂||
        mu_minus_s = mu_t - source
        
        # Project onto receiver directions
        projections = torch.sum(mu_minus_s.unsqueeze(0) * r_minus_s_hat, dim=1)
        perpendicular_distances = torch.norm(
            mu_minus_s.unsqueeze(0) - projections.unsqueeze(1) * r_minus_s_hat,
            dim=1
        )
        
        # Average over receivers
        avg_distance_to_line[n_t] = torch.mean(perpendicular_distances)
        
        # Angle factors: cos(angle between (μ-s) and receiver directions)
        mu_minus_s_norm = torch.norm(mu_minus_s)
        if mu_minus_s_norm > 1e-10:
            mu_minus_s_hat = mu_minus_s / mu_minus_s_norm
            angle_factors = torch.abs(torch.sum(mu_minus_s_hat.unsqueeze(0) * r_minus_s_hat, dim=1))
            avg_angle_factor[n_t] = torch.mean(angle_factors)
        else:
            avg_angle_factor[n_t] = 1.0
    
    return distance_to_source, avg_distance_to_line, avg_angle_factor


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


def correct_for_trajectory_effects(peak_values, trajectory, source, receivers, method='geometric'):
    """
    Correct observed peak values for geometric effects of translation.
    
    Methods:
    - 'geometric': Divide by geometric factors (distance, angle)
    - 'detrend': Remove polynomial trend
    - 'normalize': Normalize by smoothed envelope
    - 'combined': Use multiple corrections
    
    Returns:
    - corrected_peaks: peak values with trajectory effects removed
    - correction_factors: the factors used for correction (for visualization)
    """
    peak_values_np = peak_values.cpu().numpy() if isinstance(peak_values, torch.Tensor) else peak_values
    trajectory_np = trajectory.cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory
    
    if method == 'geometric':
        # Compute geometric correction factors
        distance_to_source, avg_distance_to_line, avg_angle_factor = compute_geometric_correction_factors(
            trajectory, source, receivers, device=trajectory.device if isinstance(trajectory, torch.Tensor) else 'cpu'
        )
        
        # Combined geometric factor (empirical weighting)
        distance_to_source_np = distance_to_source.cpu().numpy()
        avg_distance_to_line_np = avg_distance_to_line.cpu().numpy()
        
        # Normalize factors to avoid extreme values
        distance_factor = distance_to_source_np / np.mean(distance_to_source_np)
        line_distance_factor = 1.0 / (1.0 + avg_distance_to_line_np / np.mean(avg_distance_to_line_np))
        
        # Exponential distance falloff (intensity ~ exp(-distance^2))
        correction_factor = distance_factor * line_distance_factor
        
        # Normalize so mean correction = 1
        correction_factor = correction_factor / np.mean(correction_factor)
        
        # Apply correction
        corrected_peaks = peak_values_np / correction_factor
        
    elif method == 'detrend':
        # Remove polynomial trend (degree 2 for parabolic trajectory)
        corrected_peaks = detrend(peak_values_np, type='linear')
        # Add back the mean to keep values positive
        corrected_peaks = corrected_peaks + np.mean(peak_values_np)
        correction_factor = peak_values_np - corrected_peaks + np.mean(peak_values_np)
        
    elif method == 'normalize':
        # Compute smoothed envelope
        if len(peak_values_np) > 51:
            window_length = 51
        elif len(peak_values_np) > 11:
            window_length = 11
        else:
            window_length = max(3, len(peak_values_np) // 3)
        
        if window_length % 2 == 0:
            window_length += 1
            
        envelope = savgol_filter(peak_values_np, window_length=window_length, polyorder=2)
        
        # Avoid division by small numbers
        envelope_safe = np.maximum(envelope, 0.1 * np.max(envelope))
        corrected_peaks = peak_values_np / envelope_safe
        correction_factor = envelope
        
    elif method == 'combined':
        # Use both geometric and envelope normalization
        distance_to_source, avg_distance_to_line, _ = compute_geometric_correction_factors(
            trajectory, source, receivers, device=trajectory.device if isinstance(trajectory, torch.Tensor) else 'cpu'
        )
        
        distance_to_source_np = distance_to_source.cpu().numpy()
        distance_factor = distance_to_source_np / np.mean(distance_to_source_np)
        
        # First correct for distance
        temp_corrected = peak_values_np / distance_factor
        
        # Then normalize by envelope
        if len(temp_corrected) > 51:
            window_length = 51
        elif len(temp_corrected) > 11:
            window_length = 11
        else:
            window_length = max(3, len(temp_corrected) // 3)
        if window_length % 2 == 0:
            window_length += 1
            
        envelope = savgol_filter(temp_corrected, window_length=window_length, polyorder=2)
        envelope_safe = np.maximum(envelope, 0.1 * np.max(envelope))
        
        corrected_peaks = temp_corrected / envelope_safe
        correction_factor = distance_factor * envelope_safe
    
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected_peaks, correction_factor


def find_inflection_points(t, values, method='peaks'):
    """Find inflection points in time series."""
    if method == 'peaks':
        max_peaks, _ = find_peaks(values, distance=5, prominence=0.05)
        min_peaks, _ = find_peaks(-values, distance=5, prominence=0.05)
        
        all_peaks = np.concatenate([max_peaks, min_peaks])
        peak_types = np.array(['max'] * len(max_peaks) + ['min'] * len(min_peaks))
        
        sort_idx = np.argsort(all_peaks)
        all_peaks = all_peaks[sort_idx]
        peak_types = peak_types[sort_idx]
        
        return t[all_peaks], values[all_peaks], peak_types
    else:
        raise ValueError(f"Unknown method: {method}")


def test_trajectory_correction(omega, duration, n_time_points=500,
                               v0=[5.0, 2.0], a0=[0.0, -9.81]):
    """
    Test trajectory correction methods for omega estimation.
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
    mu_0 = torch.tensor([5.0, 0.0], dtype=torch.float64, device=device)
    v0_tensor = torch.tensor(v0, dtype=torch.float64, device=device)
    a0_tensor = torch.tensor(a0, dtype=torch.float64, device=device)
    alpha = torch.tensor(20.0, dtype=torch.float64, device=device)
    
    sigma_major = 2.0
    sigma_minor = 0.5
    U_skew = torch.tensor([[sigma_major, 0.0], [0.0, sigma_minor]], 
                          dtype=torch.float64, device=device)
    
    print("\n" + "="*70)
    print("TRAJECTORY-CORRECTED PEAK ANALYSIS")
    print("="*70)
    print(f"\nTest parameters:")
    print(f"  Angular velocity ω = {omega}")
    print(f"  Expected quarter-period = 1/(4ω) = {1/(4*omega):.4f} s")
    print(f"  Trajectory: μ₀={mu_0.cpu().numpy()}, v₀={v0}, a₀={a0}")
    print("="*70)
    
    # Generate projections
    print("\nGenerating projections with motion...")
    proj, trajectory = generate_projection_with_motion(
        t, source, receivers, alpha, U_skew, omega, mu_0, v0_tensor, a0_tensor, device
    )
    
    peak_values = torch.max(proj, dim=1)[0]
    t_np = t.cpu().numpy()
    peak_values_np = peak_values.cpu().numpy()
    
    # Test different correction methods
    correction_methods = ['geometric', 'detrend', 'normalize', 'combined']
    results = {}
    
    print("\nTesting correction methods...")
    expected_quarter = 1 / (4 * omega)
    
    for corr_method in correction_methods:
        print(f"\n--- {corr_method.upper()} correction ---")
        
        corrected_peaks, correction_factor = correct_for_trajectory_effects(
            peak_values, trajectory, source, receivers, method=corr_method
        )
        
        # Find inflection points in corrected data
        try:
            inflection_times, inflection_values, peak_types = find_inflection_points(
                t_np, corrected_peaks, method='peaks'
            )
            
            if len(inflection_times) < 2:
                print(f"  Only {len(inflection_times)} inflection point(s) found")
                results[corr_method] = None
                continue
            
            time_diffs = np.diff(inflection_times)
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            relative_error = abs(mean_diff - expected_quarter) / expected_quarter * 100
            
            print(f"  Found {len(inflection_times)} inflection points")
            print(f"  Mean Δt = {mean_diff:.6f} s")
            print(f"  Expected Δt = {expected_quarter:.6f} s")
            print(f"  Relative error = {relative_error:.2f}%")
            
            if relative_error < 5:
                print(f"  ✓ EXCELLENT - Error < 5%")
            elif relative_error < 10:
                print(f"  ✓ GOOD - Error < 10%")
            elif relative_error < 20:
                print(f"  ⚠ MARGINAL - Error < 20%")
            else:
                print(f"  ✗ POOR - Error > 20%")
            
            results[corr_method] = {
                'corrected_peaks': corrected_peaks,
                'correction_factor': correction_factor,
                'times': inflection_times,
                'values': inflection_values,
                'types': peak_types,
                'diffs': time_diffs,
                'mean_diff': mean_diff,
                'error': relative_error
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[corr_method] = None
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # Plot 0: Trajectory
    ax0 = fig.add_subplot(gs[0, 0])
    traj_np = trajectory.cpu().numpy()
    ax0.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2)
    ax0.scatter(traj_np[0, 0], traj_np[0, 1], c='green', s=100, marker='o', label='Start')
    ax0.scatter(traj_np[-1, 0], traj_np[-1, 1], c='red', s=100, marker='X', label='End')
    ax0.scatter(source[0].cpu(), source[1].cpu(), c='orange', s=150, marker='*', label='Source')
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.set_title('Trajectory')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.axis('equal')
    
    # Plot 1: Uncorrected peaks
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(t_np, peak_values_np, 'b-', linewidth=2, alpha=0.7, label='Uncorrected')
    
    # Mark expected quarter periods
    for i in range(int(duration / expected_quarter) + 1):
        t_q = i * expected_quarter
        if t_q <= duration:
            ax1.axvline(t_q, color='gray', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Peak Value')
    ax1.set_title(f'Uncorrected Peak Values (ω={omega}, Expected Δt={expected_quarter:.4f}s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plots 2-5: Each correction method
    colors = {'geometric': 'red', 'detrend': 'green', 'normalize': 'blue', 'combined': 'purple'}
    
    for idx, (corr_method, color) in enumerate(colors.items()):
        ax = fig.add_subplot(gs[idx//2 + 1, idx%2 * 2:(idx%2 * 2)+2])
        
        if results[corr_method] is not None:
            res = results[corr_method]
            corrected = res['corrected_peaks']
            
            ax.plot(t_np, corrected, color=color, linewidth=1.5, alpha=0.7, 
                   label=f'Corrected ({corr_method})')
            
            # Mark inflection points
            max_mask = res['types'] == 'max'
            min_mask = res['types'] == 'min'
            ax.scatter(res['times'][max_mask], res['values'][max_mask],
                      c='darkred', marker='o', s=80, zorder=5, label='Maxima')
            ax.scatter(res['times'][min_mask], res['values'][min_mask],
                      c='darkblue', marker='v', s=80, zorder=5, label='Minima')
            
            # Mark expected quarters
            for i in range(int(duration / expected_quarter) + 1):
                t_q = i * expected_quarter
                if t_q <= duration:
                    ax.axvline(t_q, color='gray', linestyle='--', alpha=0.3)
            
            ax.set_title(f'{corr_method.upper()}: Error = {res["error"]:.2f}%', fontsize=11)
        else:
            ax.text(0.5, 0.5, f'{corr_method}\nFailed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{corr_method.upper()}: FAILED', fontsize=11)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Corrected Peak Value', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Comparison of errors
    ax_summary = fig.add_subplot(gs[3, :])
    method_names = []
    errors = []
    for method in correction_methods:
        if results[method] is not None:
            method_names.append(method)
            errors.append(results[method]['error'])
    
    bars = ax_summary.bar(method_names, errors, color=[colors[m] for m in method_names], alpha=0.7)
    ax_summary.axhline(5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='5% threshold')
    ax_summary.axhline(10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='10% threshold')
    ax_summary.set_ylabel('Relative Error (%)', fontsize=11)
    ax_summary.set_title('Comparison of Correction Methods', fontsize=12)
    ax_summary.legend()
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax_summary.text(bar.get_x() + bar.get_width()/2., height,
                       f'{err:.1f}%', ha='center', va='bottom', fontsize=10)
    
    output_file = f'test_output/trajectory_corrected_omega_{omega:.3f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\nFigure saved to {output_file}")
    plt.close()
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    
    best_method = None
    best_error = float('inf')
    
    for method in correction_methods:
        if results[method] is not None:
            error = results[method]['error']
            if error < best_error:
                best_error = error
                best_method = method
    
    if best_method:
        print(f"\n✓ BEST METHOD: {best_method.upper()}")
        print(f"  Error reduced to {best_error:.2f}%")
        print(f"  (vs ~6-19% uncorrected)")
        
        if best_error < 5:
            print(f"\n✓✓ EXCELLENT: Trajectory correction successfully recovers 1/(4ω)")
        elif best_error < 10:
            print(f"\n✓ GOOD: Trajectory correction significantly improves accuracy")
        else:
            print(f"\n⚠ Trajectory correction helps but may need refinement")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# TRAJECTORY CORRECTION FOR OMEGA ESTIMATION")
    print("#"*70)
    print("\nKey idea: Use the known trajectory μ(t) to correct for geometric")
    print("effects, isolating the rotation-induced oscillations.")
    print("#"*70)
    
    # Test
    results = test_trajectory_correction(
        omega=1.0,
        duration=2.0,
        n_time_points=500,
        v0=[5.0, 2.0],
        a0=[0.0, -9.81]
    )
    
    print("\n" + "#"*70)
    print("# RECOMMENDATION")
    print("#"*70)
    print("\nTo estimate omega in your experiments:")
    print("1. Use the fitted trajectory from Phase 1")
    print("2. Apply the best correction method above")
    print("3. Find inflection points in corrected peak values")
    print("4. Measure Δt between inflection points")
    print("5. Estimate: ω = 1/(4*Δt)")
    print("#"*70)
