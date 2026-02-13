"""
Using trajectory knowledge to correct inflection point timing.

Key insight: The inflection points occur when the anisotropic Gaussian is
aligned/perpendicular to the projection direction. In the fixed case, this
happens at precise 1/(4ω) intervals. With motion, the GEOMETRY changes,
affecting when we observe these orientations.

Strategy: Compute the "effective projection angle" as a function of time,
accounting for the changing geometry due to trajectory. Use this to transform
the time axis and recover the clean 1/(4ω) spacing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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


def compute_projection_angles(trajectory, source, receivers, device='cpu'):
    """
    Compute the effective projection angles as the Gaussian moves.
    
    For each time point, compute:
    1. The dominant projection direction (from source through Gaussian to receivers)
    2. How this direction changes over time
    
    Returns:
    - angles: effective projection angles at each time (radians)
    - angle_rates: rate of change of projection angle (rad/s)
    """
    n_times = len(trajectory)
    
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, dtype=torch.float64, device=device)
    if not isinstance(receivers, torch.Tensor):
        receivers = torch.tensor(receivers, dtype=torch.float64, device=device)
    
    angles = torch.zeros(n_times, dtype=torch.float64, device=device)
    
    # Compute the center of receiver array (dominant direction)
    receiver_center = torch.mean(receivers, dim=0)
    
    for n_t in range(n_times):
        mu_t = trajectory[n_t]
        
        # Direction from source to Gaussian
        direction = mu_t - source
        direction_norm = torch.norm(direction)
        
        if direction_norm > 1e-10:
            # Angle of this direction (relative to x-axis)
            angles[n_t] = torch.atan2(direction[1], direction[0])
        else:
            angles[n_t] = 0.0
    
    # Compute angular velocity (rate of change of projection angle)
    # This tells us how fast the geometry is changing
    angle_rates = torch.zeros(n_times, dtype=torch.float64, device=device)
    if n_times > 1:
        # Use central differences for interior points
        angle_rates[1:-1] = (angles[2:] - angles[:-2]) / 2.0
        angle_rates[0] = angles[1] - angles[0]
        angle_rates[-1] = angles[-1] - angles[-2]
    
    return angles, angle_rates


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


def compute_rotation_in_projection_frame(t, trajectory, source, omega, device='cpu'):
    """
    Compute the net rotation angle in the projection frame.
    
    In the fixed case: rotation_angle = 2π×ω×t
    In the moving case: need to account for how the projection direction changes
    
    Net angle = intrinsic rotation - geometric rotation due to motion
    """
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, dtype=torch.float64, device=device)
    
    n_times = len(t)
    
    # Intrinsic rotation (what the Gaussian is actually doing)
    intrinsic_rotation = 2 * np.pi * omega * t.cpu().numpy()
    
    # Geometric angle changes (how the projection direction changes)
    geometric_angles = torch.zeros(n_times, dtype=torch.float64, device=device)
    
    for n_t in range(n_times):
        mu_t = trajectory[n_t]
        direction = mu_t - source
        if torch.norm(direction) > 1e-10:
            geometric_angles[n_t] = torch.atan2(direction[1], direction[0])
    
    geometric_angles_np = geometric_angles.cpu().numpy()
    
    # The "effective" rotation in the projection frame
    # This is what we actually observe
    effective_rotation = intrinsic_rotation - geometric_angles_np
    
    return intrinsic_rotation, geometric_angles_np, effective_rotation


def correct_timing_using_geometry(t, peak_values, trajectory, source, receivers, omega, device='cpu'):
    """
    Correct the timing of inflection points based on geometric considerations.
    
    Approach: The geometry changes affect when we observe max/min. Compute
    a "corrected time" that accounts for these geometric effects.
    """
    # Compute projection angles
    proj_angles, angle_rates = compute_projection_angles(trajectory, source, receivers, device)
    
    proj_angles_np = proj_angles.cpu().numpy()
    angle_rates_np = angle_rates.cpu().numpy()
    t_np = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
    
    # Compute rotation angles
    intrinsic_rot, geometric_angles, effective_rot = compute_rotation_in_projection_frame(
        t, trajectory, source, omega, device
    )
    
    # Method 1: Correct based on effective rotation angle
    # Use the effective rotation to define a "corrected time"
    # where t_corrected corresponds to the effective rotation phase
    
    # Unwrap angles to avoid discontinuities
    effective_rot_unwrapped = np.unwrap(effective_rot)
    
    # The corrected time is when the effective rotation would occur at constant rate
    # Assuming the effective rotation should be linear: θ_eff = 2π×ω×t_corrected
    # So: t_corrected = θ_eff / (2π×ω)
    
    if omega > 0:
        t_corrected = effective_rot_unwrapped / (2 * np.pi * omega)
    else:
        t_corrected = t_np.copy()
    
    # Interpolate peak values to corrected time grid
    # Create interpolator
    interp = interp1d(t_np, peak_values, kind='cubic', fill_value='extrapolate')
    
    # Resample at uniform intervals in corrected time
    t_corrected_uniform = np.linspace(t_corrected[0], t_corrected[-1], len(t_np))
    peak_values_corrected = interp(t_corrected_uniform)
    
    return t_corrected_uniform, peak_values_corrected, {
        'proj_angles': proj_angles_np,
        'angle_rates': angle_rates_np,
        'intrinsic_rot': intrinsic_rot,
        'geometric_angles': geometric_angles,
        'effective_rot': effective_rot_unwrapped,
        't_original': t_np
    }


def find_inflection_points(t, values, method='peaks'):
    """Find inflection points in time series."""
    if method == 'peaks':
        max_peaks, _ = find_peaks(values, distance=5, prominence=0.01)
        min_peaks, _ = find_peaks(-values, distance=5, prominence=0.01)
        
        all_peaks = np.concatenate([max_peaks, min_peaks])
        peak_types = np.array(['max'] * len(max_peaks) + ['min'] * len(min_peaks))
        
        sort_idx = np.argsort(all_peaks)
        all_peaks = all_peaks[sort_idx]
        peak_types = peak_types[sort_idx]
        
        return t[all_peaks], values[all_peaks], peak_types
    else:
        raise ValueError(f"Unknown method: {method}")


def test_geometric_time_correction(omega, duration, n_time_points=500,
                                   v0=[5.0, 2.0], a0=[0.0, -9.81]):
    """
    Test geometric time correction for recovering 1/(4ω) spacing.
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
    print("GEOMETRIC TIME CORRECTION TEST")
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
    
    peak_values = torch.max(proj, dim=1)[0].cpu().numpy()
    t_np = t.cpu().numpy()
    
    # Apply geometric time correction
    print("\nApplying geometric time correction...")
    t_corrected, peaks_corrected, correction_info = correct_timing_using_geometry(
        t, peak_values, trajectory, source, receivers, omega, device
    )
    
    # Find inflection points in both
    print("\nFinding inflection points...")
    expected_quarter = 1 / (4 * omega)
    
    # Original timing
    print("\n--- ORIGINAL (uncorrected) ---")
    try:
        inflect_times_orig, inflect_vals_orig, types_orig = find_inflection_points(
            t_np, peak_values, method='peaks'
        )
        
        if len(inflect_times_orig) >= 2:
            diffs_orig = np.diff(inflect_times_orig)
            mean_orig = np.mean(diffs_orig)
            error_orig = abs(mean_orig - expected_quarter) / expected_quarter * 100
            
            print(f"  Found {len(inflect_times_orig)} inflection points")
            print(f"  Mean Δt = {mean_orig:.6f} s")
            print(f"  Expected = {expected_quarter:.6f} s")
            print(f"  Error = {error_orig:.2f}%")
        else:
            inflect_times_orig = None
            error_orig = None
    except:
        inflect_times_orig = None
        error_orig = None
        print("  Failed to find inflection points")
    
    # Corrected timing
    print("\n--- GEOMETRY-CORRECTED ---")
    try:
        inflect_times_corr, inflect_vals_corr, types_corr = find_inflection_points(
            t_corrected, peaks_corrected, method='peaks'
        )
        
        if len(inflect_times_corr) >= 2:
            diffs_corr = np.diff(inflect_times_corr)
            mean_corr = np.mean(diffs_corr)
            error_corr = abs(mean_corr - expected_quarter) / expected_quarter * 100
            
            print(f"  Found {len(inflect_times_corr)} inflection points")
            print(f"  Mean Δt = {mean_corr:.6f} s")
            print(f"  Expected = {expected_quarter:.6f} s")
            print(f"  Error = {error_corr:.2f}%")
            
            if error_corr < 1:
                print(f"  ✓✓ EXCELLENT - Error < 1%")
            elif error_corr < 5:
                print(f"  ✓ GOOD - Error < 5%")
            else:
                print(f"  ⚠ Error > 5%")
        else:
            inflect_times_corr = None
            error_corr = None
    except Exception as e:
        print(f"  Failed: {e}")
        inflect_times_corr = None
        error_corr = None
    
    # Visualization
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)
    
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
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.axis('equal')
    
    # Plot 1: Projection angles
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_np, np.degrees(correction_info['proj_angles']), 'b-', linewidth=2, label='Projection angle')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Projection Direction vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rotation angles breakdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t_np, np.degrees(correction_info['intrinsic_rot']), 'b-', linewidth=2, label='Intrinsic rotation (2πωt)')
    ax2.plot(t_np, np.degrees(correction_info['geometric_angles']), 'r--', linewidth=2, label='Geometric angle')
    ax2.plot(t_np, np.degrees(correction_info['effective_rot']), 'g-', linewidth=2, label='Effective rotation (observed)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Rotation Angle Decomposition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Original peaks
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(t_np, peak_values, 'b-', linewidth=2, alpha=0.7, label='Original peaks')
    
    if inflect_times_orig is not None:
        max_mask = types_orig == 'max'
        min_mask = types_orig == 'min'
        ax3.scatter(inflect_times_orig[max_mask], inflect_vals_orig[max_mask],
                   c='darkred', marker='o', s=100, zorder=5, label='Maxima')
        ax3.scatter(inflect_times_orig[min_mask], inflect_vals_orig[min_mask],
                   c='darkblue', marker='v', s=100, zorder=5, label='Minima')
    
    for i in range(int(duration / expected_quarter) + 1):
        t_q = i * expected_quarter
        if t_q <= duration:
            ax3.axvline(t_q, color='gray', linestyle='--', alpha=0.3)
    
    error_str = f"{error_orig:.2f}%" if error_orig is not None else "N/A"
    ax3.set_title(f'Original Peaks (Error = {error_str})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Peak Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Corrected peaks
    ax4 = fig.add_subplot(gs[3, :])
    ax4.plot(t_corrected, peaks_corrected, 'g-', linewidth=2, alpha=0.7, label='Corrected peaks')
    
    if inflect_times_corr is not None:
        max_mask = types_corr == 'max'
        min_mask = types_corr == 'min'
        ax4.scatter(inflect_times_corr[max_mask], inflect_vals_corr[max_mask],
                   c='darkred', marker='o', s=100, zorder=5, label='Maxima')
        ax4.scatter(inflect_times_corr[min_mask], inflect_vals_corr[min_mask],
                   c='darkblue', marker='v', s=100, zorder=5, label='Minima')
    
    # Mark expected quarters on corrected time
    t_corr_duration = t_corrected[-1] - t_corrected[0]
    for i in range(int(t_corr_duration / expected_quarter) + 1):
        t_q = t_corrected[0] + i * expected_quarter
        if t_q <= t_corrected[-1]:
            ax4.axvline(t_q, color='gray', linestyle='--', alpha=0.3)
    
    error_str = f"{error_corr:.2f}%" if error_corr is not None else "N/A"
    ax4.set_title(f'Geometry-Corrected Peaks (Error = {error_str})')
    ax4.set_xlabel('Corrected Time (s)')
    ax4.set_ylabel('Peak Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Time mapping
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.plot(t_np, t_corrected, 'b-', linewidth=2)
    ax5.plot([t_np[0], t_np[-1]], [t_np[0], t_np[-1]], 'r--', linewidth=1, alpha=0.5, label='Identity')
    ax5.set_xlabel('Original Time (s)')
    ax5.set_ylabel('Corrected Time (s)')
    ax5.set_title('Time Transformation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error comparison
    ax6 = fig.add_subplot(gs[4, 1])
    methods = []
    errors = []
    
    if error_orig is not None:
        methods.append('Original')
        errors.append(error_orig)
    if error_corr is not None:
        methods.append('Corrected')
        errors.append(error_corr)
    
    if len(methods) > 0:
        bars = ax6.bar(methods, errors, color=['blue', 'green'][:len(methods)], alpha=0.7)
        ax6.axhline(1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='1%')
        ax6.axhline(5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='5%')
        ax6.set_ylabel('Relative Error (%)')
        ax6.set_title('Error Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, err in zip(bars, errors):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{err:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    output_file = f'test_output/geometric_correction_omega_{omega:.3f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\nFigure saved to {output_file}")
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if error_orig is not None and error_corr is not None:
        improvement = error_orig - error_corr
        print(f"\nOriginal error: {error_orig:.2f}%")
        print(f"Corrected error: {error_corr:.2f}%")
        print(f"Improvement: {improvement:.2f}%")
        
        if error_corr < 1:
            print("\n✓✓ SUCCESS: Geometry correction recovered <1% accuracy!")
        elif error_corr < error_orig * 0.5:
            print("\n✓ GOOD: Significant improvement")
        elif error_corr < error_orig:
            print("\n⚠ Modest improvement")
        else:
            print("\n✗ Correction didn't help")
    
    print("="*70)


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# GEOMETRIC TIME CORRECTION")
    print("#"*70)
    print("\nConcept: As the Gaussian moves, the projection geometry changes.")
    print("This affects WHEN we observe the rotation-induced max/min.")
    print("\nStrategy: Compute 'effective rotation' accounting for geometric changes,")
    print("then transform to a corrected time where the spacing should be 1/(4ω).")
    print("#"*70)
    
    test_geometric_time_correction(
        omega=1.0,
        duration=2.0,
        n_time_points=500,
        v0=[5.0, 2.0],
        a0=[0.0, -9.81]
    )
