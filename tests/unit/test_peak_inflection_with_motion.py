"""
Test peak inflection timing with PROJECTILE MOTION included.

This extends the previous test by adding trajectory motion:
    mu(t) = mu_0 + v0*t + 0.5*a0*t^2

This will reveal whether the combination of rotation + translation
affects the observed 1/(4*omega) inflection point spacing.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


def construct_rotation_matrix(t, omega, device='cpu'):
    """Construct a 2D rotation matrix: θ(t) = 2π×ω×t"""
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
    """
    Compute projectile motion trajectory.
    
    mu(t) = mu_0 + v0*t + 0.5*a0*t^2
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64, device=device)
    if not isinstance(mu_0, torch.Tensor):
        mu_0 = torch.tensor(mu_0, dtype=torch.float64, device=device)
    if not isinstance(v0, torch.Tensor):
        v0 = torch.tensor(v0, dtype=torch.float64, device=device)
    if not isinstance(a0, torch.Tensor):
        a0 = torch.tensor(a0, dtype=torch.float64, device=device)
    
    # Handle both scalar and array time
    if t.dim() == 0:
        return mu_0 + v0 * t + 0.5 * a0 * t**2
    else:
        t_reshaped = t.unsqueeze(1)  # [N] -> [N, 1]
        return mu_0 + v0 * t_reshaped + 0.5 * a0 * t_reshaped**2


def generate_projection_with_motion(t, source, receivers, alpha, U_skew, omega, 
                                    mu_0, v0, a0, device='cpu'):
    """
    Generate projection data for a rotating AND translating anisotropic Gaussian.
    
    Combines:
    - Rotation: U(t) = U_0 @ R(t)^T where θ(t) = 2π×ω×t
    - Translation: μ(t) = μ_0 + v0*t + 0.5*a0*t^2
    
    Parameters:
    - t: time array
    - source: source position (2D)
    - receivers: array of receiver positions (N_receivers x 2)
    - alpha: attenuation coefficient
    - U_skew: initial upper triangular skewness matrix (2x2)
    - omega: angular velocity parameter
    - mu_0: initial center position
    - v0: initial velocity (2D)
    - a0: acceleration (2D)
    - device: torch device
    
    Returns:
    - proj: projection values (len(t) x N_receivers)
    - trajectory: center positions over time (len(t) x 2)
    """
    n_times = len(t)
    n_receivers = len(receivers)
    
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
    EPS = 1e-10
    
    proj = torch.zeros(n_times, n_receivers, dtype=torch.float64, device=device)
    trajectory = torch.zeros(n_times, 2, dtype=torch.float64, device=device)
    
    # Convert inputs to tensors
    if not isinstance(source, torch.Tensor):
        source = torch.tensor(source, dtype=torch.float64, device=device)
    if not isinstance(receivers, torch.Tensor):
        receivers = torch.tensor(receivers, dtype=torch.float64, device=device)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float64, device=device)
    if not isinstance(U_skew, torch.Tensor):
        U_skew = torch.tensor(U_skew, dtype=torch.float64, device=device)
    
    # Precompute receiver geometry terms (constant over time)
    r_minus_s = receivers - source
    r_minus_s_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True)
    
    for n_t, t_n in enumerate(t):
        # Get rotation matrix at time t
        R_t = construct_rotation_matrix(t_n, omega, device)
        
        # Rotate the skewness matrix: U(t) = U_0 @ R(t)^T
        U_t = U_skew @ R_t.mT
        
        # Compute trajectory position at time t
        mu_t = compute_trajectory(t_n, mu_0, v0, a0, device)
        trajectory[n_t] = mu_t
        
        # Compute projection terms (vectorized over receivers)
        U_r_hat = U_t @ r_minus_s_hat.T
        U_r = U_t @ r_minus_s.T
        U_diff = U_t @ (source - mu_t).unsqueeze(1)  # source - mu(t)
        
        norm_term = torch.norm(U_r_hat, dim=0)
        quotient_term = sqrt_pi * alpha / (norm_term + EPS)
        
        inner_prod_sq = torch.sum(U_r * U_diff, dim=0) ** 2
        divisor = torch.norm(U_r, dim=0) ** 2 + EPS
        subtractor = torch.norm(U_diff, dim=0) ** 2
        
        exp_arg = inner_prod_sq / divisor - subtractor
        exp_term = torch.exp(exp_arg)
        
        proj[n_t] = quotient_term * exp_term
    
    return proj, trajectory


def find_inflection_points(t, values, method='peaks'):
    """Find inflection points (local min and max) in the time series."""
    if method == 'peaks':
        # Find local maxima
        max_peaks, _ = find_peaks(values, distance=5, prominence=0.1)
        # Find local minima (by inverting)
        min_peaks, _ = find_peaks(-values, distance=5, prominence=0.1)
        
        # Combine and sort
        all_peaks = np.concatenate([max_peaks, min_peaks])
        peak_types = np.array(['max'] * len(max_peaks) + ['min'] * len(min_peaks))
        
        sort_idx = np.argsort(all_peaks)
        all_peaks = all_peaks[sort_idx]
        peak_types = peak_types[sort_idx]
        
        inflection_times = t[all_peaks]
        inflection_values = values[all_peaks]
        
    elif method == 'derivative':
        # Smooth the data first
        if len(values) > 11:
            values_smooth = savgol_filter(values, window_length=11, polyorder=3)
        else:
            values_smooth = values
        
        # Compute first derivative
        dt = np.diff(t)
        dv_dt = np.diff(values_smooth) / dt
        
        # Find where derivative changes sign
        sign_changes = np.where(np.diff(np.sign(dv_dt)))[0]
        
        inflection_times = []
        inflection_values = []
        peak_types = []
        
        for idx in sign_changes:
            if idx > 0 and idx < len(dv_dt) - 1:
                if dv_dt[idx-1] > 0 and dv_dt[idx+1] < 0:
                    inflection_times.append(t[idx+1])
                    inflection_values.append(values[idx+1])
                    peak_types.append('max')
                elif dv_dt[idx-1] < 0 and dv_dt[idx+1] > 0:
                    inflection_times.append(t[idx+1])
                    inflection_values.append(values[idx+1])
                    peak_types.append('min')
        
        inflection_times = np.array(inflection_times)
        inflection_values = np.array(inflection_values)
        peak_types = np.array(peak_types)
    
    return inflection_times, inflection_values, peak_types


def test_with_projectile_motion(omega, duration, n_time_points=500,
                                v0=[5.0, 2.0], a0=[0.0, -9.81]):
    """
    Test whether peak inflection timing holds with projectile motion included.
    
    Parameters:
    - omega: angular velocity parameter
    - duration: total time duration
    - n_time_points: number of time samples
    - v0: initial velocity [vx, vy]
    - a0: acceleration [ax, ay] (default includes gravity)
    """
    device = 'cpu'
    
    # Time array
    t = torch.linspace(0, duration, n_time_points, dtype=torch.float64, device=device)
    
    # Setup geometry
    source = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    # Receivers
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
    
    # Attenuation
    alpha = torch.tensor(20.0, dtype=torch.float64, device=device)
    
    # Anisotropic skewness matrix
    sigma_major = 2.0
    sigma_minor = 0.5
    U_skew = torch.tensor([
        [sigma_major, 0.0],
        [0.0, sigma_minor]
    ], dtype=torch.float64, device=device)
    
    print("\n" + "="*70)
    print("PEAK INFLECTION TIMING TEST WITH PROJECTILE MOTION")
    print("="*70)
    print(f"\nTest parameters:")
    print(f"  Angular velocity ω = {omega}")
    print(f"  Duration = {duration} s")
    print(f"  Expected quarter-period = 1/(4ω) = {1/(4*omega):.4f} s")
    print(f"  Initial position μ₀ = {mu_0.cpu().numpy()}")
    print(f"  Initial velocity v₀ = {v0}")
    print(f"  Acceleration a₀ = {a0}")
    print(f"  Anisotropy ratio = {sigma_major/sigma_minor:.2f}")
    print(f"  Number of time points = {n_time_points}")
    print("="*70)
    
    # Generate projections WITH motion
    print("\nGenerating projections with projectile motion...")
    proj, trajectory = generate_projection_with_motion(
        t, source, receivers, alpha, U_skew, omega, mu_0, v0_tensor, a0_tensor, device
    )
    
    # Extract peak value at each time
    peak_values = torch.max(proj, dim=1)[0]
    peak_values_np = peak_values.cpu().numpy()
    t_np = t.cpu().numpy()
    trajectory_np = trajectory.cpu().numpy()
    
    # Find inflection points
    print("\nFinding inflection points...")
    
    methods = ['peaks', 'derivative']
    results = {}
    
    for method in methods:
        print(f"\nMethod: {method}")
        inflection_times, inflection_values, peak_types = find_inflection_points(
            t_np, peak_values_np, method=method
        )
        
        if len(inflection_times) < 2:
            print(f"  WARNING: Only {len(inflection_times)} inflection point(s) found")
            results[method] = None
            continue
        
        # Compute time differences
        time_diffs = np.diff(inflection_times)
        expected_quarter_period = 1 / (4 * omega)
        
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        relative_error = abs(mean_diff - expected_quarter_period) / expected_quarter_period * 100
        
        print(f"  Found {len(inflection_times)} inflection points")
        print(f"  Time differences (Δt):")
        for i, dt in enumerate(time_diffs[:10]):  # Show first 10
            print(f"    {i+1}: {dt:.6f} s (error: {abs(dt - expected_quarter_period)/expected_quarter_period*100:.2f}%)")
        if len(time_diffs) > 10:
            print(f"    ... ({len(time_diffs) - 10} more)")
        print(f"  Mean Δt = {mean_diff:.6f} s (std: {std_diff:.6f})")
        print(f"  Expected Δt = 1/(4ω) = {expected_quarter_period:.6f} s")
        print(f"  Relative error = {relative_error:.2f}%")
        
        results[method] = {
            'times': inflection_times,
            'values': inflection_values,
            'types': peak_types,
            'diffs': time_diffs,
            'mean_diff': mean_diff,
            'expected': expected_quarter_period,
            'error': relative_error
        }
    
    # Visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Plot 1: Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(trajectory_np[0, 0], trajectory_np[0, 1], c='green', s=100, 
               marker='o', zorder=5, label='Start')
    ax1.scatter(trajectory_np[-1, 0], trajectory_np[-1, 1], c='red', s=100, 
               marker='X', zorder=5, label='End')
    ax1.scatter(source[0].cpu(), source[1].cpu(), c='orange', s=150, 
               marker='*', zorder=5, label='Source')
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_title('Gaussian Trajectory (Projectile Motion)', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Position components vs time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_np, trajectory_np[:, 0], 'b-', linewidth=2, label='x(t)')
    ax2.plot(t_np, trajectory_np[:, 1], 'r-', linewidth=2, label='y(t)')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position (m)', fontsize=11)
    ax2.set_title('Position Components vs Time', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Peak values vs time with inflection points
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_np, peak_values_np, 'b-', linewidth=1.5, alpha=0.7, label='Peak value')
    
    colors_method = {'peaks': 'red', 'derivative': 'green'}
    markers_method = {'peaks': 'o', 'derivative': 's'}
    
    for method in methods:
        if results[method] is not None:
            res = results[method]
            max_mask = res['types'] == 'max'
            min_mask = res['types'] == 'min'
            
            ax3.scatter(res['times'][max_mask], res['values'][max_mask], 
                       c=colors_method[method], marker=markers_method[method], 
                       s=100, zorder=5, label=f'{method}: maxima')
            ax3.scatter(res['times'][min_mask], res['values'][min_mask], 
                       c=colors_method[method], marker=markers_method[method], 
                       s=100, alpha=0.5, zorder=5, label=f'{method}: minima')
    
    # Mark quarter-period intervals
    expected_quarter = 1 / (4 * omega)
    for i in range(int(duration / expected_quarter) + 1):
        t_quarter = i * expected_quarter
        if t_quarter <= duration:
            ax3.axvline(t_quarter, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Peak Value', fontsize=12)
    ax3.set_title(f'Peak Values vs Time WITH Motion (ω = {omega}, Expected Δt = {expected_quarter:.4f} s)', 
                 fontsize=13)
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time differences histogram
    ax4 = fig.add_subplot(gs[2, 0])
    for method in methods:
        if results[method] is not None:
            ax4.hist(results[method]['diffs'], bins=15, alpha=0.5, 
                    label=f"{method} (mean: {results[method]['mean_diff']:.4f})",
                    color=colors_method[method])
    ax4.axvline(expected_quarter_period, color='k', linestyle='--', linewidth=2, 
               label=f'Expected: {expected_quarter_period:.4f}')
    ax4.set_xlabel('Time Difference Δt (s)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Distribution of Inflection Point Spacings', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Error analysis
    ax5 = fig.add_subplot(gs[2, 1])
    for method in methods:
        if results[method] is not None:
            errors = (results[method]['diffs'] - expected_quarter_period) / expected_quarter_period * 100
            ax5.plot(range(len(errors)), errors, 
                    marker=markers_method[method], linestyle='-',
                    color=colors_method[method], label=method, linewidth=2, markersize=8)
    ax5.axhline(0, color='k', linestyle='--', linewidth=1)
    ax5.axhline(10, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='±10% threshold')
    ax5.axhline(-10, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax5.set_xlabel('Interval Index', fontsize=11)
    ax5.set_ylabel('Relative Error (%)', fontsize=11)
    ax5.set_title('Error in Each Time Interval', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Full projection heatmap
    ax6 = fig.add_subplot(gs[3, :])
    proj_np = proj.cpu().numpy()
    im = ax6.imshow(proj_np.T, aspect='auto', origin='lower', 
                   extent=[t_np[0], t_np[-1], 0, n_receivers],
                   cmap='hot', interpolation='bilinear')
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Receiver Index', fontsize=11)
    ax6.set_title('Full Projection Data (Heatmap) - WITH Motion', fontsize=12)
    plt.colorbar(im, ax=ax6, label='Intensity')
    
    # Mark quarter periods
    for i in range(int(duration / expected_quarter) + 1):
        t_quarter = i * expected_quarter
        if t_quarter <= duration:
            ax6.axvline(t_quarter, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    
    output_file = f'test_output/peak_inflection_WITH_motion_omega_{omega:.3f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_file}")
    plt.close()
    
    # Final verdict
    print("\n" + "="*70)
    print("TEST VERDICT:")
    print("="*70)
    
    best_method = None
    best_error = float('inf')
    
    for method in methods:
        if results[method] is not None:
            error = results[method]['error']
            print(f"\n{method.upper()} method:")
            print(f"  Mean error: {error:.2f}%")
            
            if error < best_error:
                best_error = error
                best_method = method
            
            if error < 5:
                print(f"  ✓ PASS - Error < 5%")
            elif error < 10:
                print(f"  ⚠ MARGINAL - Error 5-10%")
            else:
                print(f"  ✗ FAIL - Error > 10%")
    
    if best_method and best_error < 10:
        print(f"\n✓ OVERALL: Peak inflection timing STILL HOLDS with projectile motion")
        print(f"  The 1/(4ω) relationship verified within {best_error:.2f}% error")
    else:
        print(f"\n✗ OVERALL: Projectile motion BREAKS the 1/(4ω) relationship")
        print(f"  Best method achieved {best_error:.2f}% error")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# TESTING PEAK INFLECTION TIMING WITH PROJECTILE MOTION")
    print("#"*70)
    print("\nThis test adds translation to see if it affects the 1/(4ω) relationship.")
    print("The Gaussian now follows: μ(t) = μ₀ + v₀*t + 0.5*a₀*t²")
    print("While also rotating: θ(t) = 2π×ω×t")
    print("#"*70)
    
    # Test configurations
    test_configs = [
        {
            'omega': 1.0, 
            'duration': 2.0, 
            'n_time_points': 500,
            'v0': [5.0, 2.0],
            'a0': [0.0, -9.81]
        },
        {
            'omega': 0.5, 
            'duration': 4.0, 
            'n_time_points': 800,
            'v0': [3.0, 3.0],
            'a0': [0.0, -9.81]
        },
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n{'#'*70}")
        print(f"# TEST {i+1}/{len(test_configs)}")
        print(f"{'#'*70}")
        
        results = test_with_projectile_motion(**config)
        
        if i < len(test_configs) - 1:
            print("\n" + "."*70)
            input("Press Enter to continue to next test...")
    
    print("\n" + "#"*70)
    print("# ALL TESTS COMPLETE")
    print("#"*70)
    print("\nCheck the generated figures in test_output/")
    print("Compare WITH motion vs WITHOUT motion to see the effect.")
