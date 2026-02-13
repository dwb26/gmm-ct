"""
Decouple trajectory effects using isotropic baseline normalization.

Key Insight:
- Anisotropic rotating Gaussian: I_rot(t) = I_baseline(t) × R(ω,t)
  where I_baseline captures trajectory effects, R captures rotation effects

- Isotropic (circular) Gaussian: I_iso(t) = I_baseline(t) × const
  (no rotation effects, but same trajectory effects)

- Therefore: I_rot(t) / I_iso(t) ≈ R(ω,t) / const
  This ratio isolates the rotation oscillations!

By normalizing the anisotropic projections by isotropic reference projections
(same trajectory, same size, but circular), we remove trajectory effects.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
                                    mu_0, v0, a0, device='cpu', rotate=True):
    """
    Generate projection data for translating Gaussian.
    
    Parameters:
    - rotate: If True, apply rotation. If False, keep orientation fixed (for baseline)
    """
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
        if rotate:
            # Apply rotation
            R_t = construct_rotation_matrix(t_n, omega, device)
            U_t = U_skew @ R_t.mT
        else:
            # No rotation - keep fixed orientation
            U_t = U_skew
        
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


def test_isotropic_baseline_normalization(omega, duration, n_time_points=500,
                                         v0=[5.0, 2.0], a0=[0.0, -9.81]):
    """
    Test decoupling trajectory effects using isotropic baseline normalization.
    
    This creates a reference projection from a circular (isotropic) Gaussian
    following the same trajectory, which has no rotation effects but all the
    same geometric trajectory effects. Normalizing by this should recover
    the clean rotation signal.
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
    
    # Anisotropic skewness matrix
    sigma_major = 2.0
    sigma_minor = 0.5
    U_anisotropic = torch.tensor([[sigma_major, 0.0], [0.0, sigma_minor]], 
                                 dtype=torch.float64, device=device)
    
    # Isotropic skewness matrix (circular - average of major and minor)
    sigma_avg = (sigma_major + sigma_minor) / 2
    U_isotropic = torch.tensor([[sigma_avg, 0.0], [0.0, sigma_avg]], 
                               dtype=torch.float64, device=device)
    
    print("\n" + "="*70)
    print("ISOTROPIC BASELINE NORMALIZATION TEST")
    print("="*70)
    print(f"\nTest parameters:")
    print(f"  Angular velocity ω = {omega}")
    print(f"  Expected quarter-period = 1/(4ω) = {1/(4*omega):.4f} s")
    print(f"  Trajectory: μ₀={mu_0.cpu().numpy()}, v₀={v0}, a₀={a0}")
    print(f"  Anisotropic: σ_major={sigma_major}, σ_minor={sigma_minor}")
    print(f"  Isotropic baseline: σ={sigma_avg}")
    print("="*70)
    
    # Generate projections for ANISOTROPIC + ROTATING Gaussian
    print("\nGenerating anisotropic + rotating projections...")
    proj_aniso_rot, trajectory = generate_projection_with_motion(
        t, source, receivers, alpha, U_anisotropic, omega, 
        mu_0, v0_tensor, a0_tensor, device, rotate=True
    )
    
    # Generate projections for ISOTROPIC (circular) Gaussian - NO ROTATION EFFECTS
    # This captures ONLY trajectory effects
    print("Generating isotropic baseline (trajectory effects only)...")
    proj_iso_baseline, _ = generate_projection_with_motion(
        t, source, receivers, alpha, U_isotropic, omega=0.0,  # omega doesn't matter for isotropic
        mu_0=mu_0, v0=v0_tensor, a0=a0_tensor, device=device, rotate=False
    )
    
    # Also generate ANISOTROPIC but NON-ROTATING for comparison
    print("Generating anisotropic non-rotating (alternative baseline)...")
    proj_aniso_norot, _ = generate_projection_with_motion(
        t, source, receivers, alpha, U_anisotropic, omega=0.0,
        mu_0=mu_0, v0=v0_tensor, a0=a0_tensor, device=device, rotate=False
    )
    
    # Extract peak values
    peaks_aniso_rot = torch.max(proj_aniso_rot, dim=1)[0].cpu().numpy()
    peaks_iso_baseline = torch.max(proj_iso_baseline, dim=1)[0].cpu().numpy()
    peaks_aniso_norot = torch.max(proj_aniso_norot, dim=1)[0].cpu().numpy()
    
    # NORMALIZATION: Divide rotating by baseline
    print("\nApplying normalization...")
    
    # Method 1: Normalize by isotropic baseline
    EPS = 1e-10
    normalized_by_iso = peaks_aniso_rot / (peaks_iso_baseline + EPS)
    
    # Method 2: Normalize by anisotropic non-rotating
    normalized_by_aniso_norot = peaks_aniso_rot / (peaks_aniso_norot + EPS)
    
    t_np = t.cpu().numpy()
    
    # Find inflection points in different signals
    print("\nFinding inflection points...")
    
    expected_quarter = 1 / (4 * omega)
    
    signals = {
        'Uncorrected (rotating)': peaks_aniso_rot,
        'Normalized by isotropic': normalized_by_iso,
        'Normalized by aniso-norot': normalized_by_aniso_norot,
    }
    
    results = {}
    
    for signal_name, signal_values in signals.items():
        print(f"\n--- {signal_name} ---")
        
        try:
            inflection_times, inflection_values, peak_types = find_inflection_points(
                t_np, signal_values, method='peaks'
            )
            
            if len(inflection_times) < 2:
                print(f"  Only {len(inflection_times)} inflection point(s) found")
                results[signal_name] = None
                continue
            
            time_diffs = np.diff(inflection_times)
            mean_diff = np.mean(time_diffs)
            std_diff = np.std(time_diffs)
            relative_error = abs(mean_diff - expected_quarter) / expected_quarter * 100
            
            print(f"  Found {len(inflection_times)} inflection points")
            print(f"  Mean Δt = {mean_diff:.6f} s")
            print(f"  Expected Δt = {expected_quarter:.6f} s")
            print(f"  Relative error = {relative_error:.2f}%")
            
            if relative_error < 1:
                print(f"  ✓✓ EXCELLENT - Error < 1%")
            elif relative_error < 5:
                print(f"  ✓ GOOD - Error < 5%")
            elif relative_error < 10:
                print(f"  ⚠ MARGINAL - Error < 10%")
            else:
                print(f"  ✗ POOR - Error > 10%")
            
            results[signal_name] = {
                'values': signal_values,
                'times': inflection_times,
                'inflection_values': inflection_values,
                'types': peak_types,
                'diffs': time_diffs,
                'mean_diff': mean_diff,
                'error': relative_error
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[signal_name] = None
    
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
    ax0.set_xlabel('x (m)', fontsize=11)
    ax0.set_ylabel('y (m)', fontsize=11)
    ax0.set_title('Gaussian Trajectory', fontsize=12)
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.axis('equal')
    
    # Plot 1: All baseline signals
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_np, peaks_aniso_rot, 'b-', linewidth=2, alpha=0.7, label='Aniso + Rotating')
    ax1.plot(t_np, peaks_iso_baseline, 'g--', linewidth=2, alpha=0.7, label='Iso baseline')
    ax1.plot(t_np, peaks_aniso_norot, 'r:', linewidth=2, alpha=0.7, label='Aniso no-rot')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Peak Value', fontsize=11)
    ax1.set_title('Raw Signals', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plots 2-4: Each normalized signal with inflection points
    plot_configs = [
        ('Uncorrected (rotating)', 'blue', gs[1, :]),
        ('Normalized by isotropic', 'green', gs[2, :]),
        ('Normalized by aniso-norot', 'red', gs[3, :]),
    ]
    
    for signal_name, color, grid_pos in plot_configs:
        ax = fig.add_subplot(grid_pos)
        
        if results[signal_name] is not None:
            res = results[signal_name]
            
            ax.plot(t_np, res['values'], color=color, linewidth=2, alpha=0.7, label=signal_name)
            
            # Mark inflection points
            max_mask = res['types'] == 'max'
            min_mask = res['types'] == 'min'
            ax.scatter(res['times'][max_mask], res['inflection_values'][max_mask],
                      c='darkred', marker='o', s=100, zorder=5, label='Maxima')
            ax.scatter(res['times'][min_mask], res['inflection_values'][min_mask],
                      c='darkblue', marker='v', s=100, zorder=5, label='Minima')
            
            # Mark expected quarter periods
            for i in range(int(duration / expected_quarter) + 1):
                t_q = i * expected_quarter
                if t_q <= duration:
                    ax.axvline(t_q, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            
            ax.set_title(f'{signal_name} | Error = {res["error"]:.2f}%', fontsize=12)
        else:
            ax.text(0.5, 0.5, f'{signal_name}\nFailed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Error comparison
    ax_err = fig.add_subplot(gs[4, :])
    
    method_names = []
    errors = []
    colors_bar = []
    color_map = {'Uncorrected (rotating)': 'blue', 'Normalized by isotropic': 'green', 
                 'Normalized by aniso-norot': 'red'}
    
    for signal_name in signals.keys():
        if results[signal_name] is not None:
            method_names.append(signal_name.replace('Uncorrected (rotating)', 'Uncorrected'))
            errors.append(results[signal_name]['error'])
            colors_bar.append(color_map[signal_name])
    
    bars = ax_err.bar(range(len(method_names)), errors, color=colors_bar, alpha=0.7)
    ax_err.set_xticks(range(len(method_names)))
    ax_err.set_xticklabels(method_names, rotation=15, ha='right')
    ax_err.axhline(1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='1% threshold')
    ax_err.axhline(5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='5% threshold')
    ax_err.set_ylabel('Relative Error (%)', fontsize=12)
    ax_err.set_title('Comparison: Effect of Baseline Normalization', fontsize=13)
    ax_err.legend()
    ax_err.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax_err.text(bar.get_x() + bar.get_width()/2., height,
                   f'{err:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    output_file = f'test_output/isotropic_baseline_omega_{omega:.3f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n\nFigure saved to {output_file}")
    plt.close()
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY: TRAJECTORY DECOUPLING")
    print("="*70)
    
    best_method = None
    best_error = float('inf')
    
    for signal_name in signals.keys():
        if results[signal_name] is not None:
            error = results[signal_name]['error']
            if error < best_error:
                best_error = error
                best_method = signal_name
    
    if best_method:
        print(f"\n✓ BEST METHOD: {best_method}")
        print(f"  Error: {best_error:.2f}%")
        
        # Compare to uncorrected
        if 'Uncorrected (rotating)' in results and results['Uncorrected (rotating)'] is not None:
            uncorr_error = results['Uncorrected (rotating)']['error']
            improvement = uncorr_error - best_error
            print(f"  Improvement: {improvement:.2f}% (from {uncorr_error:.2f}%)")
        
        if best_error < 1:
            print(f"\n✓✓ SUCCESS: Trajectory effects FULLY DECOUPLED!")
            print(f"   We've recovered the stationary case accuracy (<1%)!")
        elif best_error < 5:
            print(f"\n✓ GOOD: Trajectory effects significantly reduced")
        else:
            print(f"\n⚠ Partial improvement, but not complete decoupling")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# TRAJECTORY EFFECT DECOUPLING VIA ISOTROPIC BASELINE")
    print("#"*70)
    print("\nConcept: Generate reference projections from an isotropic (circular)")
    print("Gaussian following the same trajectory. This baseline has NO rotation")
    print("effects but ALL the same geometric trajectory effects.")
    print("\nBy normalizing: anisotropic_rotating / isotropic_baseline,")
    print("we remove trajectory effects and isolate pure rotation signal!")
    print("#"*70)
    
    # Test
    results = test_isotropic_baseline_normalization(
        omega=1.0,
        duration=2.0,
        n_time_points=500,
        v0=[5.0, 2.0],
        a0=[0.0, -9.81]
    )
    
    print("\n" + "#"*70)
    print("# PRACTICAL APPLICATION")
    print("#"*70)
    print("\nTo use this in your omega estimation:")
    print("\n1. After Phase 1, you have the fitted trajectory μ(t)")
    print("\n2. Generate isotropic baseline projections:")
    print("   - Use CIRCULAR Gaussian (σ_x = σ_y = average)")
    print("   - Use the SAME trajectory μ(t)")
    print("   - This gives you the trajectory-only signal")
    print("\n3. Normalize observed peaks:")
    print("   normalized_peaks = observed_peaks / baseline_peaks")
    print("\n4. Find inflection points in normalized_peaks")
    print("   → Should recover 1/(4ω) spacing!")
    print("#"*70)
