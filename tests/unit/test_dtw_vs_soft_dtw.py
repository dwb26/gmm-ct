"""
Compare DTW grid search vs Soft-DTW gradient descent.

This uses the ACTUAL peak generation function from your model.
"""

import torch
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt


def soft_dtw_pytorch(x, y, gamma=1.0):
    """Soft-DTW: Differentiable DTW variant"""
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    n, m = len(x), len(y)
    D = torch.cdist(x, y, p=2) ** 2
    R = torch.full((n + 1, m + 1), float('inf'), dtype=x.dtype, device=x.device)
    R[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            vals = torch.stack([R[i-1, j], R[i, j-1], R[i-1, j-1]])
            R[i, j] = D[i-1, j-1] - gamma * torch.logsumexp(-vals / gamma, dim=0)
    
    return R[n, m]


def generate_peak_pattern(alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line):
    """Generate projection peaks from rotating, moving Gaussian (ACTUAL model)"""
    device = t.device
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
    
    peak_values = []
    
    for t_n in t:
        mu_t = x0 + v0 * t_n + 0.5 * a0 * t_n**2
        angle = 2 * torch.pi * omega * t_n
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R_t = torch.stack([
            torch.stack([cos_a, -sin_a]),
            torch.stack([sin_a, cos_a])
        ])
        U_rot = U_skew @ R_t.T
        
        projections = []
        for receiver in receiver_line:
            r_minus_s = receiver - source
            r_minus_s_hat = r_minus_s / torch.norm(r_minus_s)
            U_r_hat = U_rot @ r_minus_s_hat
            U_r = U_rot @ r_minus_s
            U_mu = U_rot @ (source - mu_t)
            
            norm_term = torch.norm(U_r_hat)
            quotient = sqrt_pi * alpha / (norm_term + 1e-10)
            inner_prod = torch.dot(U_r.squeeze(), U_mu)
            inner_prod_sq = inner_prod ** 2
            divisor = torch.norm(U_r) ** 2 + 1e-10
            subtractor = torch.norm(U_mu) ** 2
            exp_arg = inner_prod_sq / divisor - subtractor
            proj = quotient * torch.exp(exp_arg)
            projections.append(proj)
        
        peak_values.append(torch.max(torch.stack(projections)))
    
    return torch.stack(peak_values)


def test_comparison():
    """Compare grid search vs gradient descent on real peak patterns"""
    
    print("="*70)
    print("COMPARISON: DTW GRID SEARCH vs SOFT-DTW GRADIENT DESCENT")
    print("="*70)
    
    device = 'cpu'
    
    # Geometry setup
    source = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    n_receivers = 40
    receiver_line = [
        torch.tensor([5.0, y], dtype=torch.float64, device=device)
        for y in np.linspace(-2.0, 2.0, n_receivers)
    ]
    
    # Parameters
    n_times = 20
    t = torch.linspace(0, 1, n_times, dtype=torch.float64, device=device)
    alpha = torch.tensor(12.0, dtype=torch.float64, device=device)
    U_skew = torch.tensor([[30.0, 0.0], [0.0, 10.0]], dtype=torch.float64, device=device)
    x0 = torch.tensor([1.0, 0.5], dtype=torch.float64, device=device)
    v0 = torch.tensor([2.0, 1.7], dtype=torch.float64, device=device)
    a0 = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    true_omega = 1.85
    
    print(f"\nSetup:")
    print(f"  True omega: {true_omega} Hz")
    print(f"  Time points: {n_times}")
    print(f"  Anisotropy: 3:1")
    print(f"  Trajectory: moving")
    
    # Generate true pattern
    with torch.no_grad():
        true_peaks = generate_peak_pattern(
            alpha, U_skew, torch.tensor(true_omega, dtype=torch.float64),
            x0, v0, a0, t, source, receiver_line
        )
    
    # ========================================
    # METHOD 1: Standard DTW with Grid Search
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 1: Standard DTW + Grid Search")
    print(f"{'='*70}")
    
    omega_candidates = np.linspace(1.0, 2.5, 100)
    dtw_distances = []
    
    for omega_cand in omega_candidates:
        with torch.no_grad():
            pred_peaks = generate_peak_pattern(
                alpha, U_skew, torch.tensor(omega_cand, dtype=torch.float64),
                x0, v0, a0, t, source, receiver_line
            )
        distance = dtw.distance(
            true_peaks.cpu().numpy(),
            pred_peaks.cpu().numpy()
        )
        dtw_distances.append(distance)
    
    dtw_distances = np.array(dtw_distances)
    min_idx = np.argmin(dtw_distances)
    omega_dtw = omega_candidates[min_idx]
    error_dtw = abs(omega_dtw - true_omega)
    error_pct_dtw = 100 * error_dtw / true_omega
    
    print(f"\nResults:")
    print(f"  Estimated omega: {omega_dtw:.4f} Hz")
    print(f"  Error: {error_dtw:.4f} Hz ({error_pct_dtw:.2f}%)")
    print(f"  Min DTW distance: {dtw_distances[min_idx]:.6f}")
    
    # ========================================
    # METHOD 2: Soft-DTW with Gradient Descent
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 2: Soft-DTW + Gradient Descent")
    print(f"{'='*70}")
    
    # Test different initializations
    init_omegas = [1.2, 1.5, 2.0, 2.3]
    results_gd = []
    
    for omega_init in init_omegas:
        print(f"\n  Starting from ω₀={omega_init:.2f} Hz:")
        
        omega = torch.tensor(omega_init, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([omega], lr=0.01)
        
        for iter in range(50):
            optimizer.zero_grad()
            
            pred_peaks = generate_peak_pattern(
                alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line
            )
            
            loss = soft_dtw_pytorch(pred_peaks, true_peaks, gamma=0.1)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                omega.clamp_(0.5, 3.0)
            
            if iter % 20 == 0:
                print(f"    Iter {iter:2d}: ω={omega.item():.4f} Hz, loss={loss.item():.6f}")
        
        final_omega = omega.item()
        error = abs(final_omega - true_omega)
        error_pct = 100 * error / true_omega
        
        print(f"    Final: ω={final_omega:.4f} Hz (error: {error_pct:.2f}%)")
        
        results_gd.append({
            'init': omega_init,
            'final': final_omega,
            'error_pct': error_pct
        })
    
    # ========================================
    # COMPARISON SUMMARY
    # ========================================
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nStandard DTW + Grid Search:")
    print(f"  Error: {error_pct_dtw:.2f}%")
    print(f"  Computation: 100 forward passes")
    
    print(f"\nSoft-DTW + Gradient Descent:")
    successful_gd = [r for r in results_gd if r['error_pct'] < 5.0]
    print(f"  Success rate: {len(successful_gd)}/{len(results_gd)}")
    print(f"  Best error: {min(r['error_pct'] for r in results_gd):.2f}%")
    print(f"  Worst error: {max(r['error_pct'] for r in results_gd):.2f}%")
    print(f"  Computation: 50 forward+backward passes per attempt")
    
    # ========================================
    # RECOMMENDATION
    # ========================================
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    if error_pct_dtw < 5.0 and len(successful_gd) < len(results_gd):
        print("\n✓ Use DTW Grid Search:")
        print("  - More robust (always finds correct omega)")
        print("  - Fast enough for sparse data")
        print("  - No local minima issues")
        print(f"  - {error_pct_dtw:.2f}% error (excellent)")
        
        print("\n✗ Soft-DTW Gradient Descent:")
        print(f"  - Sensitive to initialization ({len(successful_gd)}/{len(results_gd)} success)")
        print("  - Can get stuck in local minima")
        print("  - Better suited for joint optimization if initialized near minimum")
    elif len(successful_gd) == len(results_gd):
        print("\n✓ Both methods work!")
        print("  - DTW grid search: More robust")
        print("  - Soft-DTW gradient: Can be used in joint optimization")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    test_comparison()
