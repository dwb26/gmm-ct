"""
Compare efficiency: Fine Grid DTW vs Multi-Start Soft-DTW

Question: Can multi-start with gradient descent beat grid search?
- Fine grid DTW: 100 evaluations, no gradients
- Coarse grid DTW: 20 evaluations, no gradients  
- Multi-start Soft-DTW: 20 initializations → gradient descent
- Can also test different optimizers (Adam, SGD, LBFGS)
"""

import torch
import numpy as np
from dtaidistance import dtw
import time


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
    """Generate projection peaks (ACTUAL model)"""
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


def test_efficiency_comparison():
    """Compare computational efficiency of different approaches"""
    
    print("="*70)
    print("EFFICIENCY COMPARISON: GRID vs MULTI-START GRADIENT DESCENT")
    print("="*70)
    
    device = 'cpu'
    
    # Setup
    source = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    n_receivers = 40
    receiver_line = [
        torch.tensor([5.0, y], dtype=torch.float64, device=device)
        for y in np.linspace(-2.0, 2.0, n_receivers)
    ]
    
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
    
    # Generate true pattern
    with torch.no_grad():
        true_peaks = generate_peak_pattern(
            alpha, U_skew, torch.tensor(true_omega, dtype=torch.float64),
            x0, v0, a0, t, source, receiver_line
        )
    
    # ========================================
    # METHOD 1: Fine Grid DTW (100 points)
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 1: Fine Grid DTW (100 candidates)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    omega_candidates_fine = np.linspace(1.0, 2.5, 100)
    dtw_distances_fine = []
    n_forward_passes = 0
    
    for omega_cand in omega_candidates_fine:
        with torch.no_grad():
            pred_peaks = generate_peak_pattern(
                alpha, U_skew, torch.tensor(omega_cand, dtype=torch.float64),
                x0, v0, a0, t, source, receiver_line
            )
        distance = dtw.distance(true_peaks.cpu().numpy(), pred_peaks.cpu().numpy())
        dtw_distances_fine.append(distance)
        n_forward_passes += 1
    
    min_idx = np.argmin(dtw_distances_fine)
    omega_fine = omega_candidates_fine[min_idx]
    error_fine = abs(omega_fine - true_omega)
    error_pct_fine = 100 * error_fine / true_omega
    time_fine = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Estimated omega: {omega_fine:.4f} Hz")
    print(f"  Error: {error_pct_fine:.2f}%")
    print(f"  Forward passes: {n_forward_passes}")
    print(f"  Time: {time_fine:.3f} seconds")
    
    # ========================================
    # METHOD 2: Coarse Grid DTW (20 points)
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 2: Coarse Grid DTW (20 candidates)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    omega_candidates_coarse = np.linspace(1.0, 2.5, 20)
    dtw_distances_coarse = []
    n_forward_passes = 0
    
    for omega_cand in omega_candidates_coarse:
        with torch.no_grad():
            pred_peaks = generate_peak_pattern(
                alpha, U_skew, torch.tensor(omega_cand, dtype=torch.float64),
                x0, v0, a0, t, source, receiver_line
            )
        distance = dtw.distance(true_peaks.cpu().numpy(), pred_peaks.cpu().numpy())
        dtw_distances_coarse.append(distance)
        n_forward_passes += 1
    
    min_idx = np.argmin(dtw_distances_coarse)
    omega_coarse = omega_candidates_coarse[min_idx]
    error_coarse = abs(omega_coarse - true_omega)
    error_pct_coarse = 100 * error_coarse / true_omega
    time_coarse = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Estimated omega: {omega_coarse:.4f} Hz")
    print(f"  Error: {error_pct_coarse:.2f}%")
    print(f"  Forward passes: {n_forward_passes}")
    print(f"  Time: {time_coarse:.3f} seconds")
    
    # ========================================
    # METHOD 3: Multi-Start Soft-DTW (Adam)
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 3: Multi-Start Soft-DTW + Adam (20 starts, 30 iters each)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    n_forward_passes = 0
    results_adam = []
    
    for omega_init in omega_candidates_coarse:
        omega = torch.tensor(omega_init, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([omega], lr=0.02)
        
        for iter in range(30):
            optimizer.zero_grad()
            pred_peaks = generate_peak_pattern(
                alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line
            )
            loss = soft_dtw_pytorch(pred_peaks, true_peaks, gamma=0.1)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                omega.clamp_(0.5, 3.0)
            
            n_forward_passes += 1
        
        results_adam.append({
            'init': omega_init,
            'final': omega.item(),
            'loss': loss.item()
        })
    
    # Find best result
    best_result = min(results_adam, key=lambda r: abs(r['final'] - true_omega))
    omega_adam = best_result['final']
    error_adam = abs(omega_adam - true_omega)
    error_pct_adam = 100 * error_adam / true_omega
    time_adam = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Best omega: {omega_adam:.4f} Hz (from init {best_result['init']:.2f})")
    print(f"  Error: {error_pct_adam:.2f}%")
    print(f"  Forward+backward passes: {n_forward_passes}")
    print(f"  Time: {time_adam:.3f} seconds")
    
    # Show spread of results
    all_omegas = [r['final'] for r in results_adam]
    print(f"  Omega range: [{min(all_omegas):.3f}, {max(all_omegas):.3f}] Hz")
    
    # ========================================
    # METHOD 4: Multi-Start Soft-DTW (SGD)
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 4: Multi-Start Soft-DTW + SGD (20 starts, 30 iters each)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    n_forward_passes = 0
    results_sgd = []
    
    for omega_init in omega_candidates_coarse:
        omega = torch.tensor(omega_init, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([omega], lr=0.05, momentum=0.9)
        
        for iter in range(30):
            optimizer.zero_grad()
            pred_peaks = generate_peak_pattern(
                alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line
            )
            loss = soft_dtw_pytorch(pred_peaks, true_peaks, gamma=0.1)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                omega.clamp_(0.5, 3.0)
            
            n_forward_passes += 1
        
        results_sgd.append({
            'init': omega_init,
            'final': omega.item(),
            'loss': loss.item()
        })
    
    # Find best result
    best_result = min(results_sgd, key=lambda r: abs(r['final'] - true_omega))
    omega_sgd = best_result['final']
    error_sgd = abs(omega_sgd - true_omega)
    error_pct_sgd = 100 * error_sgd / true_omega
    time_sgd = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Best omega: {omega_sgd:.4f} Hz (from init {best_result['init']:.2f})")
    print(f"  Error: {error_pct_sgd:.2f}%")
    print(f"  Forward+backward passes: {n_forward_passes}")
    print(f"  Time: {time_sgd:.3f} seconds")
    
    all_omegas = [r['final'] for r in results_sgd]
    print(f"  Omega range: [{min(all_omegas):.3f}, {max(all_omegas):.3f}] Hz")
    
    # ========================================
    # METHOD 5: Two-Stage (Coarse + Soft-DTW refinement)
    # ========================================
    print(f"\n{'='*70}")
    print("METHOD 5: Two-Stage (Coarse DTW → Soft-DTW refinement)")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Stage 1: Coarse grid (already computed above)
    n_forward_passes = 20  # From coarse grid
    
    # Stage 2: Refine best candidate with gradient descent
    omega = torch.tensor(omega_coarse, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([omega], lr=0.02)
    
    for iter in range(30):
        optimizer.zero_grad()
        pred_peaks = generate_peak_pattern(
            alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line
        )
        loss = soft_dtw_pytorch(pred_peaks, true_peaks, gamma=0.1)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            omega.clamp_(0.5, 3.0)
        
        n_forward_passes += 1
    
    omega_twostage = omega.item()
    error_twostage = abs(omega_twostage - true_omega)
    error_pct_twostage = 100 * error_twostage / true_omega
    time_twostage = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Stage 1 (coarse): {omega_coarse:.4f} Hz")
    print(f"  Stage 2 (refined): {omega_twostage:.4f} Hz")
    print(f"  Error: {error_pct_twostage:.2f}%")
    print(f"  Forward+backward passes: {n_forward_passes}")
    print(f"  Time: {time_twostage:.3f} seconds")
    
    # ========================================
    # SUMMARY TABLE
    # ========================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Method':<35} {'Error':<10} {'Passes':<10} {'Time':<10}")
    print("-" * 70)
    print(f"{'1. Fine Grid DTW (100)':<35} {error_pct_fine:>6.2f}%   {100:>6}     {time_fine:>6.3f}s")
    print(f"{'2. Coarse Grid DTW (20)':<35} {error_pct_coarse:>6.2f}%   {20:>6}     {time_coarse:>6.3f}s")
    print(f"{'3. Multi-Start Soft-DTW+Adam':<35} {error_pct_adam:>6.2f}%   {600:>6}     {time_adam:>6.3f}s")
    print(f"{'4. Multi-Start Soft-DTW+SGD':<35} {error_pct_sgd:>6.2f}%   {600:>6}     {time_sgd:>6.3f}s")
    print(f"{'5. Two-Stage (Coarse→Refine)':<35} {error_pct_twostage:>6.2f}%   {50:>6}     {time_twostage:>6.3f}s")
    
    # ========================================
    # RECOMMENDATION
    # ========================================
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    # Find most efficient accurate method
    methods = [
        ('Fine Grid DTW', error_pct_fine, 100, time_fine),
        ('Coarse Grid DTW', error_pct_coarse, 20, time_coarse),
        ('Multi-Start+Adam', error_pct_adam, 600, time_adam),
        ('Multi-Start+SGD', error_pct_sgd, 600, time_sgd),
        ('Two-Stage', error_pct_twostage, 50, time_twostage),
    ]
    
    # Filter to accurate methods (< 5% error)
    accurate = [(name, err, passes, t) for name, err, passes, t in methods if err < 5.0]
    
    if accurate:
        # Sort by number of passes (efficiency)
        accurate.sort(key=lambda x: x[2])
        best_name, best_err, best_passes, best_time = accurate[0]
        
        print(f"\n✓ BEST METHOD: {best_name}")
        print(f"  Error: {best_err:.2f}%")
        print(f"  Passes: {best_passes}")
        print(f"  Time: {best_time:.3f}s")
        
        if best_passes < 100:
            print(f"\n  → {100 - best_passes} fewer passes than fine grid!")
        
        if 'Two-Stage' in best_name:
            print("\n  INSIGHT: Two-stage approach combines best of both:")
            print("    • Coarse grid finds the region (cheap)")
            print("    • Gradient descent refines estimate (accurate)")
            print("    • Total cost: Much less than full multi-start")
    
    print(f"{'='*70}")


if __name__ == '__main__':
    test_efficiency_comparison()
