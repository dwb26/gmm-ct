"""
Test Soft-DTW for differentiability and integration with PyTorch optimization.

Soft-DTW is a differentiable relaxation of DTW that can be used with gradient-based
optimization. This test demonstrates:
1. Soft-DTW is differentiable (gradients flow through it)
2. It can be integrated into a PyTorch loss function
3. Gradient descent can optimize omega using Soft-DTW

We implement a simple soft-DTW in PyTorch for full gradient support.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def soft_dtw_pytorch(x, y, gamma=1.0):
    """
    Compute Soft-DTW distance using PyTorch (fully differentiable).
    
    Soft-DTW is a differentiable relaxation of DTW that uses soft-min
    instead of hard-min for computing the alignment cost.
    
    Parameters
    ----------
    x, y : torch.Tensor, shape (n,) or (n, 1)
        Time series to compare
    gamma : float
        Smoothing parameter (smaller = closer to standard DTW)
    
    Returns
    -------
    distance : torch.Tensor (scalar with gradient)
        Soft-DTW distance
    
    References
    ----------
    Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for
    Time-Series", ICML 2017
    """
    # Ensure 2D
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    n, m = len(x), len(y)
    
    # Compute pairwise squared distances
    # D[i, j] = ||x[i] - y[j]||^2
    D = torch.cdist(x, y, p=2) ** 2
    
    # Initialize cost matrix with inf
    R = torch.full((n + 1, m + 1), float('inf'), dtype=x.dtype, device=x.device)
    R[0, 0] = 0.0
    
    # Soft-min function
    def soft_min(a, b, c, gamma):
        """Soft-min of three values using log-sum-exp trick"""
        vals = torch.stack([a, b, c])
        return -gamma * torch.logsumexp(-vals / gamma, dim=0)
    
    # Dynamic programming with soft-min
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = D[i-1, j-1]
            R[i, j] = cost + soft_min(R[i-1, j], R[i, j-1], R[i-1, j-1], gamma)
    
    return R[n, m]


def generate_peak_pattern_torch(alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line):
    """
    Generate projection peak values for a moving, rotating anisotropic Gaussian.
    
    This is a TORCH version that preserves gradients for backpropagation.
    
    Parameters
    ----------
    omega : torch.Tensor (scalar with requires_grad=True)
        Angular velocity (Hz) - THIS IS WHAT WE OPTIMIZE
    
    Returns
    -------
    peak_values : torch.Tensor
        Peak projection values at each time point (with gradient tracking)
    """
    device = t.device
    sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))
    
    peak_values = []
    
    for t_n in t:
        # Trajectory: μ(t) = x0 + v0*t + 0.5*a0*t^2
        mu_t = x0 + v0 * t_n + 0.5 * a0 * t_n**2
        
        # Rotation matrix at time t (depends on omega!)
        angle = 2 * torch.pi * omega * t_n  # This makes omega differentiable
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R_t = torch.stack([
            torch.stack([cos_a, -sin_a]),
            torch.stack([sin_a, cos_a])
        ])
        
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
        
        # Extract peak value (max across receivers)
        peak_values.append(torch.max(torch.stack(projections)))
    
    return torch.stack(peak_values)


def soft_dtw_loss(omega, true_peaks, alpha, U_skew, x0, v0, a0, t, source, receiver_line, gamma=1.0):
    """
    Soft-DTW loss function for omega estimation.
    
    This is a DIFFERENTIABLE loss function that can be used with PyTorch optimizers.
    
    Parameters
    ----------
    omega : torch.Tensor (scalar with requires_grad=True)
        Angular velocity to optimize
    true_peaks : torch.Tensor
        Observed peak pattern (target)
    gamma : float
        Smoothing parameter for Soft-DTW (smaller = closer to standard DTW)
    
    Returns
    -------
    loss : torch.Tensor (scalar with gradient)
        Soft-DTW distance between predicted and true peaks
    """
    # Generate predicted peaks with current omega
    pred_peaks = generate_peak_pattern_torch(
        alpha, U_skew, omega, x0, v0, a0, t, source, receiver_line
    )
    
    # Compute Soft-DTW distance (fully differentiable in PyTorch)
    loss = soft_dtw_pytorch(pred_peaks, true_peaks, gamma=gamma)
    
    return loss, pred_peaks


def test_soft_dtw_differentiability():
    """
    Test 1: Verify that Soft-DTW is differentiable.
    """
    print("="*70)
    print("TEST 1: SOFT-DTW DIFFERENTIABILITY")
    print("="*70)
    
    device = 'cpu'
    
    # Simple setup
    source = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    n_receivers = 40
    receiver_line = [
        torch.tensor([5.0, y], dtype=torch.float64, device=device)
        for y in np.linspace(-2.0, 2.0, n_receivers)
    ]
    
    n_times = 20
    t = torch.linspace(0, 1, n_times, dtype=torch.float64, device=device)
    
    # Parameters
    alpha = torch.tensor(12.0, dtype=torch.float64, device=device)
    U_skew = torch.tensor([[30.0, 0.0], [0.0, 10.0]], dtype=torch.float64, device=device)
    x0 = torch.tensor([1.0, 0.5], dtype=torch.float64, device=device)
    v0 = torch.tensor([2.0, 1.7], dtype=torch.float64, device=device)
    a0 = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    # True omega
    true_omega = 1.85
    
    # Generate true pattern
    true_peaks = generate_peak_pattern_torch(
        alpha, U_skew, torch.tensor(true_omega, dtype=torch.float64, device=device),
        x0, v0, a0, t, source, receiver_line
    )
    
    print(f"\nTrue omega: {true_omega} Hz")
    print(f"True peaks shape: {true_peaks.shape}")
    
    # Test gradient computation at several omega values
    test_omegas = [1.5, 1.85, 2.2]
    
    print(f"\nTesting Soft-DTW gradient computation:")
    for omega_val in test_omegas:
        omega = torch.tensor(omega_val, dtype=torch.float64, device=device, requires_grad=True)
        
        # Compute loss
        loss, pred_peaks = soft_dtw_loss(
            omega, true_peaks, alpha, U_skew, x0, v0, a0, t, source, receiver_line, gamma=1.0
        )
        
        # Backward pass - compute gradient automatically!
        loss.backward()
        
        print(f"  ω={omega_val:.2f} Hz: loss={loss.item():.6f}, ∂loss/∂ω={omega.grad.item():.6f}")
    
    print("\n✓ Soft-DTW is smooth and differentiable with PyTorch autograd!")


def test_soft_dtw_optimization():
    """
    Test 2: Use Soft-DTW with gradient descent to find omega.
    """
    print("\n" + "="*70)
    print("TEST 2: GRADIENT DESCENT WITH SOFT-DTW")
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
    
    # Fixed parameters
    alpha = torch.tensor(12.0, dtype=torch.float64, device=device)
    U_skew = torch.tensor([[30.0, 0.0], [0.0, 10.0]], dtype=torch.float64, device=device)
    x0 = torch.tensor([1.0, 0.5], dtype=torch.float64, device=device)
    v0 = torch.tensor([2.0, 1.7], dtype=torch.float64, device=device)
    a0 = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    
    # True omega
    true_omega = 1.85
    
    # Generate true pattern
    true_peaks = generate_peak_pattern_torch(
        alpha, U_skew, torch.tensor(true_omega, dtype=torch.float64, device=device),
        x0, v0, a0, t, source, receiver_line
    )
    
    print(f"\nTrue omega: {true_omega} Hz")
    
    # Test gradient descent from different starting points
    initial_omegas = [1.2, 1.5, 2.0, 2.3]
    
    results = []
    
    for omega_init in initial_omegas:
        print(f"\nStarting from ω₀ = {omega_init:.2f} Hz")
        
        # Initialize omega with gradient tracking
        omega = torch.tensor(omega_init, dtype=torch.float64, device=device, requires_grad=True)
        
        # Use PyTorch optimizer
        optimizer = torch.optim.Adam([omega], lr=0.01)
        
        n_iterations = 50
        
        losses = []
        omegas = []
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Forward pass with automatic differentiation
            loss, pred_peaks = soft_dtw_loss(
                omega, true_peaks, alpha, U_skew, x0, v0, a0, t, source, receiver_line, gamma=1.0
            )
            
            # Backward pass - compute gradients automatically!
            loss.backward()
            
            # Update omega
            optimizer.step()
            
            # Clamp to reasonable range (in-place)
            with torch.no_grad():
                omega.clamp_(0.5, 3.0)
            
            losses.append(loss.item())
            omegas.append(omega.item())
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: ω={omega.item():.4f} Hz, loss={loss.item():.6f}, grad={omega.grad.item():.6f}")
        
        final_omega = omega.item()
        final_error = abs(final_omega - true_omega)
        final_error_pct = 100 * final_error / true_omega
        
        print(f"  Final: ω={final_omega:.4f} Hz (error: {final_error_pct:.2f}%)")
        
        results.append({
            'init': omega_init,
            'final': final_omega,
            'error': final_error,
            'losses': losses,
            'omegas': omegas
        })
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Convergence trajectories
    for res in results:
        ax1.plot(res['omegas'], label=f"Init: {res['init']:.2f} Hz", linewidth=2)
    
    ax1.axhline(true_omega, color='r', linestyle='--', linewidth=2, label='True ω')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Omega (Hz)', fontsize=12)
    ax1.set_title('Gradient Descent Convergence', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Loss curves
    for res in results:
        ax2.semilogy(res['losses'], label=f"Init: {res['init']:.2f} Hz", linewidth=2)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Soft-DTW Loss (log scale)', fontsize=12)
    ax2.set_title('Loss Reduction', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soft_dtw_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: soft_dtw_optimization.png")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"True omega: {true_omega:.4f} Hz")
    print(f"\nGradient descent results:")
    for res in results:
        print(f"  Init {res['init']:.2f} → Final {res['final']:.4f} Hz "
              f"(error: {100*res['error']/true_omega:.2f}%)")
    
    # Check success rate
    successful = sum(1 for res in results if 100*res['error']/true_omega < 5.0)
    print(f"\nSuccess rate (< 5% error): {successful}/{len(results)}")
    
    if successful == len(results):
        print("\n✓ SUCCESS: Soft-DTW gradient descent works reliably!")
    elif successful > len(results) / 2:
        print("\n⚠ PARTIAL: Soft-DTW gradient descent works but sensitive to initialization")
    else:
        print("\n✗ FAILED: Soft-DTW gradient descent unreliable")


if __name__ == '__main__':
    test_soft_dtw_differentiability()
    test_soft_dtw_optimization()
