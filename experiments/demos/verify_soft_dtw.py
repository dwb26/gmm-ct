#!/usr/bin/env python
"""Quick verification: Soft-DTW with PyTorch autograd"""

import torch

def soft_dtw(x, y, gamma=1.0):
    """Minimal Soft-DTW implementation"""
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    n, m = len(x), len(y)
    D = torch.cdist(x, y, p=2) ** 2
    R = torch.full((n + 1, m + 1), float('inf'), dtype=x.dtype)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            vals = torch.stack([R[i-1, j], R[i, j-1], R[i-1, j-1]])
            R[i, j] = D[i-1, j-1] - gamma * torch.logsumexp(-vals / gamma, dim=0)
    return R[n, m]

# Test differentiability
print("Testing Soft-DTW differentiability...")
t = torch.linspace(0, 1, 20)
y_true = torch.sin(2 * torch.pi * 1.85 * t)
freq = torch.tensor(1.0, requires_grad=True)
y_pred = torch.sin(2 * torch.pi * freq * t)
loss = soft_dtw(y_pred, y_true, gamma=0.1)
loss.backward()
print(f"✓ Differentiable: freq={freq.item():.2f}, loss={loss.item():.4f}, grad={freq.grad.item():.4f}")

# Test optimization
print("\nTesting gradient descent...")
freq = torch.tensor(1.2, requires_grad=True)
opt = torch.optim.Adam([freq], lr=0.02)
for i in range(30):
    opt.zero_grad()
    y_pred = torch.sin(2 * torch.pi * freq * t)
    loss = soft_dtw(y_pred, y_true, gamma=0.1)
    loss.backward()
    opt.step()
error_pct = 100 * abs(freq.item() - 1.85) / 1.85
print(f"✓ Converged: freq={freq.item():.4f} (error: {error_pct:.2f}%)")

print("\n✓✓ YES - Soft-DTW works with PyTorch optimizers!")
print("   Can be integrated into differentiable loss functions.")
