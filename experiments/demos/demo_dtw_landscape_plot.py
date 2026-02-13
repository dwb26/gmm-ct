"""
Demo: What the DTW landscape plots will look like.

This simulates the plotting functionality that's now integrated into models.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw


def generate_peak_pattern(omega, t, anisotropy=3.0):
    """Simplified peak pattern for demo"""
    angle = 2 * np.pi * omega * t
    modulation = 1.0 + 0.5 * (anisotropy - 1) * np.cos(2 * angle)
    baseline = np.exp(-(t - 0.5)**2 / 0.1)
    return modulation * baseline


# Simulate scenario
print("="*70)
print("DEMO: DTW LANDSCAPE VISUALIZATION")
print("="*70)

true_omega = 1.85
t = np.linspace(0, 1, 20)
observed_peaks = generate_peak_pattern(true_omega, t, anisotropy=3.0)

# Grid search
omega_candidates = np.linspace(1.0, 2.5, 30)
dtw_distances = []

for omega_cand in omega_candidates:
    pred_peaks = generate_peak_pattern(omega_cand, t, anisotropy=3.0)
    distance = dtw.distance(observed_peaks, pred_peaks)
    dtw_distances.append(distance)

dtw_distances = np.array(dtw_distances)
min_idx = np.argmin(dtw_distances)
omega_est = omega_candidates[min_idx]

print(f"\nTrue omega: {true_omega:.3f} Hz")
print(f"Estimated omega: {omega_est:.3f} Hz")
print(f"Error: {abs(omega_est - true_omega):.3f} Hz ({100*abs(omega_est - true_omega)/true_omega:.2f}%)")
print(f"Min DTW: {dtw_distances.min():.6f}")

# Create the plot (similar to what models.py will generate)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: DTW landscape
ax1.plot(omega_candidates, dtw_distances, 'b-', linewidth=2, label='DTW Distance')
ax1.plot(omega_est, dtw_distances[min_idx], 'ro', 
        markersize=12, label=f'Estimated ω = {omega_est:.3f} Hz')
ax1.axvline(omega_est, color='r', linestyle='--', alpha=0.5)
ax1.axvline(true_omega, color='g', linestyle=':', linewidth=2, alpha=0.7, label=f'True ω = {true_omega:.3f} Hz')

ax1.set_xlabel('Omega (Hz)', fontsize=12)
ax1.set_ylabel('DTW Distance', fontsize=12)
ax1.set_title('DTW Landscape (Loss vs Omega)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add info text
info_text = f'Min DTW: {dtw_distances.min():.6f}\n'
info_text += f'Range: [{omega_candidates.min():.2f}, {omega_candidates.max():.2f}] Hz\n'
info_text += f'Candidates: {len(omega_candidates)}\n'
info_text += f'Data points: {len(observed_peaks)}'
ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Right panel: Peak pattern comparison
predicted_peaks = generate_peak_pattern(omega_est, t, anisotropy=3.0)

ax2.plot(t, observed_peaks, 'go-', linewidth=2, markersize=8, 
        label='True Pattern', alpha=0.7)
ax2.plot(t, predicted_peaks, 'b^--', linewidth=2, markersize=6,
        label=f'Predicted (ω={omega_est:.3f} Hz)', alpha=0.7)

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Peak Value', fontsize=12)
ax2.set_title('Peak Pattern Match', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Compute match quality
residual = np.abs(observed_peaks - predicted_peaks)
mean_residual = np.mean(residual)
max_residual = np.max(residual)

match_text = f'Mean |residual|: {mean_residual:.4f}\n'
match_text += f'Max |residual|: {max_residual:.4f}\n'
match_text += f'DTW distance: {dtw_distances.min():.6f}\n'
match_text += f'Error: {100*abs(omega_est - true_omega)/true_omega:.2f}%'
ax2.text(0.02, 0.98, match_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('demo_dtw_landscape.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved demo plot: demo_dtw_landscape.png")
print("\nThis shows what you'll see for each Gaussian during omega estimation!")
print("="*70)

plt.show()
