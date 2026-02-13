"""
Direct rotation matrix verification test.

This script tests the rotation matrices directly by:
1. Applying them to fixed points and tracking their rotation
2. Computing the actual rotation angle from the matrix elements
3. Comparing with the expected angle = 2*pi*omega*t
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def construct_rotation_matrix(t, omega, device='cpu'):
    """
    Construct a 2D rotation matrix using the same formulation as in models.py.
    
    Angular position: theta(t) = 2*pi*omega*t
    """
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


def extract_angle_from_rotation_matrix(R):
    """
    Extract the rotation angle from a 2D rotation matrix.
    
    For a rotation matrix:
    R = [cos(θ), -sin(θ)]
        [sin(θ),  cos(θ)]
    
    We can recover θ using arctan2(R[1,0], R[0,0])
    
    Returns angle in radians.
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    
    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    return angle_rad


def test_rotation_matrix_directly(omega, test_times, output_file='test_output/rotation_matrix_test.png'):
    """
    Test the rotation matrix by applying it to fixed points and visualizing.
    """
    device = 'cpu'
    
    # Define test points (unit vectors and a point off-axis)
    points = np.array([
        [1.5, 0],      # Point on x-axis
        [0, 1.5],      # Point on y-axis
        [1.0, 1.0],    # Point at 45°
    ]).T  # Shape: (2, 3)
    
    n_times = len(test_times)
    fig, axes = plt.subplots(2, n_times, figsize=(4*n_times, 8))
    
    if n_times == 1:
        axes = axes.reshape(-1, 1)
    
    print("\n" + "="*70)
    print("DIRECT ROTATION MATRIX TEST")
    print("="*70)
    print(f"Testing with ω = {omega}")
    print(f"Expected: θ(t) = 2π×ω×t")
    print("="*70)
    
    for i, t in enumerate(test_times):
        # Get rotation matrix
        R_t = construct_rotation_matrix(t, omega, device).cpu().numpy()
        
        # Expected angle
        expected_angle_rad = 2 * np.pi * omega * t
        expected_angle_deg = np.degrees(expected_angle_rad)
        
        # Extract angle from matrix
        computed_angle_rad = extract_angle_from_rotation_matrix(R_t)
        computed_angle_deg = np.degrees(computed_angle_rad)
        
        # Rotate the test points
        rotated_points = R_t @ points
        
        # Plot 1: Points rotation visualization
        ax1 = axes[0, i]
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        
        # Plot original points (in gray)
        ax1.scatter(points[0, :], points[1, :], c='gray', s=100, alpha=0.5, label='Original')
        
        # Plot rotated points (in color)
        colors = ['red', 'blue', 'green']
        for j in range(points.shape[1]):
            ax1.scatter(rotated_points[0, j], rotated_points[1, j], 
                       c=colors[j], s=100, marker='o', label=f'Point {j+1}')
            # Draw arrows from origin to rotated points
            ax1.arrow(0, 0, rotated_points[0, j], rotated_points[1, j],
                     head_width=0.1, head_length=0.1, fc=colors[j], ec=colors[j],
                     alpha=0.6, linewidth=2)
            # Draw arrows from original to rotated (dashed)
            ax1.plot([points[0, j], rotated_points[0, j]], 
                    [points[1, j], rotated_points[1, j]],
                    '--', color='gray', alpha=0.4, linewidth=1)
        
        ax1.set_title(f't = {t:.3f} s\nω×t = {omega*t:.3f}', fontsize=11)
        if i == 0:
            ax1.set_ylabel('Point Rotation', fontsize=12)
        if i == n_times - 1:
            ax1.legend(fontsize=8, loc='upper right')
        
        # Plot 2: Rotation matrix visualization
        ax2 = axes[1, i]
        
        # Display rotation matrix as a heatmap
        im = ax2.imshow(R_t, cmap='RdBu', vmin=-1, vmax=1, aspect='equal')
        
        # Add text annotations
        for row in range(2):
            for col in range(2):
                text = ax2.text(col, row, f'{R_t[row, col]:.3f}',
                               ha="center", va="center", color="black", fontsize=10)
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['x', 'y'])
        ax2.set_yticklabels(['x', 'y'])
        ax2.set_title(f'R(t) matrix', fontsize=11)
        
        if i == 0:
            ax2.set_ylabel('Matrix Elements', fontsize=12)
        
        # Add colorbar only for the last subplot
        if i == n_times - 1:
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Print diagnostics
        print(f"\nTime t = {t:.4f} s (ω×t = {omega*t:.4f}):")
        print(f"  Expected angle: {expected_angle_deg:.2f}° ({expected_angle_rad:.4f} rad)")
        print(f"  Matrix-derived angle: {computed_angle_deg:.2f}° ({computed_angle_rad:.4f} rad)")
        print(f"  Difference: {abs(expected_angle_rad - computed_angle_rad):.6f} rad")
        print(f"  Rotation matrix determinant: {np.linalg.det(R_t):.6f} (should be 1.0)")
        print(f"  Matrix orthogonality check (R^T R):")
        RTR = R_t.T @ R_t
        print(f"    [{RTR[0,0]:.6f}, {RTR[0,1]:.6f}]")
        print(f"    [{RTR[1,0]:.6f}, {RTR[1,1]:.6f}]")
        print(f"    (should be identity matrix)")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_file}")
    plt.close()


def test_continuous_angle_tracking(omega, duration=4.0, n_points=100):
    """
    Track the rotation angle continuously over time and plot it.
    
    This shows whether the angle increases linearly as expected.
    """
    device = 'cpu'
    
    t_array = np.linspace(0, duration, n_points)
    expected_angles = 2 * np.pi * omega * t_array
    computed_angles = np.zeros_like(t_array)
    
    for i, t in enumerate(t_array):
        R_t = construct_rotation_matrix(t, omega, device)
        computed_angles[i] = extract_angle_from_rotation_matrix(R_t)
    
    # Unwrap computed angles to handle the [-pi, pi] discontinuity
    computed_angles_unwrapped = np.unwrap(computed_angles)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Raw angles (with wrapping)
    ax1.plot(t_array, expected_angles, 'b-', linewidth=2, label='Expected: 2π×ω×t')
    ax1.plot(t_array, computed_angles, 'r--', linewidth=2, label='Computed (wrapped)')
    ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax1.axhline(y=np.pi, color='g', linewidth=0.5, alpha=0.3, linestyle='--')
    ax1.axhline(y=-np.pi, color='g', linewidth=0.5, alpha=0.3, linestyle='--')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Angle (radians)', fontsize=12)
    ax1.set_title(f'Rotation Angle vs Time (ω = {omega})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Unwrapped angles
    ax2.plot(t_array, expected_angles, 'b-', linewidth=2, label='Expected: 2π×ω×t')
    ax2.plot(t_array, computed_angles_unwrapped, 'r--', linewidth=2, label='Computed (unwrapped)')
    
    # Mark complete rotations
    for n in range(int(omega * duration) + 1):
        t_rotation = n / omega if omega != 0 else 0
        if t_rotation <= duration:
            ax2.axvline(x=t_rotation, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
            ax2.axhline(y=2*np.pi*n, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Angle (radians, unwrapped)', fontsize=12)
    ax2.set_title('Continuous Angle Tracking (unwrapped)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'test_output/continuous_angle_tracking.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nContinuous angle tracking figure saved to {output_file}")
    plt.close()
    
    # Calculate statistics
    angle_difference = np.abs(expected_angles - computed_angles_unwrapped)
    print(f"\nAngle tracking statistics:")
    print(f"  Mean absolute error: {np.mean(angle_difference):.6e} rad")
    print(f"  Max absolute error: {np.max(angle_difference):.6e} rad")
    print(f"  Final expected angle: {expected_angles[-1]:.4f} rad = {np.degrees(expected_angles[-1]):.2f}°")
    print(f"  Final computed angle: {computed_angles_unwrapped[-1]:.4f} rad = {np.degrees(computed_angles_unwrapped[-1]):.2f}°")


if __name__ == "__main__":
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("\n" + "="*70)
    print("ROTATION MATRIX DIRECT VERIFICATION")
    print("="*70)
    
    omega = 1.0
    
    # Test 1: Rotation matrix at specific times
    print("\nTest 1: Direct matrix verification at specific times")
    print("-"*70)
    test_rotation_matrix_directly(omega, test_times=[0, 0.25, 0.5, 0.75, 1.0])
    
    # Test 2: Continuous angle tracking
    print("\n\nTest 2: Continuous angle tracking over time")
    print("-"*70)
    test_continuous_angle_tracking(omega, duration=3.0, n_points=200)
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - test_output/rotation_matrix_test.png")
    print("  - test_output/continuous_angle_tracking.png")
    print("\nThe rotation matrices should be orthogonal (det=1, R^T R = I)")
    print("and the angle should increase linearly: θ(t) = 2π×ω×t")
    print("="*70)
