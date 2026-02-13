"""
Test script to visualize and verify rotation model behavior.

This script creates an animation of an anisotropic Gaussian rotating about 
its fixed origin using the rotation model from models.py. The animation 
helps verify that when omega*t is an integer, we observe exactly one full 
rotation as expected from angular velocity = 2*pi*omega*t.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches

def construct_rotation_matrix(t, omega, device='cpu'):
    """
    Construct a 2D rotation matrix using the same formulation as in models.py.
    
    Angular position: theta(t) = 2*pi*omega*t
    When omega*t = n (integer), we should have completed n full rotations.
    
    Parameters:
    - t: time value (will be converted to tensor)
    - omega: angular velocity parameter (will be converted to tensor)
    - device: torch device
    
    Returns:
    - 2x2 rotation matrix
    """
    # Convert to tensors if needed
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


def create_anisotropic_covariance(sigma_major=2.0, sigma_minor=0.5, initial_angle_deg=30.0):
    """
    Create an initial anisotropic covariance matrix (ellipse).
    
    Parameters:
    - sigma_major: standard deviation along major axis
    - sigma_minor: standard deviation along minor axis  
    - initial_angle_deg: initial rotation angle in degrees
    
    Returns:
    - 2x2 covariance matrix
    """
    # Create diagonal covariance in canonical orientation
    Lambda = np.diag([sigma_major**2, sigma_minor**2])
    
    # Initial rotation matrix
    theta_rad = np.deg2rad(initial_angle_deg)
    R0 = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    
    # Rotated covariance: R0 @ Lambda @ R0^T
    Sigma0 = R0 @ Lambda @ R0.T
    
    return Sigma0


def rotate_covariance(Sigma0, t, omega, device='cpu'):
    """
    Rotate the initial covariance matrix using the rotation model.
    
    Sigma(t) = R(t) @ Sigma0 @ R(t)^T
    where R(t) is the rotation matrix at time t.
    
    Parameters:
    - Sigma0: initial covariance matrix (2x2 numpy array)
    - t: time value
    - omega: angular velocity parameter
    - device: torch device
    
    Returns:
    - Rotated covariance matrix as numpy array
    """
    # Convert to torch
    Sigma0_torch = torch.tensor(Sigma0, dtype=torch.float64, device=device)
    
    # Get rotation matrix
    R_t = construct_rotation_matrix(t, omega, device)
    
    # Apply rotation: Sigma(t) = R(t) @ Sigma0 @ R(t)^T
    Sigma_t = R_t @ Sigma0_torch @ R_t.T
    
    return Sigma_t.cpu().numpy()


def get_ellipse_params(Sigma):
    """
    Extract ellipse parameters from covariance matrix.
    
    Parameters:
    - Sigma: 2x2 covariance matrix
    
    Returns:
    - width, height, angle (in degrees) for matplotlib Ellipse
    
    Note: The angle returned by arctan2 is in the range [-180, 180], 
    which can cause discontinuities when tracking rotation angle.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Width and height (2 standard deviations for visualization)
    width = 2 * np.sqrt(eigenvalues[0])
    height = 2 * np.sqrt(eigenvalues[1])
    
    # Angle of major axis
    # Note: this returns angle in [-180, 180] range
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    return width, height, angle


def create_rotation_animation(omega, duration, fps=30, output_file='rotation_test.gif'):
    """
    Create an animation showing the rotation of an anisotropic Gaussian.
    
    Parameters:
    - omega: angular velocity parameter (when omega*t=1, we should see 1 full rotation)
    - duration: total duration in seconds
    - fps: frames per second
    - output_file: output filename for the animation
    """
    device = 'cpu'
    
    # Create initial anisotropic covariance
    Sigma0 = create_anisotropic_covariance(sigma_major=2.0, sigma_minor=0.5, initial_angle_deg=30.0)
    
    # Time array
    n_frames = int(duration * fps)
    t_array = np.linspace(0, duration, n_frames)
    
    # Calculate omega*t for each frame to track rotations
    omega_t_array = omega * t_array
    
    # Setup the figure
    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Main plot: the rotating ellipse
    ax_main.set_xlim(-3, 3)
    ax_main.set_ylim(-3, 3)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='k', linewidth=0.5)
    ax_main.axvline(x=0, color='k', linewidth=0.5)
    ax_main.set_xlabel('x', fontsize=14)
    ax_main.set_ylabel('y', fontsize=14)
    ax_main.set_title('Rotating Anisotropic Gaussian', fontsize=16)
    
    # Add reference axes showing the initial orientation
    width0, height0, angle0 = get_ellipse_params(Sigma0)
    ellipse_ref = Ellipse((0, 0), width0, height0, angle=angle0, 
                         facecolor='none', edgecolor='gray', 
                         linestyle='--', linewidth=1.5, alpha=0.5)
    ax_main.add_patch(ellipse_ref)
    
    # The rotating ellipse (will be updated)
    ellipse = Ellipse((0, 0), width0, height0, angle=angle0,
                     facecolor='blue', edgecolor='darkblue', 
                     alpha=0.3, linewidth=2)
    ax_main.add_patch(ellipse)
    
    # Info plot: tracking omega*t
    ax_info.set_xlim(0, duration)
    ax_info.set_ylim(-0.1, max(3, omega * duration + 0.5))
    ax_info.grid(True, alpha=0.3)
    ax_info.set_xlabel('Time (s)', fontsize=14)
    ax_info.set_ylabel('ω×t (rotations)', fontsize=14)
    ax_info.set_title('Rotation Progress', fontsize=16)
    
    # Add horizontal lines for integer rotations
    for n in range(int(omega * duration) + 2):
        ax_info.axhline(y=n, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Progress line
    line, = ax_info.plot([], [], 'b-', linewidth=2)
    current_point, = ax_info.plot([], [], 'ro', markersize=10)
    
    # Text annotations
    time_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    rotation_text = ax_main.text(0.02, 0.88, '', transform=ax_main.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    angle_text = ax_main.text(0.02, 0.78, '', transform=ax_main.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def init():
        """Initialize animation."""
        line.set_data([], [])
        current_point.set_data([], [])
        return ellipse, line, current_point, time_text, rotation_text, angle_text
    
    def animate(frame):
        """Update function for animation."""
        t = t_array[frame]
        omega_t = omega_t_array[frame]
        
        # Rotate the covariance matrix
        Sigma_t = rotate_covariance(Sigma0, t, omega, device)
        
        # Get ellipse parameters
        width, height, angle = get_ellipse_params(Sigma_t)
        
        # Update ellipse
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle
        
        # Update progress plot
        line.set_data(t_array[:frame+1], omega_t_array[:frame+1])
        current_point.set_data([t], [omega_t])
        
        # Update text
        time_text.set_text(f'Time: {t:.4f} s')
        rotation_text.set_text(f'ω×t = {omega_t:.4f} rotations')
        
        # Calculate the actual angle rotated (in degrees)
        total_angle = (2 * np.pi * omega * t) * (180 / np.pi)
        angle_text.set_text(f'Total angle: {total_angle:.1f}°')
        
        # Highlight when we're at an integer rotation
        if abs(omega_t - round(omega_t)) < 0.02:  # Within 2% of integer
            rotation_text.set_bbox(dict(boxstyle='round', facecolor='yellow', alpha=0.9))
        else:
            rotation_text.set_bbox(dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return ellipse, line, current_point, time_text, rotation_text, angle_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save animation
    print(f"Creating animation with ω = {omega}")
    print(f"Duration: {duration} s, Total rotations: {omega * duration}")
    print(f"Saving to {output_file}...")
    
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")
    
    plt.close()
    
    return anim


def test_rotation_at_specific_times(omega, test_times=[0, 0.5, 1.0, 1.5, 2.0]):
    """
    Test rotation at specific time points and display the results.
    
    This verifies that the rotation model behaves correctly at known time points.
    
    Parameters:
    - omega: angular velocity parameter
    - test_times: list of time values to test (in units where omega*t gives rotations)
    """
    device = 'cpu'
    
    # Create initial covariance
    Sigma0 = create_anisotropic_covariance(sigma_major=2.0, sigma_minor=0.5, initial_angle_deg=0.0)
    
    print(f"\nTesting rotation model with ω = {omega}")
    print("="*60)
    print(f"Angular velocity: θ(t) = 2π×ω×t = {2*np.pi*omega:.4f}×t rad/s")
    print("="*60)
    
    n_tests = len(test_times)
    fig, axes = plt.subplots(1, n_tests, figsize=(4*n_tests, 4))
    
    if n_tests == 1:
        axes = [axes]
    
    for i, t in enumerate(test_times):
        ax = axes[i]
        
        # Rotate covariance
        Sigma_t = rotate_covariance(Sigma0, t, omega, device)
        
        # Get the rotation matrix directly for comparison
        R_t = construct_rotation_matrix(t, omega, device).cpu().numpy()
        
        # Get ellipse parameters
        width, height, angle = get_ellipse_params(Sigma_t)
        
        # Expected angle (in degrees)
        expected_angle_rad = 2 * np.pi * omega * t
        expected_angle_deg = expected_angle_rad * (180 / np.pi)
        
        # Normalize expected angle to [-180, 180] for fair comparison
        expected_angle_norm = ((expected_angle_deg + 180) % 360) - 180
        
        # Plot
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        # Reference (initial) ellipse
        ellipse_ref = Ellipse((0, 0), width, height, angle=0,
                             facecolor='none', edgecolor='gray',
                             linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_patch(ellipse_ref)
        
        # Rotated ellipse
        ellipse = Ellipse((0, 0), width, height, angle=angle,
                         facecolor='blue', edgecolor='darkblue',
                         alpha=0.3, linewidth=2)
        ax.add_patch(ellipse)
        
        # Add arrow showing direction of major axis
        arrow_length = width / 2
        arrow_x = arrow_length * np.cos(np.deg2rad(angle))
        arrow_y = arrow_length * np.sin(np.deg2rad(angle))
        ax.arrow(0, 0, arrow_x, arrow_y, head_width=0.2, head_length=0.2,
                fc='red', ec='red', linewidth=2, alpha=0.7)
        
        ax.set_title(f't = {t:.2f}\nω×t = {omega*t:.2f}', fontsize=12)
        ax.text(0.02, 0.98, f'Angle: {angle:.1f}°\nExpected: {expected_angle_norm:.1f}°',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Console output
        print(f"\nTime t = {t:.4f} s:")
        print(f"  ω×t = {omega*t:.4f} (rotations)")
        print(f"  Expected angle: {expected_angle_deg:.1f}° (raw), {expected_angle_norm:.1f}° (normalized)")
        print(f"  Computed angle: {angle:.1f}°")
        print(f"  Difference: {abs(angle - expected_angle_norm):.4f}°")
        print(f"  Rotation matrix R(t):")
        print(f"    [{R_t[0,0]:8.5f}, {R_t[0,1]:8.5f}]")
        print(f"    [{R_t[1,0]:8.5f}, {R_t[1,1]:8.5f}]")
    
    plt.tight_layout()
    output_file = f'test_output/rotation_snapshots_omega_{omega:.3f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSnapshot figure saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Create test_output directory if it doesn't exist
    import os
    os.makedirs('test_output', exist_ok=True)
    
    print("="*70)
    print("ROTATION MODEL VERIFICATION TEST")
    print("="*70)
    print("\nThis test verifies that the rotation model correctly implements:")
    print("  θ(t) = 2π×ω×t")
    print("\nWhen ω×t = n (integer), we should observe exactly n full rotations.")
    print("="*70)
    
    # Test 1: Snapshots at specific times
    omega_test = 1.0
    print("\n\nTest 1: Rotation snapshots at specific times")
    print("-"*70)
    test_rotation_at_specific_times(omega_test, test_times=[0, 0.25, 0.5, 0.75, 1.0])
    
    # Test 2: Animation showing continuous rotation
    print("\n\nTest 2: Creating animation of continuous rotation")
    print("-"*70)
    omega_anim = 0.5  # 0.5 rotations per second
    duration = 4.0     # 4 seconds -> should see 2 complete rotations
    
    anim = create_rotation_animation(
        omega=omega_anim,
        duration=duration,
        fps=30,
        output_file='test_output/rotation_animation.gif'
    )
    
    print("\n" + "="*70)
    print("VERIFICATION INSTRUCTIONS:")
    print("="*70)
    print(f"\n1. Check the snapshot figure: test_output/rotation_snapshots_omega_{omega_test:.3f}.png")
    print("   - At ω×t = 0: ellipse at 0° (horizontal)")
    print("   - At ω×t = 0.25: ellipse at 90° (vertical)")
    print("   - At ω×t = 0.5: ellipse at 180° (horizontal, flipped)")
    print("   - At ω×t = 1.0: ellipse at 360° = 0° (back to initial)")
    print(f"\n2. View the animation: test_output/rotation_animation.gif")
    print(f"   - With ω = {omega_anim}, duration = {duration} s")
    print(f"   - Should show exactly {omega_anim * duration} complete rotations")
    print("   - Watch when ω×t reaches integers (marked by red lines)")
    print("   - The ellipse should return to its initial orientation at ω×t = 1, 2, 3...")
    print("\n" + "="*70)
