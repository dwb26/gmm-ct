import numpy as np

# The issue: DTW uses FIXED U_skew [[30, 5], [0, 20]] but true morphology is different

# Simulate typical true velocity
v0_true = np.array([0.7, 0.8])
v_norm = np.linalg.norm(v0_true)
v_hat = v0_true / v_norm
v_perp = np.array([-v_hat[1], v_hat[0]])

print(f"True velocity: {v0_true}, direction: {v_hat}")
print(f"True velocity angle: {np.degrees(np.arctan2(v_hat[1], v_hat[0])):.1f}°\n")

# TRUE morphology (velocity-aligned, what data is actually generated from)
major_scale = 15.0  # Small precision → large covariance → major axis
minor_scale = 30.0  # Large precision → small covariance → minor axis
U_true = np.column_stack([v_hat * major_scale, v_perp * minor_scale])
precision_true = U_true.T @ U_true
cov_true = np.linalg.inv(precision_true)

print("TRUE U_skew (velocity-aligned):")
print(U_true)
print("\nTRUE covariance:")
print(cov_true)
eigenvals_true = np.linalg.eigvalsh(cov_true)
print(f"TRUE eigenvalues: {eigenvals_true}")
print(f"TRUE aspect ratio: {eigenvals_true[1] / eigenvals_true[0]:.2f}:1")

# Get eigenvectors to see orientation
eigenvals, eigenvecs = np.linalg.eigh(cov_true)
major_axis_angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
print(f"TRUE major axis angle: {major_axis_angle:.1f}°\n")

print("="*60)

# INITIALIZED morphology (what DTW uses for prediction)
U_init = np.array([[30.0, 5.0], [0, 20.0]])
precision_init = U_init.T @ U_init
cov_init = np.linalg.inv(precision_init)

print("\nINITIALIZED U_skew (fixed):")
print(U_init)
print("\nINITIALIZED covariance:")
print(cov_init)
eigenvals_init = np.linalg.eigvalsh(cov_init)
print(f"INITIALIZED eigenvalues: {eigenvals_init}")
print(f"INITIALIZED aspect ratio: {eigenvals_init[1] / eigenvals_init[0]:.2f}:1")

eigenvals, eigenvecs = np.linalg.eigh(cov_init)
major_axis_angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
print(f"INITIALIZED major axis angle: {major_axis_angle:.1f}°\n")

print("="*60)
print("\n⚠️  PROBLEM: DTW uses INITIALIZED morphology to predict peaks,")
print("   but TRUE data has DIFFERENT morphology (orientation, aspect ratio).")
print("   This causes peak modulation patterns to mismatch!")
print("\n💡 SOLUTION: Use velocity-aligned U_skew initialization for DTW")
