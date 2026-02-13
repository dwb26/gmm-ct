# Rotation Model Verification Results

## Summary

The rotation model in `models.py` has been verified and is **working correctly**.

## Test Results

### 1. Rotation Matrix Properties
All rotation matrices satisfy:
- **Determinant = 1.0** (proper rotation, no reflection)
- **Orthogonality**: R^T R = I (preserves distances and angles)
- **Angle accuracy**: Mean error < 4e-17 radians (essentially machine precision)

### 2. Angular Velocity Formula
The model correctly implements:
```
θ(t) = 2π × ω × t
```

Where:
- ω is the angular velocity parameter
- When ω×t = n (integer), the object completes exactly n full rotations
- The rotation angle increases linearly with time

### 3. Test Cases Verified

#### At ω = 1.0:
- t = 0.00 s: ω×t = 0.00 → 0° rotation ✓
- t = 0.25 s: ω×t = 0.25 → 90° rotation ✓
- t = 0.50 s: ω×t = 0.50 → 180° rotation ✓
- t = 0.75 s: ω×t = 0.75 → 270° rotation ✓
- t = 1.00 s: ω×t = 1.00 → 360° = 0° (complete rotation) ✓

### 4. Visual Verification

The following files have been generated for visual inspection:

1. **`test_output/rotation_animation.gif`**
   - Animation showing continuous rotation with ω = 0.5
   - Duration: 4 seconds → 2 complete rotations
   - Red lines mark when ω×t reaches integer values
   - Yellow highlighting indicates near-integer ω×t values

2. **`test_output/rotation_snapshots_omega_1.000.png`**
   - Snapshots of ellipse at key time points
   - Shows anisotropic Gaussian orientation at different rotation phases

3. **`test_output/rotation_matrix_test.png`**
   - Direct visualization of rotation matrices and their effect on test points
   - Shows matrix elements at different times

4. **`test_output/continuous_angle_tracking.png`**
   - Plot of angle vs time (both wrapped and unwrapped)
   - Confirms linear relationship between angle and time
   - Green lines mark complete rotations (ω×t = integers)

## Angle Extraction Ambiguity

**Important Note**: When extracting angles from eigendecomposition of covariance matrices, there's an inherent 180° ambiguity because ellipses are symmetric. The angle returned by `arctan2(eigenvector_y, eigenvector_x)` is in the range [-180°, 180°], which can cause apparent discontinuities.

However, this is **not a problem with the rotation model itself** - the rotation matrices are correct. The ambiguity only affects how we visualize the orientation, not the actual physics.

## Conclusion

The rotation model in `models.py` using the formula:
```python
angle = 2 * torch.pi * omega * t
rot_mat[0, 0] = torch.cos(angle)
rot_mat[0, 1] = -torch.sin(angle)
rot_mat[1, 0] = torch.sin(angle)
rot_mat[1, 1] = torch.cos(angle)
```

is **mathematically correct** and produces proper rotation matrices that:
1. Preserve the properties of rotations (orthogonality, determinant = 1)
2. Rotate objects by the expected angle θ = 2π×ω×t
3. Complete exactly n full rotations when ω×t = n

## Recommendation

If you're still seeing issues in your experiments, they are likely due to:
1. How angles are being extracted/interpreted from covariance matrices
2. Initial conditions or boundary effects
3. Numerical precision in the optimization
4. Other parts of the pipeline (not the rotation model itself)

The rotation model implementation is correct per the specified angular velocity formula.
