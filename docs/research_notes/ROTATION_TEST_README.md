# Rotation Model Testing Scripts

This directory contains tests to verify the rotation model implementation in `models.py`.

## Files

### Test Scripts

1. **`test_rotation_animation.py`**
   - Creates animations and snapshots of rotating anisotropic Gaussians
   - Visualizes covariance matrix rotation over time
   - Generates:
     - `test_output/rotation_animation.gif` - Animated rotation
     - `test_output/rotation_snapshots_omega_*.png` - Key time points

2. **`test_rotation_matrix_direct.py`**
   - Direct verification of rotation matrix properties
   - Tests orthogonality, determinant, and angle extraction
   - Tracks continuous angle evolution
   - Generates:
     - `test_output/rotation_matrix_test.png` - Matrix visualization
     - `test_output/continuous_angle_tracking.png` - Angle vs time

### Results

**`ROTATION_VERIFICATION_RESULTS.md`** - Summary of verification results

## Running the Tests

```bash
# Create animation and snapshots
python test_rotation_animation.py

# Run direct matrix verification
python test_rotation_matrix_direct.py
```

## Key Findings

✅ The rotation model correctly implements θ(t) = 2π×ω×t  
✅ Rotation matrices are orthogonal with determinant = 1  
✅ When ω×t = n (integer), exactly n complete rotations occur  
✅ Angle tracking has < 4e-17 radians error (machine precision)

## Parameters You Can Modify

In `test_rotation_animation.py`:
- `omega`: Angular velocity (rotations per unit time)
- `duration`: Animation length
- `fps`: Frames per second
- `sigma_major`, `sigma_minor`: Ellipse shape
- `initial_angle_deg`: Starting orientation

In `test_rotation_matrix_direct.py`:
- `omega`: Angular velocity
- `test_times`: Specific times to test
- `duration`: Duration for continuous tracking

## Understanding ω×t

The key relationship is:
- **ω×t = 0.25** → 90° rotation (quarter turn)
- **ω×t = 0.50** → 180° rotation (half turn)
- **ω×t = 0.75** → 270° rotation (three-quarter turn)
- **ω×t = 1.00** → 360° rotation (complete rotation back to start)
- **ω×t = 2.00** → 720° rotation (two complete rotations)

Watch the animation to visually confirm this behavior!
