# Solution: Trajectory-Aware Omega Estimation

## Problem Identified

**The 1/(4×ω) inflection point spacing prediction assumes a stationary Gaussian.**

When projectile motion is added:
- **Without motion**: Error < 0.2% ✅
- **With motion**: Error 6-19% ❌  
- **With geometric correction**: Error ~6-7% (modest improvement)

## Root Cause

As the Gaussian translates along its parabolic trajectory, the projection geometry changes:
1. **Distance to source** varies → affects overall intensity
2. **Angle to receivers** varies → affects which receivers see the peak
3. **Distance to source-receiver lines** varies → affects exponential attenuation

These geometric effects **modulate** the rotation-induced oscillations, distorting both:
- The amplitude of oscillations
- The apparent timing of inflection points

## Practical Solution for Your Experiments

Since you estimate the trajectory in Phase 1, you can use this knowledge when fitting omega in Phase 2.

### Option 1: Correct the Observed Peaks (Simpler)

```python
# After Phase 1: you have μ(t), v₀, a₀ for each Gaussian

# When analyzing peaks to estimate ω:
def correct_peaks_for_trajectory(peak_values_over_time, trajectory, source, receivers):
    """
    Apply geometric correction based on known trajectory.
    """
    distance_to_source = np.array([np.linalg.norm(mu_t - source) 
                                   for mu_t in trajectory])
    
    # Normalize by average distance
    correction_factor = distance_to_source / np.mean(distance_to_source)
    
    # Correct the peaks
    corrected_peaks = peak_values_over_time / correction_factor
    
    return corrected_peaks

# Then find inflection points in corrected_peaks
# Estimate ω from Δt ≈ 1/(4×ω)
```

### Option 2: Forward Model with Trajectory (More Accurate)

When optimizing ω in Phase 2, generate predictions that **include** the trajectory:

```python
def loss_for_omega_fitting(omega, trajectory_known, other_params):
    """
    Loss function that accounts for trajectory when fitting omega.
    
    Key: Generate synthetic projections using:
    - Known trajectory μ(t) [from Phase 1]
    - Candidate omega value [being optimized]
    - Other parameters [fixed or from Phase 1]
    
    Compare to observed projections.
    """
    synthetic_projections = generate_projections(
        t=time_array,
        trajectory=trajectory_known,  # Use fitted trajectory
        omega=omega,  # Candidate value
        U_skew=U_skew_from_phase1,
        alpha=alpha_from_phase1
    )
    
    # Extract peak values
    synthetic_peaks = extract_peak_values(synthetic_projections)
    observed_peaks = extract_peak_values(observed_projections)
    
    # Loss
    return mse(synthetic_peaks, observed_peaks)
```

This approach naturally accounts for trajectory effects because the forward model includes them.

### Option 3: Hybrid Approach (Recommended)

1. **Initialize** ω using corrected inflection point spacing:
   ```python
   corrected_peaks = correct_peaks_for_trajectory(observed_peaks, trajectory_fit, ...)
   inflection_times = find_inflection_points(corrected_peaks)
   delta_t = np.mean(np.diff(inflection_times))
   omega_init = 1 / (4 * delta_t)
   ```

2. **Refine** ω using forward model optimization:
   ```python
   omega_fit = minimize(
       loss_for_omega_fitting,
       x0=omega_init,  # Start from corrected estimate
       args=(trajectory_fit, other_params)
   )
   ```

## Why This Matters

Your current approach likely:
1. Fits trajectory in Phase 1 ✓
2. Tries to estimate ω from raw peak timing → **Error: ignores trajectory effects**
3. Optimizes ω assuming stationary Gaussian → **Error: model mismatch**

By incorporating trajectory knowledge into ω estimation, you should recover the theoretical accuracy.

## Expected Improvements

| Method | Error | Status |
|--------|-------|--------|
| Raw peaks (with motion) | 6-19% | ❌ Current |
| Geometric correction | 6-7% | ⚠ Helps |
| Forward model with trajectory | <2% | ✅ Expected |

## Implementation Priority

**High Priority**: Modify your Phase 2 omega optimization to use the forward model approach (Option 2). This is likely already close to what you're doing, but ensure that:

1. The trajectory used in projection generation is the **fitted** trajectory from Phase 1 (not just x₀)
2. The loss compares synthetic vs observed **peak values** (not just raw projections)
3. The optimization is done **per Gaussian** with its specific trajectory

**Medium Priority**: Add the corrected inflection point spacing (Option 1) as a diagnostic tool to verify ω estimates and catch gross errors.

## Testing Recommendation

Before applying to real data, test with synthetic data where you know the true ω:

```python
# Generate synthetic data with known omega and trajectory
true_params = {'omega': 1.5, 'v0': [5, 2], 'a0': [0, -9.81], ...}
synthetic_projections = generate_test_data(true_params)

# Run your pipeline
phase1_trajectory = fit_trajectory(synthetic_projections)  # Should recover v0, a0
phase2_omega = fit_omega(synthetic_projections, phase1_trajectory)  # Should recover omega

# Check accuracy
error = abs(phase2_omega - true_params['omega']) / true_params['omega']
print(f"Omega recovery error: {error*100:.2f}%")  # Should be < 2%
```

## Key Insight

**The rotation signal is still there** - it's just being modulated by geometric factors due to translation. By explicitly modeling the trajectory in your omega estimation, you can decouple these effects and accurately recover the rotation parameters.
