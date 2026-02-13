# Peak Inflection Timing Test Results

## Summary

**The 1/(4×ω) relationship is VERIFIED in the sanitized experiment.**

All three test cases passed with errors < 0.2%, confirming that the theoretical prediction holds when:
- Rotation model is applied correctly ✓
- Projection formula is applied correctly ✓
- Peak extraction is done properly ✓

## Test Results

### Test 1: ω = 1.0, Duration = 2.0 s
- Expected quarter-period: **0.2500 s**
- Measured mean: **0.2505 s**
- **Error: 0.20%** ✓ PASS

### Test 2: ω = 0.5, Duration = 4.0 s
- Expected quarter-period: **0.5000 s**
- Measured mean: **0.4998 s**
- **Error: 0.04%** ✓ PASS

### Test 3: ω = 2.0, Duration = 1.0 s
- Expected quarter-period: **0.1250 s**
- Measured mean: **0.1253 s**
- **Error: 0.20%** ✓ PASS

## Key Observations

1. **Both detection methods agree**: The 'peaks' and 'derivative' methods find the same inflection points and give consistent results.

2. **Errors are tiny**: All errors < 1% on individual intervals, and mean errors < 0.2%. These are essentially numerical/sampling artifacts.

3. **Consistent across different ω values**: The relationship holds for ω = 0.5, 1.0, and 2.0.

4. **Visual confirmation**: The heatmaps clearly show the quarter-period structure, with vertical lines at 1/(4ω) intervals aligning with the oscillation patterns.

## Physical Interpretation

The inflection points (local max/min in peak values) occur when:
- **Local maxima**: The anisotropic Gaussian is aligned with the projection direction (maximum attenuation)
- **Local minima**: The Gaussian is perpendicular to the projection direction (minimum attenuation)

With angular velocity θ(t) = 2π×ω×t:
- Quarter rotation (90°) takes time: Δt = 1/(4×ω)
- This matches the spacing between consecutive extrema

## Implications for Your Stability Experiments

Since the sanitized test **passes**, the issue in your stability experiments is likely due to:

### Possible Causes of Discrepancy:

1. **Multiple Gaussians interfering**: In your real experiments, you have multiple rotating Gaussians whose peaks may overlap or interfere, making it harder to track individual Gaussian inflection points.

2. **Trajectory motion**: In the sanitized test, the Gaussian is stationary (only rotating). In your experiments, the Gaussians are also translating (projectile motion), which could complicate the peak tracking.

3. **Peak assignment/tracking errors**: The algorithm that assigns peaks to specific Gaussians over time may be making errors, especially if:
   - Gaussians cross paths
   - Peaks merge or split
   - The Hungarian algorithm assignment is suboptimal

4. **Initialization or optimization artifacts**: If the fitted ω values are incorrect, you'd naturally see the wrong period.

5. **Anisotropy ratio**: If the anisotropy is weak (σ_major ≈ σ_minor), the oscillations will be small and harder to detect reliably.

6. **Sampling rate**: If time sampling is too coarse, you might miss or misplace the inflection points.

## Next Steps

To diagnose the issue in your actual experiments:

1. **Check individual Gaussian tracking**: Verify that peaks are correctly assigned to the same Gaussian over time (not mixing up different Gaussians).

2. **Test with simpler scenarios first**:
   - Single Gaussian with translation + rotation
   - Two Gaussians with different ω values
   - Gradually add complexity

3. **Visualize peak assignments**: Plot which peaks are assigned to which Gaussian over time (color-coded).

4. **Compare fitted vs true ω**: In synthetic experiments where you know the true ω, check if the discrepancy exists there too.

5. **Check the anisotropy of fitted Gaussians**: If optimization converges to nearly isotropic Gaussians, rotation won't produce visible oscillations.

## Conclusion

✅ The rotation model works correctly  
✅ The projection formula works correctly  
✅ The 1/(4×ω) theoretical prediction is validated  

The issue must be in how these components interact with other aspects of your pipeline (multiple Gaussians, trajectory estimation, peak tracking, or optimization).
