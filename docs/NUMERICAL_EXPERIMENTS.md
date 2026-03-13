# Numerical Experiments — Planning Document

> Reference document for wrapping up GMM-CT for a top-level journal submission.
> Current baseline: N = 5, seeds 1–10, ω ∈ [2, 6] Hz, 65 time steps, 1.5 s duration,
> single fan-beam source, noiseless projections.

---

## Status Summary

| Experiment | Infrastructure | Results |
|---|---|---|
| Noiseless batch (N=5, seeds 1–10) | ✅ Complete | ✅ Complete |
| N-scaling study | ✅ Code exists | ❌ No results |
| Noise robustness | ❌ Needs implementation | ❌ No results |
| Sparse projections | ❌ Needs config sweep | ❌ No results |
| Baseline comparison | ❌ Needs implementation | ❌ No results |
| Shape/α quantification | ✅ Data exists | ❌ Not extracted |
| ω-range / identifiability | ❌ Needs config sweep | ❌ No results |
| Trajectory separation | ❌ Needs config sweep | ❌ No results |

---

## Experiment 1 — Noise Robustness (Critical)

**Motivation:** All current results use noiseless projections. No top journal will accept
an inverse problems paper without a noise study. This is the single most important
missing experiment.

**Noise model:** Poisson (physically correct for X-ray photon counting):
$$y_{t,r} \sim \text{Poisson}(I_0 \cdot e^{-p_{t,r}})$$
where $p_{t,r}$ is the clean line-integral. Equivalently, the noisy log-sinogram is:
$$\tilde{p}_{t,r} = -\log(y_{t,r} / I_0)$$
Add Gaussian additive noise as a secondary (simpler) variant for comparison.

**Sweep:**
- $I_0 \in \{10^2, 5\times10^2, 10^3, 5\times10^3, 10^4, 10^5, \infty\}$ (noiseless)
- Fixed: N = 5, seeds 1–10, 65 time steps, 1.5 s

**Metrics per noise level:**
- ω RMSE across all Gaussians and seeds
- Relative ω error: $|\hat\omega_k - \omega_k^*| / |\omega_k^*|$
- Shape Frobenius error: $\|U_k - \hat{U}_k\|_F / \|U_k\|_F$
- Attenuation relative error: $|\hat\alpha_k - \alpha_k^*| / \alpha_k^*$
- Supremum projection residual (in-distribution check)
- Fraction of seeds where Hungarian assignment is correct

**Implementation notes:**
- Add a `noise: poisson_I0: <float>` field to `configs/simulate.yaml`
- Inject noise in `gmm_ct/simulation.py` after computing projections, before saving
- Run reconstruction unchanged (reconstruction does not know noise level)

**Expected output:** A table + log-scale plot of ω RMSE vs $I_0$. A critical threshold
$I_0^*$ at which accuracy degrades significantly is the key result.

---

## Experiment 2 — N-Scaling Study (High priority)

**Motivation:** Infrastructure exists in `experiments/stability/` but no results CSV has
been generated. This shows scalability and is central to the paper's claims.

**Sweep:**
- $N \in \{1, 2, 3, 5, 7, 10\}$, 10 seeds per value of N
- Fixed: 65 time steps, 1.5 s, noiseless

**Metrics per N:**
- ω RMSE (mean ± std across seeds and Gaussians)
- Shape Frobenius error (mean ± std)
- Attenuation error (mean ± std)
- Wall-clock reconstruction time (seconds)
- Fraction of seeds with correct Hungarian assignment
- Fraction of seeds that converge (projection residual < threshold)

**Implementation notes:**
- Use `scripts/run_experiments.py` with `--N_values 1 2 3 5 7 10 --seeds 1..10`
- Existing `experiments/stability/stability_experiment.py` already implements this sweep
- Output: `data/results/scaling_study/batch_summary.csv`

**Expected output:** A table of accuracy vs N, a runtime-vs-N plot (expect roughly
$O(N^2)$ due to Hungarian), and a convergence fraction plot.

---

## Experiment 3 — Sparse Projections Study (High priority)

**Motivation:** In real experiments the number of projection angles (time steps) is
limited. This reveals the minimum data requirement for reliable reconstruction and
connects to Nyquist-type informational limits.

**Sweep:**
- $T \in \{10, 15, 20, 30, 45, 65, 100\}$ time steps, fixed duration 1.5 s
- Fixed: N = 5, seeds 1–10, noiseless

**Theoretical lower bound to motivate:** To resolve a rotation at frequency ω Hz over
duration $\tau$ seconds, the object completes $\omega\tau$ full rotations. At ω = 2 Hz,
τ = 1.5 s → 3 full rotations observable. The Stage 1.5 model-fit needs enough temporal
samples to fit the two-frequency model $g_k(t)^{-2} = c_0 + c_1\cos(\xi) + c_2\sin(\xi)$.
Minimum: ~5–8 samples per half-period. This motivates $T \gtrsim 20$ as a lower bound.

**Metrics:** Same as Experiment 1.

**Expected output:** A plot of ω RMSE vs T with a clear "knee" showing the minimum
adequate projection count. This T* value is a key practical recommendation of the paper.

---

## Experiment 4 — Baseline Comparison (Critical)

**Motivation:** Without a competing method, reviewers will reject the paper. Need at
least one meaningful baseline showing what is lost by ignoring rotation and/or coupling.

### Baseline A — Static Per-Frame Reconstruction
For each time step $t$ independently, fit N anisotropic Gaussians to the single
projection without enforcing trajectory continuity or rotation coupling.
- Shows: the benefit of temporal coupling and the joint model
- Expected failure mode: shape estimates are highly noisy; ω is unrecoverable

### Baseline B — No-Rotation Model
Run the full pipeline but constrain $\omega_k \equiv 0$ for all $k$ (isotropic projection
assumed static in shape). Measure the projection residual relative to the true rotating
model.
- Shows: how much of the signal is attributable to rotation
- Already partially implemented: Stage 1 uses isotropic Gaussians — extend this

### Baseline C — Isotropic Shape Recovery
Run the pipeline with $U_k = \sigma_k I$ (scalar width only, no skew/anisotropy). Measure
ω and trajectory estimation quality.
- Shows: that anisotropy is required to identify ω at all

**Metrics for baselines:** Projection residual, ω RMSE (for Baseline B/C where ω is
estimated), shape error, computation time.

---

## Experiment 5 — Shape and Attenuation Recovery (Low effort, high value)

**Motivation:** Current batch results (seeds 1–10, N=5) already contain all the data;
it just has not been extracted systematically. This can be done immediately.

**Tasks:**
- Post-process each `data/results/*/` directory
- Extract $\|U_k - \hat{U}_k\|_F / \|U_k\|_F$ and $|\hat\alpha_k - \alpha_k^*|/\alpha_k^*$
  for all k, all seeds
- Add columns to `data/results/batch_summary.csv`
- Report mean ± std in a parameter recovery table

**Expected output:** A 3-column summary table (ω error, shape error, α error) across
seeds, plus per-Gaussian breakdown. Should be straightforward given the saved `.pt` files.

---

## Experiment 6 — ω Range / Identifiability Limits (Medium priority)

**Motivation:** All current experiments use ω ∈ [2, 6] Hz. How does reconstruction
degrade for slow (< 1 Hz) or fast (> 12 Hz) rotations? This reveals identifiability
limits and connects to sampling theory.

**Sweep:**
- Range A: ω ∈ [0.5, 1.5] Hz (slow — < 1 full rotation in 1.5 s for smallest ω)
- Range B: ω ∈ [2.0, 6.0] Hz (current baseline)
- Range C: ω ∈ [6.0, 12.0] Hz (fast)
- Range D: ω ∈ [12.0, 20.0] Hz (very fast — near-Nyquist aliasing regime)
- Fixed: N = 5, seeds 1–10, 65 time steps

**Key question:** At what ω does accuracy degrade? Is degradation gradual or a sharp
cliff? The near-aliasing guard-band in `generate_true_param` means the hardest cases
are the near-boundary ω values.

**Expected output:** A plot of ω RMSE vs true ω grouped by range, showing the
identifiable regime.

---

## Experiment 7 — Trajectory Separation and Occlusion (Medium priority)

**Motivation:** All Gaussians start at $x_0 = [1, 1]$ and separate purely by velocity.
Near-colliding trajectories will confuse the Hungarian-algorithm assignment in Stage 1.

**Sweep:** Vary the **initial velocity spread** parameter (currently hardcoded in
`generators.py` as Uniform(0.75, 2.25) horizontal, Uniform(−2.25, 2.25) vertical).
Test:
- Wide spread (current)
- Medium spread (halved range)
- Narrow spread (all near-parallel trajectories — near-occlusion regime)

**Metric:** Hungarian assignment accuracy (fraction of seeds where all K assignments
are correct), ω RMSE conditional on correct assignment.

**Expected output:** Shows robustness of Stage 1 and the regime where the pipeline
needs a better initialisation strategy.

---

## Suggested Execution Order

1. **Now:** Experiment 5 — post-process existing results. Zero compute cost.
2. **Next:** Experiment 2 — N-scaling. Infrastructure ready; just run.
3. **Then:** Experiment 3 — sparse projections. Only config changes needed.
4. **Parallel with above:** Add Poisson noise injection (Experiment 1 set-up).
5. **After noise injection is ready:** Run Experiment 1 sweep (most compute-intensive).
6. **Alongside:** Implement Baseline A (Experiment 4). Needed early for reviewers.
7. **Later:** Experiments 6 and 7 as time permits.

---

## Target Venues

| Venue | Scope | Impact |
|---|---|---|
| *SIAM Journal on Imaging Sciences* | Computational imaging, inverse problems | High |
| *Inverse Problems* (IOP) | Mathematical inverse problems | High |
| *IEEE Transactions on Computational Imaging* | Engineering + algorithms | High |
| *Journal of Computational Physics* | Physics + numerics | Medium-high |

All four require: noise study, baseline comparison, and a clear mathematical formulation
section. *SIAM J. Imaging* and *Inverse Problems* will also expect identifiability
analysis (Experiment 6).

---

## Notes on the "Numerical Results" Section

See the main write-up notes below for structuring the section. Key principles:

- **Lead with a representative example** — a single seed, show sinograms,
  trajectory overlay, and parameter table side-by-side. This is the figure reviewers
  look at first.
- **Separate accuracy from robustness** — accuracy (noiseless) in one subsection,
  noise robustness in another.
- **Tables for numbers, figures for trends** — don't put a sweep result in a table;
  show it as a curve. Reserve tables for the final parameter-error summary.
- **Always report error bars** — mean ± std across seeds, not just mean.
- **Cite compute environment** — CPU/GPU, RAM, average runtime per run.
