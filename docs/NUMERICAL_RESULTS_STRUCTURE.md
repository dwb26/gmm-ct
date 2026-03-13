# "Numerical Results" Section — Writing Structure

> Bullet-point skeleton to be fleshed out in prose.
> All quantitative claims below are targets/anticipated results based on
> the existing noiseless baseline (N = 5, seeds 1–10).

---

## §R.1 — Experimental Setup

- State the CT geometry precisely: fan-beam source at $[-1, -1]$, 128 receivers
  on $x_1 = 4$ with $y \in [-3, 1]$, duration $\tau = 1.5$ s, $T = 65$ uniformly
  spaced projection angles.
- State what is assumed known by the algorithm: initial positions $x_0^k$,
  gravitational acceleration $a_0 = [0, -9.81]^\top$, number of components $N$,
  rotation frequency bounds $[\omega_{\min}, \omega_{\max}]$.
- State what is estimated: initial velocities $v_0^k$, angular velocities $\omega_k$,
  shape matrices $U_k$ (upper-triangular Cholesky factors), attenuation
  coefficients $\alpha_k$.
- State the parameter generation ranges (from `gmm_ct/utils/generators.py`):
  $\alpha_k \sim 15 + 5k + \mathcal{N}(0,1)$;
  $U_k$ diagonal entries $\sim \mathrm{Unif}(7.5, 25.5)$, off-diagonal
  $\sim 10 + \mathcal{N}(0,1)$;
  $\omega_k \sim \mathrm{Unif}(2, 6)$ Hz with aliasing guard-band;
  $v_0^k$ horizontal $\sim \mathrm{Unif}(0.75, 2.25)$, vertical
  $\sim \mathrm{Unif}(-2.25, 2.25)$.
- Give the compute environment: hardware (CPU/GPU model), RAM, average
  wall-clock time per reconstruction run.
- Note that all experiments use 10 independent random seeds to assess
  statistical variability, and that results are reported as mean ± standard
  deviation across seeds unless stated otherwise.

---

## §R.2 — Qualitative Illustration

> One representative seed (suggest seed 10). The visual centrepiece of the results.

- **Sinogram comparison figure** (side by side, or difference map):
  observed projections $p^\mathrm{obs}(t, r)$ vs reconstructed
  $p^\mathrm{rec}(t, r)$. Annotate the colourbar with attenuation units.
  A clean difference panel (residual sinogram) demonstrates near-zero error.
- **Trajectory overlay figure**: true and estimated Gaussian centres
  $\mu_k(t)$ as smooth curves in 2D, overlaid on the CT field of view,
  with the source and detector line drawn for geometry context.
- **GMM snapshot figures**: stills of the true and reconstructed Gaussian
  mixture at 3–4 selected time points (e.g., $t = 0, 0.5, 1.0, 1.5$ s),
  showing shape and orientation recovery.
- **Parameter recovery table**: for each Gaussian $k$, list
  $(\omega_k^*, \hat\omega_k, |\Delta\omega|)$,
  $(\alpha_k^*, \hat\alpha_k, |\Delta\alpha|/\alpha_k^*)$,
  $(\|U_k^*\|_F, \|\hat U_k\|_F, \|U_k^* - \hat U_k\|_F / \|U_k^*\|_F)$.
  This table is the primary quantitative result for the single-seed illustration.

---

## §R.3 — Quantitative Accuracy (Noiseless Baseline)

> Aggregate results across all 10 seeds, N = 5.

- **Main result statement**: quote the mean ± std of the relative ω error
  across all Gaussians and seeds (target: < 1%).
- **Parameter recovery table** (3 rows × 3 columns): one row each for
  $\omega$, $\alpha$, $\|U - \hat U\|_F / \|U\|_F$; columns = mean, std, max.
- Note the Hungarian assignment was correct on all 10 seeds (if true).
- Comment briefly on the contribution of each stage to the final accuracy
  (e.g., "Stage 1.5 reduces the ω initialisation error from ~X% to ~Y%,
  enabling Stage 2 to converge reliably").
- **Supremum projection residual**: quote $\max_{t,r}|p^\mathrm{obs} -
  p^\mathrm{rec}|$ as an absolute measure of fit quality.

---

## §R.4 — Scalability Study (N-Scaling)

> Sweep $N \in \{1, 2, 3, 5, 7, 10\}$, 10 seeds per N.

- **Plot**: ω RMSE (mean ± std) vs N on a single axis. Expect a slight
  upward trend as $N$ increases due to trajectory overlap and harder
  assignment.
- **Plot**: wall-clock reconstruction time vs N. Draw attention to the
  $O(N^2)$ Hungarian component.
- **Table**: for each N, list (ω RMSE, shape Frobenius error, α rel. error,
  runtime [s], Hungarian correct fraction). This is the core scalability claim.
- Interpret: identify the largest N at which all 10 seeds converge correctly.
  This is the practical operating range of the method.

---

## §R.5 — Minimum Data Requirements (Sparse Projections)

> Sweep $T \in \{10, 15, 20, 30, 45, 65, 100\}$, fixed $\tau = 1.5$ s,
> N = 5, 10 seeds.

- **Motivating argument**: connect to the Stage 1.5 model-fit. To resolve
  the two-frequency signal $g_k(t)^{-2} = c_0 + c_1\cos\xi_k + c_2\sin\xi_k$,
  a minimum of ~5 samples per half-period is needed, suggesting
  $T \gtrsim 20$ as a practical floor for ω ∈ [2, 6] Hz over 1.5 s.
- **Plot**: ω RMSE vs T with a vertical dashed line at the predicted minimum
  $T^*$. The empirical knee in the curve validates the theoretical argument.
- **Plot** (or table): convergence fraction vs T — shows at what projection
  count the algorithm begins failing entirely, not just degrading gracefully.
- State the recommended minimum T for the given ω range and duration.

---

## §R.6 — Noise Robustness

> Poisson noise sweep: $I_0 \in \{10^2, 5\times10^2, 10^3, 5\times10^3,
> 10^4, 10^5, \infty\}$, N = 5, 10 seeds.

- **Noise model paragraph**: explain the Poisson measurement model
  $y_{t,r} \sim \mathrm{Poisson}(I_0 e^{-p_{t,r}^*})$ and the log-sinogram
  $\tilde p = -\log(y/I_0)$. Cite the physical justification (photon counting).
- **Main plot**: ω RMSE vs $\log_{10}(I_0)$; shade ± std band. Identify the
  breakdown photon count $I_0^*$ below which accuracy degrades sharply.
- **Parameter table at 3 noise levels** (e.g., $I_0 \in \{10^3, 10^4, \infty\}$):
  ω error, shape error, α error — allowing direct comparison to noiseless case.
- Comment on which parameter is most sensitive to noise: expect $U_k$ (shape)
  to degrade before $\omega_k$, and trajectory estimates to be most robust
  (since Stage 1 uses peak heights, which are relatively noise-tolerant
  aggregate statistics).

---

## §R.7 — Comparison with Baselines

> Baselines: (A) static per-frame, (B) no-rotation model, (C) isotropic shape.

- Introduce each baseline in one sentence each. Emphasise that they represent
  natural simplifications of the full GMM-CT model, not arbitrary competitors.
- **Table**: for each method × metric (ω RMSE, shape error, projection residual,
  runtime). Highlight GMM-CT's advantage in each column.
- **Key qualitative point for Baseline A (static per-frame)**: per-frame fits
  are individually underdetermined — one projection angle cannot identify
  orientation. Show a figure of the per-frame shape estimates to make this vivid.
- **Key qualitative point for Baseline B (no rotation)**: the projection
  residual is measurably higher, shown by the sinogram difference. This
  quantifies how much information rotation carries.
- **Key qualitative point for Baseline C (isotropic)**: ω estimation fails
  entirely (ω is unidentifiable from isotropic projections). State this
  as an identifiability theorem result and reference the relevant section.

---

## §R.8 — ω-Range and Identifiability Limits (if included)

> Sweep ω range: [0.5, 1.5], [2, 6] (baseline), [6, 12], [12, 20] Hz.
> Optional — include if space and time permit.

- **Plot**: ω RMSE (pooled across seeds and Gaussians) vs ω range midpoint.
  Expected U-shape: degradation at both very slow (< 1 full rotation visible)
  and very fast (near-Nyquist aliasing) extremes.
- Interpret the slow-rotation regime: Stage 1.5 requires the viewing-angle
  variation $\phi_k(t)$ to span enough of the $\cos/\sin$ cycle.
- Interpret the fast-rotation regime: aliasing ambiguity in ω — relate to
  the aliasing guard-band in `generate_true_param`.
- If space is tight, this can be folded into a "Discussion" subsection.

---

## General Writing Principles

- **Lead with a representative example** — reviewers look at this figure first.
- **Separate accuracy from robustness** — noiseless and noisy results in
  distinct subsections with a clear transition sentence.
- **Tables for numbers, figures for trends** — a sweep result belongs on a
  curve, not in a table. Reserve tables for final summary statistics.
- **Always report error bars** — mean ± std across seeds; never mean alone.
- **Cite compute environment** — CPU/GPU model, RAM, average runtime.
- **Consistent notation throughout** — match latex macros used in the
  Methods / Reconstruction Algorithm sections.
