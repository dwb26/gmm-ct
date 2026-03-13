# "Reconstruction Algorithm" Section — Writing Structure

> Notes for structuring this section in the manuscript.
>
> **Context:** The mathematical derivations for the trajectory optimisation
> step (Stage 1 + Stage 1.5 velocity refinement) have already been completed
> elsewhere in the manuscript. This section therefore focuses on:
> - giving a high-level overview of the full four-stage pipeline
> - providing the mathematical detail for Stages 2–4 (morphology / ω recovery)
> - justifying the design choices for each stage
> - providing pseudocode or an algorithm box for reproducibility

---

## §A.1 — Overview and Design Philosophy

> One or two paragraphs at the top of the section. Set the scene before
> any equations appear.

- Open by restating the inverse problem in compact notation: given
  projections $\{p^\mathrm{obs}(t_i, r_j)\}_{i,j}$, recover
  $\Theta = \{(\alpha_k, U_k, \omega_k, v_0^k)\}_{k=1}^N$ jointly.
- Motivate the **staged decomposition**: the joint parameter space is
  high-dimensional and severely non-convex. Direct joint optimisation
  from a random initialisation fails reliably. The key insight is that
  trajectory estimation and morphology/rotation estimation are separable
  to leading order — trajectory dominates the peak positions in the
  sinogram, while shape and rotation govern the peak widths and
  amplitudes.
- State the four stages by name in a single structural sentence, then
  refer forward/back to the subsections for detail. A schematic flowchart
  or stage pipeline diagram here is strongly recommended.
- Note that Stages 1 and 1.5 (trajectory estimation and velocity
  refinement) have already been derived in Section X. This section
  picks up at Stage 1.5 conclusion and covers the full morphology
  and rotation recovery.

---

## §A.2 — Isotropic Initialisation of Shape Matrices (Bridge from Stage 1 to Stage 2)

> A short "bridge" subsection — may be as little as half a page.
> Covers the transition between the trajectory-only stage and the
> full anisotropic optimisation.

- After Stage 1, each Gaussian $k$ has an estimated trajectory
  $\hat\mu_k(t)$ but no shape estimate yet.
- Define the **velocity-aligned initialisation**: set
  $\hat U_k^{(0)} = \sigma_k\, Q_k$
  where $Q_k$ is a rotation that aligns the principal axis of $U_k$
  with the estimated velocity direction $\hat v_0^k / \|\hat v_0^k\|$.
  This is motivated by the observation that an elongated fragment
  travelling through air tends to align its long axis with velocity
  (aerodynamically). Even when this is not the case, it provides a
  better initialisation than the identity.
- State also how $\omega_k^{(0)}$ is set: from Stage 1.5 (ω initialisation
  via the model-fit grid search on $g_k(t)^{-2}$). The mathematical
  derivation of this is in §A.3.

---

## §A.3 — Angular Velocity Initialisation (Stage 1.5ω — Model-Fit Grid Search)

> This is the ω-specific part of Stage 1.5 and likely needs its own
> subsection since it is an original contribution.

- **Setup:** after trajectory estimation, the estimated peak width at time
  $t$ for Gaussian $k$ is $g_k(t) = \|U_k R_k(t) \hat e_k(t)\|$ where
  $\hat e_k(t) = (r_k^\mathrm{peak}(t) - s) / \|r_k^\mathrm{peak}(t) - s\|$
  is the unit ray direction from source $s$ to the estimated peak receiver
  position at time $t$.
- **Key identity:** for an elliptical Gaussian (diagonalisable $U_k$),
  derive the equation
  $$g_k(t)^{-2} = c_0 + c_1\cos\bigl(4\pi\omega_k t - 2\phi_k(t)\bigr)
  + c_2\sin\bigl(4\pi\omega_k t - 2\phi_k(t)\bigr)$$
  where $\phi_k(t)$ is the **trajectory-corrected viewing angle** (the
  angle from source to $\hat\mu_k(t)$ in world coordinates) and
  $c_0, c_1, c_2$ are functions of the semi-axes of $U_k$.
  Emphasise that $\phi_k(t)$ is **not** constant — it drifts as the
  Gaussian moves along its ballistic trajectory — and that ignoring
  this drift causes 6–19% error in ω (see docs/research_notes).
- **Grid search procedure:** for each candidate $\omega$ on a uniform
  grid $[\omega_{\min}, \omega_{\max}]$ (with step $\Delta\omega$),
  form the regressors $\xi_k^{(i)} = 4\pi\omega t_i - 2\phi_k(t_i)$,
  solve the ordinary least-squares problem
  $$\min_{c_0, c_1, c_2} \sum_i \bigl(g_k(t_i)^{-2} -
  c_0 - c_1\cos\xi_k^{(i)} - c_2\sin\xi_k^{(i)}\bigr)^2$$
  in closed form, and record the residual. Set $\hat\omega_k^{(0)}$
  to the grid point with minimum residual.
- **Complexity:** $O(T \cdot M)$ where $M$ is the number of grid points.
  For $T = 65$, $M \approx 400$ (grid from 2 to 6 Hz at 0.01 Hz steps),
  this is negligible.
- State the grid resolution used and its justification (should be fine
  enough that Stage 2 L-BFGS can converge from the grid initialisation).

---

## §A.4 — Stage 2: Multi-Start Joint Morphology Optimisation

> The main joint optimisation stage. This is the core numerical contribution
> beyond the trajectory work already written.

- **Parameters optimised:** $\Theta_\mathrm{morph} = \{(\alpha_k, U_k,
  \omega_k)\}_{k=1}^N$ with trajectories $\{v_0^k\}$ held fixed.
- **Loss function:** define the Huber (smooth $\ell^1$) loss on the
  full projection residual:
  $$\mathcal{L}(\Theta_\mathrm{morph}) = \sum_{i=1}^T \sum_{j=1}^R
  \ell_\beta\!\left(p(t_i, r_j;\,\Theta_\mathrm{morph}) -
  p^\mathrm{obs}(t_i, r_j)\right)$$
  where $\ell_\beta(z) = z^2/(2\beta)$ for $|z| \leq \beta$ and
  $|z| - \beta/2$ otherwise (with $\beta = 0.3$ used throughout).
  Justify the Huber choice: more robust than squared loss to occasional
  large projection residuals that occur during early iterations when
  $\omega_k$ is still mis-estimated.
- **Forward model reminder** (one equation, cross-reference §Methods):
  restate the closed-form line-integral formula for a rotated anisotropic
  Gaussian, emphasising that it is differentiable with respect to all
  parameters in $\Theta_\mathrm{morph}$.
- **Multi-start strategy:** run $n_\mathrm{starts}$ L-BFGS optimisations
  from perturbed copies of the Stage 1.5 initialisation. Perturbations
  are drawn as small multiplicative noise on $\alpha_k$, $U_k$ diagonal
  entries, and additive noise on $\omega_k$. Select the trial with the
  smallest supremum projection error $\max_{i,j}|p - p^\mathrm{obs}|$.
- **Early stopping:** optionally halt a trial early if the supremum error
  falls below a threshold $\tau$ (e.g., $\tau = 0.5$); this criterion
  is checked every $n_\mathrm{check}$ iterations.
- **Convergence note:** state that L-BFGS with full-batch forward model
  evaluation is used (no stochastic gradient). The forward model is
  $O(N \times T \times R)$ per evaluation, which is tractable for the
  problem sizes considered.

---

## §A.5 — Stage 3: Fine Grid Search Refinement of ω

> Short subsection — essentially a polishing step.

- **Motivation:** the Stage 2 L-BFGS can get trapped in local minima
  of the ω landscape because the loss has periodic structure with period
  $\approx 1/(2\Delta\phi)$. A local grid search around the Stage 2 estimate
  provides a cheap escape.
- **Procedure:** for each Gaussian independently, evaluate the loss on a
  grid of $\omega$ values in $[\hat\omega_k^{(2)} - 3, \hat\omega_k^{(2)} + 3]$
  Hz at 0.1 Hz steps, holding $\alpha_k$ and $U_k$ fixed. Set
  $\hat\omega_k^{(3)}$ to the grid minimiser.
- **Complexity:** $O(N \times 60 \times T \times R)$ forward evaluations.
  Since this is $O(N)$ independent 1D searches, it is fast relative to Stage 2.
- Connect this to the FFT-based ω estimation described in `gmm_ct/estimation/omega.py`
  (now superseded but conceptually related — can mention in passing or footnote).

---

## §A.6 — Stage 4: Final Joint Polish

> Short subsection.

- **Parameters optimised:** $\{(\alpha_k, U_k, \omega_k)\}$ jointly,
  warm-started from Stage 3.
- **Method:** L-BFGS, 200 iterations, tolerance $10^{-8}$.
- **Loss function:** same Huber loss as Stage 2 (consistent throughout Stages 2–4).
- This stage exploits the good ω initialisation from Stage 3 to fine-tune
  the shape and attenuation parameters, which were held fixed during the
  Stage 3 grid search.

---

## §A.7 — Algorithm Box (Pseudocode)

> An algorithm box numbered "Algorithm 1: GMM-CT Reconstruction".
> Place after §A.6. Essential for reproducibility.

```
Input:  projections p_obs, time points {t_i}, geometry (sources, receivers),
        known x0, a0, N, [omega_min, omega_max]

Stage 1: Trajectory estimation
  for trial = 1, ..., max(10, 2N):
    initialise v0 ~ random perturbation of prior mean
    minimise (L2 on peak heights) via L-BFGS (isotropic Gaussians)
    record best (v0, loss)
  Stage 1.5a: refine v0 via Newton-Raphson on analytic peak derivative
  Initialise U_k <- velocity-aligned anisotropic matrices

Stage 1.5b: omega initialisation (per Gaussian)
  for k = 1, ..., N:
    extract peak widths g_k(t_i) from observed sinogram
    for omega in grid [omega_min, omega_max]:
      form regressors xi(t_i) = 4 pi omega t_i - 2 phi_k(t_i)
      fit c0, c1, c2 via OLS; record residual
    omega_k <- grid argmin

Stage 2: Multi-start joint morphology optimisation
  for trial = 1, ..., n_starts:
    perturb (alpha, U, omega) from Stage 1.5 initialisation
    minimise Huber(p_model - p_obs) via L-BFGS (trajectories fixed)
    record (alpha, U, omega, sup_error) for this trial
  select trial with minimum sup_error

Stage 3: Fine omega grid search (per Gaussian, independent)
  for k = 1, ..., N:
    grid search omega in [omega_k +/- 3 Hz] at 0.1 Hz, alpha/U fixed
    omega_k <- grid argmin

Stage 4: Final polish
  minimise Huber(p_model - p_obs) via L-BFGS, warm-started from Stage 3
  200 iterations, tol = 1e-8

Output: theta_hat = {alpha_k, U_k, omega_k, v0_k}_{k=1}^N
```

---

## §A.8 — Implementation Details (short paragraph or bullet list)

> For the journal version, these belong at the end of this section or in
> a supplementary appendix. They are essential for reproducibility.

- **Software:** PyTorch (automatic differentiation for L-BFGS gradients),
  SciPy `linear_sum_assignment` for Hungarian matching.
- **Precision:** all computations in 64-bit floating point.
- **Device:** CPU or CUDA GPU; runs on CPU without code changes.
- **Parameter bounds:** $\omega_k$ is unconstrained in L-BFGS but initialised
  within $[\omega_{\min}, \omega_{\max}]$; $\alpha_k > 0$ enforced via
  reparametrisation $\alpha_k = \exp(\tilde\alpha_k)$; upper-triangularity
  of $U_k$ maintained by zeroing lower-triangular gradient components.
- **Peak detection:** 3-point sliding-window scan on each projection column
  at each time step; peaks assigned to Gaussians by Hungarian algorithm on
  the predicted vs observed peak height matrix.
- **Code availability:** the full implementation is available at
  [repo URL] under the MIT licence.

---

## Writing Notes

- **Cross-reference discipline:** every equation number used in this section
  should appear in the Numerical Results section when a stage-specific result
  is discussed (e.g., "using the Stage 1.5ω initialisation described in §A.3").
- **Do not re-derive the trajectory equations** — cross-reference the earlier
  manuscript section and pick up from the output of Stage 1.5a.
- **Justify every design choice** in 1–2 sentences: why Huber not squared loss,
  why multi-start not single-start, why a grid search after L-BFGS. Reviewers
  will ask if these are not addressed.
- **Algorithm box is mandatory** — reproducibility is a hard requirement for
  SIAM, IOP, and IEEE TCI.
- **Stage 1.5 decomposition** — Stage 1.5a (velocity refinement) and Stage 1.5ω
  (ω initialisation) are logically distinct despite sharing a label. Consider
  calling them "Stage 1b" and "Stage 1c" for clarity in the final manuscript,
  or clarify their relationship in §A.1.
