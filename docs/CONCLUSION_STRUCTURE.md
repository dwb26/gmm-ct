# Conclusion — Suggested Structure

## §C.1 Problem restatement (1–2 sentences)

Restate the inverse problem in plain language: recovering the trajectory, rotation,
and morphology of multiple rotating anisotropic Gaussian fragments from a sparse
fan-beam projection sequence. No new results here — just ground the reader.

---

## §C.2 Summary of contributions (1 paragraph)

Walk through the four-stage pipeline at a high level, naming each stage's specific
contribution:

- **Stage 1** — Trajectory recovery via projection-mode peak assignment and the
  Hungarian algorithm.
- **Stage 1.5** — Closed-form angular velocity initialisation via OLS regression on
  the amplitude-oscillation structure of the projections.
- **Stage 1.5b** — Non-negative least-squares initialisation of attenuation
  coefficients, exploiting the linearity of the forward model in {α_n}.
- **Stages 2–4** — Joint Huber-loss refinement over the full sinogram, with a fine
  angular-velocity grid search and a final L-BFGS polish.

Emphasise that each stage exploits a distinct structural property of the forward
model (linearity in α, periodicity in ω, peak-locality for trajectories), giving the
pipeline a principled multi-resolution character rather than a black-box optimisation.

---

## §C.3 Empirical findings (1 paragraph)

Summarise the key numerical results:

- N = 1–5 accuracy figures from the stability experiment (headline numbers from the
  summary table, e.g. mean parameter accuracy and mean projection accuracy per N).
- Observation that projection-space accuracy is systematically higher than
  parameter-space accuracy, and why this is physically meaningful: many parameter
  combinations produce nearly identical projections (near-identifiability boundary).
- Notable failure modes observed (e.g. high anisotropy degrading Stage 1.5,
  closely-spaced trajectories reducing Stage 1 reliability).

---

## §C.4 Limitations (short paragraph or bullet list)

Be direct. Suggested items:

- Noiseless setting assumed throughout — performance under Poisson noise untested.
- All Gaussians share a common initial position and gravitational acceleration;
  these are treated as known physics inputs.
- Stage 1.5 angular-velocity estimates degrade for strongly anisotropic shapes
  (ellipse eccentricity approaching 1).
- The current stability study covers N ≤ 5; scalability for larger N not characterised.

---

## §C.5 Future work (short paragraph or bullet list)

- **Noise robustness** — Poisson noise model, varying photon count I₀; characterise
  the SNR threshold below which reconstruction fails.
- **Unknown geometry** — Relax the assumption of known source/detector positions;
  joint calibration and reconstruction.
- **Real experimental validation** — Synchrotron or laboratory CT data with known
  ground truth (e.g. rotating phantom).
- **Beyond Gaussian morphology** — Generalise the forward model to mixtures of
  non-Gaussian basis functions or learned shape representations.

---

## Writing principles

- Keep §C.1–§C.2 tight: the reader has just finished reading your full paper.
- §C.3 should cite specific numbers from the results — do not be vague.
- §C.4 and §C.5 are expected by reviewers; omitting them reads as overconfidence.
- Total target length: 400–600 words.
