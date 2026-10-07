"""Figure: individual Gaussian shapes (simulated, Stage 1 init, reconstructed), one row per particle."""

import matplotlib.pyplot as plt
import numpy as np

from .data import Run
from .poster_style import WIDTH
from .primitives import match_to_truth
from .registry import figure

ERR_FLOOR = 1e-6   # log10 relative error is clipped here
ERR_VMIN = -6.0
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
QUANTILE_NAMES = {0.0: "Best", 0.25: "25th pct.", 0.5: "Median", 0.75: "75th pct.", 1.0: "Worst"}


def _aligned_precision(P_true: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Precision with the eigenvalues of ``P`` placed on the principal axes of ``P_true``."""
    _, V_true = np.linalg.eigh(P_true)
    lam, _ = np.linalg.eigh(P)
    return V_true @ np.diag(lam) @ V_true.T


def _gaussian_image(alpha: float, P: np.ndarray, grid: np.ndarray) -> np.ndarray:
    quad = np.einsum("...i,ij,...j->...", grid, P, grid)
    return alpha * np.exp(-0.5 * quad)


def _pick_by_quantile(errors: np.ndarray) -> list[tuple[int, float]]:
    """(particle index, quantile) at the best, 25th, median, 75th and worst error (best first)."""
    order = np.argsort(errors)
    picked, seen = [], set()
    for q in QUANTILES:
        k = int(order[int(round(q * (len(order) - 1)))])
        if k not in seen:
            seen.add(k)
            picked.append((k, q))
    return picked


@figure("gaussians")
def gaussians(run: Run, resolution: int = 256) -> plt.Figure:
    """One row per particle, ordered from best to worst reconstructed.

    Columns: simulated | Stage 1 init | init error | reconstruction | reconstruction error.
    Each Gaussian is drawn at the origin in its fixed body frame (alpha, U), with the
    reconstruction and the init rotated onto the principal axes of the true particle. The
    five particles are those at the best, 25th, median, 75th percentile and worst
    reconstruction error (mean absolute image error). All rows share axis limits, so
    differences in particle size are visible, and share one intensity scale.
    """
    if run.theta_est is None:
        raise ValueError("The gaussians figure needs reconstruction.pt")

    est_perm = match_to_truth(run, run.theta_est)
    init = run.theta_init
    init_perm = match_to_truth(run, init) if init is not None else None

    P_true = [U.T @ U for U in run.theta_true["U_skews"]]
    cov_true = [np.linalg.inv(P) for P in P_true]

    # Common half-width: 3 sigma along the longest axis of the largest selected true particle
    half = 1e-3
    coords = None

    def render(theta, perm, k):
        j = perm[k]
        P = theta["U_skews"][j].T @ theta["U_skews"][j]
        return _gaussian_image(float(theta["alphas"][j]), _aligned_precision(P_true[k], P), coords)

    # Rank every particle by its reconstruction error on a fixed grid
    rank_half = 3.0 * max(np.sqrt(np.linalg.eigvalsh(c).max()) for c in cov_true)
    ax_ = np.linspace(-rank_half, rank_half, resolution)
    coords = np.stack(np.meshgrid(ax_, ax_), axis=-1)
    errors = np.array([
        np.abs(_gaussian_image(float(run.theta_true["alphas"][k]), P_true[k], coords)
               - render(run.theta_est, est_perm, k)).mean()
        for k in range(run.N)
    ])
    selected = _pick_by_quantile(errors)

    half = max(3.0 * np.sqrt(np.linalg.eigvalsh(cov_true[k]).max()) for k, _ in selected)
    ax_ = np.linspace(-half, half, resolution)
    coords = np.stack(np.meshgrid(ax_, ax_), axis=-1)
    extent = (-half, half, -half, half)

    rows = []
    for k, q in selected:
        img_true = _gaussian_image(float(run.theta_true["alphas"][k]), P_true[k], coords)
        img_est = render(run.theta_est, est_perm, k)
        img_init = render(init, init_perm, k) if init is not None else None
        rows.append((k, q, img_true, img_init, img_est))

    vmax = max(img.max() for _, _, t, i, e in rows for img in (t, e) + ((i,) if i is not None else ()))
    log_err = lambda a, b: np.log10(np.clip(np.abs(a - b) / a.max(), ERR_FLOOR, None))  # a = truth; a.max() = its peak
    err_max = max([ERR_VMIN + 1e-3] + [log_err(t, e).max() for _, _, t, _, e in rows]
                  + [log_err(t, i).max() for _, _, t, i, _ in rows if i is not None])

    columns = ["Simulated"]
    if init is not None:
        columns += ["Initialization", "Init Error"]
    columns += ["Reconstruction", "Reconstruction Error"]
    n_rows, n_cols = len(rows), len(columns)

    cell = (WIDTH - 0.8) / n_cols
    fig = plt.figure(figsize=(WIDTH, cell * n_rows + 1.0), layout="constrained")
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.0] * n_rows + [0.05])
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes[0, 0] if (r or c) else None,
                                         sharey=axes[0, 0] if (r or c) else None)
            axes[r, c].tick_params(labelbottom=r == n_rows - 1, labelleft=c == 0)

    im_gauss = im_err = None
    for r, (k, q, img_true, img_init, img_est) in enumerate(rows):
        panels = [("g", img_true)]
        if img_init is not None:
            panels += [("g", img_init), ("e", log_err(img_true, img_init))]
        panels += [("g", img_est), ("e", log_err(img_true, img_est))]
        for c, (kind, img) in enumerate(panels):
            ax = axes[r, c]
            ax.grid(False)
            if kind == "g":
                im_gauss = ax.imshow(img, extent=extent, origin="lower", cmap="viridis",
                                     vmin=0.0, vmax=vmax, aspect="equal")
                ax.contour(ax_, ax_, img, levels=3, colors="black", linewidths=0.6, alpha=0.4)
            else:
                im_err = ax.imshow(img, extent=extent, origin="lower", cmap="Greys",
                                   vmin=ERR_VMIN, vmax=err_max, aspect="equal")
                ax.contour(ax_, ax_, img, levels=3, colors="white", linewidths=0.6, alpha=0.4)
            if r == 0:
                ax.set_title(columns[c], fontweight="bold")
        axes[r, 0].set_ylabel(f"{QUANTILE_NAMES[q]}\n\nHeight", fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Depth", fontweight="bold")

    split = 2 if init is not None else 1
    fig.colorbar(im_gauss, cax=fig.add_subplot(gs[-1, :split]), orientation="horizontal",
                 label="Attenuation")
    fig.colorbar(im_err, cax=fig.add_subplot(gs[-1, split:]), orientation="horizontal",
                 label="Log$_{10}$ Error (fraction of peak)")
    return fig
