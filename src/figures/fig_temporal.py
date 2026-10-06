"""Figure: true vs estimated particle states at chosen times, with overlaid projections."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .data import Run
from .poster_style import STATE_FACE, WIDTH, particle_colors, slice_colors
from .primitives import draw_gaussians, draw_geometry, match_to_truth
from .registry import figure


def _reordered(theta: dict, perm: np.ndarray) -> dict:
    return {k: v[perm] for k, v in theta.items()}


@figure("temporal")
def temporal(run: Run, times=None) -> plt.Figure:
    """One row per time: simulated state | true vs estimated projection | estimated state.

    ``times`` are in seconds (any number, default two), snapped to the nearest sample. The
    estimated particles are matched to the true ones so each keeps the same colour.
    """
    if run.theta_est is None:
        raise ValueError("The temporal figure needs reconstruction.pt")
    times = run.default_times() if times is None else times
    idxs = [run.time_index(tv) for tv in times]
    cols = slice_colors(len(idxs))
    colors = particle_colors(run.N)

    # est = _reordered(run.theta_init, match_to_truth(run, run.theta_init))
    est = _reordered(run.theta_est, match_to_truth(run, run.theta_est))
    proj_est = run.project(est)

    centers = run.centers(run.theta_true, run.t)
    xlim = (min(run.source[0], centers[..., 0].min()) - 0.3, run.detector_x + 0.2)
    ylim = (run.y[0] - 0.2, run.y[-1] + 0.2)

    n = len(idxs)
    fig = plt.figure(figsize=(WIDTH, 3.0 * n + 0.4), layout="constrained")
    gs = fig.add_gridspec(n, 3, width_ratios=[1.2, 0.8, 1.2])
    # titles = ("Simulated", "Projections", "Stage 2 Initialization")
    titles = ("Simulated", "Projections", "Reconstructed")

    for r, (idx, col) in enumerate(zip(idxs, cols)):
        t_val = run.t[idx]
        last = r == n - 1
        ax_true = fig.add_subplot(gs[r, 0])
        ax_proj = fig.add_subplot(gs[r, 1], sharey=ax_true)
        # Note: sharex is removed here so ax_est can be flipped independently
        ax_est = fig.add_subplot(gs[r, 2], sharey=ax_true)

        # 1. Draw Simulated (Left)
        draw_gaussians(ax_true, run, run.theta_true, t_val, colors)
        draw_geometry(ax_true, run)
        ax_true.set_xlim(*xlim)
        ax_true.set_ylim(*ylim)
        ax_true.set_aspect("equal", adjustable="datalim")
        ax_true.set_facecolor(STATE_FACE)
        ax_true.set_xlabel("Depth" if last else "", fontweight="bold")
        ax_true.set_ylabel(f"t = {t_val:.2f} \nDetector height", fontweight="bold")

        # 2. Draw Reconstructed Mirrored (Right)
        draw_gaussians(ax_est, run, est, t_val, colors)
        draw_geometry(ax_est, run)
        ax_est.set_xlim(xlim[1], xlim[0])  # Reversed limits flip the x-axis horizontally
        ax_est.set_ylim(*ylim)
        ax_est.set_aspect("equal", adjustable="datalim")
        ax_est.set_facecolor(STATE_FACE)
        ax_est.set_xlabel("Depth" if last else "", fontweight="bold")
        ax_est.tick_params(labelleft=False)

        # 3. Draw Projections (Center)
        ax_proj.plot(run.proj[idx], run.y, color="black", linewidth=1.5, label="True")
        ax_proj.plot(proj_est[idx], run.y, color="red", linestyle="--", linewidth=1.5, label="Estimated")
        ax_proj.set_xlabel("Intensity" if last else "", fontweight="bold")
        ax_proj.tick_params(labelleft=False)
        ax_proj.grid(True, alpha=0.3, linestyle="--")
        
        ax_true.tick_params(labelsize=11)
        ax_proj.tick_params(labelsize=11)
        ax_est.tick_params(labelsize=11)

        if r == 0:
            for ax in (ax_true, ax_proj, ax_est):
                ax.tick_params(labelbottom=False)
            for ax, title in zip((ax_true, ax_proj, ax_est), titles):
                ax.set_title(title, fontweight="bold")
            # ax_proj.legend(frameon=False, loc="best")
            # ax_true.legend(handles=[Patch(facecolor=c, edgecolor="black", label=f"$\\rho_{{{k+1}}}$")
                                    # for k, c in enumerate(colors)], loc="upper left", frameon=True)
            # ax_est.legend(handles=[Patch(facecolor=c, edgecolor="black", label=f"$\\widehat\\rho_{{{k+1}}}$")
                                #    for k, c in enumerate(colors)], loc="upper right", frameon=True)

    return fig