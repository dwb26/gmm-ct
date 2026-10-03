"""Figure: particle states and their projections at two times, plus the full sinogram."""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data import Run
from .primitives import draw_geometry, draw_gaussians, draw_projection_profile
from .registry import figure
from .poster_style import SINOGRAM_CMAP, STATE_FACE, WIDTH, particle_colors, slice_colors


@figure("state_sinogram")
def state_sinogram(run: Run, times=None) -> plt.Figure:
    """Top: (state, projection) at each of two times. Bottom: the full sinogram.

    ``times`` is a pair of times in seconds, snapped to the nearest sample.
    """
    times = run.default_times() if times is None else times
    idxs = [run.time_index(tv) for tv in times]
    cols = slice_colors(len(idxs))
    gauss_cols = particle_colors(run.N)

    fig = plt.figure(figsize=(WIDTH, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])

    ymin, ymax = run.y[0], run.y[-1]
    centers = run.centers(run.theta_true, run.t)
    xlo = min(run.source[0], centers[..., 0].min()) - 0.3
    xhi = run.detector_x + 0.2

    first_state = None
    for i, (idx, col) in enumerate(zip(idxs, cols)):
        ax_s = fig.add_subplot(gs[0, i], sharey=first_state)
        draw_gaussians(ax_s, run, run.theta_true, run.t[idx], gauss_cols)
        draw_geometry(ax_s, run)
        ax_s.set_xlim(xlo, xhi)
        ax_s.set_ylim(ymin, ymax)
        ax_s.set_aspect("equal", adjustable="box")
        ax_s.set_xlabel("Depth", fontweight="bold")
        ax_s.set_ylabel("Detector height" if i == 0 else "", fontweight="bold")
        ax_s.tick_params(labelleft=False)
        ax_s.set_facecolor(STATE_FACE)
        ax_s.set_title(f"State Snapshot (t = {run.t[idx]:.2f} s)", fontweight="bold")

        # Attached axes keep the projection the same height as the state panel
        ax_p = make_axes_locatable(ax_s).append_axes("right", size="35%", pad=0.08, sharey=ax_s)
        draw_projection_profile(ax_p, run, idx, "black")
        ax_p.lines[0].set_alpha(0.85)
        ax_p.set_xlabel("Intensity", fontweight="bold")
        ax_p.tick_params(labelleft=False)
        first_state = first_state or ax_s

    ax_sino = fig.add_subplot(gs[1, :])
    im = ax_sino.imshow(
        run.proj.T, aspect="auto", origin="lower", cmap='gray', interpolation="nearest",
        extent=[run.t[0], run.t[-1], ymin, ymax],
    ) # cmap=SINOGRAM_CMAP
    for idx, col in zip(idxs, cols):
        ax_sino.axvline(run.t[idx], color="lime", linestyle="--", linewidth=1.2)
    ax_sino.set_xlabel("Time", fontweight="bold")
    ax_sino.set_ylabel("Detector height", fontweight="bold")
    ax_sino.set_title("Dynamic Sinogram", fontweight="bold")
    ax_sino.grid(True, color="white", alpha=0.35, linewidth=0.5)
    lo, hi = run.active_window()
    pad = 0.05 * (hi - lo)
    ax_sino.set_xlim(max(run.t[0], lo - pad), min(run.t[-1], hi + pad))
    fig.colorbar(im, ax=ax_sino, pad=0.01, fraction=0.03, label="Projection intensity")
    return fig
