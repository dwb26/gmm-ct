"""Figure: detected modes, initial trajectories and fitted trajectories."""

import logging

import matplotlib.pyplot as plt

from .data import Run
from .primitives import draw_mode_data, draw_mode_trajectories, draw_projection_profile
from .registry import figure
from .poster_style import WIDTH, particle_colors, slice_colors

logger = logging.getLogger(__name__)


@figure("modes")
def modes(run: Run, times=None, crop_time: bool = True) -> plt.Figure:
    """Left: projection and detected modes at two times (colour = time).
    Middle: all detected modes with the best initial trajectories.
    Right: all detected modes with the fitted trajectories.

    ``times`` is a pair of times in seconds, snapped to the nearest sample.
    ``crop_time`` limits the time axis to the window where objects are on the detector.
    """
    times = run.default_times() if times is None else times
    idxs = [run.time_index(tv) for tv in times]
    cols = slice_colors(len(idxs))
    pcols = particle_colors(run.N)
    modes_per_time = run.detect_modes()

    fig = plt.figure(figsize=(WIDTH, 4.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.8, 1.8])

    ax_mid = fig.add_subplot(gs[:, 1])
    ax_right = fig.add_subplot(gs[:, 2], sharex=ax_mid, sharey=ax_mid)
    prev = None
    for row, (idx, col) in enumerate(zip(idxs, cols)):
        ax = fig.add_subplot(gs[row, 0], sharey=ax_mid)
        draw_projection_profile(ax, run, idx, col, modes_per_time[idx])
        ax.set_title(f"Projection (t = {run.t[idx]:.2f} s)", color=col, fontweight="bold")
        ax.set_ylabel("Detector height", fontweight="bold")
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Projection intensity", fontweight="bold")
        for target in (ax_mid, ax_right):
            target.axvline(run.t[idx], color=col, linewidth=1.0, zorder=1)
            target.plot([run.t[idx]] * len(modes_per_time[idx]), modes_per_time[idx], "o",
                        color=col, markeredgecolor="black", markeredgewidth=0.5, markersize=4.5, zorder=4)

    for ax in (ax_mid, ax_right):
        draw_mode_data(ax, run, modes_per_time)
        ax.set_xlabel("Time", fontweight="bold")
        ax.tick_params(labelleft=False)
    ax_mid.set_ylim(run.y[0], run.y[-1])
    if crop_time:
        lo, hi = run.active_window()
        pad = 0.05 * (hi - lo)
        ax_mid.set_xlim(lo - pad, hi + pad)
    else:
        ax_mid.set_xlim(run.t[0], run.t[-1])

    if run.theta_init is not None:
        draw_mode_trajectories(ax_mid, run, run.theta_init, pcols)
    else:
        logger.warning("No theta_pre_stage1_5 in %s; skipping initial trajectories.", run.exp_dir)
    ax_mid.set_title("Observed Modes + Initial Trajectories", fontweight="bold")

    if run.theta_est is not None:
        draw_mode_trajectories(ax_right, run, run.theta_est, pcols,
                               labels=[rf"$r^*_{k + 1}(t)$" for k in range(run.N)])
        ax_right.legend(loc="upper right", ncols=2, handlelength=1.5, columnspacing=1.0)
    ax_right.set_title("Observed Modes + Fitted Trajectories", fontweight="bold")
    return fig
