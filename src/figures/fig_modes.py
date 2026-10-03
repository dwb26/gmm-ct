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
    y_max = 0.0
    symbols = ["^", "d"]
    for row, (idx, col) in enumerate(zip(idxs, cols)):
        ax = fig.add_subplot(gs[row, 0])
        y_max_c = max(run.proj[idx])
        if y_max_c > y_max: 
            y_max = y_max_c
        draw_projection_profile(ax, run, idx, col, modes_per_time[idx], marker=symbols[row])
        ax.set_title(f"Projection (t = {run.t[idx]:.2f})", fontweight="bold")
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Detector height", fontweight="bold")
        ax.set_ylabel("Projection intensity", fontweight="bold")
        for target in (ax_mid, ax_right):
            target.plot([run.t[idx]] * len(modes_per_time[idx]), modes_per_time[idx], symbols[row],
                        color="black", markersize=7, linestyle="none", zorder=4, markerfacecolor="None", 
                        markeredgecolor="black")
    for row, (idx, col) in enumerate(zip(idxs, cols)):
        ax = fig.axes[row]
        ax.set(ylim=(-.5, y_max))
        ax.tick_params(labelsize=12)

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
        draw_mode_trajectories(ax_mid, run, run.theta_init, pcols, alpha=0.6)
    else:
        logger.warning("No theta_pre_stage1_5 in %s; skipping initial trajectories.", run.exp_dir)
    ax_mid.set_title("Observed Modes + Initial Trajectories", fontweight="bold")
    ax_mid.set_ylabel("Detector Height", fontweight="bold")
    ax_mid.tick_params(labelleft=True, labelsize=12)
    ax_right.tick_params(labelsize=12)

    if run.theta_est is not None:
        draw_mode_trajectories(ax_right, run, run.theta_est, pcols,
                               labels=[rf"$r^*_{k + 1}(t)$" for k in range(run.N)], alpha=0.6)
    ax_right.set_title("Observed Modes + Fitted Trajectories", fontweight="bold")
    return fig
