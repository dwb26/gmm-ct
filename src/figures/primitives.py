"""Axes-level drawing helpers. Each takes an ``ax`` and a ``Run`` and draws one thing."""

import numpy as np
from matplotlib.patches import Ellipse

from .data import Run
from scipy.optimize import linear_sum_assignment

from .poster_style import COLOR_DATA, COLOR_DETECTOR, COLOR_SOURCE, draw_rays

def draw_gaussians(ax, run: Run, theta: dict, t_val: float, colors, trail=True):
    """Poster style: faint trajectories and three nested black-edged ellipses (1, 2, 3 sigma)."""
    for k in range(run.N):
        if trail:
            path = run.centers(theta, run.t)[k]
            ax.plot(path[:, 0], path[:, 1], color=colors[k], alpha=0.4, linewidth=1.5, zorder=1)
        mu = run.centers(theta, [t_val])[k, 0]
        evals, evecs = np.linalg.eigh(run.covariance(theta, k, t_val))
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        weight = float(theta["alphas"][k])
        for i, scale in enumerate((1.0, 2.0, 3.0)):
            w, h = 2 * scale * np.sqrt(evals)
            ax.add_patch(Ellipse(
                mu, w, h, angle=angle, facecolor=colors[k], edgecolor="black",
                alpha=min(0.8, max(0.1, weight * (1.0 - 0.2 * i))),
                linewidth=1.5 - 0.3 * i, zorder=10 + i,
            ))


def draw_geometry(ax, run: Run):
    """Source (red dot), receivers (blue dots) and gold rays."""
    draw_rays(ax, run.source, run.detector_x, run.y)
    ax.plot(*run.source, "o", color=COLOR_SOURCE, markersize=5, alpha=0.7, zorder=50, label="Source")
    ax.plot([run.detector_x] * len(run.y), run.y, "o", color=COLOR_DETECTOR,
            markersize=1, alpha=0.5, zorder=50, label="Detectors")


def draw_projection_profile(ax, run: Run, idx: int, color, modes=None, marker="*"):
    """Projection at time index ``idx`` (detector height on x) with its modes marked by ``marker``."""
    # ax.plot(run.y, run.proj[idx], color="black", linewidth=1.0)
    ax.plot(run.proj[idx], run.y, color="black", linewidth=1.0)
    if modes is not None and len(modes):
        ax.plot(modes, 0.01 * np.ones(len(modes)), marker, markersize=6, linestyle="none",
                zorder=4, markerfacecolor="None", markeredgecolor="red", label="Modes")
        ax.legend(frameon=True)


def draw_mode_data(ax, run: Run, modes_per_time):
    t = np.concatenate([[tv] * len(m) for tv, m in zip(run.t, modes_per_time)])
    y = np.concatenate(modes_per_time)
    ax.plot(t, y, ".", color=COLOR_DATA, markersize=3, linestyle="none", zorder=2, label="Detected modes")


def match_to_truth(run: Run, theta: dict) -> np.ndarray:
    """Permutation so that ``theta`` particle ``perm[k]`` is the one closest to true particle k."""
    clip = lambda h: np.clip(np.nan_to_num(h, nan=0.0, posinf=10, neginf=-10), -10, 10)
    true_h, est_h = clip(run.mode_heights(run.theta_true)), clip(run.mode_heights(theta))
    cost = ((true_h[:, None] - est_h[None]) ** 2).mean(-1)
    return linear_sum_assignment(cost)[1]


def draw_mode_trajectories(ax, run: Run, theta: dict, colors, linewidth=1.0, labels=None, alpha=0):
    """One solid line per particle; ``colors[k]`` is the colour of the matched true particle."""
    heights = run.mode_heights(theta)
    # Hide the divergent pieces where the ray leaves the detector
    heights = np.where((heights >= run.y[0]) & (heights <= run.y[-1]), heights, np.nan)
    perm = match_to_truth(run, theta)
    for k, j in enumerate(perm):
        ax.plot(run.t, heights[j], color=colors[k], linewidth=linewidth, zorder=3,
                label=None if labels is None else labels[k], alpha=alpha)
