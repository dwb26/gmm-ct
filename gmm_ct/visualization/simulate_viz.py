"""Simulation output visualization for GMM-CT.

Provides standalone functions for loading and visualising the ``.pt`` files
saved by ``gmm-ct simulate``.  Works for both 2D and 3D simulations without
requiring a live reconstruction model instance.

Public API
----------
plot_simulation_summary
    Load a simulation directory and produce a multi-panel summary figure.
plot_projection_frames
    Grid of detector frames at selected time points.
plot_true_trajectories
    True GMM centroid trajectories (2D panels or 3D projections).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

# Clean white background with subtle grid lines
sns.set_theme(
    style="whitegrid",
    context="paper",  # Keeps line weights precise
    font="sans-serif",
)

# Refine grid line aesthetics so they don't overpower the GMM ellipses
plt.rcParams["grid.color"] = "#e5e7eb"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from .publication import (
    plot_acquisition_geometry,
    plot_gmm_snapshot_animated,
    plot_trajectories_single,
)

logger = logging.getLogger(__name__)

_LABEL_FONTSIZE = 16
_TITLE_FONTSIZE = 18
_TICK_FONTSIZE = 13


# ─── helpers ─────────────────────────────────────────────────────────────────

def _get_colors(N: int):
    return cm.rainbow(np.linspace(0, 1, N))


def _infer_detector_shape(receivers) -> tuple[int, int]:
    """Return ``(n_y, n_z)`` from a 3-D receiver list by counting unique coords."""
    pts = receivers[0]
    ys = torch.stack([r[1] for r in pts])
    zs = torch.stack([r[2] for r in pts])
    n_y = int(len(torch.unique(ys)))
    n_z = int(len(torch.unique(zs)))
    return n_y, n_z


def _compute_centroids(theta_true: dict, t: torch.Tensor, N: int) -> np.ndarray:
    """Return centroid array of shape ``(N, T, d)``."""
    t_np = t.cpu().numpy()
    d = theta_true['x0s'][0].shape[0]
    T = len(t_np)
    centroids = np.empty((N, T, d))
    for k in range(N):
        x0 = theta_true['x0s'][k].cpu().numpy()
        v0 = theta_true['v0s'][k].cpu().numpy()
        a0 = theta_true['a0s'][k].cpu().numpy()
        for i, ti in enumerate(t_np):
            centroids[k, i] = x0 + v0 * ti + 0.5 * a0 * ti ** 2
    return centroids


# ─── main entry point ────────────────────────────────────────────────────────

def plot_simulation_summary(
    sim_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> None:
    """Load a simulation directory and save a multi-panel summary figure.

    Dispatches to a 2D or 3D layout based on dimensionality stored in the
    ground-truth config.

    Parameters
    ----------
    sim_dir:
        Path to a simulation output directory containing ``projections.pt``
        and ``ground_truth.pt``.
    output_dir:
        Where to save figures.  Defaults to *sim_dir*.
    """
    sim_dir = Path(sim_dir)
    output_dir = Path(output_dir) if output_dir else sim_dir

    proj_data = torch.load(sim_dir / "projections.pt", weights_only=True)
    gt_data = torch.load(sim_dir / "ground_truth.pt", weights_only=True)

    projs: torch.Tensor = proj_data["projections"]   # (T, n_rcvrs)
    t: torch.Tensor = proj_data["times"]
    theta_true: dict = gt_data["theta_true"]
    receivers = gt_data["receivers"]
    config: dict = gt_data["config"]
    d: int = config["d"]
    N: int = config["N"]

    if d == 2:
        _summary_2d(projs, t, theta_true, receivers, N, output_dir)
    elif d == 3:
        _summary_3d(projs, t, theta_true, receivers, N, output_dir)
    else:
        logger.warning("plot_simulation_summary: unsupported d=%d", d)


# ─── 2-D summary ─────────────────────────────────────────────────────────────

def _summary_2d(
    projs: torch.Tensor,
    t: torch.Tensor,
    theta_true: dict,
    receivers,
    N: int,
    output_dir: Path,
) -> None:
    projs_np = projs.cpu().numpy()          # (T, n_rcvrs)
    t_np = t.cpu().numpy()
    rcvr_ys = np.array([r[1].item() for r in receivers[0]])
    T = projs_np.shape[0]
    colors = _get_colors(N)
    centroids = _compute_centroids(theta_true, t, N)  # (N, T, 2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                             gridspec_kw={'wspace': 0.35},
                             layout='constrained')

    # Panel 1 — sinogram
    ax = axes[0]
    im = ax.imshow(
        projs_np.T,
        aspect='auto',
        origin='lower',
        extent=[t_np[0], t_np[-1], rcvr_ys.min(), rcvr_ys.max()],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Time (s)', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Receiver height', fontsize=_LABEL_FONTSIZE)
    ax.set_title('Sinogram', fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)

    # Panel 2 — single-frame projection profile
    ax = axes[1]
    mid_t = T // 2
    ax.plot(rcvr_ys, projs_np[mid_t], color='steelblue', lw=1.5)
    ax.set_xlabel('Receiver height', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Intensity', fontsize=_LABEL_FONTSIZE)
    ax.set_title(f'Profile at t = {t_np[mid_t]:.3f} s', fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 3 — true GMM trajectories
    ax = axes[2]
    for k in range(N):
        ax.plot(centroids[k, :, 0], centroids[k, :, 1],
                color=colors[k], lw=1.5, label=f'$\\rho_{{{k+1}}}$')
        ax.scatter(centroids[k, 0, 0], centroids[k, 0, 1],
                   color=colors[k], s=60, zorder=5, marker='o')
    ax.set_xlabel('x (depth)', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('y (height)', fontsize=_LABEL_FONTSIZE)
    ax.set_title('True GMM trajectories', fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Simulation summary (2D)', fontsize=_TITLE_FONTSIZE + 2,
                 fontweight='bold')
    outpath = output_dir / 'simulation_summary_2d.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved 2D simulation summary to %s", outpath)


# ─── 3-D summary ─────────────────────────────────────────────────────────────

def _summary_3d(
    projs: torch.Tensor,
    t: torch.Tensor,
    theta_true: dict,
    receivers,
    N: int,
    output_dir: Path,
) -> None:
    projs_np = projs.cpu().numpy()          # (T, n_y * n_z)
    t_np = t.cpu().numpy()
    n_y, n_z = _infer_detector_shape(receivers)
    T = len(t_np)
    centroids = _compute_centroids(theta_true, t, N)  # (N, T, 3)
    colors = _get_colors(N)

    # ── Figure 1: grid of detector frames ────────────────────────────────
    n_frames = min(6, T)
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4),
                             gridspec_kw={'wspace': 0.3},
                             layout='constrained')
    if n_frames == 1:
        axes = [axes]
    vmax = projs_np.max()
    for j, fi in enumerate(frame_indices):
        frame = projs_np[fi].reshape(n_y, n_z)
        im = axes[j].imshow(frame, origin='lower', cmap='viridis',
                            vmin=0, vmax=vmax, aspect='auto')
        fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        axes[j].set_title(f't = {t_np[fi]:.3f} s', fontsize=_TICK_FONTSIZE)
        axes[j].set_xlabel('z index', fontsize=_TICK_FONTSIZE - 1)
        if j == 0:
            axes[j].set_ylabel('y index', fontsize=_TICK_FONTSIZE - 1)
        else:
            axes[j].set_yticks([])
    fig.suptitle('Detector frames (3D simulation)', fontsize=_TITLE_FONTSIZE,
                 fontweight='bold')
    out1 = output_dir / 'simulation_detector_frames_3d.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved detector frames to %s", out1)

    # ── Figure 2: trajectory projections + signal statistics ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6),
                             gridspec_kw={'wspace': 0.35},
                             layout='constrained')

    # Panel 1 — x–y projection
    ax = axes[0]
    for k in range(N):
        ax.plot(centroids[k, :, 0], centroids[k, :, 1],
                color=colors[k], lw=1.5, label=f'$\\rho_{{{k+1}}}$')
        ax.scatter(centroids[k, 0, 0], centroids[k, 0, 1],
                   color=colors[k], s=60, zorder=5, marker='o')
    ax.set_xlabel('x (depth)', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('y', fontsize=_LABEL_FONTSIZE)
    ax.set_title('True trajectories — x–y', fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 2 — x–z projection
    ax = axes[1]
    for k in range(N):
        ax.plot(centroids[k, :, 0], centroids[k, :, 2],
                color=colors[k], lw=1.5, label=f'$\\rho_{{{k+1}}}$')
        ax.scatter(centroids[k, 0, 0], centroids[k, 0, 2],
                   color=colors[k], s=60, zorder=5, marker='o')
    ax.set_xlabel('x (depth)', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('z', fontsize=_LABEL_FONTSIZE)
    ax.set_title('True trajectories — x–z', fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 3 — projection signal statistics over time
    ax = axes[2]
    mean_signal = projs_np.mean(axis=1)
    max_signal = projs_np.max(axis=1)
    ax.plot(t_np, mean_signal, label='Mean', color='steelblue', lw=1.5)
    ax.plot(t_np, max_signal, label='Max', color='coral', lw=1.5)
    ax.set_xlabel('Time (s)', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Projection signal', fontsize=_LABEL_FONTSIZE)
    ax.set_title('Projection statistics over time', fontsize=_TITLE_FONTSIZE)
    ax.tick_params(labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Simulation summary (3D)', fontsize=_TITLE_FONTSIZE + 2,
                 fontweight='bold')
    out2 = output_dir / 'simulation_summary_3d.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved 3D simulation summary to %s", out2)


# ─── standalone public helpers ────────────────────────────────────────────────

def plot_projection_frames(
    projs: torch.Tensor,
    t: torch.Tensor,
    receivers,
    n_frames: int = 6,
    output_path: Optional[str | Path] = None,
) -> None:
    """Save (or show) a grid of detector frames at evenly-spaced time points.

    Parameters
    ----------
    projs:
        Projection tensor of shape ``(T, n_rcvrs)``.
    t:
        Time vector of shape ``(T,)``.
    receivers:
        Receiver list as saved by the simulation (``ground_truth["receivers"]``).
    n_frames:
        Number of frames to display.
    output_path:
        If provided, save to this path; otherwise call ``plt.show()``.
    """
    projs_np = projs.cpu().numpy()
    t_np = t.cpu().numpy()
    T = len(t_np)
    d = receivers[0][0].shape[0]
    n_frames = min(n_frames, T)
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4),
                             gridspec_kw={'wspace': 0.3})
    if n_frames == 1:
        axes = [axes]

    vmax = projs_np.max()
    for j, fi in enumerate(frame_indices):
        ax = axes[j]
        ax.set_title(f't = {t_np[fi]:.3f} s', fontsize=_TICK_FONTSIZE)
        if d == 2:
            rcvr_ys = np.array([r[1].item() for r in receivers[0]])
            ax.plot(rcvr_ys, projs_np[fi], color='steelblue', lw=1.2)
            ax.set_ylim(0, vmax * 1.05)
            ax.set_xlabel('Receiver y', fontsize=_TICK_FONTSIZE - 1)
            if j == 0:
                ax.set_ylabel('Intensity', fontsize=_TICK_FONTSIZE - 1)
        else:
            n_y, n_z = _infer_detector_shape(receivers)
            frame = projs_np[fi].reshape(n_y, n_z)
            im = ax.imshow(frame, origin='lower', cmap='viridis',
                           vmin=0, vmax=vmax, aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel('z index', fontsize=_TICK_FONTSIZE - 1)
            if j == 0:
                ax.set_ylabel('y index', fontsize=_TICK_FONTSIZE - 1)
            else:
                ax.set_yticks([])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_true_trajectories(
    theta_true: dict,
    t: torch.Tensor,
    N: int,
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot true GMM centroid trajectories.

    For 2D: single x–y panel.  For 3D: x–y and x–z panels side by side.

    Parameters
    ----------
    theta_true:
        Parameter dictionary as saved in ``ground_truth["theta_true"]``.
    t:
        Time vector.
    N:
        Number of Gaussians.
    output_path:
        If provided, save to this path; otherwise call ``plt.show()``.
    """
    d = theta_true['x0s'][0].shape[0]
    centroids = _compute_centroids(theta_true, t, N)  # (N, T, d)
    colors = _get_colors(N)

    if d == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        panel_axes = [(ax, 0, 1, 'x (depth)', 'y')]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                       gridspec_kw={'wspace': 0.35})
        panel_axes = [
            (ax1, 0, 1, 'x (depth)', 'y'),
            (ax2, 0, 2, 'x (depth)', 'z'),
        ]

    for ax, xi, yi, xlabel, ylabel in panel_axes:
        for k in range(N):
            ax.plot(centroids[k, :, xi], centroids[k, :, yi],
                    color=colors[k], lw=1.5, label=f'$\\rho_{{{k+1}}}$')
            ax.scatter(centroids[k, 0, xi], centroids[k, 0, yi],
                       color=colors[k], s=60, zorder=5, marker='o')
        ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=_LABEL_FONTSIZE)
        ax.tick_params(labelsize=_TICK_FONTSIZE)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('True GMM trajectories', fontsize=_TITLE_FONTSIZE + 2,
                 fontweight='bold')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ─── 3D animation helpers ────────────────────────────────────────────────────

def _compute_rotation_matrix_3d(omegas_k, t_val: float) -> np.ndarray:
    """Replicate ``construct_rotation_matrix_funcs`` from model.py in pure NumPy.

    Returns the 3×3 rotation matrix

        R(t) = G_{01}(t) · G_{02}(t) · G_{12}(t)

    where each G_{ij}(t) is a Givens rotation in plane (i, j) with angular
    frequency ``omegas_k[n]``.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    R = np.eye(3)
    for n, (i, j) in enumerate(pairs):
        angle = 2.0 * np.pi * float(omegas_k[n]) * t_val
        c, s = np.cos(angle), np.sin(angle)
        G = np.eye(3)
        G[i, i] = c
        G[i, j] = -s
        G[j, i] = s
        G[j, j] = c
        R = R @ G
    return R


def _plot_gmm_3d_marginal(
    ax,
    theta_true: dict,
    t_val: float,
    N: int,
    plane: tuple,
    colors,
    artists: list,
) -> None:
    """Draw animated marginal 2D ellipses for a 3D GMM on a coordinate plane.

    Computes the marginal covariance by extracting the (i, j) sub-block of
    the full rotated 3D covariance Σ(t) = R(t) Σ₀ R(t)ᵀ, then draws the
    same concentric chi² ellipse style used in ``plot_gmm_snapshot_animated``.

    Parameters
    ----------
    plane:
        ``(i, j)`` index pair selecting which two coordinate axes to project
        onto — e.g. ``(0, 1)`` for the x–y plane.
    artists:
        Mutable list; all new ``Ellipse`` patches are appended so the caller
        can remove them on the next animation frame.
    """
    from matplotlib.patches import Ellipse

    i_ax, j_ax = plane
    chi2_vals = [1.0, 4.0, 9.0]

    for k in range(N):
        x0 = theta_true['x0s'][k].cpu().numpy()
        v0 = theta_true['v0s'][k].cpu().numpy()
        a0 = theta_true['a0s'][k].cpu().numpy()
        U_skew = theta_true['U_skews'][k].cpu().numpy()
        omegas_k = theta_true['omegas'][k]
        alpha = float(theta_true['alphas'][k])

        mu = x0 + v0 * t_val + 0.5 * a0 * t_val ** 2
        mu_2d = mu[[i_ax, j_ax]]

        R = _compute_rotation_matrix_3d(omegas_k, t_val)
        precision = U_skew.T @ U_skew
        covariance = np.linalg.inv(precision)
        Sigma_full = R @ covariance @ R.T
        Sigma_2d = Sigma_full[np.ix_([i_ax, j_ax], [i_ax, j_ax])]

        for lvl, chi2 in enumerate(chi2_vals):
            Sigma_scaled = chi2 * Sigma_2d
            alpha_ellipse = min(0.8, max(0.1, alpha * (1.0 - lvl * 0.2)))
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma_scaled)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(np.maximum(eigenvalues, 0.0))
            ellipse = Ellipse(
                xy=mu_2d, width=width, height=height, angle=angle,
                facecolor=colors[k], edgecolor='black',
                alpha=alpha_ellipse, linewidth=1.5 - lvl * 0.3,
                linestyle='-', zorder=10 + lvl,
            )
            ax.add_patch(ellipse)
            artists.append(ellipse)


# ─── simulation animation ─────────────────────────────────────────────────────

def animate_simulation(
    sim_dir: str | Path,
    output_path: Optional[str | Path] = None,
    fps: int = 15,
    upsample: int = 8,
    show_trajectories: bool = True,
    title_fontsize: int = 18,
    label_fontsize: int = 16,
    tick_fontsize: int = 13,
):
    """Animate simulation data: GMM in motion (left) + projection profile (right).

    Loads ``ground_truth.pt`` and ``projections.pt`` from *sim_dir* and
    produces a 2-panel animation: the true GMM evolving in space on the left,
    and the corresponding projection profile on the right.

    GMM shapes are re-evaluated on an upsampled time grid to eliminate
    stroboscopic aliasing for fast-rotating Gaussians; projection data uses
    nearest-neighbour lookup on the original saved grid.

    Parameters
    ----------
    sim_dir:
        Simulation output directory (must contain ``ground_truth.pt`` and
        ``projections.pt``).
    output_path:
        Where to save the animation (``.mp4`` or ``.gif``).  Requires
        ``ffmpeg`` on ``$PATH``.  If *None* the animation object is returned
        without saving.
    fps:
        Frames per second for the saved file.
    upsample:
        Integer upsampling factor applied to the saved time grid for GMM
        shape rendering (default 8).
    show_trajectories:
        If *True*, draw full centroid trajectories as a static background.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    from .publication import (
        plot_acquisition_geometry,
        plot_gmm_snapshot_animated,
        plot_trajectories_single,
    )

    sim_dir = Path(sim_dir)
    proj_data = torch.load(sim_dir / "projections.pt", weights_only=True)
    gt_data = torch.load(sim_dir / "ground_truth.pt", weights_only=True)

    projs: torch.Tensor = proj_data["projections"]   # (T, n_rcvrs)
    t: torch.Tensor = proj_data["times"]
    theta_true: dict = gt_data["theta_true"]
    receivers = gt_data["receivers"]
    sources = gt_data["sources"]
    config: dict = gt_data["config"]
    d: int = config["d"]
    N: int = config["N"]

    if d not in (2, 3):
        raise ValueError(f"Unsupported dimensionality d={d}")

    if d == 3:
        # ── 3D animation: x–y panel | x–z panel | detector heatmap ──────
        from matplotlib.animation import FuncAnimation
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Patch

        projs_np = projs.cpu().numpy()      # (T, n_y * n_z)
        t_np = t.cpu().numpy()
        T = len(t_np)
        n_y, n_z = _infer_detector_shape(receivers)
        colors = _get_colors(N)
        proj_max = float(projs_np.max()) or 1.0

        # Geometry extents
        src_pos = sources[0].cpu().numpy()
        rcvr_pts = np.array([r.cpu().numpy() for r in receivers[0]])
        src_x = float(src_pos[0])
        rcvr_x = float(rcvr_pts[0, 0])
        y_min_v, y_max_v = float(rcvr_pts[:, 1].min()), float(rcvr_pts[:, 1].max())
        z_min_v, z_max_v = float(rcvr_pts[:, 2].min()), float(rcvr_pts[:, 2].max())
        x_margin = 0.4
        x_lims = (src_x - x_margin, rcvr_x + x_margin)

        # ── Figure: 3 panels ──────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(
            1, 3, figure=fig, width_ratios=[1.4, 1.4, 1.0],
            wspace=0.12, left=0.06, right=0.97, top=0.88, bottom=0.12,
        )
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_det = fig.add_subplot(gs[0, 2])

        # Spatial panels
        ax_xy.set_xlim(*x_lims)
        ax_xy.set_ylim(y_min_v - 0.2, y_max_v + 0.2)
        ax_xy.set_xlabel('Depth x (m)', fontweight='bold', fontsize=label_fontsize)
        ax_xy.set_ylabel('Width y (m)', fontweight='bold', fontsize=label_fontsize)
        ax_xy.tick_params(labelsize=tick_fontsize)
        ax_xy.grid(True, alpha=0.3, linestyle='--')
        ax_xy.set_facecolor('#f8f9fa')

        ax_xz.set_xlim(*x_lims)
        ax_xz.set_ylim(z_min_v - 0.2, z_max_v + 0.2)
        ax_xz.set_xlabel('Depth x (m)', fontweight='bold', fontsize=label_fontsize)
        ax_xz.set_ylabel('Height z (m)', fontweight='bold', fontsize=label_fontsize)
        ax_xz.tick_params(labelsize=tick_fontsize)
        ax_xz.grid(True, alpha=0.3, linestyle='--')
        ax_xz.set_facecolor('#f8f9fa')

        # Static geometry: source star + receiver dots (marginal positions)
        y_unique = np.unique(rcvr_pts[:, 1])
        z_unique = np.unique(rcvr_pts[:, 2])
        ax_xy.scatter([src_pos[0]], [src_pos[1]], marker='*', color='crimson',
                      s=200, zorder=10, clip_on=False)
        ax_xy.scatter(np.full_like(y_unique, rcvr_x), y_unique,
                      color='steelblue', s=12, alpha=0.5, zorder=5)
        ax_xz.scatter([src_pos[0]], [src_pos[2]], marker='*', color='crimson',
                      s=200, zorder=10, clip_on=False)
        ax_xz.scatter(np.full_like(z_unique, rcvr_x), z_unique,
                      color='steelblue', s=12, alpha=0.5, zorder=5)

        # Static trajectories
        if show_trajectories:
            centroids = _compute_centroids(theta_true, t, N)  # (N, T, 3)
            for k in range(N):
                ax_xy.plot(centroids[k, :, 0], centroids[k, :, 1],
                           color=colors[k], lw=1.5, alpha=0.6, linestyle='--', zorder=3)
                ax_xz.plot(centroids[k, :, 0], centroids[k, :, 2],
                           color=colors[k], lw=1.5, alpha=0.6, linestyle='--', zorder=3)

        legend_elems = [Patch(facecolor=colors[k], edgecolor='black',
                              label=f'$\\rho_{{{k+1}}}$') for k in range(N)]
        ax_xy.legend(handles=legend_elems, loc='upper left',
                     fontsize=13, framealpha=0.9)

        # Detector heatmap
        # frame.reshape(n_y, n_z): rows=y (y_max at row 0 due to CT flip), cols=z
        # .T → rows=z (z_min at row 0), cols=y (y_max at col 0)
        # fliplr → cols now y_min…y_max (left to right)
        # origin='lower' → z_min at bottom, z_max at top  (height increases upward)
        # extent=[y_min, y_max, z_min, z_max] gives correct coordinate axes
        def _det_frame(idx: int) -> np.ndarray:
            return np.fliplr(projs_np[idx].reshape(n_y, n_z).T)

        im_det = ax_det.imshow(
            _det_frame(0), origin='lower', cmap='viridis',
            vmin=0, vmax=proj_max, aspect='auto',
            extent=[y_min_v, y_max_v, z_min_v, z_max_v],
        )
        fig.colorbar(im_det, ax=ax_det, fraction=0.046, pad=0.04)
        ax_det.set_xlabel('Width y (m)', fontweight='bold', fontsize=label_fontsize)
        ax_det.set_ylabel('Height z (m)', fontweight='bold', fontsize=label_fontsize)
        ax_det.tick_params(labelsize=tick_fontsize)
        ax_det.set_title('Detector', fontweight='bold',
                         fontsize=title_fontsize, pad=10)

        # Upsampled time grid
        upsample = max(1, int(upsample))
        t_start, t_end = float(t_np[0]), float(t_np[-1])
        n_frames = T * upsample
        t_anim = np.linspace(t_start, t_end, n_frames)

        spatial_artists: list = []

        def init_3d():
            ax_xy.set_title(f'x–y view  (t = {t_start:.3f} s)',
                            fontweight='bold', fontsize=title_fontsize, pad=10)
            ax_xz.set_title(f'x–z view  (t = {t_start:.3f} s)',
                            fontweight='bold', fontsize=title_fontsize, pad=10)
            return []

        def update_3d(frame):
            for artist in spatial_artists:
                artist.remove()
            spatial_artists.clear()

            t_val = t_anim[frame]
            data_idx = int(np.argmin(np.abs(t_np - t_val)))

            ax_xy.set_title(f'x–y view  (t = {t_val:.3f} s)',
                            fontweight='bold', fontsize=title_fontsize, pad=10)
            ax_xz.set_title(f'x–z view  (t = {t_val:.3f} s)',
                            fontweight='bold', fontsize=title_fontsize, pad=10)

            _plot_gmm_3d_marginal(ax_xy, theta_true, t_val, N, (0, 1),
                                  colors, spatial_artists)
            _plot_gmm_3d_marginal(ax_xz, theta_true, t_val, N, (0, 2),
                                  colors, spatial_artists)
            im_det.set_data(_det_frame(data_idx))

            return spatial_artists + [im_det]

        interval_ms = (t_end - t_start) * 1000 / n_frames
        anim_3d = FuncAnimation(
            fig, update_3d, init_func=init_3d, frames=n_frames,
            interval=interval_ms, blit=False, repeat=True,
        )

        if output_path:
            output_path = Path(output_path)
            fps_save = n_frames / (t_end - t_start)
            logger.info("Saving 3D simulation animation to %s ...", output_path)
            anim_3d.save(str(output_path), writer='ffmpeg', fps=fps_save)
            logger.info("Saved: %s", output_path)

        return anim_3d

    projs_np = projs.cpu().numpy()          # (T, n_rcvrs)
    t_np = t.cpu().numpy()
    T = len(t_np)

    # Receiver heights and sort order for the profile panel
    rcvr_heights = np.array([r[1].item() for r in receivers[0]])
    sort_idx = np.argsort(rcvr_heights)
    sorted_heights = rcvr_heights[sort_idx]

    proj_max = float(projs_np.max())
    proj_margin = proj_max * 0.05
    colors = _get_colors(N)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.10, width_ratios=[1.5, 1.0],
                  left=0.07, right=0.97, top=0.88, bottom=0.12)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ── Left panel — spatial view ─────────────────────────────────────────
    src_x = sources[0][0].item()
    rcvr_x = receivers[0][0][0].item()
    x_margin = 0.4
    y_min = sorted_heights.min() - 0.2
    y_max = sorted_heights.max() + 0.2

    ax_left.set_xlim(src_x - x_margin, rcvr_x + x_margin)
    ax_left.set_ylim(y_min, y_max)
    ax_left.set_xlabel('Depth (m)', fontweight='bold', fontsize=label_fontsize)
    ax_left.set_ylabel('Height (m)', fontweight='bold', fontsize=label_fontsize)
    ax_left.tick_params(labelsize=tick_fontsize)
    ax_left.grid(True, alpha=0.3, linestyle='--')
    ax_left.set_facecolor('#f8f9fa')

    legend_elems = [Patch(facecolor=colors[k], edgecolor='black',
                          label=f'$\\rho_{{{k+1}}}$') for k in range(N)]
    ax_left.legend(handles=legend_elems, loc='upper left',
                   fontsize=13, framealpha=0.9)

    if show_trajectories:
        plot_trajectories_single(ax_left, theta_true, t, N, colors, mirror=False)
    plot_acquisition_geometry(ax_left, sources, receivers, d, mirror=False)

    # ── Right panel — projection profile ─────────────────────────────────
    ax_right.set_xlim(-proj_margin, proj_max + proj_margin)
    ax_right.set_ylim(y_min, y_max)
    ax_right.set_xlabel('Intensity', fontweight='bold', fontsize=label_fontsize)
    ax_right.tick_params(axis='x', labelsize=tick_fontsize)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.grid(True, alpha=0.3, linestyle='--')
    ax_right.set_facecolor('#ffffff')
    ax_right.set_title('Projection', fontweight='bold',
                       fontsize=title_fontsize, pad=10)

    # ── Upsampled time grid for GMM rendering ─────────────────────────────
    upsample = max(1, int(upsample))
    t_start, t_end = float(t_np[0]), float(t_np[-1])
    n_frames = T * upsample
    t_anim = np.linspace(t_start, t_end, n_frames)

    left_artists: list = []
    right_artists: list = []

    def init():
        ax_left.set_title(f'Simulated GMM  (t = {t_start:.3f} s)',
                          fontweight='bold', fontsize=title_fontsize, pad=10)
        return []

    def update(frame):
        for artist in left_artists + right_artists:
            artist.remove()
        left_artists.clear()
        right_artists.clear()

        t_val = t_anim[frame]
        data_idx = int(np.argmin(np.abs(t_np - t_val)))

        ax_left.set_title(f'Simulated GMM  (t = {t_val:.3f} s)',
                          fontweight='bold', fontsize=title_fontsize, pad=10)

        # Animated GMM ellipses
        plot_gmm_snapshot_animated(ax_left, theta_true, t_val, N, d, colors,
                                   left_artists, is_true=True, mirror=False)

        # Projection profile at nearest data frame
        proj_frame = projs_np[data_idx][sort_idx]
        (line,) = ax_right.plot(proj_frame, sorted_heights,
                                color='black', lw=2.0, alpha=0.85)
        right_artists.append(line)

        return left_artists + right_artists

    interval_ms = (t_end - t_start) * 1000 / n_frames
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=interval_ms, blit=False, repeat=True)

    if output_path:
        output_path = Path(output_path)
        fps_save = n_frames / (t_end - t_start)
        logger.info("Saving simulation animation to %s ...", output_path)
        anim.save(str(output_path), writer='ffmpeg', fps=fps_save)
        logger.info("Saved: %s", output_path)

    return anim


# ─── interactive 3D animation (Plotly) ─────────────────────────────────────
def animate_simulation_interactive(
    sim_dir: str | Path,
    output_path: Optional[str | Path] = None,
    upsample: int = 4,
    show_trajectories: bool = True,
    detector_panel: bool = False,
    n_ring_points: int = 40,
    chi2_levels: tuple = (1.0, 4.0),
    scene_padding: float = 0.5,
    camera: Optional[dict] = None,
    title: Optional[str] = None,
):
    """Animate 3D GMM simulation as a lightweight interactive Plotly HTML.

    Each Gaussian is drawn as three principal-axis wireframe rings
    (``go.Scatter3d``) rather than a surface mesh.  This is ~50× lighter per
    frame, giving smooth playback without JavaScript lag.

    Layout
    ------
    Left : 3D object space — ellipsoid rings, centroid markers, static
           trajectory lines, source star, detector-plane outline.
    Right: detector heatmap (optional), synchronised to the animation time.

    Parameters
    ----------
    sim_dir:
        Path to a simulation output directory.
    output_path:
        Write to an HTML file (``include_plotlyjs='cdn'``).  If *None*, calls
        ``fig.show()``.
    upsample:
        Integer upsampling factor applied to the saved time grid (default 4).
    show_trajectories:
        Draw static full centroid trajectories in the 3D scene.
    detector_panel:
        Include the synchronised detector heatmap subplot.
    n_ring_points:
        Points per ring arc (default 60; higher → smoother circles).
    chi2_levels:
        χ² confidence levels to draw (default ``(1.0, 4.0)`` ≈ 1σ / 2σ).
    scene_padding:
        Extra margin (m) added around the scene on all sides (default 1.5).
        Increase to zoom out; decrease to zoom in.
    camera:
        Plotly camera ``dict(eye=dict(x=..., y=..., z=...))``; larger eye
        values zoom out further.  Defaults to a wide isometric-ish view.
    title:
        Figure title string.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sim_dir   = Path(sim_dir)
    proj_data = torch.load(sim_dir / "projections.pt", weights_only=True)
    gt_data   = torch.load(sim_dir / "ground_truth.pt", weights_only=True)

    projs: torch.Tensor = proj_data["projections"]   # (T, n_rcvrs)
    t: torch.Tensor     = proj_data["times"]
    theta_true: dict    = gt_data["theta_true"]
    receivers           = gt_data["receivers"]
    sources             = gt_data["sources"]
    config: dict        = gt_data["config"]
    d: int              = config["d"]
    N: int              = config["N"]
    assert d == 3, "animate_simulation_interactive requires d=3."

    projs_np = projs.cpu().numpy()
    t_np     = t.cpu().numpy()
    T        = len(t_np)
    n_y, n_z = _infer_detector_shape(receivers)

    colors = [
        f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        for c in _get_colors(N)
    ]

    upsample = max(1, int(upsample))
    t_start, t_end = float(t_np[0]), float(t_np[-1])
    n_frames = T * upsample
    t_anim   = np.linspace(t_start, t_end, n_frames)
    # Cap at 30 FPS — browsers can't render faster and it just causes lag
    frame_ms = max((t_end - t_start) * 1000.0 / n_frames, 33.0)

    # ── Geometry ──────────────────────────────────────────────────────────
    src_pos  = sources[0].cpu().numpy()
    rcvr_pts = np.array([r.cpu().numpy() for r in receivers[0]])
    rcvr_x   = float(rcvr_pts[0, 0])
    y_min_v, y_max_v = float(rcvr_pts[:, 1].min()), float(rcvr_pts[:, 1].max())
    z_min_v, z_max_v = float(rcvr_pts[:, 2].min()), float(rcvr_pts[:, 2].max())

    # ── Detector surface grids (constant; reused in every frame) ──────────
    y_1d = np.linspace(y_min_v, y_max_v, n_y)
    z_1d = np.linspace(z_min_v, z_max_v, n_z)
    Z_surf, Y_surf = np.meshgrid(z_1d, y_1d)   # (n_y, n_z)
    X_surf = np.full_like(Y_surf, rcvr_x)

    # ── Pre-compute trajectories (used for axis bounds and static traces) ──
    centroids = _compute_centroids(theta_true, t, N)  # (N, T, 3)
    all_pts   = centroids.reshape(-1, 3)
    pad = scene_padding
    x_range = [min(float(all_pts[:, 0].min()), src_pos[0]) - pad,
               max(float(all_pts[:, 0].max()), rcvr_x) + pad]
    y_range = [min(float(all_pts[:, 1].min()), y_min_v) - pad * 0.4,
               max(float(all_pts[:, 1].max()), y_max_v) + pad * 0.4]
    z_range = [min(float(all_pts[:, 2].min()), z_min_v) - pad * 0.4,
               max(float(all_pts[:, 2].max()), z_max_v) + pad * 0.4]

    # ── Helpers ───────────────────────────────────────────────────────────
    def _gmm_at(t_val):
        """Return list of (mu, Sigma) per Gaussian at scalar t_val."""
        out = []
        for k in range(N):
            x0     = theta_true['x0s'][k].cpu().numpy()
            v0     = theta_true['v0s'][k].cpu().numpy()
            a0     = theta_true['a0s'][k].cpu().numpy()
            U_skew = theta_true['U_skews'][k].cpu().numpy()
            mu     = x0 + v0 * t_val + 0.5 * a0 * t_val ** 2
            R      = _compute_rotation_matrix_3d(theta_true['omegas'][k], t_val)
            Sigma  = R @ np.linalg.inv(U_skew.T @ U_skew) @ R.T
            out.append((mu, Sigma))
        return out

    def _rings(mu, Sigma, chi2):
        """Three principal-axis rings for the ellipsoid at chi2 confidence level."""
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        radii = np.sqrt(np.maximum(eigvals, 0.0) * chi2)
        th    = np.linspace(0, 2 * np.pi, n_ring_points)
        rings = []
        for i_ax, j_ax in [(0, 1), (0, 2), (1, 2)]:
            pts = (np.outer(np.cos(th), eigvecs[:, i_ax] * radii[i_ax])
                   + np.outer(np.sin(th), eigvecs[:, j_ax] * radii[j_ax])
                   + mu)
            rings.append(pts)   # (n_ring_points, 3)
        return rings

    def _det_frame(data_idx):
        return np.fliplr(projs_np[data_idx].reshape(n_y, n_z).T)

    # ── Figure ────────────────────────────────────────────────────────────
    specs  = [[{"type": "scene"}]]
    if detector_panel:
        specs[0].append({"type": "xy"})
    fig = make_subplots(
        rows=1, cols=2 if detector_panel else 1,
        specs=specs,
        subplot_titles=["3D object space", "Detector"] if detector_panel
                       else ["3D object space"],
        column_widths=[0.65, 0.35] if detector_panel else [1.0],
        horizontal_spacing=0.06,
    )

    # ── Static traces ─────────────────────────────────────────────────────
    # Added first so their indices are 0 … n_static-1 and never touched by
    # animation frames.
    n_static = 0

    if show_trajectories:
        for k in range(N):
            fig.add_trace(go.Scatter3d(
                x=centroids[k, :, 0], y=centroids[k, :, 1], z=centroids[k, :, 2],
                mode='lines',
                line=dict(color=colors[k], width=3),
                name=f'Traj ρ{k+1}',
                hoverinfo='skip',
            ), row=1, col=1)
            n_static += 1

    # Source
    fig.add_trace(go.Scatter3d(
        x=[src_pos[0]], y=[src_pos[1]], z=[src_pos[2]],
        mode='markers',
        marker=dict(symbol='diamond', size=8, color='red'),
        name='Source', hoverinfo='skip',
    ), row=1, col=1)
    n_static += 1

    # X-ray beams — four corner rays from source to detector
    ray_x, ray_y, ray_z = [], [], []
    for cy, cz in [(y_min_v, z_min_v), (y_max_v, z_min_v),
                   (y_max_v, z_max_v), (y_min_v, z_max_v)]:
        ray_x += [float(src_pos[0]), rcvr_x, None]
        ray_y += [float(src_pos[1]), cy, None]
        ray_z += [float(src_pos[2]), cz, None]
    fig.add_trace(go.Scatter3d(
        x=ray_x, y=ray_y, z=ray_z,
        mode='lines',
        line=dict(color='rgba(255,255,180,0.85)', width=2),
        name='X-ray beams', hoverinfo='skip', showlegend=True,
    ), row=1, col=1)
    n_static += 1

    # Detector-plane border
    fig.add_trace(go.Scatter3d(
        x=[rcvr_x]*5,
        y=[y_min_v, y_max_v, y_max_v, y_min_v, y_min_v],
        z=[z_min_v, z_min_v, z_max_v, z_max_v, z_min_v],
        mode='lines',
        line=dict(color='steelblue', width=2, dash='dot'),
        name='Detector border', hoverinfo='skip', showlegend=False,
    ), row=1, col=1)
    n_static += 1

    # ── Animated trace skeleton at t=0 ────────────────────────────────────
    # Layout: [chi2 levels × N gaussians (3 rings combined)] + [N centroids]
    #         + [detector surface] + [heatmap?]
    n_chi2     = len(chi2_levels)
    n_animated = n_chi2 * N + N + 1 + (1 if detector_panel else 0)
    anim_indices = list(range(n_static, n_static + n_animated))

    _proj_min = float(projs_np.min())
    _proj_max = float(projs_np.max())

    gmm0 = _gmm_at(t_anim[0])

    for lvl_idx, chi2 in enumerate(chi2_levels):
        lw      = max(1, 4 - lvl_idx)
        opacity = max(0.3, 0.9 - lvl_idx * 0.35)
        for k, (mu, Sigma) in enumerate(gmm0):
            rx, ry, rz = [], [], []
            for ring in _rings(mu, Sigma, chi2):
                rx.extend(ring[:, 0].tolist() + [None])
                ry.extend(ring[:, 1].tolist() + [None])
                rz.extend(ring[:, 2].tolist() + [None])
            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode='lines',
                line=dict(color=colors[k], width=lw),
                opacity=opacity,
                name=f'ρ{k+1} χ²={chi2:.0f}',
                showlegend=(lvl_idx == 0),
                hoverinfo='skip',
            ), row=1, col=1)

    for k, (mu, _) in enumerate(gmm0):
        fig.add_trace(go.Scatter3d(
            x=[mu[0]], y=[mu[1]], z=[mu[2]],
            mode='markers',
            marker=dict(size=7, color=colors[k],
                        line=dict(color='black', width=1)),
            name=f'ρ{k+1}', showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

    # Animated detector surface — flat screen coloured by projection intensity
    fig.add_trace(go.Surface(
        x=X_surf, y=Y_surf, z=Z_surf,
        surfacecolor=projs_np[0].reshape(n_y, n_z),
        colorscale='Viridis',
        cmin=_proj_min, cmax=_proj_max,
        showscale=False,
        opacity=0.95,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
        hoverinfo='skip',
        name='Detector screen',
    ), row=1, col=1)

    if detector_panel:
        y_ax = np.linspace(y_min_v, y_max_v, n_y)
        z_ax = np.linspace(z_min_v, z_max_v, n_z)
        fig.add_trace(go.Heatmap(
            z=_det_frame(0), x=y_ax, y=z_ax,
            colorscale='Viridis',
            zmin=float(projs_np.min()), zmax=float(projs_np.max()),
            colorbar=dict(title='Intensity', len=0.6, y=0.5),
        ), row=1, col=2)

    # ── Animation frames ──────────────────────────────────────────────────
    plotly_frames = []
    for frame_idx, t_val in enumerate(t_anim):
        data_idx  = int(np.argmin(np.abs(t_np - t_val)))
        gmm_t     = _gmm_at(t_val)
        frame_data = []

        for lvl_idx, chi2 in enumerate(chi2_levels):
            for k, (mu, Sigma) in enumerate(gmm_t):
                rx, ry, rz = [], [], []
                for ring in _rings(mu, Sigma, chi2):
                    rx.extend(ring[:, 0].tolist() + [None])
                    ry.extend(ring[:, 1].tolist() + [None])
                    rz.extend(ring[:, 2].tolist() + [None])
                frame_data.append(go.Scatter3d(x=rx, y=ry, z=rz, mode='lines'))

        for k, (mu, _) in enumerate(gmm_t):
            frame_data.append(go.Scatter3d(
                x=[mu[0]], y=[mu[1]], z=[mu[2]], mode='markers',
            ))

        # Detector surface — only surfacecolor changes; x/y/z kept from base trace
        frame_data.append(go.Surface(
            surfacecolor=projs_np[data_idx].reshape(n_y, n_z),
        ))

        if detector_panel:
            frame_data.append(go.Heatmap(z=_det_frame(data_idx)))

        base_title = title or "3D GMM-CT Simulation"
        plotly_frames.append(go.Frame(
            data=frame_data,
            traces=anim_indices,
            name=str(frame_idx),
            layout=dict(title=dict(text=f"{base_title}  —  t = {t_val:.3f} s")),
        ))

    fig.frames = plotly_frames

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title or "3D GMM-CT Simulation", x=0.5),
        height=660,
        margin=dict(l=10, r=10, t=65, b=10),
        updatemenus=[{
            "type": "buttons",
            "showactive": True,
            "y": 1.06, "x": 0.5, "xanchor": "center",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": frame_ms, "redraw": True},
                        "transition": {"duration": 0},
                        "fromcurrent": True, "mode": "immediate",
                    }],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    }],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {
                "prefix": "t = ", "suffix": " s",
                "visible": True, "xanchor": "center",
            },
            "pad": {"t": 50, "b": 10},
            "steps": [
                {
                    "args": [[str(i)], {
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    }],
                    "label": f"{t_anim[i]:.3f}",
                    "method": "animate",
                }
                for i in range(n_frames)
            ],
        }],
    )

    # Lock scene axes — prevents drift across frames
    ax_range_kw = dict(autorange=False, showspikes=False)
    fig.update_scenes(
        xaxis=dict(title="x (depth)", range=x_range, **ax_range_kw),
        yaxis=dict(title="y (width)", range=y_range, **ax_range_kw),
        zaxis=dict(title="z (height)", range=z_range, **ax_range_kw),
        aspectmode="data",
        camera=camera or dict(eye=dict(x=1.3, y=1.8, z=0.5)),
    )
    if detector_panel:
        fig.update_xaxes(title_text="Width y (m)", row=1, col=2)
        fig.update_yaxes(title_text="Height z (m)", row=1, col=2)

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs='cdn')
        logger.info("Saved interactive animation to %s", output_path)
    else:
        fig.show()

    return fig



def export_poster_gmm_figure(
    sim_dir: str | Path,
    timestamps: list[float] = [0.37, 0.83, 0.98],
    output_path: str = "poster_gmm_snapshots.pdf",
):
    """Generates a 3-row figure showing key snapshots of the dynamic 2D GMM simulation

    and its projection profiles for poster inclusion.
    """
    sim_dir = Path(sim_dir)
    proj_data = torch.load(sim_dir / "projections.pt", weights_only=True)
    gt_data = torch.load(sim_dir / "ground_truth.pt", weights_only=True)

    projs_np = proj_data["projections"].cpu().numpy()  # (T, n_rcvrs)
    t_np = proj_data["times"].cpu().numpy()
    theta_true = gt_data["theta_true"]
    receivers = gt_data["receivers"]
    sources = gt_data["sources"]
    config = gt_data["config"]
    d, N = config["d"], config["N"]

    # Receiver geometry & heights
    rcvr_heights = np.array([r[1].item() for r in receivers[0]])
    sort_idx = np.argsort(rcvr_heights)
    sorted_heights = rcvr_heights[sort_idx]

    # Limits & geometry
    src_x = sources[0][0].item()
    rcvr_x = receivers[0][0][0].item()
    x_margin = 0.4
    y_min = sorted_heights.min() - 0.2
    y_max = sorted_heights.max() + 0.5
    proj_max = float(projs_np.max())
    proj_margin = proj_max * 0.05
    colors = _get_colors(N)

    # Figure Layout: 3 rows (one per timestamp), 2 columns (spatial left, projection right)
    n_rows = len(timestamps)
    fig = plt.figure(figsize=(10, 3.8 * n_rows), dpi=300)
    gs = GridSpec(
        n_rows, 2, figure=fig, wspace=0.05, hspace=0.25, width_ratios=[1.1, 1.0]
    )

    for i, t_target in enumerate(timestamps):
        ax_left = fig.add_subplot(gs[i, 0])
        ax_right = fig.add_subplot(gs[i, 1])

        # Find closest time index in simulation data
        data_idx = int(np.argmin(np.abs(t_np - t_target)))
        t_actual = t_np[data_idx]

        # ── LEFT PANEL: Spatial GMM Motion ─────────────────────────────────
        ax_left.set_xlim(src_x - x_margin, rcvr_x + x_margin)
        ax_left.set_ylim(y_min, y_max)
        if i == len(timestamps) - 1:
            ax_left.set_xlabel("Depth (m)", fontweight="bold", fontsize=12)
        ax_left.set_ylabel("Detector Height (m)", fontweight="bold", fontsize=12)
        ax_left.tick_params(axis="both", labelsize=14)
        ax_left.grid(True, alpha=0.3, linestyle="--")
        ax_left.set_facecolor("#f8f9fa")
        ax_left.set_title(
            f"State Snapshot (t = {t_actual:.2f} s)",
            fontweight="bold",
            fontsize=14,
        )

        # Draw acquisition geometry, static trajectories, and animated GMM ellipses
        plot_trajectories_single(
            ax_left, theta_true, proj_data["times"], N, colors, mirror=False
        )
        plot_acquisition_geometry(ax_left, sources, receivers, d, mirror=False)

        dummy_artists = []
        plot_gmm_snapshot_animated(
            ax_left,
            theta_true,
            t_actual,
            N,
            d,
            colors,
            dummy_artists,
            is_true=True,
            mirror=False,
        )

        if i == 0:
            legend_elems = [
                Patch(
                    facecolor=colors[k],
                    edgecolor="black",
                    label=f"$\\rho_{{{k+1}}}$",
                )
                for k in range(N)
            ]
            ax_left.legend(
                handles=legend_elems, loc="upper left", fontsize=10, framealpha=0.9
            )

        # ── RIGHT PANEL: Projection Profile ──────────────────────────────
        ax_right.set_xlim(-proj_margin, proj_max + proj_margin)
        ax_right.set_ylim(y_min, y_max)
        if i < len(timestamps) - 1:
            ax_right.set_xticklabels([])
        if i == len(timestamps) - 1:
            ax_right.set_xlabel("Intensity", fontweight="bold", fontsize=12)
        ax_right.tick_params(axis="both", labelsize=14)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.grid(True, alpha=0.3, linestyle="--")
        ax_right.set_facecolor("#ffffff")
        ax_right.set_title(
            f"Projection Profile (t = {t_actual:.2f} s)",
            fontweight="bold",
            fontsize=14,
        )

        proj_frame = projs_np[data_idx][sort_idx]
        ax_right.plot(
            proj_frame, sorted_heights, color="black", lw=2.0, alpha=0.85
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Poster figure saved successfully to {output_path}")


def export_poster_snapshot_sinogram_figure(
    sim_dir: str | Path,
    timestamps: list[float] = [0.37, 0.83, 0.98],
    output_path: str = "poster_gmm_and_sinogram.pdf",
):
    sim_dir = Path(sim_dir)
    proj_data = torch.load(sim_dir / "projections.pt", weights_only=True)
    gt_data = torch.load(sim_dir / "ground_truth.pt", weights_only=True)

    projs_np = proj_data["projections"].cpu().numpy()  # (T, n_rcvrs)
    t_np = proj_data["times"].cpu().numpy()
    theta_true = gt_data["theta_true"]
    receivers = gt_data["receivers"]
    sources = gt_data["sources"]
    config = gt_data["config"]
    d, N = config["d"], config["N"]

    rcvr_heights = np.array([r[1].item() for r in receivers[0]])
    sort_idx = np.argsort(rcvr_heights)
    sorted_heights = rcvr_heights[sort_idx]
    sorted_projs = projs_np[:, sort_idx]  # (T, n_rcvrs)

    src_x = sources[0][0].item()
    rcvr_x = receivers[0][0][0].item()
    x_margin = 0.4
    y_min, y_max = sorted_heights.min() - 0.2, sorted_heights.max() + 0.2
    colors = _get_colors(N)

    # ── Master Figure (2 Rows: Top 1x3 snapshots, Bottom wide sinogram) ──────
    fig = plt.figure(figsize=(14, 9), dpi=300)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.2, 1.2],
        wspace=0.18,
        hspace=0.35,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.08,
    )

    # ── ROW 1: SPATIAL SNAPSHOTS (3 Columns) ──────────────────────────────
    for i, t_target in enumerate(timestamps):
        ax = fig.add_subplot(gs[0, i])
        data_idx = int(np.argmin(np.abs(t_np - t_target)))
        t_actual = t_np[data_idx]

        ax.set_xlim(src_x - x_margin, rcvr_x + x_margin)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Depth (m)", fontweight="bold", fontsize=18)
        if i == 0:
            ax.set_ylabel("Detector Height (m)", fontweight="bold", fontsize=18)
        else:
            ax.tick_params(labelleft=False)

        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#f8f9fa")
        ax.set_title(
            f"State Snapshot ($t = {t_actual:.2f}\\text{{s}}$)",
            fontweight="bold",
            fontsize=18,
        )

        plot_trajectories_single(
            ax, theta_true, proj_data["times"], N, colors, mirror=False
        )
        plot_acquisition_geometry(ax, sources, receivers, d, mirror=False)

        dummy_artists = []
        plot_gmm_snapshot_animated(
            ax,
            theta_true,
            t_actual,
            N,
            d,
            colors,
            dummy_artists,
            is_true=True,
            mirror=False,
        )

    # ── ROW 2: FULL OBSERVED SINOGRAM (Spans all 3 columns) ─────────────────
    ax_sino = fig.add_subplot(gs[1, :])

    # Plot sinogram heatmap (Time on X, Height on Y)
    # sorted_projs shape is (T, n_rcvrs), transpose to (n_rcvrs, T) for imshow
    im = ax_sino.imshow(
        sorted_projs.T,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[t_np.min(), t_np.max(), sorted_heights.min(), sorted_heights.max()],
    )
    t_start_crop = 0.5
    t_end_crop = 1.3
    ax_sino.set_xlim(t_start_crop, t_end_crop)

    cbar = fig.colorbar(im, ax=ax_sino, pad=0.015, fraction=0.02)
    cbar.set_label(
        "Projection intensity", fontweight="bold", fontsize=18, labelpad=10
    )

    ax_sino.tick_params(axis="both", labelsize=14)
    ax_sino.set_xlabel("Time (s)", fontweight="bold", fontsize=18)
    ax_sino.set_ylabel("Detector Height (m)", fontweight="bold", fontsize=18)
    ax_sino.set_title("Dynamic Sinogram", fontweight="bold", fontsize=18, pad=8)

    # Add vertical indicator lines pointing to the timestamps from Row 1
    # for t_target in timestamps:
    #     ax_sino.axvline(
    #         x=t_target, color="white", linestyle="--", linewidth=1.5, alpha=0.85
    #     )

    plt.savefig(output_path, bbox_inches="tight")
    print(f"Composite figure saved successfully to {output_path}")