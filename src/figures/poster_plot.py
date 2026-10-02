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

from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from ..visualization.publication import (
    plot_acquisition_geometry,
    plot_gmm_snapshot_animated,
    plot_trajectories_single,
)

_LABEL_FONTSIZE = 16
_TITLE_FONTSIZE = 18
_TICK_FONTSIZE = 13

def _get_colors(N: int):
    return cm.rainbow(np.linspace(0, 1, N))

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

    projs_np = proj_data["projections"].cpu().detach().numpy()  # (T, n_rcvrs)
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

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(sim_dir / output_path, bbox_inches="tight")


export_poster_gmm_figure("/Users/danburrows/Documents/Projects/gmm-ct/data/snr40_N8_nproj128_seed42")