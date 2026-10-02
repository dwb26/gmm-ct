"""Shared poster-style look for every figure: rcParams, palette and helpers."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Polygon

DPI = 300
WIDTH = 10.0  # inches
SPINE_GREY = "#BBBBBB"

# One colour per particle n = 1..5 (blue, red, green, purple, orange); reused in every panel
PARTICLE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# Accent colours for the selected time slices (purple, coral)
SLICE_COLORS = ["#7B2CBF", "#E8685F"]

COLOR_DATA = "black"
COLOR_DETECTOR = "blue"
COLOR_SOURCE = "red"
COLOR_RAY = "gold"
STATE_FACE = "#f8f9fa"
GRID_COLOR = "#e5e7eb"
COLOR_FAN = "#FFB000"
SINOGRAM_CMAP = "viridis"

FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND = 9, 11, 12, 9

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": SPINE_GREY,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.7,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.6,
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "savefig.dpi": DPI,
}


def style_context():
    return mpl.rc_context(RC)


def particle_colors(n: int):
    """Rainbow palette, as in poster_plot.py."""
    return [tuple(c) for c in mpl.colormaps["rainbow"](np.linspace(0, 1, n))]


def slice_colors(n: int):
    return [SLICE_COLORS[k % len(SLICE_COLORS)] for k in range(n)]


def draw_rays(ax, source, detector_x: float, heights):
    """Faint gold rays from the source to every receiver."""
    for n_h, h in enumerate(heights):
        ax.plot([source[0], detector_x], [source[1], h], color=COLOR_RAY, linewidth=0.3,
                alpha=0.15, zorder=5, label="Rays" if n_h == 0 else None)


def save_figure(fig, path: Path, preview: bool = True) -> Path:
    """Save ``path`` (vector for pdf/svg) and, for vector formats, a PNG preview alongside."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=DPI, facecolor="white")
    if preview and path.suffix.lower() in (".pdf", ".svg", ".eps"):
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=DPI, facecolor="white")
    return path
