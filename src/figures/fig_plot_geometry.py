"""Figure: particle states and their projections at two times, plus the full sinogram."""

import matplotlib.pyplot as plt

from .data import Run
from .primitives import draw_geometry
from .registry import figure
from .poster_style import WIDTH


@figure("plot_geometry")
def plot_geometry(run: Run) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 1)

    ymin, ymax = run.y[0] - 0.1, run.y[-1] + 0.1
    centers = run.centers(run.theta_true, run.t)
    xlo = min(run.source[0], centers[..., 0].min()) - 0.3
    xhi = run.detector_x + 0.2
    
    ax = fig.add_subplot(gs[0])
    draw_geometry(ax, run)
    
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Depth", fontweight="bold", fontsize=18)
    ax.set_ylabel("Detector height", fontweight="bold", fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    
    ax.legend(fontsize=16, fancybox=True, markerscale=1.5, frameon=True)    
    
    return fig