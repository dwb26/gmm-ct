"""Manuscript figures. ``load_run`` once, then ``make_figures(run, names=[...])``."""

from . import fig_gaussians, fig_modes, fig_plot_geometry, fig_state_sinogram, fig_temporal  # noqa: F401  (register figures)
from .data import Run, load_run
from .registry import FIGURES, figure, make_figures

__all__ = ["Run", "load_run", "FIGURES", "figure", "make_figures"]
