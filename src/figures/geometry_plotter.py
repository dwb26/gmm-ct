"""Figure: particle states and their projections at two times, plus the full sinogram."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data import Run
from .primitives import draw_geometry, draw_gaussians, draw_projection_profile
from .registry import figure
from .poster_style import SINOGRAM_CMAP, STATE_FACE, WIDTH, particle_colors, slice_colors


@figure("plot_geometry")
