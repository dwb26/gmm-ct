"""
Gaussian Mixture CT Reconstruction

A Python package for reconstructing dynamic objects in CT imaging using
Gaussian Mixture Models with motion estimation.
"""

import logging

__version__ = "0.2.0"
__author__ = "Daniel Burrows"

# Library-level null handler — callers configure their own logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import main classes and functions for convenient access
from .config import (
    GRAVITATIONAL_ACCELERATION,
    load_reconstruct_config,
    load_simulate_config,
    ReconstructConfig,
    SimulateConfig,
)
from .model import GMM_reco, NewtonRaphsonLBFGS
from .utils import construct_receivers, generate_true_param, set_random_seeds, export_parameters
from .simulation import run_simulation
from .reconstruct import run_reconstruction, analyse_results
from .visualization.animations import (
    save_GMM_animation,
    save_projection_comparison_animation,
    save_GMM_with_projection_comparison,
    save_optimization_stages_animation,
)
from .visualization.publication import (
    plot_individual_gaussian_reconstruction,
    plot_temporal_gmm_comparison,
    animate_temporal_gmm_comparison,
    reorder_theta_to_match_true,
    plot_parameter_recovery,
    plot_error_analysis,
    plot_sinogram_comparison,
    plot_sinogram,
    plot_projection_modes,
    plot_trajectory_comparison,
    create_publication_figure,
    plot_projection_modes_and_trajectories,
)

# Define what gets imported with "from gmm_ct import *"
__all__ = [
    # Core model
    'GMM_reco',
    'NewtonRaphsonLBFGS',

    # Config & runners
    'GRAVITATIONAL_ACCELERATION',
    'load_reconstruct_config',
    'load_simulate_config',
    'ReconstructConfig',
    'SimulateConfig',
    'run_simulation',
    'run_reconstruction',
    'analyse_results',

    # Parameter generation and utilities
    'generate_true_param',
    'construct_receivers',
    'set_random_seeds',
    'export_parameters',

    # Visualization - animations
    'save_GMM_animation',
    'save_projection_comparison_animation',
    'save_GMM_with_projection_comparison',
    'save_optimization_stages_animation',

    # Visualization - publication plots
    'plot_individual_gaussian_reconstruction',
    'plot_temporal_gmm_comparison',
    'animate_temporal_gmm_comparison',
    'reorder_theta_to_match_true',
    'plot_parameter_recovery',
    'plot_error_analysis',
    'plot_sinogram_comparison',
    'plot_sinogram',
    'plot_projection_modes',
    'plot_trajectory_comparison',
    'create_publication_figure',
    'plot_projection_modes_and_trajectories',
]
