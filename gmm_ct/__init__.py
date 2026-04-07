"""
Gaussian Mixture CT Reconstruction

A Python package for reconstructing dynamic objects in CT imaging using
Gaussian Mixture Models with motion estimation.
"""

import logging

__version__ = "0.1.0"
__author__ = "Daniel Burrows"

# Library-level null handler — callers configure their own logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import main classes and functions for convenient access
from .config.defaults import ReconstructionConfig
from .config.yaml_config import load_reconstruct_config, load_simulate_config
from .core.reconstruction import GMM_reco
from .core.solvers import NewtonRaphsonLBFGS
from .reconstruct import run_reconstruction, analyse_results
from .simulation import run_simulation
from .utils.generators import generate_true_param
from .utils.geometry import construct_receivers
from .utils.helpers import set_random_seeds, export_parameters
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
)
from .config.defaults import ReconstructionConfig
from .config.yaml_config import load_reconstruct_config, load_simulate_config
from .simulation import run_simulation
from .reconstruct import run_reconstruction, analyse_results

# Define what gets imported with "from gmm_ct import *"
__all__ = [
    # Core model
    'GMM_reco',
    'ReconstructionConfig',
    
    # YAML config & runners
    'load_reconstruct_config',
    'load_simulate_config',
    'run_simulation',
    'run_reconstruction',
    'analyse_results',
    
    # Parameter generation and utilities
    'generate_true_param',
    'construct_receivers',
    'set_random_seeds',
    'export_parameters',
    'NewtonRaphsonLBFGS',
    
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
]
