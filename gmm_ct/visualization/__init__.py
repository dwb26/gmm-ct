"""Visualization and plotting utilities."""

from .animations import (
    save_GMM_animation,
    save_projection_comparison_animation,
    save_GMM_with_projection_comparison,
    save_optimization_stages_animation,
)
from .diagnostics import (
    plot_trajectory_estimations,
    plot_heights_by_assignment,
    plot_raw_receiver_heights,
)
from .publication import (
    plot_individual_gaussian_reconstruction,
    plot_temporal_gmm_comparison,
    animate_temporal_gmm_comparison,
    reorder_theta_to_match_true,
    plot_parameter_recovery,
    plot_error_analysis,
    plot_sinogram_comparison,
    plot_trajectory_comparison,
    create_publication_figure,
)

__all__ = [
    # Animations
    'save_GMM_animation',
    'save_projection_comparison_animation',
    'save_GMM_with_projection_comparison',
    'save_optimization_stages_animation',
    # Diagnostics
    'plot_trajectory_estimations',
    'plot_heights_by_assignment',
    'plot_raw_receiver_heights',
    # Publication plots
    'plot_individual_gaussian_reconstruction',
    'plot_temporal_gmm_comparison',
    'animate_temporal_gmm_comparison',
    'reorder_theta_to_match_true',
    'plot_parameter_recovery',
    'plot_error_analysis',
    'plot_sinogram_comparison',
    'plot_trajectory_comparison',
    'create_publication_figure',
]
