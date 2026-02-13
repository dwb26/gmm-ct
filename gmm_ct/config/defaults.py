"""Default configuration and constants for GMM CT reconstruction."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

# Physical constants
GRAVITATIONAL_ACCELERATION = 9.81  # m/s^2

# Random seed for reproducibility
RANDOM_SEED = 99


@dataclass
class ReconstructionConfig:
    """Configuration for GMM reconstruction.
    
    This class encapsulates all parameters needed for GMM-based CT reconstruction
    with motion estimation.
    
    Parameters
    ----------
    n_gaussians : int
        Number of Gaussian components in the mixture model
    dimensionality : int, optional
        Spatial dimensionality of the problem (default: 2)
    omega_range : Tuple[float, float], optional
        Range of angular velocities for initialization (min, max) in rad/s (default: (0.0, 10.0))
    device : str, optional
        Computation device ('cuda', 'cpu', or None for auto-detection) (default: None)
    n_omega_inits : int, optional
        Number of random multi-start trials for omega search.
        If None, uses max(15, 3*n_gaussians) (default: None)
    n_traj_trials : int, optional
        Number of trajectory optimization trials (default: None)
    use_fft_omega : bool, optional
        Enable FFT-based omega estimation (default: True)
    max_iterations : int, optional
        Maximum optimization iterations (default: 500)
    tolerance : float, optional
        Convergence tolerance (default: 1e-5)
    line_search : str, optional
        Line search method for optimization ('strong_wolfe', etc.) (default: 'strong_wolfe')
    output_dir : str or Path, optional
        Directory for saving outputs (default: 'results')
    save_animations : bool, optional
        Whether to save animation files (default: True)
    save_plots : bool, optional
        Whether to save plot files (default: True)
    verbose : bool, optional
        Enable verbose output during reconstruction (default: False)
    
    Examples
    --------
    >>> config = ReconstructionConfig(
    ...     n_gaussians=3,
    ...     omega_range=(0, 10),
    ...     device='cuda',
    ...     verbose=True
    ... )
    >>> model = GMM_reco.from_config(config)
    """
    
    # Core parameters
    n_gaussians: int
    dimensionality: int = 2
    
    # Motion parameters
    omega_range: Tuple[float, float] = (0.0, 10.0)
    
    # Computation parameters
    device: Optional[str] = None
    n_omega_inits: Optional[int] = None
    n_traj_trials: Optional[int] = None
    
    # Feature flags
    use_fft_omega: bool = True
    
    # Optimization parameters
    max_iterations: int = 500
    tolerance: float = 1e-5
    line_search: str = 'strong_wolfe'
    
    # Output parameters
    output_dir: Path = field(default_factory=lambda: Path('results'))
    save_animations: bool = True
    save_plots: bool = True
    
    # Logging
    verbose: bool = False
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert output_dir to Path if it's a string
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Validate n_gaussians
        if self.n_gaussians < 1:
            raise ValueError(f"n_gaussians must be >= 1, got {self.n_gaussians}")
        
        # Validate dimensionality
        if self.dimensionality not in [2, 3]:
            raise ValueError(f"dimensionality must be 2 or 3, got {self.dimensionality}")
        
        # Validate omega_range
        if self.omega_range[0] >= self.omega_range[1]:
            raise ValueError(
                f"omega_range must be (min, max) with min < max, "
                f"got {self.omega_range}"
            )
        
        # Set default n_omega_inits if not provided
        if self.n_omega_inits is None:
            self.n_omega_inits = max(15, 3 * self.n_gaussians)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def omega_min(self) -> float:
        """Minimum angular velocity."""
        return self.omega_range[0]
    
    @property
    def omega_max(self) -> float:
        """Maximum angular velocity."""
        return self.omega_range[1]
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns
        -------
        dict
            Configuration as a dictionary
        """
        config_dict = {
            'n_gaussians': self.n_gaussians,
            'dimensionality': self.dimensionality,
            'omega_range': self.omega_range,
            'device': self.device,
            'n_omega_inits': self.n_omega_inits,
            'n_traj_trials': self.n_traj_trials,
            'use_fft_omega': self.use_fft_omega,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'line_search': self.line_search,
            'output_dir': str(self.output_dir),
            'save_animations': self.save_animations,
            'save_plots': self.save_plots,
            'verbose': self.verbose,
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ReconstructionConfig':
        """Create configuration from dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary
            
        Returns
        -------
        ReconstructionConfig
            Configuration object
        """
        return cls(**config_dict)
