"""
YAML configuration loading and validation for GMM-CT.

Provides functions to load reconstruction and simulation configurations
from YAML files, with validation and sensible defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import yaml

from .defaults import GRAVITATIONAL_ACCELERATION


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeometryConfig:
    """CT geometry specification (sources and receivers).

    Parameters
    ----------
    sources : list of list of float
        Source positions, each a d-dimensional coordinate.
    receivers : dict
        Receiver specification with keys:
        ``n_receivers``, ``x_coordinate``, ``y_min``, ``y_max``.
    """

    sources: List[List[float]]
    receivers: dict

    @property
    def dimensionality(self) -> int:
        """Infer spatial dimensionality from source coordinates."""
        return len(self.sources[0])

    def to_tensors(self, device: torch.device):
        """Convert geometry to torch tensors on *device*.

        Returns
        -------
        sources_t : list of torch.Tensor
        receivers_t : list of list of torch.Tensor
        """
        from ..utils.geometry import construct_receivers

        sources_t = [
            torch.tensor(s, dtype=torch.float64, device=device)
            for s in self.sources
        ]

        rcv = self.receivers
        receivers_t = construct_receivers(
            device,
            (rcv["n_receivers"], rcv["x_coordinate"], rcv["y_min"], rcv["y_max"]),
        )

        return sources_t, receivers_t


@dataclass
class PhysicsConfig:
    """Known physical parameters.

    Parameters
    ----------
    initial_positions : list of list of float
        Known initial position for each Gaussian (or a single position
        broadcast to all).
    accelerations : list of list of float
        Known acceleration for each Gaussian (or a single acceleration
        broadcast to all).
    omega_range : tuple of float
        ``(omega_min, omega_max)`` angular velocity search bounds.
    """

    initial_positions: List[List[float]]
    accelerations: List[List[float]]
    omega_range: Tuple[float, float] = (-24.0, -16.0)

    def to_tensors(self, n_gaussians: int, device: torch.device):
        """Convert to per-Gaussian tensor lists, broadcasting if needed.

        Returns
        -------
        x0s : list of torch.Tensor
        a0s : list of torch.Tensor
        """
        x0s = self._broadcast(self.initial_positions, n_gaussians, device)
        a0s = self._broadcast(self.accelerations, n_gaussians, device)
        return x0s, a0s

    @staticmethod
    def _broadcast(values, n, device):
        tensors = [
            torch.tensor(v, dtype=torch.float64, device=device) for v in values
        ]
        if len(tensors) == 1:
            tensors = [tensors[0].clone() for _ in range(n)]
        if len(tensors) != n:
            raise ValueError(
                f"Expected 1 or {n} entries, got {len(tensors)}"
            )
        return tensors


@dataclass
class ReconstructionSettings:
    """Tuning knobs for the reconstruction algorithm.

    Parameters
    ----------
    n_trajectory_trials : int or None
        Multi-start trials for trajectory optimisation.
    n_omega_inits : int or None
        Random omega initialisations for joint optimisation.
    max_iterations : int
        Maximum L-BFGS iterations.
    tolerance : float
        Convergence tolerance.
    """

    n_trajectory_trials: Optional[int] = None
    n_omega_inits: Optional[int] = None
    max_iterations: int = 500
    tolerance: float = 1e-5


@dataclass
class OutputConfig:
    """Output and logging settings.

    Parameters
    ----------
    directory : str or Path
        Where to write results.
    save_plots : bool
        Save diagnostic plots.
    save_animations : bool
        Save animations.
    verbose : bool
        Verbose console output.
    """

    directory: Union[str, Path] = "results"
    save_plots: bool = True
    save_animations: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.directory = Path(self.directory)


@dataclass
class AnalysisConfig:
    """Settings for post-reconstruction analysis.

    Controls whether error metrics are computed, comparison plots are
    generated, and animations are rendered after reconstruction.

    Parameters
    ----------
    enabled : bool
        Run analysis automatically after reconstruction (requires
        ground-truth data alongside the projection file).
    skip_errors : bool
        Skip parameter & projection error computation.
    skip_plots : bool
        Skip static comparison plots.
    skip_animations : bool
        Skip animation rendering (can be slow).
    time_indices : list of int or None
        Time indices for temporal comparison plots.  ``None`` means
        auto-select.
    """

    enabled: bool = True
    skip_errors: bool = False
    skip_plots: bool = False
    skip_animations: bool = False
    time_indices: Optional[List[int]] = None


@dataclass
class SimulationSettings:
    """Settings specific to synthetic data generation.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_projections : int
        Number of time steps to simulate.
    duration : float
        Time window in seconds.
    initial_velocity : list of float
        Base initial velocity vector (perturbations added per Gaussian).
    """

    seed: int = 40
    n_projections: int = 65
    duration: float = 2.0
    initial_velocity: List[float] = field(default_factory=lambda: [0.75, 0.5])


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


@dataclass
class ReconstructConfig:
    """Complete configuration for running a reconstruction.

    Loaded by ``load_reconstruct_config(path)``.
    """

    data_path: str  # path to projection data (.pt or .npy)
    n_gaussians: int
    geometry: GeometryConfig
    physics: PhysicsConfig
    reconstruction: ReconstructionSettings = field(
        default_factory=ReconstructionSettings
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    device: Optional[str] = None


@dataclass
class SimulateConfig:
    """Complete configuration for generating synthetic data.

    Loaded by ``load_simulate_config(path)``.
    """

    n_gaussians: int
    geometry: GeometryConfig
    physics: PhysicsConfig
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _parse_geometry(raw: dict) -> GeometryConfig:
    sources = raw["sources"]
    receivers = raw["receivers"]
    return GeometryConfig(sources=sources, receivers=receivers)


def _parse_physics(raw: dict) -> PhysicsConfig:
    omega = raw.get("omega_range", [-24.0, -16.0])
    return PhysicsConfig(
        initial_positions=raw["initial_positions"],
        accelerations=raw["accelerations"],
        omega_range=tuple(omega),
    )


def _parse_reconstruction(raw: dict) -> ReconstructionSettings:
    return ReconstructionSettings(
        n_trajectory_trials=raw.get("n_trajectory_trials"),
        n_omega_inits=raw.get("n_omega_inits"),
        max_iterations=raw.get("max_iterations", 500),
        tolerance=raw.get("tolerance", 1e-5),
    )


def _parse_output(raw: dict) -> OutputConfig:
    return OutputConfig(
        directory=raw.get("directory", "results"),
        save_plots=raw.get("save_plots", True),
        save_animations=raw.get("save_animations", True),
        verbose=raw.get("verbose", False),
    )


def _parse_simulation(raw: dict) -> SimulationSettings:
    return SimulationSettings(
        seed=raw.get("seed", 40),
        n_projections=raw.get("n_projections", 65),
        duration=raw.get("duration", 2.0),
        initial_velocity=raw.get("initial_velocity", [0.75, 0.5]),
    )


def load_reconstruct_config(path: Union[str, Path]) -> ReconstructConfig:
    """Load a reconstruction config from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    ReconstructConfig

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    analysis_raw = raw.get("analysis", {})
    analysis_cfg = AnalysisConfig(
        enabled=analysis_raw.get("enabled", True),
        skip_errors=analysis_raw.get("skip_errors", False),
        skip_plots=analysis_raw.get("skip_plots", False),
        skip_animations=analysis_raw.get("skip_animations", False),
        time_indices=analysis_raw.get("time_indices"),
    )

    return ReconstructConfig(
        data_path=raw["data"]["projections"],
        n_gaussians=raw["model"]["n_gaussians"],
        geometry=_parse_geometry(raw["geometry"]),
        physics=_parse_physics(raw["physics"]),
        reconstruction=_parse_reconstruction(raw.get("reconstruction", {})),
        output=_parse_output(raw.get("output", {})),
        analysis=analysis_cfg,
        device=raw.get("device"),
    )


def load_simulate_config(path: Union[str, Path]) -> SimulateConfig:
    """Load a simulation config from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SimulateConfig

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return SimulateConfig(
        n_gaussians=raw["model"]["n_gaussians"],
        geometry=_parse_geometry(raw["geometry"]),
        physics=_parse_physics(raw["physics"]),
        simulation=_parse_simulation(raw.get("simulation", {})),
        output=_parse_output(raw.get("output", {})),
        device=raw.get("device"),
    )
