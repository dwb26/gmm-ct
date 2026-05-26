"""Configuration loading and constants for GMM-CT.

All configuration dataclasses and YAML loaders live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import yaml

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

GRAVITATIONAL_ACCELERATION = 9.81  # m/s²

# ---------------------------------------------------------------------------
# Geometry and physics
# ---------------------------------------------------------------------------


@dataclass
class GeometryConfig:
    """CT geometry: sources and receiver array specification."""

    sources: List[List[float]]
    receivers: dict

    @property
    def dimensionality(self) -> int:
        return len(self.sources[0])

    def to_tensors(self, device: torch.device):
        """Return (sources, receivers) as torch tensors on *device*."""
        from .utils import construct_receivers

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
    """Known physical parameters for all Gaussians."""

    initial_positions: List[List[float]]
    accelerations: List[List[float]]
    omega_range: Tuple[float, float] = (-24.0, -16.0)

    def to_tensors(self, n_gaussians: int, device: torch.device):
        """Return (x0s, a0s) as per-Gaussian tensor lists on *device*."""
        x0s = self._broadcast(self.initial_positions, n_gaussians, device)
        a0s = self._broadcast(self.accelerations, n_gaussians, device)
        return x0s, a0s

    @staticmethod
    def _broadcast(values, n, device):
        tensors = [torch.tensor(v, dtype=torch.float64, device=device) for v in values]
        if len(tensors) == 1:
            tensors = [tensors[0].clone() for _ in range(n)]
        if len(tensors) != n:
            raise ValueError(f"Expected 1 or {n} entries, got {len(tensors)}")
        return tensors


# ---------------------------------------------------------------------------
# Algorithm and output settings
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionSettings:
    """Tuning knobs for the 4-stage reconstruction pipeline."""

    n_trajectory_trials: Optional[int] = None
    n_omega_inits: Optional[int] = None
    max_iterations: int = 500
    tolerance: float = 1e-5


@dataclass
class OutputConfig:
    """Output directory and what to save."""

    directory: Union[str, Path] = "results"
    save_plots: bool = True
    save_animations: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.directory = Path(self.directory)


@dataclass
class AnalysisConfig:
    """Post-reconstruction analysis settings."""

    enabled: bool = True
    skip_errors: bool = False
    skip_plots: bool = False
    skip_animations: bool = False
    time_indices: Optional[List[int]] = None


@dataclass
class SimulationSettings:
    """Settings for synthetic data generation."""

    seed: int = 40
    n_projections: int = 65
    duration: float = 2.0
    initial_velocity: List[float] = field(default_factory=lambda: [0.75, 0.5])


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


@dataclass
class ReconstructConfig:
    """Complete configuration for a reconstruction run."""

    data_path: str
    n_gaussians: int
    geometry: GeometryConfig
    physics: PhysicsConfig
    reconstruction: ReconstructionSettings = field(default_factory=ReconstructionSettings)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    device: Optional[str] = None


@dataclass
class SimulateConfig:
    """Complete configuration for a simulation run."""

    n_gaussians: int
    geometry: GeometryConfig
    physics: PhysicsConfig
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------


def load_reconstruct_config(path: Union[str, Path]) -> ReconstructConfig:
    """Load a reconstruction config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    analysis_raw = raw.get("analysis", {})
    return ReconstructConfig(
        data_path=raw["data"]["projections"],
        n_gaussians=raw["model"]["n_gaussians"],
        geometry=_parse_geometry(raw["geometry"]),
        physics=_parse_physics(raw["physics"]),
        reconstruction=_parse_reconstruction(raw.get("reconstruction", {})),
        output=_parse_output(raw.get("output", {})),
        analysis=AnalysisConfig(
            enabled=analysis_raw.get("enabled", True),
            skip_errors=analysis_raw.get("skip_errors", False),
            skip_plots=analysis_raw.get("skip_plots", False),
            skip_animations=analysis_raw.get("skip_animations", False),
            time_indices=analysis_raw.get("time_indices"),
        ),
        device=raw.get("device"),
    )


def load_simulate_config(path: Union[str, Path]) -> SimulateConfig:
    """Load a simulation config from a YAML file."""
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


def _parse_geometry(raw: dict) -> GeometryConfig:
    return GeometryConfig(sources=raw["sources"], receivers=raw["receivers"])


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
