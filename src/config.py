"""Configuration loading and constants for GMM-CT.

All configuration dataclasses and YAML loaders live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import yaml


GRAVITATIONAL_ACCELERATION = 9.81  # m/s²


# ---------------------------------------------------------------------------
# Section Dataclasses
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
        d = self.dimensionality
        if d == 2:
            receivers_t = construct_receivers(
                device,
                (rcv["n_receivers"], rcv["x_coordinate"], rcv["y_min"], rcv["y_max"]),
            )
        elif d == 3:
            receivers_t = construct_receivers(
                device,
                (
                    rcv["n_receivers_y"], rcv["n_receivers_z"], rcv["x_coordinate"], 
                    rcv["y_min"], rcv["y_max"], rcv["z_min"], rcv["z_max"]
                ),
            )
        else:
            raise ValueError(f"Unsupported dimensionality: {d}")
        return sources_t, receivers_t
    

@dataclass
class PhysicsConfig:
    """Physical parameters and forward simulation dynamics."""
    
    initial_positions: List[List[float]]
    initial_accelerations: List[List[float]]
    omega_range: Tuple[float, float] = (2.0, 6.0)
    n_projections: int = 150
    duration: float = 1.5
    initial_velocities: List[float] = field(default_factory=lambda: [0.75, 0.5])
    
    def to_tensors(self, n_gaussians: int, device: torch.device):
        """Return (x0s, a0s) as per-Gaussian tensor lists on device."""
        x0s = self._broadcast(self.initial_positions, n_gaussians, device)
        a0s = self._broadcast(self.initial_accelerations, n_gaussians, device)
        return x0s, a0s
    
    @staticmethod
    def _broadcast(values, n, device):
        tensors = [torch.tensor(v, dtype=torch.float64, device=device) for v in values]
        if len(tensors) == 1:
            tensors = [tensors[0].clone() for _ in range(n)]
        if len(tensors) != n:
            raise ValueError(f"Expected 1 or {n} entries, got {len(tensors)}")
        return tensors
    

@dataclass
class ReconstructionConfig:
    """Tuning knobs for the 4-stage reconstruction pipeline."""
    
    n_trajectory_trials: Optional[int] = None
    n_omega_inits: Optional[int] = None
    max_iterations: int = 500
    tolerance: float = 1e-5
    
    
@dataclass
class AnalysisConfig:
    """Post-reconstruction analysis settings."""
    
    enabled: bool = True
    skip_errors: bool = False
    skip_plots: bool = False
    skip_animations: bool = False
    time_indices: Optional[List[int]] = None
    

@dataclass
class OutputConfig:
    """Output directory and artifact toggles."""
    
    directory: Union[str, Path] = "data/"
    save_plots: bool = True
    save_animations: bool = True
    verbose: bool = True
    
    def __post_init__(self): 
        self.directory = Path(self.directory)
        
        
# ---------------------------------------------------------------------------
# Master Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Single master configuration for simulation, reconstruction, and analysis."""
    
    geometry: GeometryConfig
    physics: PhysicsConfig
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    sim_n_gaussians: int = 5
    reco_n_gaussians: Optional[int] = None
    exp_dir: Optional[Union[str, Path]] = None
    device: Optional[str] = None
    seed: int = 9
    add_sino_noise: bool = False
    snr_db: float = 0.0
    
    def __post_init__(self):
        if self.reco_n_gaussians is None:
            self.reco_n_gaussians = self.sim_n_gaussians
        if self.exp_dir is None:
            self.exp_dir = self.output.directory / "projections.pt"
        else:
            self.exp_dir = Path(self.exp_dir)
            

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load unified ExperimentConfig directly from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
        
    model_raw = raw.get("model", {})
    sim_n = model_raw.get("sim_n_gaussians", model_raw.get("n_gaussians", 5))
    reco_n = model_raw.get("reco_n_gaussians", sim_n)
    
    physics_raw = raw.get("physics", {})
    omega = physics_raw.get("omega_range", [2.0, 6.0])
    
    return ExperimentConfig(
        geometry=GeometryConfig(
            sources=raw["geometry"]["sources"],
            receivers=raw["geometry"]["receivers"],
        ),
        physics=PhysicsConfig(
            initial_positions=physics_raw["initial_positions"],
            initial_accelerations=physics_raw["initial_accelerations"],
            omega_range=tuple(omega),
            n_projections=physics_raw.get("n_projections", 150),
            duration=physics_raw.get("duration", 1.5),
            initial_velocities=physics_raw.get("initial_velocities", [0.75, 0.5])
        ),
        reconstruction=ReconstructionConfig(**raw.get("reconstruction", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
        output=OutputConfig(**raw.get("output", {})),
        sim_n_gaussians=sim_n,
        reco_n_gaussians=reco_n,
        exp_dir=raw.get("data", {}).get("projections"),
        device=raw.get("device"),
        seed=raw.get("seed", 9),
        add_sino_noise=raw.get("add_sino_noise", False),
        snr_db=raw.get("snr_db", 0.0),
    )