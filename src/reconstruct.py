"""
Reconstruction and analysis runner for GMM-CT.

Loads observed projection data from disk, instantiates ``GMM_reco`` from a
YAML config, runs the 4-stage reconstruction pipeline and saves the results.
"""

import logging
from pathlib import Path
from time import time as wall_clock

import numpy as np
import torch

from .config import ExperimentConfig
from .model import GMM_reco
from .utils import export_parameters, set_random_seeds

logger = logging.getLogger(__name__)


# ======================================================================
# Data Loading Helpers
# ======================================================================

def _load_projection_data(
    exp_dir: str, 
    device: torch.device
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Load projection measurements and time steps (.pt or .npy format)."""
    path = Path(exp_dir)
    if not path.exists():
        raise FileNotFoundError(f"Projection data not found: {path}")

    if path.suffix == ".pt":
        bundle = torch.load(path, map_location=device, weights_only=False)
        proj_data = bundle["projections"].to(device)
        t = bundle["times"].to(device)
    elif path.suffix == ".npy":
        proj_np = np.load(path)
        proj_data = torch.tensor(proj_np, dtype=torch.float64, device=device)
        times_path = path.parent / "times.npy"
        if not times_path.exists():
            raise FileNotFoundError(f"Expected companion file {times_path} alongside {path}")
        t = torch.tensor(np.load(times_path), dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unsupported data format '{path.suffix}'. Use .pt or .npy")
    
    # Ensure single-source 2D tensor is wrapped in a list expected by forward model
    if isinstance(proj_data, torch.Tensor) and proj_data.dim() == 2:
        proj_data = [proj_data]
        
    return proj_data, t

# ======================================================================
# Orchestration Engine
# ======================================================================

def run_reconstruction(cfg: ExperimentConfig) -> GMM_reco:
    """Run the full reconstruction pipeline and optional analysis from config 
    (though this to be decoupled)."""
    start = wall_clock()

    # --- Reproducibility & Device ---
    set_random_seeds(cfg.seed + 10000)
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load Projections and Instantiate Model ---
    exp_dir = Path(cfg.exp_dir)
    proj_data, t = _load_projection_data(exp_dir / 'projections.pt', device)
    model = GMM_reco.from_config(cfg)
    
    # --- Run reconstruction ---
    soln_dict = model.fit(proj_data, t)

    # --- Export Human-Readable Parameter Estimates ---
    export_parameters(
        soln_dict,
        exp_dir / "estimated_parameters.md",
        title="Estimated Parameters",
    )

    # --- Save Standalone Reconstruction Checkpoint ---
    torch.save(
        {
            "params": {
                "v0s": soln_dict['v0s'],             # Shape: [N, 2]
                "x0s": soln_dict['x0s'],             # Shape: [N, 2]
                "a0s": soln_dict['a0s'],             # Shape: [N, 2]
                "alphas": soln_dict['alphas'],      # Shape: [N]
                "omegas": soln_dict['omegas'],      # Shape: [N]
                "U_skews": soln_dict['U_skews'],    # Shape: [N, 2, 2]
            },
            "theta_init": getattr(model, "theta_pre_stage1_5", None),
            "theta_est": soln_dict,  # Full estimated parameter vector
            "runtime_seconds": wall_clock() - start,
            "config": {
                "n_gaussians": cfg.reco_n_gaussians,
                "omega_range": list(cfg.physics.omega_range),
                "exp_dir": str(cfg.exp_dir),
                "device": str(device),
            },
        },
        exp_dir / "reconstruction.pt",
    )
    
    elapsed = wall_clock() - start
    logger.info("Recontruction finished in %.1fs", elapsed)
    
    return model