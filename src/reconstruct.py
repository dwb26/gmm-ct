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
    """Run the full reconstruction pipeline."""
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

    # Optional sensitivity perturbations of what the reconstruction assumes
    rc = cfg.reconstruction
    model.v0_init_mean = getattr(rc, "init_v_mean", None) or model.v0_init_mean
    model.v0_init_std = getattr(rc, "init_v_std", None) or model.v0_init_std
    offset = getattr(rc, "x0_offset", None)
    if offset is not None:
        shift = torch.tensor(offset, dtype=torch.float64, device=model.x0s[0].device)
        model.x0s = [x0 + shift for x0 in model.x0s]
    
    # --- Run reconstruction ---
    pipeline_mode = getattr(cfg.reconstruction, "pipeline_mode", "gmm-ct")
    if pipeline_mode == "gmm-ct":
        logger.info("Executing GMM-CT: Trajectory Recovery (Hausdorff/Peaks) -> Multi-Start Least Squares")
        soln_dict = model.fit(proj_data=proj_data, t=t)
        
    elif pipeline_mode == "direct-ls":
        logger.info("Executing Direct Frame-by-Frame Least Squares Baseline with no Staging/Pipelining")
        soln_dict = model.direct_ls(proj_data=proj_data, t=t)
        
    elif pipeline_mode == "decoupled-ls":
        logger.info("Executing: Least-Squares, but with Decoupling")
        soln_dict = model.decoupled_ls(proj_data=proj_data, t=t, intermediate_initialization=False)

    # --- Export Human-Readable Parameter Estimates ---
    export_parameters(
        soln_dict,
        exp_dir / "estimated_parameters.md",
        title="Estimated Parameters",
    )

    peak_data = getattr(model, "peak_data", None)
    detected_modes = dict(peak_data.receiver_heights_by_time) if peak_data is not None else None
    detected_modes = detected_modes or None

    # --- Save Standalone Reconstruction Checkpoint ---
    torch.save(
        {
            "params": {
                "v0s": soln_dict['v0s'],            # Shape: [N, 2]
                "x0s": soln_dict['x0s'],            # Shape: [N, 2]
                "a0s": soln_dict['a0s'],            # Shape: [N, 2]
                "alphas": soln_dict['alphas'],      # Shape: [N]
                "omegas": soln_dict['omegas'],      # Shape: [N]
                "U_skews": soln_dict['U_skews'],    # Shape: [N, 2, 2]
            },
            "theta_pre_stage_2": getattr(model, "theta_pre_stage_2", None),
            "theta_est": soln_dict,
            "detected_modes": detected_modes,
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