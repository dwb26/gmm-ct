"""
Reconstruction and analysis runner for GMM-CT.

Loads observed projection data from disk, instantiates ``GMM_reco`` from a
YAML config, runs the 4-stage reconstruction pipeline, saves the results,
and — when ground-truth data is available — automatically runs error
analysis and generates publication-quality plots.
"""

import logging
from datetime import datetime
from pathlib import Path
from time import time as wall_clock

import numpy as np
import torch

from .config import AnalysisConfig, ReconstructConfig
from .model import GMM_reco
from .utils import export_parameters

logger = logging.getLogger(__name__)


# ======================================================================
# Data & Ground-Truth Loading Helpers
# ======================================================================

def _load_projection_data(
    data_path: str, 
    device: torch.device
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Load projection measurements and time steps (.pt or .npy format)."""
    path = Path(data_path)
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


def _try_load_ground_truth(data_path: Path, device: torch.device) -> None:
    """Attempt to load companion ground_truth.pt from the input data directory."""
    gt_path = data_path.parent / "ground_truth.pt"
    if gt_path.exists():
        try:
            return torch.load(gt_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning("Failed to load ground_truth.pt despite file existing %s", e)
    return None


# ======================================================================
# Orchestration Engine
# ======================================================================

def run_reconstruction(cfg: ReconstructConfig) -> dict:
    """Run the full reconstruction pipeline and optional analysis from config 
    (though this to be decoupled)."""
    start = wall_clock()

    # --- Device ---
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device selected: %s", device)

    # --- Load Projection Measurements ---
    data_path = Path(cfg.data_path)
    proj_data, t = _load_projection_data(data_path, device)
    logger.info("Loaded projections shape: %s", proj_data[0].shape)
    logger.info("Time mesh: %d steps (%.3fs – %.3fs)", t.shape[0], t[0].item(), t[-1].item())
    
    # --- Fetch Ground Truth if Available ---
    gt = _try_load_ground_truth(data_path, device)
    seed_str = str(gt.get("config", {}).get("seed", "unknown")) if gt else "unknown"

    # --- Output Directory Management ---
    N = cfg.n_gaussians
    out_dir = Path(cfg.output.directory)
    
    if getattr(cfg.output, "use_timestamp", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_seed{seed_str}_N{N}"
    else:
        folder_name = f"reco_seed{seed_str}_N{N}"
        
    experiment_dir = out_dir / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # --- Model Instantiation ---
    model = GMM_reco.from_config(cfg)
    model.output_dir = experiment_dir
    if gt is not None and "theta_true" in gt:
        model.theta_true = gt["theta_true"]

    # --- Run reconstruction ---
    logger.info("Starting GMM reconstruction optimization...")
    soln_dict, best_res = model.fit(proj_data, t)

    # --- Export Human-Readable Parameter Estimates ---
    export_parameters(
        soln_dict,
        experiment_dir / "estimated_parameters.md",
        title="Estimated Parameters",
    )

    # --- Save Standalone Reconstruction Checkpoint ---
    theta_init = getattr(model, "theta_pre_stage2", None)
    theta_stage1_init = getattr(model, "theta_pre_stage1_5", None)
    
    torch.save(
        {
            "theta_est": soln_dict,
            "theta_init": theta_init,
            "config": {
                "n_gaussians": cfg.n_gaussians,
                "omega_range": list(cfg.physics.omega_range),
                "data_path": str(cfg.data_path),
                "device": str(device),
            },
        },
        experiment_dir / "reconstruction.pt",
    )
    
    elapsed = wall_clock() - start
    logger.info("Recontruction finished in %.1fs", elapsed)
    
    # --- Analysis & Figure Generation
    if gt is not None:
        gt_cfg = gt.get("config", {})
        theta_true = gt["theta_true"]
        sources, receivers = gt["sources"], gt["receivers"]
        
        # Save unified results bundle for paper figures
        torch.save(
            {
                "theta_true": theta_true,
                "theta_est": soln_dict,
                "theta_init": theta_init,
                "theta_stage1_init": theta_stage1_init,
                "proj_data": proj_data,
                "t": t,
                "sources": sources,
                "receivers": receivers,
                "config": {
                    "d": gt_cfg.get("d", 2),
                    "N": cfg.n_gaussians,
                    "seed": gt_cfg.get("seed", -1),
                    "omega_min": cfg.physics.omega_range[0],
                    "omega_max": cfg.physics.omega_range[1],
                    "device": str(device),
                },
            },
            experiment_dir / "results.pt",
        )
           
    #     if cfg.analysis.enabled:
    #         logger.info("Running automatic error analysis and plot generation...")
    #         analyze_results(
    #             theta_true=theta_true,
    #             theta_est=soln_dict,
    #             theta_init=theta_init,
    #             theta_stage1_init=theta_stage1_init,
    #             proj_data=proj_data,
    #             t=t,
    #             sources=sources,
    #             receivers=receivers,
    #             d=gt_cfg.get("d", 2),
    #             N=cfg.n_gaussians,
    #             omega_min=cfg.physics.omega_range[0],
    #             omega_max=cfg.physics.omega_range[1],
    #             device=device,
    #             experiment_dir=experiment_dir,
    #             analysis_cfg=cfg.analysis,
    #             res=best_res,
    #         )
    # else:
    #     logger.info("No ground_truth.pt found alongside data; post-analysis skipped.")
            
    return soln_dict


# ======================================================================
# Post-Reconstruction Analysis & Visualizations
# ======================================================================

