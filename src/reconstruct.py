"""
Reconstruction and analysis runner for GMM-CT.

Loads observed projection data from disk, instantiates ``GMM_reco`` from a
YAML config, runs the 4-stage reconstruction pipeline, saves the results,
and — when ground-truth data is available — automatically runs error
analysis and generates publication-quality plots.
"""

import logging
from pathlib import Path
from time import time as wall_clock

import numpy as np
import torch
import pandas as pd

from .config import ExperimentConfig
from .model import GMM_reco
from .utils import export_parameters, set_random_seeds
from .visualization.simulate_viz import animate_simulation

logger = logging.getLogger(__name__)


# ======================================================================
# Data & Ground-Truth Loading Helpers
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

def _try_load_ground_truth(exp_dir: Path, device: torch.device) -> None:
    """Attempt to load companion ground_truth.pt from the input data directory."""
    gt_path = exp_dir.parent / "ground_truth.pt"
    if gt_path.exists():
        try:
            return torch.load(gt_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning("Failed to load ground_truth.pt despite file existing %s", e)
    return None

import torchmin

def fit_gmm_1d_single_k(
    proj_row: torch.Tensor, 
    receiver_coords: torch.Tensor, 
    k: int,
) -> tuple[torch.Tensor, float]:
    """Fits k 1D Gaussians to a projection row and returns (peaks, loss_value)."""
    device = proj_row.device
    dtype = torch.float64
    proj_row = proj_row.to(dtype=dtype)
    receiver_coords = receiver_coords.to(device=device, dtype=dtype)

    if k == 0:
        loss_val = torch.mean(proj_row ** 2).item()
        return torch.empty((0, 3), dtype=dtype, device=device), loss_val

    # 1. Initialize mu across active detector span
    r_min, r_max = receiver_coords.min(), receiver_coords.max()
    span = r_max - r_min
    mu_init = torch.linspace(r_min + 0.1 * span, r_max - 0.1 * span, k, device=device, dtype=dtype)
    
    # 2. Log-parameterized initializations
    target_A = (proj_row.max() / k).clamp(min=1e-3)
    target_sigma = torch.tensor(0.15, device=device, dtype=dtype)
    
    log_A_init = torch.full((k,), torch.log(target_A).item(), device=device, dtype=dtype)
    log_sigma_init = torch.full((k,), torch.log(target_sigma).item(), device=device, dtype=dtype)

    # Parameters: [log_A (k), mu (k), log_sigma (k)]
    params = torch.cat([log_A_init, mu_init, log_sigma_init]).requires_grad_(True)
    r = receiver_coords.unsqueeze(1)  # [R, 1]

    def loss_fn(p):
        A = torch.exp(p[:k])
        mu = p[k:2*k]
        sigma = torch.exp(p[2*k:]) + 1e-4
        
        # [R, k]
        diff = (r - mu) / sigma
        pred = torch.sum(A * torch.exp(-0.5 * (diff ** 2)), dim=1)
        return torch.mean((pred - proj_row) ** 2)

    res = torchmin.minimize(
        loss_fn, 
        params, 
        method='l-bfgs', 
        options={'max_iter': 500, 'gtol': 1e-6, 'disp': False}
    )
    
    p_opt = res.x
    A_opt = torch.exp(p_opt[:k])
    mu_opt = p_opt[k:2*k]
    sigma_opt = torch.exp(p_opt[2*k:]) + 1e-4
    
    peaks = torch.stack([mu_opt, A_opt, sigma_opt], dim=1)
    loss_val = res.fun.item() if isinstance(res.fun, torch.Tensor) else float(res.fun)
    
    # Sort peaks spatially by mu (first column)
    sort_idx = torch.argsort(mu_opt)
    peaks = torch.stack([mu_opt[sort_idx], A_opt[sort_idx], sigma_opt[sort_idx]], dim=1)
    return peaks, loss_val

def fit_gmm_torchmin(
    proj_row: torch.Tensor, 
    receiver_coords: torch.Tensor, 
    N: int,
) -> torch.Tensor:
    """Fits 1D GMM evaluating all k in [0, 1, ..., N] and returns parameters with lowest MSE."""
    best_loss = float('inf')
    best_peaks = torch.empty((0, 3), dtype=torch.float64, device=proj_row.device)

    for k in range(N + 1):
        peaks, loss_val = fit_gmm_1d_single_k(proj_row, receiver_coords, k)
        if loss_val < best_loss:
            best_loss = loss_val
            best_peaks = peaks

    return best_peaks

# ======================================================================
# Orchestration Engine
# ======================================================================

def run_reconstruction(cfg: ExperimentConfig) -> GMM_reco:
    """Run the full reconstruction pipeline and optional analysis from config 
    (though this to be decoupled)."""
    start = wall_clock()

    # --- Reproducibility & Device ---
    set_random_seeds(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load Projection Measurements ---
    exp_dir = Path(cfg.exp_dir)
    proj_data, t = _load_projection_data(exp_dir / 'projections.pt', device)
    
    # --- Fetch Ground Truth if Available ---
    gt = _try_load_ground_truth(exp_dir, device)

    # --- Model Instantiation ---
    model = GMM_reco.from_config(cfg)
    if gt is not None and "theta_true" in gt:
        model.theta_true = gt["theta_true"]
        
    # --- Save projections ---
    proj_tensor = model.process_projections(proj_data)
    receiver_coords = torch.tensor([r[1] for r in model.receivers[0]])
    # import matplotlib.pyplot as plt
    fitted_gaussian_peaks = {}
    records = []
    
    for n_p, proj_row in enumerate(proj_tensor):
        peaks = fit_gmm_torchmin(
            proj_row=proj_row,
            receiver_coords=receiver_coords,
            N = cfg.reco_n_gaussians,
        )
        if len(peaks) > 0:
            # fig, axs = plt.subplots(ncols=1)
            time_val = t[n_p]
            detected_heights = []
            
            for n_d, data in enumerate(peaks):
                mu, A, sigma = data
                time_val_scalar = time_val.item() if isinstance(time_val, torch.Tensor) else float(time_val)
                detected_heights.append(mu.item())
                receiver_idx = torch.argmin(torch.abs(mu - receiver_coords)).item()
                
                records.append({
                    'time_idx': int(n_p),
                    'time_val': float(time_val_scalar),
                    'receiver_idx': int(receiver_idx),
                    'receiver_pos': float(receiver_coords[receiver_idx].item()),
                    'peak_val': float(A.item()),
                    'gaussian_idx': int(n_d),
                })
                
            fitted_gaussian_peaks[time_val_scalar] = detected_heights.copy()
                
                # axs.plot(receiver_coords, proj_row, label='data')
                # axs.vlines(mu, 0.0, A)
            # out_dir = exp_dir / "fitted_plots"
            # out_dir.mkdir(parents=True, exist_ok=True)        
            # plt.savefig(out_dir / f"n_{n_p}.png", dpi=150, bbox_inches='tight')
            # plt.close()

    # --- Run reconstruction ---
    logger.info("Starting GMM reconstruction optimization...")
    # model.peak_detection_records = pd.DataFrame(records)
    # model.fitted_gaussian_peaks = fitted_gaussian_peaks.copy()
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
            "theta_est": soln_dict,
            "theta_init": getattr(model, "theta_pre_stage2", None),
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