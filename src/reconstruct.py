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
# import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .model import GMM_reco
from .utils import export_parameters, set_random_seeds

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

def fit_gmm_1d_fixed_N(
    proj_row: torch.Tensor, 
    receiver_coords: torch.Tensor, 
    N: int,
    min_amplitude_threshold: float = 1e-4,
) -> tuple[torch.Tensor, float]:
    """Fits fixed N 1D Gaussians allowing A -> 0 via softplus parameterization."""
    device = proj_row.device
    dtype = torch.float64
    proj_row = proj_row.to(dtype=dtype)
    receiver_coords = receiver_coords.to(device=device, dtype=dtype)
    
    if N == 0:
        loss_val = torch.mean(proj_row ** 2).item()
        return torch.empty((0, 3), dtype=dtype, device=device), loss_val

    r_min, r_max = receiver_coords.min(), receiver_coords.max()
    span = r_max - r_min
    
    # 1. Initialize mu raw values near center via inverse sigmoid
    mu_grid = torch.linspace(r_min + 0.1 * span, r_max - 0.1 * span, N, device=device, dtype=dtype)
    # Inverse sigmoid: logit((mu - r_min) / span)
    mu_raw_init = torch.logit((mu_grid - r_min) / span)

    # 2. Initializations
    target_A = (proj_row.max() / N).clamp(min=1e-3)
    target_sigma = torch.tensor(0.15, device=device, dtype=dtype)
    
    log_A_init = torch.full((N,), torch.log(target_A).item(), device=device, dtype=dtype)
    log_sigma_init = torch.full((N,), torch.log(target_sigma).item(), device=device, dtype=dtype)

    params = torch.cat([log_A_init, mu_raw_init, log_sigma_init]).requires_grad_(True)
    r = receiver_coords.unsqueeze(1)  # [R, 1]

    def loss_fn(p):
        A = torch.exp(p[:N])
        # Sigmoid parameterization guarantees mu in (r_min, r_max)
        mu = r_min + span * torch.sigmoid(p[N:2*N])
        sigma = torch.exp(p[2*N:]) + 1e-4
        
        diff = (r - mu) / sigma
        pred = torch.sum(A * torch.exp(-0.5 * (diff ** 2)), dim=1)
        return torch.mean((pred - proj_row) ** 2)

    res = torchmin.minimize(
        loss_fn, 
        params, 
        method='l-bfgs', 
        options={'max_iter': 5000, 'gtol': 1e-6, 'disp': False}
    )
    
    p_opt = res.x
    A_opt = torch.exp(p_opt[:N])
    mu_opt = r_min + span * torch.sigmoid(p_opt[N:2*N])
    sigma_opt = torch.exp(p_opt[2*N:]) + 1e-4

    # Spatial sort
    sort_idx = torch.argsort(mu_opt)
    peaks = torch.stack([mu_opt[sort_idx], A_opt[sort_idx], sigma_opt[sort_idx]], dim=1)
    loss_val = res.fun.item() if isinstance(res.fun, torch.Tensor) else float(res.fun)
    
    return peaks, loss_val


def fit_gmm_torchmin(
    proj_row: torch.Tensor, 
    receiver_coords: torch.Tensor, 
    N: int,
) -> torch.Tensor:
    """Fits 1D GMM evaluating all k in [0, 1, ..., N] and returns parameters with lowest MSE."""
    best_peaks = torch.empty((0, 3), dtype=torch.float64, device=proj_row.device)
    best_peaks, _ = fit_gmm_1d_fixed_N(proj_row, receiver_coords, N)
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

    # --- 1. Peak Detection via Gaussian Fitting ---
    proj_tensor = model.process_projections(proj_data)
    device = proj_tensor.device

    # Extract receiver 1D positions as a clean float64 tensor on the same device
    rcv_coords = torch.tensor(
        [r[1] for r in model.receivers[0]], dtype=torch.float64, device=device
    )

    fitted_gaussian_params = {}
    records = []
    data = []

    for n_p, proj_row in enumerate(proj_tensor):
        params = fit_gmm_torchmin(
            proj_row=proj_row,
            receiver_coords=rcv_coords,
            N=cfg.reco_n_gaussians,
        )

        if len(params) > 0:
            params = params[params[:, 1] > 0.075]
            
            time_val_scalar = (
                t[n_p].item() if isinstance(t[n_p], torch.Tensor) else float(t[n_p])
            )

            mu = params[:, 0]
            A = params[:, 1]
            sigma = params[:, 2]

            data.append(
                {
                    "time_idx": int(n_p),
                    "time_val": time_val_scalar,
                    "mu": mu.detach().cpu(),
                    "A": A.detach().cpu(),
                    "sigma": sigma.detach().cpu(),
                }
            )

            detected_heights = []

            # Unroll per-Gaussian components so peak records maintain 1-to-1 row structure
            for n_d in range(len(params)):
                mu_k = mu[n_d].item()
                A_k = A[n_d].item()
                sigma_k = sigma[n_d].item()

                detected_heights.append(mu_k)

                # Nearest receiver index on spatial grid
                rcv_idx = torch.argmin(torch.abs(mu[n_d] - rcv_coords)).item()

                records.append(
                    {
                        "time_idx": int(n_p),
                        "time_val": time_val_scalar,
                        "receiver_idx": int(rcv_idx),
                        "receiver_pos": float(rcv_coords[rcv_idx].item()),
                        "mu": float(mu_k),
                        "peak_val": float(A_k),
                        "sigma": float(sigma_k),
                        "gaussian_idx": int(n_d),
                    }
                )

            fitted_gaussian_params[time_val_scalar] = detected_heights

    peak_detection_records = pd.DataFrame(records)
    data_df = pd.DataFrame(data)

    # --- 2. Diagnostic Spatial Profile Plots ---
    out_dir = exp_dir / "fitted_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rcv_coords_np = rcv_coords.cpu().numpy()
    time_vals_in_df = data_df["time_val"].to_numpy(dtype=np.float64)

    for n_p, proj_row in enumerate(proj_tensor):
        time_val_scalar = (
            t[n_p].item() if isinstance(t[n_p], torch.Tensor) else float(t[n_p])
        )

        if np.any(np.isclose(time_vals_in_df, time_val_scalar, atol=1e-8)):
            fig, ax = plt.subplots(figsize=(8, 4))

            # True projection curve
            ax.plot(
                rcv_coords_np,
                proj_row.cpu().numpy(),
                label="True",
                lw=3,
                color="black",
                alpha=0.3,
            )

            # Retrieve row matching frame time
            sub_df = data_df[
                np.isclose(time_vals_in_df, time_val_scalar, atol=1e-8)
            ].iloc[0]

            mu = sub_df["mu"].to(device=device)
            A = sub_df["A"].to(device=device)
            sigma = sub_df["sigma"].to(device=device)

            # Vectorized 2D GMM evaluation: [R, 1] - [K] -> [R, K]
            diff = rcv_coords.unsqueeze(1) - mu.unsqueeze(0)
            y_fit = torch.sum(A * torch.exp(-0.5 * (diff / sigma) ** 2), dim=1)

            ax.plot(
                rcv_coords_np,
                y_fit.cpu().numpy(),
                label="GMM Fit",
                color="crimson",
                linestyle="--",
                lw=1.8,
            )

            ax.set_title(f"Time: {time_val_scalar:.3f} s (Frame {n_p})")
            ax.set_xlabel("Receiver Position")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")

            plt.savefig(out_dir / f"obs_{n_p:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --- 3. Trajectory Time-Series Plot ---
    if not peak_detection_records.empty:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Filter out negligible amplitude noise components
        valid_peaks = peak_detection_records[peak_detection_records["peak_val"] > 1e-4]

        ax.scatter(
            valid_peaks["time_val"],
            valid_peaks["mu"],
            c=valid_peaks["peak_val"],
            cmap="viridis",
            s=15,
            alpha=0.8,
            edgecolors="none",
        )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Fitted Mean Position (μ)")
        ax.set_title("Peak Trajectories Over Time")
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.savefig(exp_dir / "time_series.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # --- Run reconstruction ---
    logger.info("Starting GMM reconstruction optimization...")
    model.peak_detection_records = pd.DataFrame(records)
    model.fitted_gaussian_params = fitted_gaussian_params.copy()
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