"""
Error analysis and publication-figure generation for GMM-CT.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Prevent truncation of columns and expand terminal display width
from .config import AnalysisConfig, ExperimentConfig
from .model import GMM_reco

from .visualization.publication import (
    animate_temporal_gmm_comparison,
    plot_acquisition_geometry_exact,
    plot_individual_gaussian_reconstruction,
    plot_temporal_gmm_comparison,
    plot_projection_modes,
    plot_sinogram,
    reorder_theta_to_match_true,
)

logger = logging.getLogger(__name__)


def compute_spatial_mesh(
    resolution: int,
    x_min: float = -1.0,
    x_max: float = 5.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
):
    device = 'cpu'
    xs = torch.linspace(x_min, x_max, resolution, dtype=torch.float64, device=device)
    ys = torch.linspace(y_min, y_max, resolution, dtype=torch.float64, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

    positions = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # [P, 2]

    # Cell area and per-time trapezoidal weights for the spatio-temporal quadrature
    dx = (x_max - x_min) / (resolution - 1)
    dy = (y_max - y_min) / (resolution - 1)
    cell_area = dx * dy
    
    return positions, cell_area

def compute_integrated_l2_density(
    exp_dir: Path,
    gt_dict: dict,
    gt_params: dict, 
    est_params: dict | None = None,
    resolution: int = 200,
) -> tuple[float, float]:
    """Computes spatio-temporal L2 density errors in a single pass.
    
    Returns:
        Tuple[float, float]: (absolute_l2_error, relative_l2_error)
    """
    d = gt_dict['config']['d']
    N = gt_dict['config']['N']
    sources, receivers = gt_dict['sources'], gt_dict['receivers']
    x0s, a0s = gt_dict['params']['x0s'], gt_dict['params']['a0s']
    omega_min, omega_max = gt_dict['config']['omega_min'], gt_dict['config']['omega_max']
    duration = gt_dict['config']['duration']
    n_proj = gt_dict['config']['n_projections']
    
    t = torch.linspace(0.0, duration, n_proj, dtype=torch.float64, device='cpu')
    
    model = GMM_reco(
        d=d, N=N, 
        sources=sources, receivers=receivers, 
        x0s=x0s, a0s=a0s,
        omega_min=omega_min, omega_max=omega_max, 
        exp_dir=exp_dir,
    )
    
    positions, cell_area = compute_spatial_mesh(resolution)
    
    rho_true = model.evaluate_density(t=t, theta_dict=gt_params, positions=positions)
    rho_true = rho_true.reshape(t.shape[0], resolution, resolution)
    
    if est_params is not None:
        rho_est = model.evaluate_density(t=t, theta_dict=est_params, positions=positions)
        rho_est = rho_est.reshape(t.shape[0], resolution, resolution)
    else:
        rho_est = torch.zeros_like(rho_true)

    # Absolute spatio-temporal L2 error
    sq_error_per_time = torch.sum((rho_true - rho_est) ** 2, dim=(1, 2)) * cell_area
    abs_l2_error = torch.trapezoid(sq_error_per_time, t).item()
    
    # Baseline ground-truth L2 norm
    sq_gt_per_time = torch.sum(rho_true ** 2, dim=(1, 2)) * cell_area
    gt_energy = torch.trapezoid(sq_gt_per_time, t).item()
    
    rel_l2_error = abs_l2_error / (gt_energy + 1e-12)
    
    return abs_l2_error, rel_l2_error

def compute_run_metrics(exp_dir: Path) -> dict:
    """Evaluates parameter and spatio-temporal density metrics for a single experiment directory."""
    gt_path = exp_dir / "ground_truth.pt"
    rec_path = exp_dir / "reconstruction.pt"

    gt = torch.load(gt_path, map_location="cpu")
    rec = torch.load(rec_path, map_location="cpu")

    N = gt["config"]["N"]

    theta_true = gt["theta_true"]
    theta_est = rec["theta_est"]
    theta_est_matched, _ = reorder_theta_to_match_true(theta_true, theta_est, N)

    # 1. Motion Errors (Kinematic)
    true_v0 = torch.stack(gt["params"]["v0s"])
    est_v0 = torch.stack(theta_est_matched["v0s"])
    v0_err = torch.sqrt(torch.sum((true_v0 - est_v0) ** 2, dim=1))  # [N]
    v0_rmse = torch.mean(v0_err).item()
    v0_median_err = torch.median(v0_err).item()

    true_omega = torch.stack(gt["params"]["omegas"])
    est_omega = torch.stack(theta_est_matched["omegas"])
    omega_err = torch.abs(true_omega - est_omega)  # [N]
    omega_rmse = torch.sqrt(torch.mean(omega_err**2)).item()
    omega_median_err = torch.median(omega_err).item()

    # 2. Morphology Errors (Static shape)
    true_alpha = torch.stack(gt["params"]["alphas"])
    est_alpha = torch.stack(theta_est_matched["alphas"])
    alpha_err = torch.abs(true_alpha - est_alpha)  # [N]
    alpha_rmse = torch.sqrt(torch.mean(alpha_err**2)).item()
    alpha_median_err = torch.median(alpha_err).item()

    true_U = torch.stack(gt["params"]["U_skews"])
    est_U = torch.stack(theta_est_matched["U_skews"])
    U_err = torch.sqrt(torch.sum((true_U - est_U) ** 2, dim=(1, 2)))  # [N]
    U_rmse = torch.mean(U_err).item()
    U_median_err = torch.median(U_err).item()

    # 3. Joint Physical Density Metric
    l2_err, rel_l2_err = compute_integrated_l2_density(
        exp_dir=exp_dir,
        gt_dict=gt,
        gt_params=theta_true,
        est_params=theta_est_matched,
    )

    return {
        "exp_dir": str(exp_dir),
        "N": N,
        "n_proj": gt["config"]["n_projections"],
        "snr_db": float(gt["config"]["snr_db"]),
        "seed": gt["config"]["seed"],
        # Mean/RMSE Parameter Errors
        "v0_rmse": v0_rmse,
        "omega_rmse": omega_rmse,
        "alpha_rmse": alpha_rmse,
        "U_rmse": U_rmse,
        # Robust Median Parameter Errors
        "v0_median_err": v0_median_err,
        "omega_median_err": omega_median_err,
        "alpha_median_err": alpha_median_err,
        "U_median_err": U_median_err,
        # Joint Spatial/Temporal Errors
        "l2_density_error": l2_err,
        "rel_l2_density_error": rel_l2_err,
        "log_l2_density_error": np.log10(l2_err + 1e-12),
        "log_rel_l2_density_error": np.log10(rel_l2_err + 1e-12),
    }

    
def run_analysis(
    exp_dir: Path,
    cfg: ExperimentConfig,
) -> None:
    """Load unified results bundle and execute complete post-reconstruction analysis."""
    exp_dir = Path(exp_dir)
    logger.info(f"Running analysis on {exp_dir}...")
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Cannot run analysis: missing {exp_dir}")
    
    logger.info(f"Loading from {exp_dir}")
    gt_data = torch.load(exp_dir / "ground_truth.pt" , map_location="cpu", weights_only=False)
    est_data = torch.load(exp_dir / "reconstruction.pt" , map_location="cpu", weights_only=False)
    proj_data = torch.load(exp_dir / "projections.pt" , map_location="cpu", weights_only=False)
    
    omega_min, omega_max = cfg.physics.omega_range
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sources, receivers = cfg.geometry.to_tensors(device=device)
    
    analyze_results(
        theta_true=gt_data["theta_true"],
        theta_est=est_data["theta_est"],
        theta_init=est_data.get("theta_init"),
        theta_stage1_init=est_data.get("theta_stage1_init"),
        proj_data=proj_data["projections"],
        t=proj_data["times"],
        sources=sources,
        receivers=receivers,
        d=cfg.geometry.dimensionality,
        N=cfg.reco_n_gaussians,
        omega_min=omega_min,
        omega_max=omega_max,
        device=cfg.device,
        exp_dir=exp_dir,
        analysis_cfg=cfg.analysis,
    )
    
    
def analyze_results(
    *,
    theta_true: dict,
    theta_est: dict,
    theta_init: dict | None,
    theta_stage1_init: dict | None = None,
    proj_data: list[torch.Tensor] | None = None,
    t: torch.Tensor,
    sources: torch.Tensor,
    receivers: torch.Tensor,
    d: int,
    N: int,
    omega_min: float,
    omega_max: float,
    device: torch.device,
    exp_dir: Path,
    analysis_cfg: AnalysisConfig | None = None,
):
    """Compute relative parameter errors and output publication PDF plots."""
    # Match permutations: align estimated indices to true particles by trajectory
    theta_est, matching_indices = reorder_theta_to_match_true(theta_true, theta_est, N)
    logger.info("Permutation matching (est -> true): %s", matching_indices)
    
    if theta_init is not None:
        theta_init, _ = reorder_theta_to_match_true(theta_true, theta_init, N)

    # --- Error analysis ---
    if not analysis_cfg.skip_errors:
        x0s, a0s = theta_true["x0s"], theta_true["a0s"]
        model = GMM_reco(
            d=d, N=N, 
            sources=sources, 
            receivers=receivers, 
            x0s=x0s, a0s=a0s,
            omega_min=omega_min, 
            omega_max=omega_max, 
            device=device,
            exp_dir=exp_dir,
        )

        errors_init = _compute_parameter_errors(theta_true, theta_init, N) if theta_init else {}
        errors_final = _compute_parameter_errors(theta_true, theta_est, N)

        proj_init = model.generate_projections(t, theta_init) if theta_init else proj_data
        proj_final = model.generate_projections(t, theta_est)
        proj_err_init = _compute_projection_error(proj_data, proj_init) if theta_init else 0.0
        proj_err_final = _compute_projection_error(proj_data, proj_final)

        if theta_init:
            _print_error_summary(errors_init, errors_final, proj_err_init, proj_err_final)
            _plot_error_table(
                errors_init, errors_final,
                proj_err_init, proj_err_final,
                exp_dir / "error_analysis.pdf",
            )

    # --- PDF Figure Generation ---
    if not analysis_cfg.skip_plots:
        logger.info("Generating figure plots...")

        plot_acquisition_geometry_exact(
            sources=sources, 
            receivers=receivers, 
            d=d,
            filename=exp_dir / "acquisition_geometry_exact.pdf",
        )

        plot_individual_gaussian_reconstruction(
            theta_true=theta_true, 
            theta_est=theta_est, 
            K=N, d=d,
            gaussian_indices=range(N),
            filename=exp_dir / "individual_gaussian_reconstruction.pdf",
            theta_init=theta_stage1_init,
        )

        if theta_init is not None:
            plot_temporal_gmm_comparison(
                sources=sources, 
                receivers=receivers, 
                theta_true=theta_true, 
                theta_est=theta_init, 
                output_dir=exp_dir,
                proj_data=proj_data,
                t=t, K=N, d=d,
                filename=exp_dir / "initial_temporal_gmm_comparison.pdf",
                title="Stage 2 Initialization",
            )

        plot_temporal_gmm_comparison(
            sources=sources,
            receivers=receivers,
            theta_true=theta_true,
            theta_est=theta_est,
            t=t, K=N, d=d,
            output_dir=exp_dir,
            proj_data=proj_data,
            filename=exp_dir / "temporal_gmm_comparison.pdf",
            title="Reconstruction",
        )
        
        proj_2d = proj_data[0] if isinstance(proj_data, (list, tuple)) else proj_data
        plot_sinogram(proj_2d, t, receivers, filename=exp_dir / "observed_sinogram.pdf")        
        plot_projection_modes(
            proj_mixture=proj_2d, 
            t=t, 
            receivers=receivers,
            title="Projection Modes",
            filename=exp_dir / "projection_modes.pdf"
        )


    # --- Animation ---
    if not analysis_cfg.skip_animations:
        logger.info("Generating animation...")
        animate_temporal_gmm_comparison(
            sources=sources, 
            receivers=receivers, 
            theta_true=theta_true, 
            theta_est=theta_est, 
            t=t, K=N, d=d,
            output_dir=exp_dir,
            proj_data=proj_data,
            filename=exp_dir / "temporal_gmm_comparison.mp4",
        )

    logger.info("All analysis outputs written to: %s", exp_dir)


# ======================================================================
# Error metric helpers
# ======================================================================

def _compute_parameter_errors(theta_true: dict, theta_est: dict, N: int) -> dict:
    """Compute relative L2 errors for each parameter stack."""
    errors = {}

    def _rel_l2(true_tensor, est_tensor):
        denom = torch.norm(true_tensor)
        return (torch.norm(true_tensor - est_tensor) / denom).item() if denom > 0 else 0.0

    for key, is_matrix in [("alphas", False), ("x0s", False), ("v0s", False),
                         ("U_skews", True), ("omegas", False)]:
        true_stack = torch.stack([
            theta_true[key][n].flatten() if is_matrix else theta_true[key][n]
            for n in range(N)
        ])
        est_stack = torch.stack([
            theta_est[key][n].flatten() if is_matrix else theta_est[key][n]
            for n in range(N)
        ])
        errors[key] = _rel_l2(true_stack, est_stack)

    return errors


def _compute_projection_error(proj_true, proj_est) -> float:
    """Relative L2 error between projection time series."""
    true_flat = torch.cat([p.flatten() for p in proj_true])
    est_flat = torch.cat([p.flatten() for p in proj_est])
    denom = torch.norm(true_flat)
    return (torch.norm(true_flat - est_flat) / denom).item() if denom > 0 else 0.0


def _print_error_summary(errors_init, errors_final, proj_err_init, proj_err_final):
    """Log error reduction table to console."""
    labels = {
        "alphas": "Amplitudes (α)",
        "x0s": "Positions  (x₀)",
        "v0s": "Velocities (v₀)",
        "U_skews": "Shape      (U)",
        "omegas": "Rotation   (ω)",
    }
    logger.info("Parameter Errors (Relative L2):")
    logger.info("  %-22s %12s  %12s  %12s", "Parameter", "Init", "Final", "Improvement")
    for key in ["alphas", "x0s", "v0s", "U_skews", "omegas"]:
        init = errors_init[key]
        final = errors_final[key]
        imp = 100 * (1 - final / init) if init > 0 else 0.0
        logger.info("  %-22s %12.4e  %12.4e  %+11.1f%%", labels[key], init, final, imp)

    imp_proj = 100 * (1 - proj_err_final / proj_err_init) if proj_err_init > 0 else 0.0
    logger.info("  %-22s %12.4e  %12.4e  %+11.1f%%", "Projections", proj_err_init, proj_err_final, imp_proj)


def _plot_error_table(errors_init, errors_final, proj_err_init, proj_err_final, output_path):
    """Save clean error summary table PDF."""
    labels = {
        "alphas": "Amplitudes (α)",
        "x0s": "Initial Positions (x₀)",
        "v0s": "Initial Velocities (v₀)",
        "U_skews": "Shape Matrices (U)",
        "omegas": "Angular Velocities (ω)",
    }

    header = ["Parameter", "Init Error", "Final Error", "Improvement", "Reduction"]
    rows = [header]

    for key in ["alphas", "x0s", "v0s", "U_skews", "omegas"]:
        init = errors_init[key]
        final = errors_final[key]
        imp = 100 * (1 - final / init) if init > 0 else 0.0
        red = init / final if final > 0 else np.inf
        rows.append([labels[key], f"{init:.4e}", f"{final:.4e}", f"{imp:.1f}%", f"{red:.1f}×"])

    imp_proj = 100 * (1 - proj_err_final / proj_err_init) if proj_err_init > 0 else 0.0
    red_proj = proj_err_init / proj_err_final if proj_err_final > 0 else np.inf
    rows.append(["Projections", f"{proj_err_init:.4e}", f"{proj_err_final:.4e}", f"{imp_proj:.1f}%", f"{red_proj:.1f}×"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    table = ax.table(cellText=rows, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.4)

    for j in range(len(header)):
        table[(0, j)].set_facecolor("#2E86AB")
        table[(0, j)].set_text_props(weight="bold", color="white", fontsize=13)
        table[(len(rows) - 1, j)].set_facecolor("#E8E8E8")
        table[(len(rows) - 1, j)].set_text_props(weight="bold")

    for i in range(1, len(rows) - 1):
        if i % 2 == 0:
            for j in range(len(header)):
                table[(i, j)].set_facecolor("#F5F5F5")

    ax.set_title("Error Analysis: Initialisation vs Optimisation", fontweight="bold", fontsize=18, pad=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Error table saved: %s", output_path)
    
    
    
    
    # def run_large_scale_analysis(
#     sim_dirs: list[Path],
#     output_path: Path | None = None,
# ) -> pd.DataFrame:
    
#     records = []
#     for exp_dir in sim_dirs:
#         logger.info(f"Analyzing for {exp_dir}...")
        
#         gt_path = exp_dir / "ground_truth.pt"
#         rec_path = exp_dir / "reconstruction.pt"
        
#         gt = torch.load(gt_path, map_location='cpu')
#         rec = torch.load(rec_path, map_location='cpu')
    
#         N = gt["config"]["N"]
        
#         # Extract the respective parameter dictionaries
#         theta_true = gt["theta_true"]
#         theta_est = rec["theta_est"]        
#         theta_est_matched, _ = reorder_theta_to_match_true(theta_true, theta_est, N)
        
#         # -------------------------------------------------------------
#         # Category 1: Kinematic / Motion Metrics (Subproblem 1)
#         # -------------------------------------------------------------    
#         true_v0 = torch.stack(gt["params"]["v0s"])          # [N, d]
#         est_v0 = torch.stack(theta_est_matched["v0s"])      # [N, d]        
#         v0_rmse = torch.sqrt(torch.mean((true_v0 - est_v0) ** 2)).item()
        
#         true_omega = torch.stack(gt["params"]["omegas"])          # [N]
#         est_omega = torch.stack(theta_est_matched["omegas"])      # [N]
#         omega_rmse = torch.sqrt(torch.mean((true_omega - est_omega) ** 2)).item()
        
#         # -------------------------------------------------------------
#         # Category 2: Static Morphology Metrics (Subproblem 2)
#         # -------------------------------------------------------------
#         true_alpha = torch.stack(gt["params"]["alphas"])  # [N]
#         est_alpha = torch.stack(theta_est_matched["alphas"])  # [N]
#         alpha_rmse = torch.sqrt(torch.mean((true_alpha - est_alpha) ** 2)).item()

#         true_U = torch.stack(gt["params"]["U_skews"])  # [N, d, d]
#         est_U = torch.stack(theta_est_matched["U_skews"])  # [N, d, d]
#         U_rmse = torch.sqrt(torch.mean((true_U - est_U) ** 2)).item()
        
#         # -------------------------------------------------------------
#         # Category 3: Joint Physical Object Metric (End-to-End)
#         # -------------------------------------------------------------
#         # Evaluates physical density matching across spatial domain & time
#         l2_density_error = compute_integrated_l2_density(
#             exp_dir=exp_dir,
#             gt_dict=gt,
#             gt_params=theta_true, 
#             est_params=theta_est_matched,
#             relative=False,
#         )
#         rel_l2_density_error = compute_integrated_l2_density(
#             exp_dir=exp_dir,
#             gt_dict=gt,
#             gt_params=theta_true,
#             est_params=theta_est_matched,
#             relative=True,
#         )
        
#         # -------------------------------------------------------------
#         # Aggregate Record
#         # -------------------------------------------------------------
#         records.append({
#             "exp_dir": str(exp_dir),
#             # Swept Metadata
#             "N": N,
#             "n_proj": gt["config"]["n_projections"],
#             "snr_db": float(gt["config"]["snr_db"]),
#             "seed": gt["config"]["seed"],
#             # 1. Motion Errors
#             "v0_rmse": v0_rmse,
#             "omega_rmse": omega_rmse,
#             # 2. Morphology Errors
#             "alpha_rmse": alpha_rmse,
#             "U_rmse": U_rmse,
#             # 3. Joint Physical Error
#             "l2_density_error": l2_density_error,
#             "log_l2_density_error": np.log10(l2_density_error),
#             "rel_log_l2_density_error": np.log10(rel_l2_density_error),
#         })
        
#     df = pd.DataFrame(records)
    
#     pd.set_option("display.max_columns", None)
#     pd.set_option("display.width", 1000)
#     pd.set_option("display.max_colwidth", None)
#     print(df.head(100))
    
#     if output_path:
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         df.to_parquet(output_path, index=False)
#         logger.info(f"Saved benchmark metrics to {output_path}")
        
#     return df