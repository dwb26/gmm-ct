"""
Error analysis and publication-figure generation for GMM-CT.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from .model import GMM_reco

from .visualization.publication import (
    animate_temporal_gmm_comparison,
    # plot_acquisition_geometry_exact,
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
# compute_run_metrics
def run_analysis(exp_dir: Path) -> dict:
    """Evaluates parameter and spatio-temporal density metrics for a single experiment directory."""
    gt_path = exp_dir / "ground_truth.pt"
    rec_path = exp_dir / "reconstruction.pt"
    proj_path = exp_dir / "projections.pt"

    gt = torch.load(gt_path, map_location="cpu")
    rec = torch.load(rec_path, map_location="cpu")
    proj_data = torch.load(proj_path, map_location="cpu", weights_only=False)

    # Read in the experiment hyperparameters
    N = gt["config"]["N"]
    d = gt['config']['d']
    sources, receivers = gt['sources'], gt['receivers']
    t = proj_data["times"]
    proj_data = proj_data["projections"]

    # And the experiment parameters
    theta_true = gt["theta_true"]
    theta_pre_stage1_5 = rec.get("theta_pre_stage1_5")
    theta_pre_stage2 = rec.get("theta_pre_stage2")
    theta_est = rec["theta_est"]
    
    theta_est_matched, _ = reorder_theta_to_match_true(theta_true, theta_est, N)
    if theta_pre_stage1_5 is not None:
        theta_pre_stage1_5, _ = reorder_theta_to_match_true(theta_true, theta_est, N)

    # 1. Motion Errors (Kinematic)
    true_v0 = torch.stack(gt["params"]["v0s"])
    est_v0 = torch.stack(theta_est_matched["v0s"])
    v0_err = torch.sqrt(torch.sum((true_v0 - est_v0) ** 2, dim=1))
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
    U_err = torch.sqrt(torch.sum((true_U - est_U) ** 2, dim=(1, 2)))
    U_rmse = torch.mean(U_err).item()
    U_median_err = torch.median(U_err).item()

    # 3. Joint Physical Density Metric
    l2_err, rel_l2_err = compute_integrated_l2_density(
        exp_dir=exp_dir,
        gt_dict=gt,
        gt_params=theta_true,
        est_params=theta_est_matched,
    )
    
    # --- PDF Figure Generation ---
    # logger.info("Generating figure plots...")

    # # Static comparison plots
    # if theta_pre_stage1_5 is not None:
    #     plot_temporal_gmm_comparison(
    #         sources=sources, 
    #         receivers=receivers, 
    #         theta_true=theta_true, 
    #         theta_est=theta_pre_stage1_5, 
    #         output_dir=exp_dir,
    #         proj_data=proj_data,
    #         t=t, K=N, d=d,
    #         filename=exp_dir / "pre_stage_1_5_temporal_gmm_comparison.pdf",
    #         title="Stage 2 Initialization",
    #     )
    #     plot_individual_gaussian_reconstruction(
    #         theta_true=theta_true, 
    #         theta_est=theta_est, 
    #         K=N, d=d,
    #         gaussian_indices=range(N),
    #         filename=exp_dir / "pre_stage_1_5_individual_gaussian_reconstruction.pdf",
    #         theta_init=theta_pre_stage1_5,
    #     )
        
    # if theta_pre_stage2 is not None:
    #     plot_temporal_gmm_comparison(
    #         sources=sources,
    #         receivers=receivers,
    #         theta_true=theta_true,
    #         theta_est=theta_pre_stage2,
    #         output_dir=exp_dir,
    #         proj_data=proj_data,
    #         t=t, K=N, d=d,
    #         filename=exp_dir / "pre_stage_2_temporal_gmm_comparison.pdf",
    #         title="Stage 2 Initialization",
    #     )
    #     plot_individual_gaussian_reconstruction(
    #         theta_true=theta_true, 
    #         theta_est=theta_est, 
    #         K=N, d=d,
    #         gaussian_indices=range(N),
    #         filename=exp_dir / "pre_stage_2_individual_gaussian_reconstruction.pdf",
    #         theta_init=theta_pre_stage2,
    #     )

    # # Time captures
    # plot_temporal_gmm_comparison(
    #     sources=sources,
    #     receivers=receivers,
    #     theta_true=theta_true,
    #     theta_est=theta_est,
    #     t=t, K=N, d=d,
    #     output_dir=exp_dir,
    #     proj_data=proj_data,
    #     filename=exp_dir / "temporal_gmm_comparison.pdf",
    #     title="Reconstruction",
    # )
    
    # proj_2d = proj_data[0] if isinstance(proj_data, (list, tuple)) else proj_data
    # plot_sinogram(proj_2d, t, receivers, filename=exp_dir / "observed_sinogram.pdf")        
    # plot_projection_modes(
    #     proj_mixture=proj_2d, 
    #     t=t, 
    #     receivers=receivers,
    #     title="Projection Modes",
    #     filename=exp_dir / "projection_modes.pdf"
    # )

    # logger.info("All analysis outputs written to: %s", exp_dir)
    
    import matplotlib.pyplot as plt
    plt.close('all')

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
    
    

    # plot_acquisition_geometry_exact(
    #     sources=sources, 
    #     receivers=receivers, 
    #     d=d,
    #     filename=exp_dir / "acquisition_geometry_exact.pdf",
    # )
    
    # --- Animation ---
    # logger.info("Generating animation...")
    # animate_temporal_gmm_comparison(
    #     sources=sources, 
    #     receivers=receivers, 
    #     theta_true=theta_true, 
    #     theta_est=theta_est, 
    #     t=t, K=N, d=d,
    #     output_dir=exp_dir,
    #     proj_data=proj_data,
    #     filename=exp_dir / "temporal_gmm_comparison.mp4",
    # )