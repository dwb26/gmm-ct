"""
Error analysis and publication-figure generation for GMM-CT.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import AnalysisConfig
from .model import GMM_reco

logger = logging.getLogger(__name__)
    
def run_analysis(
    experiment_dir: Path,
    analysis_cfg: AnalysisConfig | None = None,
) -> None:
    """Load unified results bundle and execute complete post-reconstruction analysis."""
    experiment_dir = Path(experiment_dir)
    results_path = experiment_dir / "results.pt"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Cannot run analysis: missing {results_path}")
    
    data = torch.load(results_path, map_location="cpu", weights_only=False)
    
    analyze_results(
        theta_true=data["theta_true"],
        theta_est=data["theta_est"]
    )
    
    analyze_results(
        theta_true=data["theta_true"],
        theta_est=data["theta_est"],
        theta_init=data.get("theta_init"),
        theta_stage1_init=data.get("theta_stage1_init"),
        proj_data=data["proj_data"],
        t=data["t"],
        sources=data["sources"],
        receivers=data["receivers"],
        d=data["config"]["d"],
        N=data["config"]["N"],
        omega_min=data["config"]["omega_min"],
        omega_max=data["config"]["omega_max"],
        device=torch.device(data["config"]["device"]),
        experiment_dir=experiment_dir,
        analysis_cfg=analysis_cfg,
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
    experiment_dir: Path,
    analysis_cfg: AnalysisConfig | None = None,
    res: dict | None = None,
):
    """Compute relative parameter errors and output publication PDF plots."""
    from .visualization.publication import (
        animate_temporal_gmm_comparison,
        plot_acquisition_geometry_exact,
        plot_individual_gaussian_reconstruction,
        plot_temporal_gmm_comparison,
        plot_projection_modes,
        plot_sinogram,
        reorder_theta_to_match_true,
    )
    from .visualization.animations import animate_GMM_motion

    if analysis_cfg is None:
        analysis_cfg = AnalysisConfig()

    # Match permutations: align estimated indices to true particles by trajectory
    theta_est, matching_indices = reorder_theta_to_match_true(theta_true, theta_est, N)
    logger.info("Permutation matching (est -> true): %s", matching_indices)
    
    if theta_init is not None:
        theta_init, _ = reorder_theta_to_match_true(theta_true, theta_init, N)

    # --- Error analysis ---
    if not analysis_cfg.skip_errors:
        x0s, a0s = theta_true["x0s"], theta_true["a0s"]
        model = GMM_reco(
            d, N, sources, receivers, x0s, a0s,
            omega_min, omega_max, device=device,
            output_dir=experiment_dir,
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
                experiment_dir / "error_analysis.pdf",
            )

    # --- PDF Figure Generation ---
    if not analysis_cfg.skip_plots:
        logger.info("Generating figure plots...")

        plot_acquisition_geometry_exact(
            sources=sources, 
            receivers=receivers, 
            d=d,
            filename=experiment_dir / "acquisition_geometry_exact.pdf",
        )

        plot_individual_gaussian_reconstruction(
            theta_true=theta_true, 
            theta_est=theta_est, 
            K=N, d=d,
            gaussian_indices=range(N),
            filename=experiment_dir / "individual_gaussian_reconstruction.pdf",
            theta_init=theta_stage1_init,
        )

        if theta_init is not None:
            plot_temporal_gmm_comparison(
                sources=sources, 
                receivers=receivers, 
                theta_true=theta_true, 
                theta_est=theta_init, 
                t=t, K=N, d=d,
                filename=experiment_dir / "initial_temporal_gmm_comparison.pdf",
                title="Stage 2 Initialization",
            )

        plot_temporal_gmm_comparison(
            sources=sources,
            receivers=receivers,
            theta_true=theta_true,
            theta_est=theta_est,
            t=t,K=N,d=d,
            filename=experiment_dir / "temporal_gmm_comparison.pdf",
            title="Reconstruction",
        )
        
        proj_2d = proj_data[0] if isinstance(proj_data, (list, tuple)) else proj_data
        plot_sinogram(proj_2d, t, receivers, filename=experiment_dir / "observed_sinogram.pdf")        
        plot_projection_modes(
            proj_mixture=proj_2d, 
            t=t, 
            receivers=receivers,
            title="Projection Modes",
            filename=experiment_dir / "projection_modes.pdf"
        )


    # --- Animation ---
    # if not analysis_cfg.skip_animations:
    #     logger.info("Generating animation...")
    #     anim = animate_temporal_gmm_comparison(
    #         sources, receivers, theta_true, theta_est, t, N, d,
    #         filename=experiment_dir / "temporal_gmm_comparison.mp4",
    #     )

    logger.info("All analysis outputs written to: %s", experiment_dir)


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