"""
Synthetic data simulation for GMM-CT.

Generates projection data from a ground-truth GMM and saves it alongside
the true parameters so that the data can later be fed into the
reconstruction pipeline without coupling to the reconstruction code.

Usage (Python API)::

    from gmm_ct.simulation import run_simulation
    from gmm_ct.config.yaml_config import load_simulate_config

    cfg = load_simulate_config("configs/simulate.yaml")
    run_simulation(cfg)

Usage (CLI)::

    gmm-ct simulate --config configs/simulate_2D.yaml
"""

import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

import torch

from .config import ExperimentConfig
from .model import GMM_reco
from .utils import export_parameters, generate_true_param, set_random_seeds

from .visualization.simulate_viz import (animate_simulation,
                                         export_poster_gmm_figure,
                                         export_poster_snapshot_sinogram_figure,
)                                         

logger = logging.getLogger(__name__)


def run_simulation(cfg: ExperimentConfig) -> Path:
    """Generate synthetic projection data from a YAML-driven config."""

    # --- Reproducibility & Device ---
    set_random_seeds(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Geometry & Physics ---
    sources, receivers = cfg.geometry.to_tensors(device)
    d = cfg.geometry.dimensionality

    N = cfg.sim_n_gaussians
    x0s, a0s = cfg.physics.to_tensors(N, device)
    omega_min, omega_max = cfg.physics.omega_range
    n_proj = cfg.physics.n_projections

    # --- Time mesh ---
    t = torch.linspace(
        0.0,
        cfg.physics.duration,
        cfg.physics.n_projections,
        dtype=torch.float64,
        device=device,
    )

    # --- Ground truth parameters ---
    v_base = torch.tensor(
        cfg.physics.initial_velocities, dtype=torch.float64, device=device
    )
    # generate_true_param also takes x0, v0, a0 base vectors
    theta_true = generate_true_param(
        d=d, N=N, 
        initial_location=x0s[0], 
        initial_velocity=v_base,    # Gets perturbed by noise to provide different true v0s
        initial_acceleration=a0s[0], 
        min_rot=omega_min, 
        max_rot=omega_max, 
        device=device,
    )

    # --- Output directory ---
    out_dir = Path(cfg.output.directory)
    if getattr(cfg.output, "use_timestamp", False):
        folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{cfg.seed}_N{N}"
    else:
        folder_name = f"seed{cfg.seed}_N{N}_nproj{n_proj}"
    
    exp_dir = out_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to: {exp_dir}")

    # --- Generate projection data ---
    logger.info(f"Generating projections...")
    model = GMM_reco(
        d=d, N=N, 
        sources=sources, 
        receivers=receivers, 
        x0s=x0s, a0s=a0s,
        omega_min=omega_min, 
        omega_max=omega_max, 
        output_dir=exp_dir,
    )
    proj_data = model.generate_projections(t, theta_true)

    # --- Save projections ---
    proj_tensor = model.process_projections(proj_data)
    torch.save(
        {
            "projections": proj_tensor,
            "times": t,
        },
        exp_dir / "projections.pt",
    )

    # --- Save ground truth ---
    torch.save(
        {
            "theta_true": theta_true,
            "sources": sources,
            "receivers": receivers,
            "config": {
                "d": d,
                "N": N,
                "seed": cfg.seed,
                "omega_min": omega_min,
                "omega_max": omega_max,
                "n_projections": n_proj,
                "duration": cfg.physics.duration,
                "device": str(device),
            },
        },
        exp_dir / "ground_truth.pt",
    )
    export_parameters(
        theta_true,
        exp_dir / "true_parameters.md",
        title="Ground Truth Parameters",
    )

    # --- Visualizations ---
    if cfg.analysis.skip_animations:
        pass
    else:
        logger.info(f"Generating the plots and animations...")
        animate_simulation(
            sim_dir=exp_dir,
            output_path=exp_dir / 'simulation_2d.mp4',
        )
    export_poster_gmm_figure(exp_dir)
    export_poster_snapshot_sinogram_figure(exp_dir)

    return exp_dir
