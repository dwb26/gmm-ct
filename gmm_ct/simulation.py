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

from .config import SimulateConfig
from .model import GMM_reco
from .utils import export_parameters, generate_true_param, set_random_seeds

from gmm_ct.visualization.simulate_viz import (animate_simulation,
                                               export_poster_gmm_figure,
                                               export_poster_snapshot_sinogram_figure,
)                                         

logger = logging.getLogger(__name__)


def run_simulation(cfg: SimulateConfig) -> Path:
    """Generate synthetic projection data from a YAML-driven config."""

    # --- Reproducibility & Device ---
    set_random_seeds(cfg.simulation.seed)
    device = torch.device(
        cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Geometry & Physics ---
    sources, receivers = cfg.geometry.to_tensors(device)
    d = cfg.geometry.dimensionality

    N = cfg.n_gaussians
    x0s, a0s = cfg.physics.to_tensors(N, device)
    omega_min, omega_max = cfg.physics.omega_range

    # --- Time mesh ---
    t = torch.linspace(
        0.0,
        cfg.simulation.duration,
        cfg.simulation.n_projections,
        dtype=torch.float64,
        device=device,
    )

    # --- Ground truth parameters ---
    v_base = torch.tensor(
        cfg.simulation.initial_velocity, dtype=torch.float64, device=device
    )
    # generate_true_param also takes x0, v0, a0 base vectors
    theta_true = generate_true_param(
        d, N, x0s[0], v_base, a0s[0], omega_min, omega_max, device=device,
    )

    # --- Generate projection data ---
    logger.info(f"Generating projections...")
    model = GMM_reco(
        d, N, sources, receivers, x0s, a0s,
        omega_min, omega_max, device=device,
    )
    proj_data = model.generate_projections(t, theta_true)

    # --- Output directory ---
    out_dir = Path(cfg.output.directory)
    if getattr(cfg.output, "use_timestamp", False):
        folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{cfg.simulation.seed}_N{N}"
    else:
        folder_name = f"sim_seed{cfg.simulation.seed}_N{N}"
    
    experiment_dir = out_dir / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # --- Save projections ---
    proj_tensor = model.process_projections(proj_data)
    torch.save(
        {
            "projections": proj_tensor,
            "times": t,
        },
        experiment_dir / "projections.pt",
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
                "seed": cfg.simulation.seed,
                "omega_min": omega_min,
                "omega_max": omega_max,
                "n_projections": cfg.simulation.n_projections,
                "duration": cfg.simulation.duration,
                "device": str(device),
            },
        },
        experiment_dir / "ground_truth.pt",
    )
    export_parameters(
        theta_true,
        experiment_dir / "true_parameters.md",
        title="Ground Truth Parameters",
    )

    # --- Visualizations ---
    logger.info(f"Generating the plots and animations...")
    animate_simulation(
        sim_dir=experiment_dir,
        output_path=experiment_dir / 'simulation_2d.mp4'
    )
    export_poster_gmm_figure(experiment_dir)
    export_poster_snapshot_sinogram_figure(experiment_dir)

    return experiment_dir
