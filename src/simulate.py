"""
Synthetic data simulation for GMM-CT.

Generates projection data from a ground-truth object and saves it.
"""

import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .config import ExperimentConfig
from .utils import (export_parameters, 
                    generate_true_param, 
                    set_random_seeds, 
                    add_sinogram_noise,)
from .model import GMM_reco
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

    # --- Fix the geometry & physics from config ---
    sources, receivers = cfg.geometry.to_tensors(device)
    d = cfg.geometry.dimensionality
    N = cfg.sim_n_gaussians
    x0s, a0s = cfg.physics.to_tensors(N, device)
    omega_min, omega_max = cfg.physics.omega_range
    n_proj = cfg.physics.n_projections
    t = torch.linspace(0.0, cfg.physics.duration, cfg.physics.n_projections, 
                       dtype=torch.float64, device=device)
    snr_db = 1e+08
    if cfg.add_sino_noise:
        snr_db = cfg.snr_db  

    # --- Simulate the ground truth parameters ---
    theta_true = generate_true_param(
        d=d, N=N, 
        initial_location=x0s[0], 
        initial_acceleration=a0s[0], 
        min_rot=omega_min, 
        max_rot=omega_max, 
        device=device,
    )
    
    # --- Setup output directory ---
    out_dir = Path(cfg.output.directory)
    if getattr(cfg.output, "use_timestamp", False):
        folder_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{cfg.seed}_N{N}"
    else:
        folder_name = f"snr{snr_db}_N{N}_nproj{n_proj}_seed{cfg.seed}"
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
        exp_dir=exp_dir,
    )
    proj_data = model.generate_projections(t, theta_true)
    if cfg.add_sino_noise:
        proj_data = [add_sinogram_noise(
            proj_data=model.process_projections(proj_data),
            snr_db=snr_db)]

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
            # Raw parameter dictionary for direct metric evaluation
            "params": {
                "v0s": theta_true['v0s'],           # Shape: [N, 2]
                "x0s": theta_true['x0s'],           # Shape: [N, 2]
                "a0s": theta_true['a0s'],           # Shape: [N, 2]
                "alphas": theta_true['alphas'],     # Shape: [N]
                "omegas": theta_true['omegas'],     # Shape: [N]
                "U_skews": theta_true['U_skews'],   # Shape: [N, 2, 2]
            },
            "theta_true": theta_true,  # Full flattened parameter vector
            "sources": sources,
            "receivers": receivers,
            "config": {
                "d": d,
                "N": N,
                "seed": cfg.seed,
                "snr_db": cfg.snr_db,
                "n_projections": n_proj,
                "duration": cfg.physics.duration,
                "device": str(device),
                "omega_min": omega_min,
                "omega_max": omega_max,
            },
        },
        exp_dir / "ground_truth.pt",
    )
    export_parameters(
        theta_true,
        exp_dir / "true_parameters.md",
        title="Ground Truth Parameters",
    )
    
    # --- Compute & Save Dataset Identifiability Score ---
    identifiability_dict = model.compute_dataset_identifiability_score(
        proj_tensor=proj_tensor,
        t=t,
        theta_dict=theta_true,
    )
    
    # Assign difficulty regime classification
    i_min = identifiability_dict.get("I_min", float('inf'))
    if i_min > 2.0:
        regime = "Regime A (Fully Identifiable)"
    elif i_min >= 1.0:
        regime = "Regime B (Partially Identifiable)"
    else:
        regime = "Regime C (Severely Unidentifiable)"

    identifiability_data = {
        "regime": regime,
        "metrics": identifiability_dict,
        "parameters": {
            "N": N,
            "n_projections": n_proj,
            "snr_db": snr_db,
            "seed": cfg.seed,
        }
    }
    
    ident_file = exp_dir / "identifiability.yaml"
    with open(ident_file, "w") as f:
        yaml.dump(identifiability_data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"The identifiability score for this simulation is {identifiability_data['metrics']['I_min']}")
    logger.info(f"This is rated to be {identifiability_data['regime']}")
    logger.info(f"Saved identifiability metrics to: {ident_file}")

    # --- Visualizations ---
    # if cfg.analysis.skip_animations:
    #     pass
    # else:
    #     logger.info(f"Generating the plots and animations...")
    #     animate_simulation(
    #         sim_dir=exp_dir,
    #         output_path=exp_dir / 'simulation_2d.mp4',
        # )
    # export_poster_gmm_figure(exp_dir)
    # export_poster_snapshot_sinogram_figure(exp_dir)

    return exp_dir