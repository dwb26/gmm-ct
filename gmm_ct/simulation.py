"""
Synthetic data simulation for GMM-CT.

Generates projection data from a ground-truth GMM and saves it alongside
the true parameters so that the data can later be fed into the
reconstruction pipeline (or to any other consumer) without coupling to
the reconstruction code.

Usage (Python API)::

    from gmm_ct.simulation import run_simulation
    from gmm_ct.config.yaml_config import load_simulate_config

    cfg = load_simulate_config("configs/simulate.yaml")
    run_simulation(cfg)

Usage (CLI)::

    gmm-ct simulate --config configs/simulate.yaml
"""

from datetime import datetime
from pathlib import Path
from time import time as wall_clock

import torch

from .config.yaml_config import SimulateConfig
from .core.reconstruction import GMM_reco
from .utils.generators import generate_true_param
from .utils.helpers import set_random_seeds, export_parameters


def run_simulation(cfg: SimulateConfig) -> Path:
    """Generate synthetic projection data from a YAML-driven config.

    Parameters
    ----------
    cfg : SimulateConfig
        Fully parsed simulation configuration.

    Returns
    -------
    Path
        Directory where outputs were written.
    """
    start = wall_clock()

    # --- Reproducibility ---
    set_random_seeds(cfg.simulation.seed)

    # --- Device ---
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Geometry ---
    sources, receivers = cfg.geometry.to_tensors(device)
    d = cfg.geometry.dimensionality

    # --- Physics ---
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
    dt = cfg.simulation.duration / (cfg.simulation.n_projections - 1)
    theta_true = generate_true_param(
        d, N, x0s[0], v_base, a0s[0], omega_min, omega_max,
        device=device, sampling_dt=dt,
    )

    # --- Generate projection data ---
    model = GMM_reco(
        d, N, sources, receivers, x0s, a0s,
        omega_min, omega_max, device=device,
    )
    proj_data = model.generate_projections(t, theta_true)

    # --- Output directory ---
    out_dir = Path(cfg.output.directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = out_dir / f"{timestamp}_seed{cfg.simulation.seed}_N{N}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # --- Save projections (the "observed" data for reconstruction) ---
    proj_tensor = model.process_projections(proj_data)
    torch.save(
        {
            "projections": proj_tensor,
            "times": t,
        },
        experiment_dir / "projections.pt",
    )

    # --- Save ground truth (kept separate — not needed for reconstruction) ---
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

    elapsed = wall_clock() - start
    print(f"\nSimulation complete in {elapsed:.1f}s")
    print(f"  Projections : {experiment_dir / 'projections.pt'}")
    print(f"  Ground truth: {experiment_dir / 'ground_truth.pt'}")
    print(f"  Parameters  : {experiment_dir / 'true_parameters.md'}")

    return experiment_dir
