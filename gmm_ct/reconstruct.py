"""
Reconstruction runner for GMM-CT.

Loads observed projection data from disk, instantiates ``GMM_reco`` from a
YAML config, runs the 4-stage reconstruction pipeline, and saves the results.

Usage (Python API)::

    from gmm_ct.reconstruct import run_reconstruction
    from gmm_ct.config.yaml_config import load_reconstruct_config

    cfg = load_reconstruct_config("configs/reconstruct.yaml")
    run_reconstruction(cfg)

Usage (CLI)::

    gmm-ct reconstruct --config configs/reconstruct.yaml
"""

from datetime import datetime
from pathlib import Path
from time import time as wall_clock

import numpy as np
import torch

from .config.yaml_config import ReconstructConfig
from .core.reconstruction import GMM_reco
from .utils.helpers import export_parameters


def _load_projection_data(data_path: str, device: torch.device):
    """Load projection data and time vector from disk.

    Supports ``.pt`` (PyTorch) and ``.npy`` (NumPy) files.

    For ``.pt`` files the expected format is a dict with keys
    ``"projections"`` and ``"times"``.

    For ``.npy`` files a companion ``times.npy`` is expected in the same
    directory.

    Returns
    -------
    proj_data : torch.Tensor
        Projection measurements, shape ``(n_times, n_receivers)``.
    t : torch.Tensor
        Time vector, shape ``(n_times,)``.
    """
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
            raise FileNotFoundError(
                f"Expected companion file {times_path} alongside {path}"
            )
        t = torch.tensor(
            np.load(times_path), dtype=torch.float64, device=device
        )
    else:
        raise ValueError(
            f"Unsupported data format '{path.suffix}'. Use .pt or .npy"
        )

    return proj_data, t


def run_reconstruction(cfg: ReconstructConfig) -> dict:
    """Run the reconstruction pipeline from a YAML-driven config.

    Parameters
    ----------
    cfg : ReconstructConfig
        Fully parsed reconstruction configuration.

    Returns
    -------
    soln_dict : dict
        Optimised parameter dictionary.
    """
    start = wall_clock()

    # --- Device ---
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load observed data ---
    proj_data, t = _load_projection_data(cfg.data_path, device)
    print(f"Loaded projections: {proj_data.shape}")
    print(f"Time steps: {t.shape[0]}  ({t[0].item():.3f} – {t[-1].item():.3f}s)")

    # --- Instantiate model from config ---
    model = GMM_reco.from_config(cfg)

    # --- Run reconstruction ---
    # fit() expects proj_data in the list-of-tensors-per-source format
    # that generate_projections returns.  If we have a single-source 2-D
    # tensor from the data file, wrap it in a list.
    if isinstance(proj_data, torch.Tensor) and proj_data.dim() == 2:
        proj_data_input = [proj_data]
    else:
        proj_data_input = proj_data

    soln_dict = model.fit(proj_data_input, t)

    # --- Output ---
    out_dir = Path(cfg.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_parameters(
        soln_dict,
        out_dir / "estimated_parameters.md",
        title="Estimated Parameters",
    )

    torch.save(
        {
            "theta_est": soln_dict,
            "theta_init": getattr(model, "theta_dict_init", None),
            "config": {
                "n_gaussians": cfg.n_gaussians,
                "omega_range": list(cfg.physics.omega_range),
                "data_path": str(cfg.data_path),
                "device": str(device),
            },
        },
        out_dir / "reconstruction.pt",
    )

    elapsed = wall_clock() - start
    print(f"\nReconstruction complete in {elapsed:.1f}s")
    print(f"  Results saved to {out_dir}")
    print(f"  Final ω: "
          f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")

    return soln_dict
