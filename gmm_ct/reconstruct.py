"""
Reconstruction (and optional analysis) runner for GMM-CT.

Loads observed projection data from disk, instantiates ``GMM_reco`` from a
YAML config, runs the 4-stage reconstruction pipeline, saves the results,
and — when ground-truth data is available — automatically runs error
analysis and generates publication-quality plots.

Usage (Python API)::

    from gmm_ct.reconstruct import run_reconstruction
    from gmm_ct.config.yaml_config import load_reconstruct_config

    cfg = load_reconstruct_config("configs/reconstruct.yaml")
    run_reconstruction(cfg)

Usage (CLI)::

    gmm-ct reconstruct --config configs/reconstruct.yaml
    gmm-ct reconstruct --config configs/reconstruct.yaml --skip-analysis
    gmm-ct reconstruct --config configs/reconstruct.yaml --skip-animations
"""

from datetime import datetime
from pathlib import Path
from time import time as wall_clock

import numpy as np
import torch

from .config.yaml_config import AnalysisConfig, ReconstructConfig
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
    
    # --- Output directory ---
    N = cfg.n_gaussians
    out_dir = Path(cfg.output.directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = out_dir / f"{timestamp}_N{N}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # --- Save estimated parameters (markdown) ---
    export_parameters(
        soln_dict,
        experiment_dir / "estimated_parameters.md",
        title="Estimated Parameters",
    )

    # --- Save reconstruction-only checkpoint ---
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
        experiment_dir / "reconstruction.pt",
    )

    # --- Save combined results.pt and run analysis if ground truth exists ---
    data_dir = Path(cfg.data_path).parent
    gt_path = data_dir / "ground_truth.pt"
    theta_init = getattr(model, "theta_dict_init", None)

    if gt_path.exists():
        gt = torch.load(gt_path, map_location=device, weights_only=False)
        gt_cfg = gt.get("config", {})
        theta_true = gt["theta_true"]
        sources = gt["sources"]
        receivers = gt["receivers"]

        torch.save(
            {
                "theta_true": theta_true,
                "theta_est": soln_dict,
                "theta_init": theta_init,
                "proj_data": proj_data_input,
                "t": t,
                "sources": sources,
                "receivers": receivers,
                "config": {
                    "d": gt_cfg.get("d", 2),
                    "N": cfg.n_gaussians,
                    "seed": gt_cfg.get("seed", -1),
                    "omega_min": cfg.physics.omega_range[0],
                    "omega_max": cfg.physics.omega_range[1],
                    "device": str(device),
                },
            },
            experiment_dir / "results.pt",
        )

        elapsed = wall_clock() - start
        print(f"\nReconstruction complete in {elapsed:.1f}s")
        print(f"  Results saved to {experiment_dir}")
        print(f"  Final ω: "
              f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")

        # --- Automatic analysis ---
        if cfg.analysis.enabled:
            print(f"\n{'='*50}")
            print("Running post-reconstruction analysis...")
            print(f"{'='*50}")
            analyse_results(
                theta_true=theta_true,
                theta_est=soln_dict,
                theta_init=theta_init,
                proj_data=proj_data_input,
                t=t,
                sources=sources,
                receivers=receivers,
                d=gt_cfg.get("d", 2),
                N=cfg.n_gaussians,
                omega_min=cfg.physics.omega_range[0],
                omega_max=cfg.physics.omega_range[1],
                device=device,
                experiment_dir=experiment_dir,
                analysis_cfg=cfg.analysis,
            )
    else:
        elapsed = wall_clock() - start
        print(f"\nReconstruction complete in {elapsed:.1f}s")
        print(f"  Results saved to {experiment_dir}")
        print(f"  Final ω: "
              f"{[f'{omega.item():.4f}' for omega in soln_dict['omegas']]}")
        print(f"  (ground_truth.pt not found in {data_dir}; "
              f"analysis skipped)")

    return soln_dict


# ======================================================================
# Analysis
# ======================================================================

def analyse_results(
    *,
    theta_true: dict,
    theta_est: dict,
    theta_init: dict | None,
    proj_data,
    t: torch.Tensor,
    sources,
    receivers,
    d: int,
    N: int,
    omega_min: float,
    omega_max: float,
    device: torch.device,
    experiment_dir: Path,
    analysis_cfg: AnalysisConfig | None = None,
):
    """Run error analysis and generate comparison plots.

    This is the logic formerly in ``scripts/analyse.py``, factored out
    so it can be called directly after reconstruction.

    Parameters
    ----------
    theta_true : dict
        Ground-truth parameter dictionary.
    theta_est : dict
        Estimated parameter dictionary (will be reordered to match true).
    theta_init : dict or None
        Initial-guess parameter dictionary (will also be reordered).
    proj_data : list of torch.Tensor
        Observed projection data.
    t : torch.Tensor
        Time vector.
    sources, receivers
        CT geometry tensors.
    d : int
        Dimensionality.
    N : int
        Number of Gaussians.
    omega_min, omega_max : float
        Angular velocity search bounds.
    device : torch.device
        Computation device.
    experiment_dir : Path
        Directory for saving outputs.
    analysis_cfg : AnalysisConfig, optional
        Fine-grained control over which analysis steps to run.
    """
    import matplotlib.pyplot as plt
    from .visualization.publication import (
        animate_temporal_gmm_comparison,
        plot_individual_gaussian_reconstruction,
        plot_temporal_gmm_comparison,
        reorder_theta_to_match_true,
    )

    if analysis_cfg is None:
        analysis_cfg = AnalysisConfig()

    # --- Match estimated Gaussians to true (by velocity) for color-coding ---
    theta_est, matching_indices = reorder_theta_to_match_true(
        theta_true, theta_est, N,
    )
    print(f"Gaussian matching (est → true): {matching_indices}")
    if theta_init is not None:
        theta_init, _ = reorder_theta_to_match_true(
            theta_true, theta_init, N,
        )

    # --- Error analysis ---
    if not analysis_cfg.skip_errors:
        x0s = theta_true["x0s"]
        a0s = theta_true["a0s"]

        model = GMM_reco(
            d, N, sources, receivers, x0s, a0s,
            omega_min, omega_max, device=device,
            output_dir=experiment_dir,
        )

        errors_init = _compute_parameter_errors(theta_true, theta_init, N)
        errors_final = _compute_parameter_errors(theta_true, theta_est, N)

        proj_init = model.generate_projections(t, theta_init)
        proj_final = model.generate_projections(t, theta_est)
        proj_err_init = _compute_projection_error(proj_data, proj_init)
        proj_err_final = _compute_projection_error(proj_data, proj_final)

        _print_error_summary(
            errors_init, errors_final, proj_err_init, proj_err_final,
        )
        _plot_error_table(
            errors_init, errors_final,
            proj_err_init, proj_err_final,
            experiment_dir / "error_analysis.pdf",
        )

    # --- Plots ---
    if not analysis_cfg.skip_plots:
        print("\nGenerating plots...")

        time_indices = analysis_cfg.time_indices or [17, 20, 22]

        plot_individual_gaussian_reconstruction(
            theta_true, theta_est, N, d,
            gaussian_indices=range(N),
            filename=experiment_dir / "individual_gaussian_reconstruction.pdf",
        )

        if theta_init is not None:
            plot_temporal_gmm_comparison(
                sources, receivers, theta_true, theta_init, t, N, d,
                time_indices=time_indices,
                filename=experiment_dir / "initial_temporal_gmm_comparison.pdf",
                title="Initial Guesses",
            )

        plot_temporal_gmm_comparison(
            sources, receivers, theta_true, theta_est, t, N, d,
            time_indices=time_indices,
            filename=experiment_dir / "temporal_gmm_comparison.pdf",
        )

    # --- Animation ---
    if not analysis_cfg.skip_animations:
        print("Generating animation (this may take a moment)...")
        animate_temporal_gmm_comparison(
            sources, receivers, theta_true, theta_est, t, N, d,
            filename=experiment_dir / "temporal_gmm_comparison.mp4",
        )

    print(f"\nAll analysis outputs in: {experiment_dir}")


# ======================================================================
# Error metric helpers
# ======================================================================

def _compute_parameter_errors(theta_true, theta_est, N):
    """Compute relative L2 errors for each parameter type."""
    errors = {}

    def _rel_error(true_stack, est_stack):
        return (torch.norm(true_stack - est_stack) / torch.norm(true_stack)).item()

    for key, flatten in [("alphas", False), ("x0s", False), ("v0s", False),
                         ("U_skews", True), ("omegas", False)]:
        true_stack = torch.stack([
            theta_true[key][k].flatten() if flatten else theta_true[key][k]
            for k in range(N)
        ])
        est_stack = torch.stack([
            theta_est[key][k].flatten() if flatten else theta_est[key][k]
            for k in range(N)
        ])
        errors[key] = _rel_error(true_stack, est_stack)

    return errors


def _compute_projection_error(proj_true, proj_est):
    """Relative L2 error between two sets of projections."""
    true_flat = torch.cat([p.flatten() for p in proj_true])
    est_flat = torch.cat([p.flatten() for p in proj_est])
    return (torch.norm(true_flat - est_flat) / torch.norm(true_flat)).item()


def _print_error_summary(errors_init, errors_final, proj_err_init, proj_err_final):
    """Print a console summary of error reduction."""
    labels = {
        "alphas": "Amplitudes (α)",
        "x0s": "Positions  (x₀)",
        "v0s": "Velocities (v₀)",
        "U_skews": "Shape      (U)",
        "omegas": "Rotation   (ω)",
    }
    print("\nParameter errors (relative L2):")
    print(f"  {'':22s} {'Init':>12s}  {'Final':>12s}  {'Improvement':>12s}")
    for key in ["alphas", "x0s", "v0s", "U_skews", "omegas"]:
        init = errors_init[key]
        final = errors_final[key]
        imp = 100 * (1 - final / init) if init > 0 else 0
        print(f"  {labels[key]:22s} {init:12.4e}  {final:12.4e}  {imp:+11.1f}%")

    imp_proj = 100 * (1 - proj_err_final / proj_err_init) if proj_err_init > 0 else 0
    print(f"\n  {'Projections':22s} {proj_err_init:12.4e}  {proj_err_final:12.4e}  {imp_proj:+11.1f}%")


def _plot_error_table(errors_init, errors_final, proj_err_init, proj_err_final,
                      output_path):
    """Save an error-comparison table as a PDF figure."""
    import matplotlib.pyplot as plt

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
        imp = 100 * (1 - final / init) if init > 0 else 0
        red = init / final if final > 0 else np.inf
        rows.append([labels[key], f"{init:.4e}", f"{final:.4e}",
                     f"{imp:.1f}%", f"{red:.1f}×"])

    imp_proj = 100 * (1 - proj_err_final / proj_err_init) if proj_err_init > 0 else 0
    red_proj = proj_err_init / proj_err_final if proj_err_final > 0 else np.inf
    rows.append(["Projections", f"{proj_err_init:.4e}", f"{proj_err_final:.4e}",
                 f"{imp_proj:.1f}%", f"{red_proj:.1f}×"])

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

    ax.set_title("Error Analysis: Initialisation vs Optimisation",
                 fontweight="bold", fontsize=18, pad=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Error table saved: {output_path}")
