#!/usr/bin/env python3
"""
Analyse results from a GMM-CT reconstruction experiment.

Loads a results.pt file produced by reconstruct.py, computes error metrics,
and generates publication-quality plots and animations.

Usage:
    python scripts/analyse.py data/results/20260213_143000_seed40_N2/
    python scripts/analyse.py data/results/20260213_143000_seed40_N2/ --skip-animations
    python scripts/analyse.py data/results/20260213_143000_seed40_N2/ --time-indices 17 20 22
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from gmm_ct import GMM_reco
from gmm_ct.visualization.publication import (
    animate_temporal_gmm_comparison,
    plot_individual_gaussian_reconstruction,
    plot_temporal_gmm_comparison,
)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def compute_parameter_errors(theta_true, theta_est, N, detailed=False):
    """
    Compute relative L2 errors for each parameter type across all Gaussians.

    Parameters
    ----------
    theta_true : dict
        Ground-truth parameter dictionary.
    theta_est : dict
        Estimated parameter dictionary.
    N : int
        Number of Gaussians.
    detailed : bool
        If True, also return per-Gaussian errors.

    Returns
    -------
    errors : dict
        Relative L2 error per parameter type.
    per_gaussian : dict or None
        Per-Gaussian errors (only when *detailed=True*).
    """
    errors = {}
    per_gaussian = {} if detailed else None

    def _rel_error(true_stack, est_stack):
        return (torch.norm(true_stack - est_stack) / torch.norm(true_stack)).item()

    def _per_gauss(key, flatten=False):
        out = []
        for k in range(N):
            t_k = theta_true[key][k].flatten() if flatten else theta_true[key][k]
            e_k = theta_est[key][k].flatten() if flatten else theta_est[key][k]
            out.append((torch.norm(t_k - e_k) / torch.norm(t_k)).item())
        return out

    for key, flatten in [("alphas", False), ("x0s", False), ("v0s", False),
                         ("U_skews", True), ("omegas", False)]:
        true_stack = torch.stack([theta_true[key][k].flatten() if flatten else theta_true[key][k] for k in range(N)])
        est_stack = torch.stack([theta_est[key][k].flatten() if flatten else theta_est[key][k] for k in range(N)])
        errors[key] = _rel_error(true_stack, est_stack)
        if detailed:
            per_gaussian[key] = _per_gauss(key, flatten)

    if detailed:
        return errors, per_gaussian
    return errors


def compute_projection_error(proj_true, proj_est):
    """Relative L2 error between two sets of projections."""
    true_flat = torch.cat([p.flatten() for p in proj_true])
    est_flat = torch.cat([p.flatten() for p in proj_est])
    return (torch.norm(true_flat - est_flat) / torch.norm(true_flat)).item()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_error_summary(errors_init, errors_final, proj_err_init, proj_err_final):
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


def plot_error_table(errors_init, errors_final, proj_err_init, proj_err_final, output_path):
    """Save an error-comparison table as a PDF figure."""
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
        rows.append([labels[key], f"{init:.4e}", f"{final:.4e}", f"{imp:.1f}%", f"{red:.1f}×"])

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

    ax.set_title("Error Analysis: Initialisation vs Optimisation", fontweight="bold", fontsize=18, pad=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Error table saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse GMM-CT reconstruction results.")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory containing results.pt")
    parser.add_argument("--time-indices", type=int, nargs="+", default=[17, 20, 22],
                        help="Time indices for temporal comparison plots (default: 17 20 22)")
    parser.add_argument("--skip-animations", action="store_true", help="Skip animation generation")
    parser.add_argument("--skip-errors", action="store_true", help="Skip error analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.experiment_dir)

    # Accept either the directory or the results.pt file directly
    if input_path.is_file() and input_path.name == "results.pt":
        results_path = input_path
        experiment_dir = input_path.parent
    else:
        experiment_dir = input_path
        results_path = experiment_dir / "results.pt"

    if not results_path.exists():
        raise FileNotFoundError(f"No results.pt found in {experiment_dir}")

    # --- Load ---
    data = torch.load(results_path, weights_only=False)
    theta_true = data["theta_true"]
    theta_est = data["theta_est"]
    theta_init = data["theta_init"]
    proj_data = data["proj_data"]
    t = data["t"]
    sources = data["sources"]
    rcvrs = data["receivers"]
    cfg = data["config"]
    d, N = cfg["d"], cfg["N"]
    omega_min, omega_max = cfg["omega_min"], cfg["omega_max"]
    device = torch.device(cfg["device"])

    print(f"Loaded experiment: N={N}, seed={cfg['seed']}, device={device}")

    # --- Error analysis ---
    if not args.skip_errors:
        x0s = theta_true["x0s"]
        a0s = theta_true["a0s"]

        GMM = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=experiment_dir)

        errors_init = compute_parameter_errors(theta_true, theta_init, N)
        errors_final = compute_parameter_errors(theta_true, theta_est, N)

        proj_init = GMM.generate_projections(t, theta_init)
        proj_final = GMM.generate_projections(t, theta_est)
        proj_err_init = compute_projection_error(proj_data, proj_init)
        proj_err_final = compute_projection_error(proj_data, proj_final)

        print_error_summary(errors_init, errors_final, proj_err_init, proj_err_final)
        plot_error_table(errors_init, errors_final, proj_err_init, proj_err_final,
                         experiment_dir / "error_analysis.pdf")

    # --- Plots ---
    print("\nGenerating plots...")

    plot_individual_gaussian_reconstruction(
        theta_true, theta_est, N, d,
        gaussian_indices=range(N),
        filename=experiment_dir / "individual_gaussian_reconstruction.pdf",
    )

    plot_temporal_gmm_comparison(
        sources, rcvrs, theta_true, theta_init, t, N, d,
        time_indices=args.time_indices,
        filename=experiment_dir / "initial_temporal_gmm_comparison.pdf",
        title="Initial Guesses",
    )

    plot_temporal_gmm_comparison(
        sources, rcvrs, theta_true, theta_est, t, N, d,
        time_indices=args.time_indices,
        filename=experiment_dir / "temporal_gmm_comparison.pdf",
    )

    # --- Animation ---
    if not args.skip_animations:
        print("Generating animation (this may take a moment)...")
        animate_temporal_gmm_comparison(
            sources, rcvrs, theta_true, theta_est, t, N, d,
            filename=experiment_dir / "temporal_gmm_comparison.mp4",
        )

    print(f"\nAll outputs in: {experiment_dir}")


if __name__ == "__main__":
    main()
