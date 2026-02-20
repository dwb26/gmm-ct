#!/usr/bin/env python3
"""
Analyse results from a GMM-CT reconstruction experiment.

Loads a results.pt file and runs error analysis + publication-quality plots.

This script is a thin wrapper around ``gmm_ct.reconstruct.analyse_results``.
In most workflows you don't need it — analysis runs automatically after
``gmm-ct reconstruct``.  Use this script when you want to re-analyse a
previous experiment without re-running reconstruction.

Usage:
    python scripts/analyse.py data/results/20260213_143000_seed40_N2/
    python scripts/analyse.py data/results/20260213_143000_seed40_N2/ --skip-animations
    python scripts/analyse.py data/results/20260213_143000_seed40_N2/ --time-indices 17 20 22
"""

import argparse
from pathlib import Path

import torch

from gmm_ct.config.yaml_config import AnalysisConfig
from gmm_ct.reconstruct import analyse_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse GMM-CT reconstruction results.",
    )
    parser.add_argument(
        "experiment_dir", type=str,
        help="Path to experiment directory containing results.pt",
    )
    parser.add_argument(
        "--time-indices", type=int, nargs="+", default=None,
        help="Time indices for temporal comparison plots (default: auto)",
    )
    parser.add_argument(
        "--skip-animations", action="store_true",
        help="Skip animation generation",
    )
    parser.add_argument(
        "--skip-errors", action="store_true",
        help="Skip error analysis",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip static comparison plots",
    )
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
    cfg = data["config"]
    device = torch.device(cfg["device"])

    print(f"Loaded experiment: N={cfg['N']}, seed={cfg['seed']}, device={device}")

    # --- Run analysis ---
    analysis_cfg = AnalysisConfig(
        enabled=True,
        skip_errors=args.skip_errors,
        skip_plots=args.skip_plots,
        skip_animations=args.skip_animations,
        time_indices=args.time_indices,
    )

    analyse_results(
        theta_true=data["theta_true"],
        theta_est=data["theta_est"],
        theta_init=data["theta_init"],
        proj_data=data["proj_data"],
        t=data["t"],
        sources=data["sources"],
        receivers=data["receivers"],
        d=cfg["d"],
        N=cfg["N"],
        omega_min=cfg["omega_min"],
        omega_max=cfg["omega_max"],
        device=device,
        experiment_dir=experiment_dir,
        analysis_cfg=analysis_cfg,
    )


if __name__ == "__main__":
    main()
