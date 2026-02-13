#!/usr/bin/env python3
"""
Run a GMM-CT reconstruction experiment.

Generates synthetic ground-truth parameters, produces projection data,
fits a GMM reconstruction model, and saves all results to an experiment
directory under data/results/.

Usage:
    python scripts/reconstruct.py                    # defaults: N=2, seed=40
    python scripts/reconstruct.py --N 5 --seed 99
    python scripts/reconstruct.py --N 3 --seed 10 --duration 3.0
"""

import argparse
from datetime import datetime
from pathlib import Path
from time import time

import torch

from gmm_ct import (
    GMM_reco,
    construct_receivers,
    export_parameters,
    generate_true_param,
    set_random_seeds,
)
from gmm_ct.config.defaults import GRAVITATIONAL_ACCELERATION
from gmm_ct.visualization.publication import reorder_theta_to_match_true


def parse_args():
    parser = argparse.ArgumentParser(description="Run a GMM-CT reconstruction experiment.")
    parser.add_argument("--N", type=int, default=2, help="Number of Gaussians (default: 2)")
    parser.add_argument("--seed", type=int, default=40, help="Random seed (default: 40)")
    parser.add_argument("--n-projections", type=int, default=65, help="Number of projection time steps (default: 65)")
    parser.add_argument("--duration", type=float, default=2.0, help="Time window in seconds (default: 2.0)")
    parser.add_argument("--omega-min", type=float, default=-24.0, help="Min angular velocity (default: -24.0)")
    parser.add_argument("--omega-range", type=float, default=8.0, help="Omega range above min (default: 8.0)")
    parser.add_argument("--n-receivers", type=int, default=128, help="Number of receivers (default: 128)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda' (default: auto)")
    return parser.parse_args()


def run_reconstruction(args):
    """Run a single reconstruction experiment and save all outputs."""

    start_time = time()

    # --- Setup ---
    set_random_seeds(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")

    # --- Hyperparameters ---
    d = 2
    N = args.N
    omega_min = args.omega_min
    omega_max = omega_min + args.omega_range
    t = torch.linspace(0.0, args.duration, args.n_projections, dtype=torch.float64, device=device)

    # --- Output directory ---
    if args.output_dir:
        experiment_dir = Path(args.output_dir)
    else:
        project_root = Path(__file__).resolve().parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = project_root / "data" / "results" / f"{timestamp}_seed{args.seed}_N{N}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {experiment_dir}")

    # --- Geometry ---
    sources = [torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)]
    x1 = sources[0][0].item() + 5.0
    x2_min = sources[0][1].item() - 2.0
    x2_max = sources[0][1].item() + 2.0
    rcvrs = construct_receivers(device, (args.n_receivers, x1, x2_min, x2_max))

    # --- Ground truth ---
    i_loc = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
    v_loc = torch.tensor([0.75, 0.5], dtype=torch.float64, device=device)
    a_loc = torch.tensor([0.0, -GRAVITATIONAL_ACCELERATION], dtype=torch.float64, device=device)

    theta_true = generate_true_param(d, N, i_loc, v_loc, a_loc, omega_min, omega_max, device=device)
    x0s = theta_true["x0s"]
    a0s = theta_true["a0s"]

    GMM_true = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=experiment_dir)
    proj_data = GMM_true.generate_projections(t, theta_true)

    # --- Reconstruction ---
    GMM = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=experiment_dir)
    soln_dict = GMM.fit(proj_data, t)
    theta_dict_init = GMM.theta_dict_init

    # Reorder estimated parameters to match ground truth
    soln_dict, matching_indices = reorder_theta_to_match_true(theta_true, soln_dict, N)
    print(f"\nMatched Gaussians: {matching_indices}")

    # --- Print omega comparison ---
    print("\nAngular velocity comparison:")
    for k in range(N):
        true_omega = theta_true["omegas"][k].item()
        est_omega = soln_dict["omegas"][k].item()
        print(f"  Gaussian {k}: true={true_omega:.4f}  est={est_omega:.4f}  |err|={abs(true_omega - est_omega):.4e}")

    # --- Save results ---
    export_parameters(theta_true, experiment_dir / "true_parameters.md", title="Ground Truth Parameters")
    export_parameters(
        soln_dict,
        experiment_dir / "estimated_parameters.md",
        title="Estimated Parameters",
        theta_true=theta_true,
        theta_init=theta_dict_init,
    )

    # Save tensors for downstream analysis
    torch.save(
        {
            "theta_true": theta_true,
            "theta_est": soln_dict,
            "theta_init": theta_dict_init,
            "proj_data": proj_data,
            "t": t,
            "sources": sources,
            "receivers": rcvrs,
            "matching_indices": matching_indices,
            "config": {
                "d": d,
                "N": N,
                "seed": args.seed,
                "omega_min": omega_min,
                "omega_max": omega_max,
                "n_receivers": args.n_receivers,
                "device": str(device),
            },
        },
        experiment_dir / "results.pt",
    )
    print(f"\nResults saved to {experiment_dir / 'results.pt'}")

    elapsed = time() - start_time
    print(f"Completed in {elapsed:.1f}s")

    return experiment_dir


if __name__ == "__main__":
    run_reconstruction(parse_args())
