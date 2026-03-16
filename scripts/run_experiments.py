"""
Batch simulate-then-reconstruct over a range of random seeds.

Usage
-----
    python scripts/run_experiments.py                          # seeds 1-10 (default)
    python scripts/run_experiments.py --seeds 1 5 42          # explicit list
    python scripts/run_experiments.py --seeds-range 1 20      # 1..20 inclusive
    python scripts/run_experiments.py --sim-config  configs/simulate.yaml \\
                                      --reco-config configs/reconstruct.yaml \\
                                      --seeds-range 1 10

Each seed runs the full simulate → reconstruct pipeline.  Results are saved to
the directories configured in the YAML files (with the simulation output
directory automatically forwarded to the reconstruction step).

A summary CSV is written to ``data/results/batch_summary.csv`` listing the
seed, experiment directory, and outcome of each run.
"""

import argparse
import sys
import traceback
import csv
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
# (e.g. `python scripts/run_experiments.py` from any working directory).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch simulate + reconstruct over multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=Path("configs/simulate.yaml"),
        help="Path to simulation YAML config",
    )
    parser.add_argument(
        "--reco-config",
        type=Path,
        default=Path("configs/reconstruct.yaml"),
        help="Path to reconstruction YAML config",
    )

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="Explicit list of seeds to run",
    )
    seed_group.add_argument(
        "--seeds-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        help="Inclusive range of seeds: START .. STOP",
    )

    parser.add_argument(
        "--skip-reconstruct",
        action="store_true",
        help="Only run simulation (useful for generating data in bulk first)",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip post-reconstruction analysis in each reconstruction run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override compute device for both simulate and reconstruct",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/results/batch_summary.csv"),
        help="Where to write the per-seed summary CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve seed list
    if args.seeds_range is not None:
        start, stop = args.seeds_range
        seeds = list(range(start, stop + 1))
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(1, 11))  # default: 1..10

    print(f"Seeds to run: {seeds}")
    print(f"Sim config:   {args.sim_config}")
    print(f"Reco config:  {args.reco_config}")
    print()

    # Lazy imports — keep startup fast and errors local
    from gmm_ct.config.yaml_config import (
        load_simulate_config,
        load_reconstruct_config,
    )
    from gmm_ct.simulation import run_simulation
    from gmm_ct.reconstruct import run_reconstruction

    sim_cfg_base  = load_simulate_config(args.sim_config)
    reco_cfg_base = load_reconstruct_config(args.reco_config)

    results = []  # list of dicts for the summary CSV

    for seed in seeds:
        print("=" * 60)
        print(f"  SEED {seed}")
        print("=" * 60)

        row = {"seed": seed, "sim_dir": "", "status": ""}

        # ------------------------------------------------------------------
        # Simulate
        # ------------------------------------------------------------------
        try:
            sim_cfg = load_simulate_config(args.sim_config)  # fresh copy
            sim_cfg.simulation.seed = seed
            if args.device:
                sim_cfg.device = args.device

            sim_dir = run_simulation(sim_cfg)
            row["sim_dir"] = str(sim_dir)
            print(f"\n  ✓ Simulation complete → {sim_dir}\n")

        except Exception:
            print(f"\n  ✗ Simulation FAILED for seed {seed}:")
            traceback.print_exc()
            row["status"] = "sim_failed"
            results.append(row)
            continue

        if args.skip_reconstruct:
            row["status"] = "sim_only"
            results.append(row)
            continue

        # ------------------------------------------------------------------
        # Reconstruct — point at the projections just produced
        # ------------------------------------------------------------------
        projections_path = sim_dir / "projections.pt"
        if not projections_path.exists():
            print(f"  ✗ projections.pt not found in {sim_dir}; skipping.")
            row["status"] = "no_projections"
            results.append(row)
            continue

        try:
            reco_cfg = load_reconstruct_config(args.reco_config)  # fresh copy
            reco_cfg.data_path = str(projections_path)
            if args.device:
                reco_cfg.device = args.device
            if args.skip_analysis:
                reco_cfg.analysis.enabled = False

            run_reconstruction(reco_cfg)
            row["status"] = "ok"
            print(f"\n  ✓ Reconstruction complete for seed {seed}\n")

        except Exception:
            print(f"\n  ✗ Reconstruction FAILED for seed {seed}:")
            traceback.print_exc()
            row["status"] = "reco_failed"

        results.append(row)

    # ------------------------------------------------------------------
    # Write summary CSV
    # ------------------------------------------------------------------
    summary_path = args.summary_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "sim_dir", "status"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("Batch complete.")
    print(f"Summary written to {summary_path}")
    ok     = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if "failed" in r["status"])
    print(f"  {ok}/{len(seeds)} succeeded,  {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
