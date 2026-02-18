"""
Command-line interface for GMM-CT.

Provides two main commands:

``gmm-ct simulate``
    Generate synthetic projection data from a YAML config.

``gmm-ct reconstruct``
    Run reconstruction on observed (or simulated) projection data.

Both commands take a ``--config`` flag pointing to a YAML file that
describes the geometry, physics, and algorithm settings.  See
``configs/`` for annotated examples.
"""

import argparse
import sys
from pathlib import Path


def _add_common_args(parser: argparse.ArgumentParser):
    """Arguments shared across subcommands."""
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override computation device (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )


def main(argv=None):
    """Entry point for the ``gmm-ct`` CLI."""
    parser = argparse.ArgumentParser(
        prog="gmm-ct",
        description="GMM-CT: Gaussian Mixture Model CT Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  gmm-ct simulate   --config configs/simulate.yaml\n"
            "  gmm-ct reconstruct --config configs/reconstruct.yaml\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- simulate --------------------------------------------------------
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Generate synthetic projection data",
        description="Generate synthetic projection data from a YAML config.",
    )
    _add_common_args(sim_parser)
    sim_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )

    # --- reconstruct -----------------------------------------------------
    reco_parser = subparsers.add_parser(
        "reconstruct",
        help="Run reconstruction on projection data",
        description="Run the 4-stage GMM reconstruction pipeline.",
    )
    _add_common_args(reco_parser)
    reco_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override projection data path from config",
    )

    # --- parse -----------------------------------------------------------
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Lazy imports to keep CLI start-up fast
    if args.command == "simulate":
        return _run_simulate(args)
    elif args.command == "reconstruct":
        return _run_reconstruct(args)
    else:
        parser.print_help()
        return 1


# -----------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------

def _run_simulate(args) -> int:
    from .config.yaml_config import load_simulate_config
    from .simulation import run_simulation

    cfg = load_simulate_config(args.config)

    # Apply CLI overrides
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output.directory = Path(args.output_dir)
    if args.seed is not None:
        cfg.simulation.seed = args.seed

    print("=" * 50)
    print("GMM-CT  –  Simulate")
    print("=" * 50)
    print(f"Config : {args.config}")
    print(f"N      : {cfg.n_gaussians}")
    print(f"Seed   : {cfg.simulation.seed}")
    print(f"Output : {cfg.output.directory}")
    print()

    run_simulation(cfg)
    return 0


def _run_reconstruct(args) -> int:
    from .config.yaml_config import load_reconstruct_config
    from .reconstruct import run_reconstruction

    cfg = load_reconstruct_config(args.config)

    # Apply CLI overrides
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output.directory = Path(args.output_dir)
    if args.data:
        cfg.data_path = args.data

    print("=" * 50)
    print("GMM-CT  –  Reconstruct")
    print("=" * 50)
    print(f"Config : {args.config}")
    print(f"Data   : {cfg.data_path}")
    print(f"N      : {cfg.n_gaussians}")
    print(f"Output : {cfg.output.directory}")
    print()

    run_reconstruction(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
