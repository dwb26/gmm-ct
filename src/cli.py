"""Command-line interface for GMM-CT."""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_common_args(parser: argparse.ArgumentParser):
    """Arguments shared across subcommands."""
    parser.add_argument(
        "--config",
        type=str,
        required=False,
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
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- run (Simulate -> Reconstruct -> Analyze)
    run_parser = subparsers.add_parser(
        "run",
        help="Run full end-to-end experiment (Simulate -> Reconstruct -> Analyze)",
    )
    _add_common_args(run_parser)
    
    # --- simulate --------------------------------------------------------
    sim_parser = subparsers.add_parser(
        "simulate",
        help="Generate synthetic projection data only",
    )
    _add_common_args(sim_parser)
    sim_parser.add_argument("--seed", type=int, default=None, help="Override seed",
    )
    # --- reconstruct -----------------------------------------------------
    reco_parser = subparsers.add_parser(
        "reconstruct",
        help="Run reconstruction on projection data",
    )
    _add_common_args(reco_parser)
    reco_parser.add_argument("--data", type=str, default=None, help="Override projection data path")
    reco_parser.add_argument("--skip-analysis", action="store_true", help="Skip post-reconstruction analysis")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return _run_experiment_cmd(args)
    if args.command == "simulate":
        return _run_simulate_cmd(args)
    elif args.command == "reconstruct":
        return _run_reconstruct_cmd(args)
    else:
        parser.print_help()
        return 1

def _run_experiment_cmd(args) -> int:
    from .config import load_experiment_config
    from .experiment import run_experiment
    
    cfg = load_experiment_config(args.config)
    _apply_cli_overrides(cfg, args)
    
    run_experiment(cfg)
    return 0

def _run_simulate_cmd(args) -> int:
    from .config import load_simulate_config
    from .simulation import run_simulation
    
    cfg = load_simulate_config(arg.config)
    _apply_cli_overrides(cfg, args)
    if getattr(args, "seed", None) is not None:
        cfg.simulation.seed = args.seed

    out_dir = run_simulation(cfg)
    logger.info(f"Simulation complete. Data saved to:\n  {out_dir}")
    # print(f"\nTo reconstruct, run:\n  gmm-ct reconstruct --config configs/reconstruct.yaml --data {out_dir}/projections.pt")
    return 0

def _run_reconstruct_cmd(args) -> int:
    from .config import load_reconstruct_config
    from .reconstruct import run_reconstruction
    
    cfg = load_reconstruct_config(args.config)
    _apply_cli_overrides(cfg, args)
    
    if getattr(args, "data", None):
        cfg.data_path = args.data
    if getattr(args, "skip_analysis", False):
        cfg.analysis_enabled = False
        
    run_reconstruction(cfg.reconstruct)
    return 0

def _apply_cli_overrides(cfg, args) -> None:
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output_directory = Path(args.output_dir)

if __name__ == "__main__":
    sys.exit(main())


    # Apply CLI overrides
    # if args.device:
    #     cfg.device = args.device
    # if args.output_dir:
    #     cfg.output.directory = Path(args.output_dir)
    # if args.data:
    #     cfg.data_path = args.data
    #     logger.info("Overriding data path: %s", cfg.data_path)
    # if args.skip_analysis:
    #     cfg.analysis.enabled = False
    # if args.skip_animations:
    #     cfg.analysis.skip_animations = True

    # logger.info("GMM-CT Reconstruct | data=%s, N=%d, output=%s",
    #             cfg.data_path, cfg.n_gaussians, cfg.output.directory)

    # run_reconstruction(cfg)
    # return 0