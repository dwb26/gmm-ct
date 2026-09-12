"""Command-line interface for GMM-CT."""

import argparse
import logging
import sys
from pathlib import Path

from .config import load_experiment_config
from .simulate import run_simulation
from .reconstruct import run_reconstruction
from .analysis import run_analysis
from .tomography_solver.solver_pipeline import GMMTomographySolver

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
    reco_parser.add_argument("--exp-dir", type=str, default=None, help="Override projection data path")


    # --- analysis -----------------------------------------------------
    ana_parser = subparsers.add_parser(
        "analysis",
        help="Run reconstruction on projection data",
    )
    _add_common_args(ana_parser)
    ana_parser.add_argument("--exp-dir", type=Path, required=True)


    # Runner calls
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
    elif args.command == "analysis":
        return _run_analysis_cmd(args)
    else:
        parser.print_help()
        return 1


# --- Task Runners -----------------------------------------------------
def _run_experiment_cmd(args) -> Path:
    """Run the full simulate -> reconstruct -> analysis pipeline.
    Run as 
        python -m src.cli run --config configs/experiment.yaml
    """
    
    cfg = load_experiment_config(args.config)
    _apply_cli_overrides(cfg, args)
    
    logger.info("=== STEP 1: SIMULATION ===")
    exp_dir = run_simulation(cfg)
    cfg.exp_dir = exp_dir
    
    # Point reconstruction config to the newly generated projection data
    logger.info("=== STEP 2: RECONSTRUCTION ===")
    run_reconstruction(cfg)
    
    if cfg.analysis.enabled:
        logger.info("=== STEP 3: ANALYSIS ===")
        run_analysis(exp_dir, cfg)
        
    logger.info("=== EXPERIMENT COMPLETE: %s ===", exp_dir)
    
    return exp_dir


def _run_simulate_cmd(args) -> Path:
    """Run the simulation component.
    Run as  
        python -m src.cli simulate --config configs/experiment.yaml --seed 1
    """
    
    cfg = load_experiment_config(args.config)
    _apply_cli_overrides(cfg, args)
    
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed

    exp_dir = run_simulation(cfg)
    logger.info(f"Simulation complete. Data saved to:\n  {exp_dir}")
    
    return exp_dir


def _run_reconstruct_cmd(args) -> GMMTomographySolver:
    """Run the reconstruction component.
    Run as
         python -m src.cli reconstruct --config configs/experiment.yaml --exp-dir data/seed1_N8_nproj150
    """
    
    cfg = load_experiment_config(args.config)
    _apply_cli_overrides(cfg, args)
    
    if getattr(args, "exp_dir", None):
        cfg.exp_dir = args.exp_dir
    
    return run_reconstruction(cfg)


def _run_analysis_cmd(args) -> Path:
    """Run the analysis component.
    Run as
        python -m src.cli analysis --config configs/experiment.yaml --exp-dir data/seed1_N8_nproj150
    """
    
    cfg = load_experiment_config(args.config)
    exp_dir = Path(args.exp_dir)
    
    run_analysis(exp_dir, cfg)
    
    return exp_dir


def _apply_cli_overrides(cfg, args) -> None:
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output_directory = Path(args.output_dir)

if __name__ == "__main__":
    sys.exit(main())