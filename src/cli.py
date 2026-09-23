"""Command-line interface for GMM-CT.

Offers the full end-to-end simulate->reconstruct->anaylsis pipeline via 

    python -m src.cli run --config configs/experiment.yaml
    
or allows each of the three steps to be called individually; i.e.

    python -m src.cli simulate --config configs/experiment.yaml
    python -m src.cli reconstruct --config configs/experiment.yaml --exp-dir data/seed1_N8_nproj150
    python -m src.cli analysis --config configs/experiment.yaml --exp-dir data/seed1_N8_nproj150
"""
import os
import torch

import argparse
import logging
import sys
from pathlib import Path

from .config import load_experiment_config
from .simulate import run_simulation
from .reconstruct import run_reconstruction
from .analysis import run_analysis
from .model import GMM_reco

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="full",
        choices=["full", "static-lstsq", "naive-fit", "no-stage-1-5", "no-trajectory"],
        help="Pipeline execution mode for ablation studies (naive-fit is basic lstsq on traj and no 1.5)",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Override random seed"
    )
    parser.add_argument(
        "--sim-n-gaussians", 
        type=int, 
        help="Override N simulation particles"
    )
    parser.add_argument(
        "--reco-n-gaussians", 
        type=int, 
        help="Override N reconstruction particles"
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
        _run_experiment_cmd(args)
        return 0
    if args.command == "simulate":
        _run_simulate_cmd(args)
        return 0
    elif args.command == "reconstruct":
        _run_reconstruct_cmd(args)
        return 0
    elif args.command == "analysis":
        _run_analysis_cmd(args)
        return 0
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
        run_analysis(exp_dir)
        
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

def _run_reconstruct_cmd(args) -> GMM_reco:
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
    exp_dir = Path(args.exp_dir)    
    record = run_analysis(exp_dir)
    
    return exp_dir

def _apply_cli_overrides(cfg, args) -> None:
    if args.device:
        cfg.device = args.device
    if args.output_dir:
        cfg.output.directory = Path(args.output_dir)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.sim_n_gaussians is not None:
        cfg.sim_n_gaussians = args.sim_n_gaussians
    if args.reco_n_gaussians is not None:
        cfg.reco_n_gaussians  = args.reco_n_gaussians
    if getattr(args, "pipeline_mode", None):
        cfg.reconstruction.pipeline_mode = args.pipeline_mode

    # Temporary guardrail: Limit threads if running the naive baseline parallel to the main sweep
    if getattr(args, "pipeline_mode", None) in ["full", "naive-fit", "no-stage-1-5", "no-trajectory"]:
        logger.info("Configuring single-shot naive baseline run (thread limit = 2)...")
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        torch.set_num_threads(2)

if __name__ == "__main__":
    sys.exit(main())