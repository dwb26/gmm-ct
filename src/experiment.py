
"""
Global experiment runner for GMM-CT.
"""

from pathlib import Path
import logging

from .simulate import run_simulation
from .reconstruct import run_reconstruction
from .analysis import run_analysis
from .config import ExperimentConfig

logger = logging.getLogger(__name__)

def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run full Simulate -> Reconstruct -> Analyse pipeline in a single pass."""
    logger.info("=== STEP 1: SIMULATION ===")
    exp_dir = run_simulation(cfg)
    cfg.data_path = exp_dir
    
    # Point reconstruction config to the newly generated projection data
    logger.info("=== STEP 2: RECONSTRUCTION ===")
    run_reconstruction(cfg)
    
    if cfg.analysis.enabled:
        logger.info("=== STEP 3: ANALYSIS ===")
        run_analysis(exp_dir, cfg)
        
    logger.info("=== EXPERIMENT COMPLETE: %s ===", exp_dir)
    return exp_dir