
"""
Global experiment runner for GMM-CT.
"""

from pathlib import Path
import logging

from .simulation import run_simulation
from .reconstruct import run_reconstruction
from .analysis import run_analysis
from .config import ExperimentConfig

logger = logging.getLogger(__name__)

def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run full Simulate -> Reconstruct -> Analyse pipeline in a single pass."""
    logger.info("=== STEP 1: SIMULATION ===")
    sim_dir = run_simulation(cfg.simulate)
    
    # Point reconstruction config to the newly generated projection data
    cfg.reconstruct.data_path = sim_dir / "projections.pt"
    
    logger.info("=== STEP 2: RECONSTRUCTION ===")
    reco_dir = run_reconstruction(cfg.reconstruct)
    
    if cfg.analysis.enabled:
        logger.info("=== STEP 3: ANALYSIS ===")
        run_analysis(reco_dir, cfg.analysis)
        
    logger.info("=== EXPERIMENT COMPLETE: %s ===", reco_dir)
    return reco_dir