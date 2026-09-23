import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# --- Experimental Grid Setup ---
N_PARTICLES: List[int] = [1, 2, 5, 8]
SEEDS: List[int] = list(range(10))

# Map pipeline CLI arguments to their isolated target subdirectories
PIPELINE_MODES: Dict[str, str] = {
    "full": "data/ablation_and_baseline/full-pipeline",
    "static-lstsq": "data/ablation_and_baseline/static-lstsq",
    "no-stage-1-5": "data/ablation_and_baseline/no-stage-1-5",
    "no-trajectory": "data/ablation_and_baseline/no-trajectory",
}

BASE_CONFIG = "configs/experiment.yaml"

def run_experiment_grid():
    """Iterate over N, seeds, and pipeline modes to run all experiments cleanly."""
    total_runs = len(N_PARTICLES) * len(SEEDS) * len(PIPELINE_MODES)
    current_run = 0
    
    logger.info(f"=== Starting Ablation & Baseline Suite ({total_runs} total executions) ===")
    
    for N in N_PARTICLES:
        for seed in SEEDS:
            for mode, output_dir in PIPELINE_MODES.items():
                current_run += 1
                
                # Ensure target subdirectory exists
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                logger.info(f"\n------------------------------------------------------------")
                logger.info(f"[Run {current_run}/{total_runs}] N={N} | Seed={seed} | Mode={mode}")
                logger.info(f"Target Output Directory: {output_dir}")
                logger.info(f"------------------------------------------------------------")
                
                # Construct CLI command overriding config parameters dynamically
                cmd = [
                    sys.executable, "-m", "src.cli",
                    "run",
                    "--config", BASE_CONFIG,
                    "--pipeline-mode", mode,
                    "--seed", str(seed),
                    "--sim-n-gaussians", str(N),
                    "--reco-n-gaussians", str(N),
                    "--output-dir", output_dir,
                ]
                
                # Execute run in a clean sub-process
                try:
                    result = subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.info(f"ERROR: Experiment failed for N={N}, seed={seed}, mode={mode}")
                    logger.info(f"Exit code: {e.returncode}")
                    continue
    logger.info("\n=== All Ablation & Baseline Experiments Completed Successfully ===")
    
if __name__ == "__main__":
    run_experiment_grid()