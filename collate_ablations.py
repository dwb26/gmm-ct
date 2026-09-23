import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from src.analysis import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# --- Target Directory & Variant Mapping ---
BASE_DIR = Path("data/ablation_and_baseline")

VARIANTS: Dict[str, str] = {
    "Direct Huber LS": "static-lstsq",
    "No Stage 1.5": "no-stage-1-5",
    "No Trajectory Matching": "no-trajectory",
    "Full GMM-CT (Ours)": "full-pipeline",
}

N_PARTICLES: List[int] = [1, 2, 5, 8]
SEEDS: List[int] = list(range(10))

def collate_results():
    summary_rows = []
    
    for N in N_PARTICLES:
        row = {"N Particles": N}
        
        for label, dir_name in VARIANTS.items():
            variant_dir = BASE_DIR / dir_name
            errors = []
            
            for seed in SEEDS:
                exp_folder = f"snr80_N{N}_nproj128_seed{seed}"
                exp_dir = variant_dir / exp_folder
                
                if not exp_dir.exists():
                    continue
                
                try:
                    record = run_analysis(exp_dir=exp_dir)
                    err = record['rel_l2_density_error']
                    
                    if np.isfinite(err):
                        errors.append(err)
                except Exception:
                    continue
                    
            n_completed = len(errors)
            
            if n_completed > 0:
                mean_err = np.mean(errors)
                # Compute std with ddof=1 when N > 1 to avoid std=0.0 on single samples
                std_err = np.std(errors, ddof=1) if n_completed > 1 else 0.0
                
                # Report fraction of successful completions relative to planned seeds
                if n_completed < len(SEEDS):
                    row[label] = f"{mean_err:.4f} ± {std_err:.4f} ({n_completed}/{len(SEEDS)} runs)"
                else:
                    row[label] = f"{mean_err:.4f} ± {std_err:.4f}"
            else:
                row[label] = "Failed (Divergent)"
                
        summary_rows.append(row)
        
    df = pd.DataFrame(summary_rows)
    df.set_index("N Particles", inplace=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("                     GMM-CT ABLATION & BASELINE SUMMARY                     ")
    logger.info("=" * 80)
    logger.info(df.to_markdown())
    logger.info("=" * 80 + "\n")

    # Export to CSV for direct import into Overleaf/LaTeX or Pandas
    output_csv = BASE_DIR / "ablation_summary.csv"
    df.to_csv(output_csv)
    logger.info(f"Summary saved to: {output_csv}")

if __name__ == "__main__":
    collate_results()