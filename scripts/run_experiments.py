import logging
from pathlib import Path
import copy
import shutil
import pandas as pd
import numpy as np

from src.config import ExperimentConfig, load_experiment_config
from src.simulate import run_simulation
from src.reconstruct import run_reconstruction
from src.analysis import run_analysis
from src.visualization.plots import generate_benchmark_plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Experiment Matrix ---
# SNR_LEVELS = [10.0, 15.0, 20.0, 40.0, 80.0]
# N_GAUSSIANS = [1, 2, 3, 5, 8, 12, 20]
# N_PROJECTIONS = [8, 16, 32, 64, 128, 256]
# SEEDS = range(1, 26)
SNR_LEVELS = [10.0, 15.0, 20.0, 40.0]
N_GAUSSIANS = [1, 2, 3, 5, 8, 10, 15]
N_PROJECTIONS = [8, 16, 32, 64, 128]
SEEDS = range(1, 21)

def generate_configs(base_config_path: Path) -> list[ExperimentConfig]:
    """Generates a list of ExperimentConfig objects by sweeping over parameters."""
    base_cfg = load_experiment_config(base_config_path)
    sweep_configs = []
    
    for snr in SNR_LEVELS:
        for N in N_GAUSSIANS:
            for n_proj in N_PROJECTIONS:
                for seed in SEEDS:
                    # Deep copy base config and override target values
                    cfg = copy.deepcopy(base_cfg)
                    cfg.seed = seed
                    cfg.sim_n_gaussians = N
                    cfg.reco_n_gaussians = N
                    cfg.physics.n_projections = n_proj
                    cfg.snr_db = snr
                    
                    out_dir = Path(cfg.output.directory)
                    folder_name = f"snr{snr}_N{N}_nproj{n_proj}_seed{seed}"
                    cfg.exp_dir = out_dir / folder_name
                    
                    sweep_configs.append(cfg)                    
                    
    return sweep_configs

def filter_completed_configs(
    configs: list[ExperimentConfig], results_path: Path
) -> list[ExperimentConfig]:
    """Filters out configs that are already recorded in the output Parquet file."""
    if not results_path.exists():
        return configs

    df_existing = pd.read_parquet(results_path)

    completed_keys = set(
        zip(
            df_existing["snr_db"],
            df_existing["N"],
            df_existing["n_proj"],
            df_existing["seed"],
        )
    )

    remaining_configs = [
        cfg
        for cfg in configs
        if (
            float(cfg.snr_db),
            int(cfg.sim_n_gaussians),
            int(cfg.physics.n_projections),
            int(cfg.seed),
        )
        not in completed_keys
    ]

    logger.info(
        f"Found {len(completed_keys)} completed runs in {results_path.name}. "
        f"Resuming with remaining {len(remaining_configs)} / {len(configs)} configs."
    )
    return remaining_configs

def _append_to_parquet(records: list[dict], path: Path):
    """Appends evaluated metric records to the master Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)
    if path.exists():
        existing_df = pd.read_parquet(path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(path, index=False)
    else:
        new_df.to_parquet(path, index=False)
    logger.info(f"Checkpoint saved: flushed {len(new_df)} records to {path}")

def run_experiment_pipeline(
    configs: list[ExperimentConfig],
    results_path: Path,
    keep_tensors: bool = False,
    batch_size: int = 50,
):
    """Executes sim, reco, metric logging, and folder cleanup in a unified stream."""
    records = []
    total = len(configs)

    for idx, cfg in enumerate(configs, start=1):
        logger.info(
            f"[{idx}/{total}] Processing SNR={cfg.snr_db}, N={cfg.sim_n_gaussians}, "
            f"n_proj={cfg.physics.n_projections}, seed={cfg.seed}"
        )

        try:
            exp_dir = run_simulation(cfg)
            run_reconstruction(cfg)
            metrics_dict = run_analysis(exp_dir)
            metrics_dict['status'] = 'success'

            # Clean up intermediate tensor directory to prevent disk bloat
            if not keep_tensors:
                shutil.rmtree(exp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Failed pipeline run for {cfg.exp_dir.name}: {e}")
            metrics_dict = {
                'exp_dir': str(cfg.exp_dir),
                'N': cfg.sim_n_gaussians,
                'n_proj': cfg.physics.n_projections,
                'snr_db': float(cfg.snr_db),
                'seed': cfg.seed,
                'status': 'failed',
                # Fill numerical errors with NaN so they don't corrupt metric calculations
                'v0_rmse': np.nan,
                'omega_rmse': np.nan,
                'alpha_rmse': np.nan,
                'U_rmse': np.nan,
                'l2_density_error': np.nan,
                'rel_l2_density_error': np.nan,
                'log_l2_density_error': np.nan,
                'log_rel_l2_density_error': np.nan,
            }
            
        records.append(metrics_dict)

        # Batch checkpoint to Parquet
        if len(records) >= batch_size:
            _append_to_parquet(records, results_path)
            records = []

    # Flush any remaining records at the end
    if records:
        _append_to_parquet(records, results_path)


# ======================================================================
# Main Entry Point
# ======================================================================

def main():
    base_config_path = Path("configs/experiment.yaml")
    results_path = Path("data/benchmark_results.parquet")
    figures_dir = Path("data/figures")
    
    # 1. Generate full parameter sweep matrix
    # all_configs = generate_configs(base_config_path)
    
    # # 2. Filter out already completed runs for automatic resumption
    # configs_to_run = filter_completed_configs(all_configs, results_path)
    
    # if not configs_to_run:
    #     logger.info("All experiments in the sweep are completed!")
    #     return

    # # 3. Stream pipeline
    # run_experiment_pipeline(
    #     configs=configs_to_run,
    #     results_path=results_path,
    #     keep_tensors=False,
    #     batch_size=50,
    # )
    
    # 4. Read Parquet results & generate plots
    generate_benchmark_plots(
        parquet_path=results_path,
        output_dir=figures_dir,
    )
    

if __name__ == "__main__":
    main()