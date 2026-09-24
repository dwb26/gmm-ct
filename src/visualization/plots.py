"""
Publication-ready plotting utilities for GMM-CT benchmark results.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Apply publication-quality aesthetic defaults
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def generate_benchmark_plots(parquet_path: Path, output_dir: Path) -> None:
    """Reads benchmark Parquet data and exports publication-ready figures."""
    if not parquet_path.exists():
        logger.error(f"Cannot generate plots: {parquet_path} does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} experiment runs from {parquet_path.name} for plotting.")
    
    # Convert N to categorical string so Seaborn treats each value as a discrete line & legend item
    n_order = sorted(df["N"].unique())
    df["N_cat"] = df["N"].astype(str)
    hue_order = [str(n) for n in n_order]

    # ------------------------------------------------------------------
    # Figure 1: Relative L2 Density Error vs. Projections (by N Gaussians)
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4.5), dpi=300, sharey=True)

    # Filter out noise-free case (inf) if plotting log-SNR, or map SNR levels
    snr_subset = df[df["snr_db"] < 100] if (df["snr_db"] < 100).any() else df

    sns.lineplot(
        data=snr_subset,
        x="n_proj",
        y="rel_l2_density_error",
        hue="N_cat",
        hue_order=hue_order,
        palette="viridis",
        estimator=np.mean,
        errorbar=None,  # 25th-75th percentile shade (IQR)
        marker="o",
        ax=axs[0],
        legend=False,
    )
    
    sns.lineplot(
        data=snr_subset,
        x="snr_db",
        y="rel_l2_density_error",
        hue="N_cat",
        hue_order=hue_order,
        palette="viridis",
        estimator=np.mean,
        errorbar=None,  # 25th-75th percentile shade (IQR)
        marker="o",
        ax=axs[1],
    )

    axs[0].set_xscale("log", base=2)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Number of Projections")
    axs[0].set_ylabel("Relative $L_2$ Density Error (Mean)")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Signal-to-Noise Ratio (dB)")
    axs[1].set_ylabel("")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    axs[1].legend(title="N Gaussians", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.suptitle("Spatio-Temporal Density Reconstruction Errors")

    plt.tight_layout()
    fig1_path = output_dir / "rel_l2_error.pdf"
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fig1_path}")

    # ------------------------------------------------------------------
    # Figure 2: Kinematic & Morphology Parameter Errors (Grid Plot)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300, sharex=True)

    metrics = [
        ("v0_rmse", r"Velocity Error $\|\mathbf{v}_0^* - \widehat{\mathbf{v}}_0\|_2$"),
        ("omega_rmse", r"Rotational Error $|\Theta^* - \widehat{\Theta}|$"),
        ("alpha_rmse", r"Amplitude Error $|\alpha^* - \widehat{\alpha}|$"),
        ("U_rmse", r"Skew Covariance Error $\|\mathbf{U}^* - \widehat{\mathbf{U}}\|_F$"),
    ]

    for (col_name, title_str), ax in zip(metrics, axes.flat):
        sns.boxplot(
            data=df,
            x="N",
            y=col_name,
            hue="n_proj",
            # palette="viridis",
            ax=ax,
            showfliers=False,  # Robust against extreme unobserved outliers
        )
        ax.set_yscale("log")
        ax.set_title(title_str)
        ax.set_xlabel("Number of Gaussians ($N$)")
        ax.set_ylabel("RMSE")
        ax.grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    fig2_path = output_dir / "parameter_errors_breakdown.pdf"
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fig2_path}")