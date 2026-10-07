"""
Publication-ready plotting utilities for GMM-CT benchmark results.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis import cap_density_error

logger = logging.getLogger(__name__)

REF_SNR_DB = 20.0   # SNR held fixed in the projection-sensitivity panel
REF_N_PROJ = 128    # projection count held fixed in the noise-sensitivity panel

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

    # Enforce the cap here too, so rows saved before it was introduced are treated identically
    capped = df["rel_l2_density_error"].map(lambda e: cap_density_error(e))
    df["rel_l2_density_error"] = capped.map(lambda c: c[0])
    df["diverged"] = capped.map(lambda c: c[1])
    
    # Convert N to categorical string so Seaborn treats each value as a discrete line & legend item
    n_order = sorted(df["N"].unique())
    df["N_cat"] = df["N"].astype(str)
    hue_order = [str(n) for n in n_order]
    # Plasma trimmed at both ends (no near-black or bright yellow), plus a marker per N
    palette = dict(zip(hue_order, plt.cm.plasma(np.linspace(0.12, 0.78, len(hue_order)))))
    markers = dict(zip(hue_order, ["o", "s", "^", "D", "v", "P", "X", "*"]))

    # ------------------------------------------------------------------
    # Figure 1: Relative L2 Density Error vs. Projections (by N Gaussians)
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4.5), dpi=300, sharey=True)

    # Each panel is a slice of the sweep with the other factor held fixed
    proj_slice = df[df["snr_db"] == REF_SNR_DB]       # left: vary n_proj at fixed SNR
    snr_slice = df[df["n_proj"] == REF_N_PROJ]        # right: vary SNR at fixed n_proj

    sns.lineplot(
        data=proj_slice,
        x="n_proj",
        y="rel_l2_density_error",
        hue="N_cat",
        hue_order=hue_order,
        palette=palette,
        style="N_cat",
        style_order=hue_order,
        markers=markers,
        dashes=False,
        estimator=np.median,
        errorbar=None,
        ax=axs[0],
        legend=False,
    )
    
    sns.lineplot(
        data=snr_slice,
        x="snr_db",
        y="rel_l2_density_error",
        hue="N_cat",
        hue_order=hue_order,
        palette=palette,
        style="N_cat",
        style_order=hue_order,
        markers=markers,
        dashes=False,
        estimator=np.median,
        errorbar=None,
        ax=axs[1],
    )
    fs = 14
    ls = 14
    axs[0].set_xscale("log", base=2)
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"Number of Projections ($N_t$)", fontweight="bold", fontsize=fs)
    axs[0].set_ylabel("Relative $L_2$ Density Error (Median)", fontweight="bold", fontsize=fs)
    axs[0].set_title(rf"Projection Sensitivity ($\mathrm{{SNR}}={REF_SNR_DB:g}$ dB)", fontweight="bold", fontsize=fs+1)
    axs[0].tick_params(labelsize=ls)
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Signal-to-Noise Ratio (dB)", fontweight="bold", fontsize=fs)
    axs[1].set_ylabel("")
    axs[1].grid(True, which="both", ls="--", alpha=0.5)
    # Errors are capped at 1, so a median on this line means at least half the runs diverged
    for ax in axs:
        ax.axhline(1.0, color="0.35", ls=":", lw=1.0, zorder=1)
        ax.set_ylim(top=1.5)
        ax.annotate(r"$>50\%$ diverged", xy=(0.4, 1.0), xycoords=ax.get_yaxis_transform(),
                    xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color="0.35")
    axs[1].legend(title="N", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=fs-1, title_fontsize=fs-1)
    axs[1].set_title(rf"Noise Sensitivity ($N_t={REF_N_PROJ}$)", fontweight="bold", fontsize=fs+1)
    axs[1].tick_params(labelsize=ls)
    
    plt.suptitle("Spatio-Temporal Density Reconstruction Errors Over 10 Randomized Trials", fontweight="bold", fontsize=fs+2)

    plt.tight_layout()
    if "diverged" in df:
        rate = df.groupby(["N", "snr_db", "n_proj"])["diverged"].mean().rename("divergence_rate")
        rate.to_csv(output_dir / "divergence_rate.csv")
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