import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from src.analysis import cap_density_error, run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# --- Target Directory & Variant Mapping ---
hyp_param = '02_02'
BASE_DIR = Path(f"data/ablation_and_baseline_{hyp_param}")
CSV_PATH = BASE_DIR / "ablation_summary.csv"
LATEX_OUTPUT_PATH = BASE_DIR / "ablation_table.tex"

# Map CSV column headers to formatted LaTeX column titles
COLUMN_MAPPINGS = {
    "Direct LS": r"\makecell{Direct LS \\ \small(No decoupling)}",
    "Decoupled LS": r"\makecell{Decoupled LS}",
    "GMM-CT": r"\makecell{GMM-CT}",
}

VARIANTS: Dict[str, str] = {
    "Direct LS": "direct-ls",
    "Decoupled LS": "decoupled-ls",
    "GMM-CT": "gmm-ct",
}

N_PARTICLES: List[int] = [1, 2, 5, 9]
SEEDS: List[int] = list(range(10))

PER_SEED_KEYS = ["v0_rmse", "omega_rmse", "alpha_rmse", "U_rmse", "traj_rmse",
                 "v0_median_err", "traj_median_err"]
DENSITY_SUCCESS = 0.1   # relative L2 below this counts as recovered
TRAJ_SUCCESS = 0.01     # trajectory RMSE below this counts as recovered


def write_success_outputs(detail: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    """Success-rate table (CSV + LaTeX) and per-seed strip plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(VARIANTS)
    rows = []
    for N, g in detail.groupby("N Particles"):
        r = {"N Particles": N}
        for lab in labels:
            x = g[g["Variant"] == lab]
            if len(x):
                x = x.iloc[0]
                r[lab] = f"{x.n_density_ok}/{x.n_runs} ({x.n_traj_ok}/{x.n_runs})"
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("N Particles")
    tab.to_csv(CSV_PATH.with_name("ablation_success.csv"))
    logger.info("\nSuccess rate: density rel-L2 < %g (trajectory RMSE < %g)\n%s",
                DENSITY_SUCCESS, TRAJ_SUCCESS, tab.to_markdown())

    lines = [r"\begin{tabular}{c" + "c" * len(labels) + "}", r"\toprule",
             " & ".join([r"$N$"] + labels) + r" \\", r"\midrule"]
    for N, r in tab.iterrows():
        lines.append(" & ".join([f"${N}$"] + [str(r.get(l, "--")) for l in labels]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    CSV_PATH.with_name("ablation_success.tex").write_text("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    rng = np.random.default_rng(0)
    colors = dict(zip(labels, ["tab:red", "tab:orange", "tab:blue"]))
    for ax, key, thr, ylab in [(axes[0], "rel_l2", DENSITY_SUCCESS, "Relative $L_2$ density error"),
                               (axes[1], "traj_rmse", TRAJ_SUCCESS, "Trajectory RMSE")]:
        for i, N in enumerate(N_PARTICLES):
            for j, lab in enumerate(labels):
                v = per_seed[(per_seed.N == N) & (per_seed.variant == lab)][key].to_numpy(float)
                v = np.clip(np.nan_to_num(v, nan=1e6), 1e-5, 1e6)
                ax.scatter(i + (j - 1) * 0.25 + rng.uniform(-0.05, 0.05, len(v)), v,
                           s=18, color=colors[lab], alpha=0.7, label=lab if i == 0 else None)
        ax.axhline(thr, color="k", ls="--", lw=0.8)
        ax.set_yscale("log"); ax.set_ylabel(ylab)
        ax.set_xticks(range(len(N_PARTICLES))); ax.set_xticklabels(N_PARTICLES)
        ax.set_xlabel("$N$ Gaussians")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CSV_PATH.with_name("ablation_per_seed.png"), dpi=200)
    plt.close(fig)


def collate_results():
    summary_rows = []
    detail_rows = []
    per_seed = []

    for N in N_PARTICLES:
        row = {"N Particles": N}

        for label, dir_name in VARIANTS.items():
            variant_dir = BASE_DIR / dir_name
            errors, traj_errs, n_failed, n_total = [], [], 0, 0

            for seed in SEEDS:
                exp_dir = variant_dir / f"snr20_N{N}_nproj128_seed{seed}"
                if not exp_dir.exists():
                    continue
                n_total += 1
                res = {}
                try:
                    res = run_analysis(exp_dir=exp_dir)
                    err, traj = res["rel_l2_density_error"], res["traj_rmse"]
                except Exception:
                    logger.exception("Analysis failed for %s", exp_dir)
                    err, traj = np.nan, np.nan
                # Errors are capped at 1 (the empty reconstruction); >= 0.99 counts as diverged
                err, diverged = cap_density_error(err)
                n_failed += diverged
                errors.append(err)
                traj_errs.append(traj)
                per_seed.append({"N": N, "variant": label, "seed": seed, "rel_l2": err,
                                 **{k: res.get(k, np.nan) for k in PER_SEED_KEYS}})

            if not errors:
                row[label] = "Missing (no run folders)"
                continue

            std = np.std(errors, ddof=1) if len(errors) > 1 else 0.0
            cell = f"{np.median(errors):.4f} ± {std:.4f}"
            if n_failed:
                cell += f" ({n_failed}/{n_total} diverged)"
            row[label] = cell

            traj_arr = np.array(traj_errs, dtype=float)
            detail_rows.append({
                "N Particles": N, "Variant": label, "n_runs": n_total, "n_failed": n_failed,
                "mean_rel_l2": float(np.mean(errors)), "median_rel_l2": float(np.median(errors)),
                "max_rel_l2": float(np.max(errors)),
                "n_density_ok": int((np.array(errors) < DENSITY_SUCCESS).sum()),
                "n_traj_ok": int((np.nan_to_num(traj_arr, nan=np.inf) < TRAJ_SUCCESS).sum()),
                "median_traj_rmse": float(np.nanmedian(traj_arr)) if np.isfinite(traj_arr).any() else np.nan,
                "max_traj_rmse": float(np.nanmax(traj_arr)) if np.isfinite(traj_arr).any() else np.nan,
            })

        summary_rows.append(row)

    pd.DataFrame(per_seed).to_csv(CSV_PATH.with_name("ablation_per_seed.csv"), index=False)
    df = pd.DataFrame(summary_rows).set_index("N Particles")
    logger.info("\n" + "=" * 80)
    logger.info("                     GMM-CT ABLATION & BASELINE SUMMARY                     ")
    logger.info("=" * 80)
    logger.info(df.to_markdown())
    logger.info("=" * 80 + "\n")

    df.to_csv(CSV_PATH)
    logger.info(f"Summary (median ± std, failures flagged) saved to: {CSV_PATH}")

    detail = pd.DataFrame(detail_rows)
    detail_path = CSV_PATH.with_name("ablation_detail.csv")
    detail.to_csv(detail_path, index=False)
    logger.info("\n" + detail.to_markdown(index=False, floatfmt=".4g"))
    write_success_outputs(detail, pd.DataFrame(per_seed))
    logger.info(f"Detail (mean/max error, trajectory RMSE, failure counts) saved to: {detail_path}")

# ========================================================================
# LaTeX Section
# ========================================================================
    
def format_latex_cell(cell_value: str) -> str:
    """Format cell value into inline math mode $...$ for numerical error & std dev."""
    if pd.isna(cell_value):
        return "--"
    
    val_str = str(cell_value).strip()
    
    if val_str in ["Failed (Divergent)", "--"]:
        return r"\text{Failed}"

    # "mean ± std (k/n diverged)"; the divergence note is set in text mode outside the math
    div_match = re.search(r"\((\d+/\d+) diverged\)", val_str)
    if div_match:
        num_part = val_str[:div_match.start()].strip().replace("±", r"\pm ")
        return f"${num_part}$ \\small({div_match.group(1)} div.)"

    # Handles optional convergence percentage if present in CSV
    conv_match = re.search(r"\((.*?\%) conv\)", val_str)
    
    if conv_match:
        conv_text = conv_match.group(1).replace("%", r"\%")
        num_part = val_str[:conv_match.start()].strip().replace("±", r"\pm ")
        return f"${num_part}$ ({conv_text})"
    else:
        num_part = val_str.replace("±", r"\pm ")
        return f"${num_part}$"

def convert_csv_to_latex():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run collate_ablations.py first.")
        return

    df = pd.read_csv(CSV_PATH, index_col=0)
    
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Ablation and Baseline Benchmark ($\text{SNR} = 20\,\text{dB}$). "
                        r"Reported values are the median relative $L_2$ error of the spatio-temporal density $\pm$ standard deviation "
                        r"across 10 random seeds. Errors are capped at 1 (the error of an empty reconstruction); "
                        r"runs with error $\geq 0.99$ or a non-finite error are counted as diverged (div.).}")
    latex_lines.append(r"\label{tab:gmm_ct_ablation}")
    latex_lines.append(r"\vspace{2mm}")
    latex_lines.append(r"\resizebox{\linewidth}{!}{%")
    
    n_cols = len(df.columns) + 1
    col_spec = "c" * n_cols
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append(r"\toprule")
    
    # Header row
    headers = [r"\textbf{$N$ Gaussians}"]
    for col in df.columns:
        headers.append(COLUMN_MAPPINGS.get(col, r"\makecell{" + str(col) + "}"))
    
    latex_lines.append(" & ".join(headers) + r" \\")
    latex_lines.append(r"\midrule")
    
    # Data rows
    for idx, row in df.iterrows():
        row_str = [f"$N = {idx}$"]
        for col in df.columns:
            formatted_val = format_latex_cell(row[col])
            row_str.append(formatted_val)
        latex_lines.append(" & ".join(row_str) + r" \\")
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table*}")
    
    latex_code = "\n".join(latex_lines)
    LATEX_OUTPUT_PATH.write_text(latex_code)
    
    print("\n" + "=" * 60)
    print("      GENERATED PUBLICATION-READY LATEX TABLE CODE          ")
    print("=" * 60 + "\n")
    print(latex_code)
    print("\n" + "=" * 60)
    print(f"LaTeX snippet saved to: {LATEX_OUTPUT_PATH}")


if __name__ == "__main__":
    collate_results()
    convert_csv_to_latex()