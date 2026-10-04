import logging
import re
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
SNR_DB = 20  # must match the experiment folder prefix, e.g. snr20_N5_...
BASE_DIR = Path(f"data/snr{SNR_DB}.0_ablation_and_baseline")
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
                exp_dir = variant_dir / f"snr{SNR_DB}_N{N}_nproj128_seed{seed}"
                if not exp_dir.exists():
                    continue
                try:
                    errors.append(run_analysis(exp_dir=exp_dir)["rel_l2_density_error"])
                except Exception:
                    logger.exception("Analysis failed for %s", exp_dir)

            if not errors:
                row[label] = "Missing (no run folders)"
                continue

            std = np.std(errors, ddof=1) if len(errors) > 1 else 0.0
            row[label] = f"{np.median(errors):.4f} ± {std:.4f}"

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows).set_index("N Particles")
    logger.info("\n" + "=" * 80)
    logger.info("                     GMM-CT ABLATION & BASELINE SUMMARY                     ")
    logger.info("=" * 80)
    logger.info(df.to_markdown())
    logger.info("=" * 80 + "\n")

    df.to_csv(CSV_PATH)
    logger.info(f"Summary (median ± std) saved to: {CSV_PATH}")

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
                        r"Reported values indicate Median Spatio-Temporal Relative $L_2$ Error $\pm$ Standard Deviation "
                        r"across 10 random seeds.}")
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