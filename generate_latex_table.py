import re
import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/ablation_and_baseline/ablation_summary.csv")
LATEX_OUTPUT_PATH = Path("data/ablation_and_baseline/ablation_table.tex")

# Map CSV column headers to formatted LaTeX column titles
COLUMN_MAPPINGS = {
    "Direct Huber LS": r"\makecell{Direct Huber LS \\ \small(Baseline)}",
    "No Stage 1.5": r"\makecell{Trajectory Opt. \\ \small(No Stage 1.5)}",
    "No Trajectory Matching": r"\makecell{Stage 1.5 Only \\ \small(No Hausdorff)}",
    "Full GMM-CT (Ours)": r"\makecell{GMM-CT}",
}


def format_latex_cell(cell_value: str) -> str:
    """Format cell value into inline math mode $...$ for numerical error & std dev."""
    if pd.isna(cell_value):
        return "--"
    
    val_str = str(cell_value).strip()
    
    if val_str in ["Failed (Divergent)", "--"]:
        return r"\text{Failed}"

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
    latex_lines.append(r"\caption{Ablation and Baseline Benchmark ($\text{SNR} = 80\,\text{dB}$). "
                        r"Reported values indicate Mean Spatio-Temporal Relative $L_2$ Error $\pm$ Standard Deviation "
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
    convert_csv_to_latex()