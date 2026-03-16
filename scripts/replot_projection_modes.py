"""
Replot projection_modes.pdf from a saved results.pt bundle without re-running
the full reconstruction pipeline.

Usage
-----
    python scripts/replot_projection_modes.py data/results/20260316_193717_seed9_N5_custm
    python scripts/replot_projection_modes.py data/results/20260316_193717_seed9_N5_custm --output my_plot.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from gmm_ct.visualization.publication import plot_projection_modes


def main():
    parser = argparse.ArgumentParser(description="Replot projection modes from results.pt")
    parser.add_argument("results_dir", type=Path, help="Path to the results directory")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output filename (default: projection_modes.pdf inside results_dir)",
    )
    parser.add_argument(
        "--title", type=str, default="Projection Modes",
        help="Figure title",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist.")
        sys.exit(1)

    bundle_path = results_dir / "results.pt"
    if not bundle_path.exists():
        print(f"Error: {bundle_path} not found.")
        sys.exit(1)

    print(f"Loading {bundle_path} ...")
    bundle = torch.load(bundle_path, weights_only=False)

    t         = bundle["t"]
    receivers = bundle["receivers"]
    proj_data = bundle["proj_data"]

    # proj_data is saved as a list-of-tensors-per-source; grab source 0
    proj_2d = proj_data[0] if isinstance(proj_data, (list, tuple)) else proj_data

    output_path = args.output or (results_dir / "projection_modes.pdf")

    print(f"Plotting → {output_path}")
    fig = plot_projection_modes(
        proj_2d, t, receivers,
        title=args.title,
        filename=output_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
