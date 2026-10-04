"""Sensitivity of the three pipelines to what the reconstruction assumes.

Re-uses the simulated data of the main snr20 study (so the truth is identical) and perturbs only the
reconstruction side: the velocity initialisation prior and the assumed initial position x0.

    python run_sensitivity.py run       # reconstruct every scenario / mode / seed
    python run_sensitivity.py collate   # print mean +/- std and median of the relative L2 error
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

SNR_DB = 20
SRC = Path(f"data/snr{SNR_DB}.0_ablation_and_baseline")
OUT = Path(f"data/snr{SNR_DB}.0_sensitivity")
N_PARTICLES = [5]
SEEDS = list(range(5))
MODES = {"Direct LS": "direct-ls", "Decoupled LS": "decoupled-ls", "GMM-CT": "gmm-ct"}

# Default prior is mean (1, 1), std (0.5, 2); the simulator draws the truth from it
SCENARIOS = {
    "baseline": [],
    "wide_v_prior": ["--init-v-std", "3", "4"],
    "shifted_v_prior": ["--init-v-mean", "3", "-3"],
    "x0_offset": ["--x0-offset", "0.3", "0.3"],
}


def run():
    for scen, flags in SCENARIOS.items():
        for N in N_PARTICLES:
            for seed in SEEDS:
                for label, mode in MODES.items():
                    name = f"snr{SNR_DB}_N{N}_nproj128_seed{seed}"
                    dst = OUT / scen / mode / name
                    if (dst / "reconstruction.pt").exists():
                        continue
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in ("ground_truth.pt", "projections.pt"):
                        shutil.copy(SRC / mode / name / f, dst / f)
                    cmd = [sys.executable, "-m", "src.cli", "reconstruct",
                           "--config", "configs/experiment.yaml", "--pipeline-mode", mode,
                           "--seed", str(seed), "--sim-n-gaussians", str(N),
                           "--reco-n-gaussians", str(N), "--exp-dir", str(dst),
                           "--output-dir", str(OUT / scen / mode), *flags]
                    print("RUN", scen, mode, N, seed, flush=True)
                    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def collate():
    from src.analysis import run_analysis
    print(f"{'scenario':16s} {'N':>2s} " + " ".join(f"{m:>34s}" for m in MODES))
    for scen in SCENARIOS:
        for N in N_PARTICLES:
            cells = []
            for mode in MODES.values():
                errs = []
                for seed in SEEDS:
                    d = OUT / scen / mode / f"snr{SNR_DB}_N{N}_nproj128_seed{seed}"
                    try:
                        e = run_analysis(exp_dir=d)["rel_l2_density_error"]
                    except Exception:
                        e = np.nan
                    errs.append(e if np.isfinite(e) else 1.0)
                errs = np.array(errs)
                cells.append(f"{errs.mean():.3f} ± {errs.std():.3f} (med {np.median(errs):.3f}, {int((errs >= 0.999).sum())} fail)")
            print(f"{scen:16s} {N:>2d} " + " ".join(f"{c:>34s}" for c in cells), flush=True)


if __name__ == "__main__":
    {"run": run, "collate": collate}[sys.argv[1]]()
