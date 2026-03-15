"""
Projection Sweep Experiment
============================
Measures absolute error across all GMM parameters as a function of the
number of projections T.  The number of projections is varied in powers of
two: T = 2^2, 2^3, ..., 2^9 (i.e. 4, 8, 16, 32, 64, 128, 256, 512).

A single fixed seed is used across all values of T so that the ground-truth
parameters are identical in every run, making the comparison fair.

Outputs
-------
results/projection_sweep_<timestamp>/
    sweep_results.csv        per-Gaussian absolute errors for every T
    projection_sweep.pdf     publication-quality 4-panel figure
    projection_sweep.png     same figure as PNG
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

from gmm_ct import GMM_reco, generate_true_param, construct_receivers, set_random_seeds
from gmm_ct.visualization.publication import reorder_theta_to_match_true
from gmm_ct.config.defaults import GRAVITATIONAL_ACCELERATION

# ---------------------------------------------------------------------------
# Geometry / physics constants (match simulate.yaml defaults)
# ---------------------------------------------------------------------------
D          = 2
N          = 5
DURATION   = 1.5          # seconds
OMEGA_MIN  = 2.0
OMEGA_MAX  = 6.0
I_LOC      = torch.tensor([1.0,  1.0], dtype=torch.float64)
V_LOC      = torch.tensor([0.75, 0.5], dtype=torch.float64)
A_LOC      = torch.tensor([0.0, -GRAVITATIONAL_ACCELERATION], dtype=torch.float64)

N_RECEIVERS = 128
SOURCE_POS  = [-1.0, -1.0]
RCVR_X      = 4.0
RCVR_Y_MIN  = -3.0
RCVR_Y_MAX  = 1.0

# Powers of 2: 2^2 … 2^9
PROJ_COUNTS = [2**p for p in range(2, 10)]   # [4, 8, 16, 32, 64, 128, 256, 512]

SEED = 7


# ---------------------------------------------------------------------------
# Per-parameter absolute error helpers
# ---------------------------------------------------------------------------

def _to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def compute_absolute_error(theta_true, theta_est, N):
    """
    Return the absolute error across ALL GMM parameters as a single scalar:
    L1 norm of the concatenated difference vector over all N Gaussians,
    [alpha_1, v0_1, U_skew_1, omega_1, ..., alpha_N, v0_N, U_skew_N, omega_N].

    Returns
    -------
    float
    """
    vecs_true, vecs_est = [], []
    for k in range(N):
        vecs_true.append(np.concatenate([
            _to_np(theta_true['alphas'][k]).flatten(),
            _to_np(theta_true['v0s'][k]).flatten(),
            _to_np(theta_true['U_skews'][k]).flatten(),
            _to_np(theta_true['omegas'][k]).flatten(),
        ]))
        vecs_est.append(np.concatenate([
            _to_np(theta_est['alphas'][k]).flatten(),
            _to_np(theta_est['v0s'][k]).flatten(),
            _to_np(theta_est['U_skews'][k]).flatten(),
            _to_np(theta_est['omegas'][k]).flatten(),
        ]))
    full_true = np.concatenate(vecs_true)
    full_est  = np.concatenate(vecs_est)
    return float(np.sum(np.abs(full_est - full_true)))


# ---------------------------------------------------------------------------
# Single-T run
# ---------------------------------------------------------------------------

def run_single(n_proj, theta_true, device, sources, receivers, x0s, a0s,
               output_dir):
    """Run reconstruction for a fixed ground truth with n_proj time steps."""
    t = torch.linspace(0.0, DURATION, n_proj, dtype=torch.float64, device=device)

    # Generate projections from ground truth
    model_true = GMM_reco(D, N, sources, receivers, x0s, a0s,
                          OMEGA_MIN, OMEGA_MAX, device=device)
    proj_list = model_true.generate_projections(t, theta_true)

    # Reconstruct
    model = GMM_reco(D, N, sources, receivers, x0s, a0s,
                     OMEGA_MIN, OMEGA_MAX, device=device,
                     output_dir=output_dir)
    theta_est = model.fit(proj_list, t)

    # Reorder estimated Gaussians to match true labelling
    theta_est, _ = reorder_theta_to_match_true(theta_true, theta_est, N)

    return compute_absolute_error(theta_true, theta_est, N)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_projection_sweep(seed=SEED, proj_counts=None, output_root=None):
    if proj_counts is None:
        proj_counts = PROJ_COUNTS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Projection counts: {proj_counts}")

    # Output directory
    project_root = Path(__file__).resolve().parent.parent.parent
    if output_root is None:
        output_root = project_root / 'plots'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root) / f"projection_sweep_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed geometry
    sources = [torch.tensor(SOURCE_POS, dtype=torch.float64, device=device)]
    receivers = construct_receivers(
        device, (N_RECEIVERS, RCVR_X, RCVR_Y_MIN, RCVR_Y_MAX)
    )
    x0s = [I_LOC.clone().to(device) for _ in range(N)]
    a0s = [A_LOC.clone().to(device) for _ in range(N)]

    # Generate ground truth ONCE using the largest T for sampling_dt
    # (dt is only used for the Nyquist aliasing check, so use finest grid)
    set_random_seeds(seed)
    dt_finest = DURATION / (max(proj_counts) - 1)
    theta_true = generate_true_param(
        D, N, I_LOC.to(device), V_LOC.to(device), A_LOC.to(device),
        OMEGA_MIN, OMEGA_MAX, device=device, sampling_dt=dt_finest,
    )

    # Sweep
    records = []
    for n_proj in proj_counts:
        print(f"\n{'='*55}")
        print(f"  T = {n_proj} projections")
        print(f"{'='*55}")
        t0 = time()
        try:
            err = run_single(n_proj, theta_true, device, sources, receivers,
                             x0s, a0s, out_dir)
            elapsed = time() - t0
            records.append({
                'n_proj':  n_proj,
                'err_abs': err,
                'time_s':  elapsed,
            })
            print(f"  Absolute error: {err:.4e}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            records.append({
                'n_proj': n_proj, 'err_abs': np.nan, 'time_s': np.nan,
            })

    df = pd.DataFrame(records)
    csv_path = out_dir / 'sweep_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_sweep_results(df, out_dir, N)
    return df, out_dir


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep_results(df, out_dir, n_gaussians):
    """Single-panel figure: collective absolute parameter error vs T."""
    proj_counts = sorted(df['n_proj'].unique())
    vals = [df.loc[df['n_proj'] == T, 'err_abs'].values[0]
            if T in df['n_proj'].values else np.nan
            for T in proj_counts]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(proj_counts, vals, 'k-o', linewidth=2, markersize=7)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of projections $T$', fontsize=11)
    ax.set_ylabel(
        r'$\|\hat{\theta} - \theta\|_1$  (all GMM parameters)',
        fontsize=11
    )
    ax.set_title(
        fr'Absolute parameter error vs projections  ($N={n_gaussians}$, seed $={SEED}$)',
        fontsize=12
    )
    ax.set_xticks(proj_counts)
    ax.set_xticklabels([str(T) for T in proj_counts], fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = out_dir / f'projection_sweep.{ext}'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_projection_sweep()
