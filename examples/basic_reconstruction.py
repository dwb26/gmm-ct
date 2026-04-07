"""
Basic GMM-CT reconstruction example.

Demonstrates the complete workflow on synthetic data:
  1. Configure CT geometry (source and receivers).
  2. Generate synthetic ground-truth GMM parameters.
  3. Simulate X-ray projection measurements.
  4. Run the four-stage GMM reconstruction.
  5. Compare estimated parameters to ground truth.
  6. Save publication-quality diagnostic figures.

Usage::

    python examples/basic_reconstruction.py

Outputs are written to plots/<timestamp>_seed<SEED>_N<N>/.
"""

import logging
from datetime import datetime
from pathlib import Path
from time import time

import torch

from gmm_ct import (
    GMM_reco,
    construct_receivers,
    export_parameters,
    generate_true_param,
    set_random_seeds,
)
from gmm_ct.config.defaults import GRAVITATIONAL_ACCELERATION
from gmm_ct.visualization.publication import (
    animate_temporal_gmm_comparison,
    plot_individual_gaussian_reconstruction,
    plot_temporal_gmm_comparison,
    reorder_theta_to_match_true,
)

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


def compute_parameter_errors(theta_true, theta_est, N):
    """Return relative L2 errors for each parameter group.

    Parameters
    ----------
    theta_true : dict
        Ground-truth parameter dictionary.
    theta_est : dict
        Estimated parameter dictionary.
    N : int
        Number of Gaussian components.

    Returns
    -------
    dict
        Mapping from parameter name to relative L2 error (scalar float).
    """
    errors = {}
    for key in ("alphas", "x0s", "v0s", "omegas"):
        true_stack = torch.stack([theta_true[key][k] for k in range(N)])
        est_stack = torch.stack([theta_est[key][k] for k in range(N)])
        errors[key] = (torch.norm(true_stack - est_stack) / torch.norm(true_stack)).item()

    U_true = torch.stack([theta_true["U_skews"][k].flatten() for k in range(N)])
    U_est = torch.stack([theta_est["U_skews"][k].flatten() for k in range(N)])
    errors["U_skews"] = (torch.norm(U_true - U_est) / torch.norm(U_true)).item()

    return errors


def main():
    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #
    SEED = 40
    N = 2           # number of Gaussian components
    D = 2           # spatial dimension
    N_PROJ = 65     # number of projection time steps
    T_MAX = 2.0     # observation window (seconds)
    OMEGA_MIN = -24.0
    OMEGA_MAX = OMEGA_MIN + 8.0

    set_random_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {SEED}  |  N: {N}")

    # Output directory
    project_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / "plots" / f"{timestamp}_seed{SEED}_N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # CT geometry                                                          #
    # ------------------------------------------------------------------ #
    sources = [torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)]
    x1 = sources[0][0].item() + 5.0
    x2_min = sources[0][1].item() - 2.0
    x2_max = sources[0][1].item() + 2.0
    rcvrs = construct_receivers(device, (128, x1, x2_min, x2_max))

    # ------------------------------------------------------------------ #
    # Ground-truth parameters and simulated projections                    #
    # ------------------------------------------------------------------ #
    t = torch.linspace(0.0, T_MAX, N_PROJ, dtype=torch.float64, device=device)

    i_loc = torch.tensor([1.0,  1.0], dtype=torch.float64, device=device)
    v_loc = torch.tensor([0.75, 0.5], dtype=torch.float64, device=device)
    a_loc = torch.tensor([0.0, -GRAVITATIONAL_ACCELERATION], dtype=torch.float64, device=device)

    theta_true = generate_true_param(
        D, N, i_loc, v_loc, a_loc, OMEGA_MIN, OMEGA_MAX, device=device
    )
    x0s = theta_true["x0s"]
    a0s = theta_true["a0s"]

    gmm_true = GMM_reco(
        D, N, sources, rcvrs, x0s, a0s, OMEGA_MIN, OMEGA_MAX,
        device=device, output_dir=out_dir,
    )
    proj_data = gmm_true.generate_projections(t, theta_true)

    # ------------------------------------------------------------------ #
    # Reconstruction                                                       #
    # ------------------------------------------------------------------ #
    t0 = time()
    gmm = GMM_reco(
        D, N, sources, rcvrs, x0s, a0s, OMEGA_MIN, OMEGA_MAX,
        device=device, output_dir=out_dir,
    )
    soln = gmm.fit(proj_data, t)
    print(f"\nReconstruction completed in {time() - t0:.1f}s")

    soln, matching = reorder_theta_to_match_true(theta_true, soln, N)
    print(f"Gaussian matching (est -> true): {matching}")

    # ------------------------------------------------------------------ #
    # Parameter comparison                                                 #
    # ------------------------------------------------------------------ #
    errors = compute_parameter_errors(theta_true, soln, N)
    print("\nRelative L2 errors:")
    for key, err in errors.items():
        print(f"  {key:<10s}: {err:.4e}")

    print("\nAngular velocity comparison:")
    for k in range(N):
        w_true = theta_true["omegas"][k].item()
        w_est = soln["omegas"][k].item()
        print(f"  Gaussian {k}: true={w_true:.4f}  est={w_est:.4f}  |err|={abs(w_true - w_est):.4e}")

    # ------------------------------------------------------------------ #
    # Export parameters                                                    #
    # ------------------------------------------------------------------ #
    export_parameters(
        theta_true, out_dir / "true_parameters.md", title="Ground Truth Parameters"
    )
    export_parameters(
        soln, out_dir / "estimated_parameters.md", title="Estimated Parameters",
        theta_true=theta_true, theta_init=gmm.theta_dict_init,
    )

    # ------------------------------------------------------------------ #
    # Figures                                                              #
    # ------------------------------------------------------------------ #
    plot_individual_gaussian_reconstruction(
        theta_true, soln, N, D,
        gaussian_indices=range(N),
        filename=out_dir / "individual_gaussian_reconstruction.pdf",
    )
    plot_temporal_gmm_comparison(
        sources, rcvrs, theta_true, soln, t, N, D,
        time_indices=[17, 20, 22],
        filename=out_dir / "temporal_gmm_comparison.pdf",
    )
    animate_temporal_gmm_comparison(
        sources, rcvrs, theta_true, soln, t, N, D,
        filename=out_dir / "temporal_gmm_comparison.mp4",
    )

    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()
