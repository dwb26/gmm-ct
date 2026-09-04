"""Utility functions for GMM-CT: geometry, data generation, and helpers."""

import logging
import math
import warnings
from datetime import datetime

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ==========================================================================
# Geometry
# ==========================================================================

def construct_receivers(device=None, *args):
    """Build a flat (parallel-beam) receiver array on a vertical line (2D) or
    a planar grid (3D).

    Parameters
    ----------
    device : torch.device, optional
        Device for the output tensors (default: CPU).
    *args : tuple
        A single tuple describing the geometry:

        * **2D** – ``(n_receivers, x1, x2_min, x2_max)``
        * **3D** – ``(n_receivers_y, n_receivers_z, x1, y_min, y_max, z_min, z_max)``

    Returns
    -------
    list of list of torch.Tensor
        ``receivers[source_idx][receiver_idx]`` → position tensor (2-D or 3-D).
    """
    if device is None:
        device = torch.device('cpu')

    params = args[0]

    if len(params) == 4:
        # 2D: receivers on a vertical line
        n_rcvrs, x1, x2_min, x2_max = params
        x2 = torch.linspace(x2_min, x2_max, n_rcvrs, dtype=torch.float64, device=device)
        x2 = torch.flip(x2, dims=[0])  # conventional CT orientation
        return [[
            torch.tensor([x1, x2_val], dtype=torch.float64, device=device)
            for x2_val in x2
        ]]

    elif len(params) == 7:
        # 3D: receivers on a flat y×z panel at fixed x1
        n_rcvrs_y, n_rcvrs_z, x1, y_min, y_max, z_min, z_max = params
        y = torch.linspace(y_min, y_max, n_rcvrs_y, dtype=torch.float64, device=device)
        y = torch.flip(y, dims=[0])  # conventional CT orientation
        z = torch.linspace(z_min, z_max, n_rcvrs_z, dtype=torch.float64, device=device)
        return [[
            torch.tensor([x1, y_val, z_val], dtype=torch.float64, device=device)
            for y_val in y
            for z_val in z
        ]]

    else:
        raise ValueError(
            f"Expected a tuple of length 4 (2D) or 7 (3D), got {len(params)}."
        )


# ==========================================================================
# Ground-truth parameter generation
# ==========================================================================

def generate_true_param(
    d: int, 
    N: int, 
    initial_location: torch.Tensor, 
    initial_velocity: torch.Tensor,
    initial_acceleration: torch.Tensor, 
    min_rot: float, 
    max_rot: float,
    device: torch.device | None = None, 
    min_diag_ratio: float = 1.5
) -> dict[str, list[torch.tensor]]:
    """Generate a complete set of synthetic GMM parameters for testing.

    Parameters
    ----------
    min_diag_ratio : float, optional
        Minimum diagonal aspect ratio for U_skew (enforces anisotropy).

    Returns
    -------
    dict
        Keys: ``'alphas', 'U_skews', 'omegas', 'x0s', 'v0s', 'a0s'``.
    """
    if device is None:
        device = torch.device('cpu')

    if len(initial_location) != d:
        raise ValueError("initial_location must have length d.")
    if len(initial_velocity) != d:
        raise ValueError("initial_velocity must have length d.")
    if len(initial_acceleration) != d:
        raise ValueError("initial_acceleration must have length d.")

    # ---- Generate attenuation coefficients
    alphas = [
        torch.tensor(15., dtype=torch.float64, device=device) + 5 * n
        + torch.randn(1, dtype=torch.float64, device=device)
        for n in range(N)
    ]

    # ---- Generate morphology precision matrices – rejection-sample to enforce minimum anisotropy
    U_ns = []
    for _ in range(N):
        for _attempt in range(500):
            mean_diag_val = 7.5
            U_n_diag = torch.rand(size=(d,), dtype=torch.float64, device=device) * 18.0 + mean_diag_val

            # Test for the anisotropy condition before constructing the full matrix. If fails, restart
            if (U_n_diag.max() / U_n_diag.min()).item() < min_diag_ratio:
                continue

            # Construct the full upper-triangular matrix with the sampled diagonal and random upper entries
            U_n_upper = 10 + torch.randn(
                size=((d - 1) * d // 2,), dtype=torch.float64, device=device
            )
            U_n = torch.zeros(d, d, dtype=torch.float64, device=device)
            triu_indices = torch.triu_indices(d, d, device=device)
            diag_idx = 0
            upper_idx = 0
            for idx in range(len(triu_indices[0])):
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                if i == j:
                    U_n[i, j] = U_n_diag[diag_idx]
                    diag_idx += 1
                else:
                    U_n[i, j] = U_n_upper[upper_idx]
                    upper_idx += 1
            break
        else:
            warnings.warn(
                f"Could not generate U_skew with diagonal ratio >= {min_diag_ratio} "
                "after 500 attempts; accepting last sample.",
                RuntimeWarning, stacklevel=2,
            )
        U_ns.append(U_n)

    # ---- Generate angular velocities
    omegas = [
        max_rot - torch.rand(size=(math.comb(d, 2),), dtype=torch.float64, device=device)
        * (max_rot - min_rot)
        for _ in range(N)
    ]

    # Initial positions (shared for all Gaussians, assumed known)
    x0s = [initial_location.to(torch.float64) for _ in range(N)]

    # Initial velocities
    hardcoded = True
    if hardcoded and N == 5 and d == 2:
        v0s = [
            torch.tensor([1.0, 3.0], dtype=torch.float64, device=device),
            torch.tensor([1.5, 1.8], dtype=torch.float64, device=device),
            torch.tensor([0.8, 2.5], dtype=torch.float64, device=device),
            torch.tensor([0.75, 1.2], dtype=torch.float64, device=device),
            torch.tensor([2., 3.], dtype=torch.float64, device=device),
        ]
    else:
        v0s = [
            initial_velocity.to(torch.float64) + (
                torch.rand(d, dtype=torch.float64, device=device) - 0.5
            ) * 4.5
            for _ in range(N)
        ]

    a0s = [initial_acceleration.to(torch.float64) for _ in range(N)]

    return {"alphas": alphas, "U_skews": U_ns, "omegas": omegas,
            "x0s": x0s, "v0s": v0s, "a0s": a0s}


# ==========================================================================
# Helpers
# ==========================================================================

def set_random_seeds(seed=42):
    """Set random seeds for PyTorch and NumPy for reproducibility.

    Parameters
    ----------
    seed : int

    Returns
    -------
    numpy.random.Generator
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def export_parameters(
    theta_dict: dict[str, list[torch.Tensor]], 
    filename: str, 
    title: str = "GMM Parameters",
    theta_true: dict[str, list[torch.Tensor]] | None = None, 
    theta_init: dict[str, list[torch.Tensor]] | None = None,
) -> None:
    """Export GMM parameters to a Markdown file."""
    with open(filename, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        if theta_true:
            f.write("## Overall Gaussian Errors\n\n")
            f.write("| Gaussian Index | Absolute Error |\n")
            f.write("|----------------|----------------|\n")
            K = len(theta_dict.get('v0s', []))
            for i in range(K):
                error = 0.0
                if 'v0s' in theta_true and 'v0s' in theta_dict:
                    error += np.linalg.norm(
                        theta_dict['v0s'][i].detach().cpu().numpy()
                        - theta_true['v0s'][i].detach().cpu().numpy()
                    )
                if 'omegas' in theta_true and 'omegas' in theta_dict:
                    error += abs(
                        theta_dict['omegas'][i].item() - theta_true['omegas'][i].item()
                    )
                if 'alphas' in theta_true and 'alphas' in theta_dict:
                    error += abs(
                        theta_dict['alphas'][i].item() - theta_true['alphas'][i].item()
                    )
                if 'U_skews' in theta_true and 'U_skews' in theta_dict:
                    error += np.linalg.norm(
                        theta_dict['U_skews'][i].detach().cpu().numpy()
                        - theta_true['U_skews'][i].detach().cpu().numpy()
                    )
                f.write(f"| {i + 1:<14} | {error:.4f}         |\n")
            f.write("\n")

        for key, value in theta_dict.items():
            f.write(f"## `{key}`\n\n")
            if not isinstance(value, list) or not value:
                f.write(f"```\n{value}\n```\n\n")
                continue

            if not isinstance(value[0], torch.Tensor):
                f.write(f"```\n{value}\n```\n\n")
                continue

            np_values = [v.detach().cpu().numpy() for v in value]
            init_values = (
                [v.detach().cpu().numpy() for v in theta_init[key]]
                if theta_init and key in theta_init else None
            )
            error_values = None
            if theta_true and key in theta_true:
                true_values = [v.detach().cpu().numpy() for v in theta_true[key]]
                if np_values[0].ndim == 0 or np_values[0].size == 1:
                    error_values = [np.abs(e - t) for e, t in zip(np_values, true_values)]
                else:
                    error_values = [np.linalg.norm(e - t) for e, t in zip(np_values, true_values)]

            if np_values[0].ndim == 0 or np_values[0].size == 1:
                header = "| Gaussian | Value "
                sep = "|----------|-------"
                if init_values:
                    header += "| Initial "
                    sep += "|--------"
                if error_values:
                    header += "| Error "
                    sep += "|-------"
                f.write(header + "|\n" + sep + "|\n")
                for i, val in enumerate(np_values):
                    row = f"| {i + 1:<8} | {float(np.squeeze(val)):.4f} "
                    if init_values:
                        row += f"| {float(np.squeeze(init_values[i])):.4f}  "
                    if error_values:
                        row += f"| {float(np.squeeze(error_values[i])):.4f} "
                    f.write(row + "|\n")
                f.write("\n")

            elif np_values[0].ndim == 1:
                nc = np_values[0].shape[0]
                header = "| Gaussian | " + " | ".join(f"Comp {j + 1}" for j in range(nc))
                sep = "|----------" + "|----------" * nc
                if error_values:
                    header += " | Error (L2)"
                    sep += "|------------"
                f.write(header + "|\n" + sep + "|\n")
                for i, vec in enumerate(np_values):
                    row = f"| {i + 1:<8} | " + " | ".join(f"{c:.4f}" for c in vec)
                    if error_values:
                        row += f" | {error_values[i]:.4f}"
                    f.write(row + "|\n")
                f.write("\n")

            elif np_values[0].ndim == 2:
                for i, matrix in enumerate(np_values):
                    f.write(f"### Gaussian {i + 1}\n")
                    if init_values and i < len(init_values):
                        f.write("#### Initial\n```\n")
                        f.write(np.array2string(init_values[i], precision=4, separator=', '))
                        f.write("\n```\n\n")
                    f.write("#### Estimated\n```\n")
                    f.write(np.array2string(matrix, precision=4, separator=', '))
                    f.write("\n```\n")
                    if error_values and i < len(error_values):
                        f.write(f"\n**Frobenius error:** {error_values[i]:.4f}\n")
                    f.write("\n")

    logger.info("Parameters exported to %s", filename)


# ==========================================================================
# L-BFGS root-finding solver (used by Newton-Raphson velocity refinement)
# ==========================================================================

def NewtonRaphsonLBFGS(
    func, 
    x0: torch.Tensor, 
    *args, 
    tol: float = 1e-5, 
    max_iter: int = 100,
    line_search_fn: str = 'strong_wolfe'
) -> torch.Tensor:
    """Find roots of func(x) = 0 by minimising ‖func(x)‖² with L-BFGS."""
    if not x0.requires_grad:
        x0.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x0], 
        max_iter=max_iter, 
        tolerance_grad=tol,
        tolerance_change=tol, 
        line_search_fn=line_search_fn,
    )

    def closure():
        optimizer.zero_grad()
        f_val = func(x0, *args)
        loss = f_val ** 2 if f_val.dim() == 0 else torch.sum(f_val ** 2)
        if loss.requires_grad:
            loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception as e:
        if "does not require grad" not in str(e):
            logger.warning("L-BFGS root-finding failed: %s", e)

    return x0