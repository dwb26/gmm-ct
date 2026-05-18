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
    """Build a flat (parallel-beam) receiver array on a vertical line.

    Parameters
    ----------
    device : torch.device, optional
        Device for the output tensors (default: CPU).
    *args : tuple
        A single tuple ``(n_receivers, x_coordinate, y_min, y_max)``.

    Returns
    -------
    list of list of torch.Tensor
        ``receivers[source_idx][receiver_idx]`` → 2-D position tensor.
    """
    if device is None:
        device = torch.device('cpu')

    n_rcvrs, x1, x2_min, x2_max = args[0]
    x2 = torch.linspace(x2_min, x2_max, n_rcvrs, dtype=torch.float64, device=device)
    x2 = torch.flip(x2, dims=[0])  # conventional CT orientation
    return [[
        torch.tensor([x1, x2_val], dtype=torch.float64, device=device)
        for x2_val in x2
    ]]


# ==========================================================================
# Ground-truth parameter generation
# ==========================================================================

def generate_true_param(d, K, initial_location, initial_velocity,
                        initial_acceleration, min_rot, max_rot,
                        device=None, sampling_dt=None,
                        min_velocity_separation=0.5, min_diag_ratio=1.5):
    """Generate a complete set of synthetic GMM parameters for testing.

    Parameters
    ----------
    d : int
        Spatial dimensionality.
    K : int
        Number of Gaussian components.
    initial_location : torch.Tensor
        Shared initial position (``d``-dimensional).
    initial_velocity : torch.Tensor
        Base initial velocity (``d``-dimensional); perturbations are added.
    initial_acceleration : torch.Tensor
        Shared acceleration (``d``-dimensional).
    min_rot, max_rot : float
        Angular velocity search bounds (Hz).
    device : torch.device, optional
        Computation device (default: CPU).
    sampling_dt : float, optional
        Projection time interval; used to screen aliased omega values.
    min_velocity_separation : float, optional
        Minimum pairwise Euclidean distance between initial velocities.
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

    # Attenuation coefficients
    alphas = [
        torch.tensor(15., dtype=torch.float64, device=device) + 5 * k
        + torch.randn(1, dtype=torch.float64, device=device)
        for k in range(K)
    ]

    # U_skew matrices – rejection-sample to enforce minimum anisotropy
    U_ks = []
    for _ in range(K):
        for _attempt in range(500):
            mean_diag_val = 7.5
            U_k_diag = torch.rand(size=(d,), dtype=torch.float64, device=device) * 18.0 + mean_diag_val
            U_k_diag = torch.abs(U_k_diag)

            if (U_k_diag.max() / U_k_diag.min()).item() < min_diag_ratio:
                continue

            U_k_upper = 10 + torch.randn(
                size=((d - 1) * d // 2,), dtype=torch.float64, device=device
            )
            U_k = torch.zeros(d, d, dtype=torch.float64, device=device)
            triu_indices = torch.triu_indices(d, d, device=device)
            diag_idx = 0
            upper_idx = 0
            for idx in range(len(triu_indices[0])):
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                if i == j:
                    U_k[i, j] = U_k_diag[diag_idx]
                    diag_idx += 1
                else:
                    U_k[i, j] = U_k_upper[upper_idx]
                    upper_idx += 1
            break
        else:
            warnings.warn(
                f"Could not generate U_skew with diagonal ratio >= {min_diag_ratio} "
                "after 500 attempts; accepting last sample.",
                RuntimeWarning, stacklevel=2,
            )
        U_ks.append(U_k)

    # Angular velocities – screen aliased values
    alias_buffer = 0.10

    def _is_aliased(omega_val):
        if sampling_dt is None:
            return False
        frac = abs(omega_val * 2 * sampling_dt) % 1
        return frac < alias_buffer or frac > 1 - alias_buffer

    omegas = []
    for _ in range(K):
        for _attempt in range(200):
            omega_k = (
                max_rot - torch.rand(size=(math.comb(d, 2),), dtype=torch.float64, device=device)
                * (max_rot - min_rot)
            )
            if not any(_is_aliased(w.item()) for w in omega_k):
                break
        omegas.append(omega_k)

    # Initial positions (shared for all Gaussians, assumed known)
    x0s = [initial_location.to(torch.float64) for _ in range(K)]

    # Initial velocities – either fixed test values or rejection-sampled
    hardcoded = True
    if hardcoded and K == 5:
        _fixed_v0s = [
            torch.tensor([1.0, 3.0], dtype=torch.float64, device=device),
            torch.tensor([1.5, 1.8], dtype=torch.float64, device=device),
            torch.tensor([0.8, 2.5], dtype=torch.float64, device=device),
            torch.tensor([0.75, 1.2], dtype=torch.float64, device=device),
            torch.tensor([2., 3.], dtype=torch.float64, device=device),
        ]
        v0s = _fixed_v0s[:K]
    else:
        def _sample_velocity():
            v_h = initial_velocity[0] + torch.rand(1, dtype=torch.float64, device=device).item() * 1.5
            v_v = (torch.rand(1, dtype=torch.float64, device=device).item() - 0.5) * 4.5
            return torch.tensor([v_h, v_v], dtype=torch.float64, device=device)

        v0s = []
        for k in range(K):
            if k == 0:
                v0s.append(_sample_velocity())
                continue
            for _attempt in range(500):
                candidate = _sample_velocity()
                if all(
                    torch.norm(candidate - accepted).item() >= min_velocity_separation
                    for accepted in v0s
                ):
                    v0s.append(candidate)
                    break
            else:
                warnings.warn(
                    f"Could not find a velocity for component {k} satisfying "
                    f"min_velocity_separation={min_velocity_separation}. "
                    "Accepting last candidate.",
                    RuntimeWarning, stacklevel=2,
                )
                v0s.append(candidate)

    a0s = [initial_acceleration.to(torch.float64) for _ in range(K)]

    return {"alphas": alphas, "U_skews": U_ks, "omegas": omegas,
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


def export_parameters(theta_dict, filename, title="GMM Parameters",
                      theta_true=None, theta_init=None):
    """Export GMM parameters to a Markdown file.

    Parameters
    ----------
    theta_dict : dict
        Estimated parameter dictionary.
    filename : str or Path
        Output file path.
    title : str, optional
        Document title.
    theta_true : dict, optional
        Ground-truth parameters (for error computation).
    theta_init : dict, optional
        Initial-guess parameters (displayed alongside estimates).
    """
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
