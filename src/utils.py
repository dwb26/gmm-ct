"""Utility functions for GMM-CT: geometry, data generation, and helpers."""

import logging
import math
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

VELOCITY_MODELS = ("clustered", "sparse")

def generate_velocity_ensemble(
    N: int,
    mean: tuple[float, float] = (1.0, 1.0),
    std: tuple[float, float] = (0.5, 2.0),
    device: torch.device = torch.device('cpu'),
    max_attempts: int = 100000,
    model: str = "sparse",
    x0: torch.Tensor | None = torch.tensor([1.0, 1.0]),
    a0: torch.Tensor | None = torch.tensor([0.0, -9.81]),
    duration: float = 2.0,
    min_separation: float = 0.2, # Both 0.2 is the safe option
    # max_overlap_frac: float = 0.25,
    max_overlap_frac: float = 0.4,
    n_times: int = 128,
) -> list[torch.Tensor]:
    """Generates N independent velocity vectors sampled from a 2D diagonal Normal distribution.
    
    Rejects samples where v_x <= 0 to ensure forward motion along the x-axis.

    ``model`` selects the sampling scheme:

    - ``"clustered"``: independent draws with only the v_x > 0 rule, so particles may
      overlap or stay close (the original scheme, also used to initialise estimates).
    - ``"sparse"``: additionally, a candidate is discarded when its vertical (y) trajectory
      y(t) = x0_y + v_y t + a0_y t^2 / 2 lies within ``min_separation`` of an already
      accepted particle for more than ``max_overlap_frac`` of the observation window,
      so that particles remain distinguishable in the receiver array. Requires ``x0``
      and ``a0``. ``max_attempts`` is then a per-particle budget.
    """
    if model not in VELOCITY_MODELS:
        raise ValueError(f"Unknown velocity model '{model}'; expected one of {VELOCITY_MODELS}.")
    reject = model == "sparse"
    if reject and (x0 is None or a0 is None):
        raise ValueError('model="sparse" requires x0 and a0.')
    dtype = torch.float64
    mu = torch.tensor(mean, dtype=dtype, device=device)
    sigma = torch.tensor(std, dtype=dtype, device=device)
    
    v0s = []
    attempts = 0
    budget = max_attempts * (N if reject else 1)

    if reject:
        t_grid = torch.linspace(0.0, duration, n_times, dtype=dtype, device=device)
        y_of = lambda v: x0[1].to(dtype) + v[1] * t_grid + 0.5 * a0[1].to(dtype) * t_grid**2
        accepted_y = []
    
    while len(v0s) < N and attempts < budget:
        attempts += 1
        # Sample independent (v_x, v_y) from bivariate normal
        v_sample = mu + sigma * torch.randn(2, dtype=dtype, device=device)
        
        # Simple rejection rule: keep only forward x-velocities
        if not v_sample[0] > 0.0:
            continue

        if reject:
            y = y_of(v_sample)
            if any(((y - y_k).abs() < min_separation).double().mean() > max_overlap_frac
                   for y_k in accepted_y):
                continue
            accepted_y.append(y)

        v_sample.requires_grad_(True)
        v0s.append(v_sample)
            
    if len(v0s) < N:
        raise RuntimeError(
            f"Could not generate {N} valid velocities with v_x > 0 after {max_attempts} attempts. "
            f"Consider increasing mean[0] or std[0]."
        )
        
    return v0s

def generate_particle_morphology(
    N: int,
    d: int = 2,
    max_diag_ratio: float = 2.5,
    diag_range: tuple[float, float] = (16.0, 58.0),
    device: torch.device = torch.device('cpu'),
) -> list[torch.Tensor]:
    """Generates upper-triangular precision factor matrices U.

    Size and shape are sampled separately so that both vary independently:
    the geometric mean of the diagonal (particle size; larger = smaller particle) is
    log-uniform over ``diag_range``, and the anisotropy ratio max/min is uniform on
    [1, ``max_diag_ratio``] (so near-isotropic particles are possible but not dominant).
    Samples whose diagonal leaves ``diag_range`` are rejected, so no particle is thinner
    than ``max_diag_ratio`` allows.
    """
    dtype = torch.float64
    lo, hi = diag_range
    log_lo, log_hi = math.log(lo), math.log(hi)
    U_ns = []

    for _ in range(N):
        for _ in range(500):
            geo_mean = math.exp(log_lo + (log_hi - log_lo) * torch.rand(1).item())
            ratio = 1.0 + (2 * max_diag_ratio - 1.0) * torch.rand(1).item()

            # Spread the ratio symmetrically in log-space around the geometric mean
            if d == 2:
                sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                log_spread = torch.tensor([sign, -sign], dtype=dtype, device=device)
            else:
                log_spread = torch.empty(d, dtype=dtype, device=device).uniform_(-1, 1)
                log_spread = log_spread / log_spread.abs().max()
            U_n_diag = geo_mean * torch.exp(0.5 * math.log(ratio) * log_spread)

            if U_n_diag.min() < lo or U_n_diag.max() > hi:
                continue

            # Zero-centered off-diagonals to avoid systematic shear bias
            num_off_diag = (d - 1) * d // 2
            U_n_upper = torch.randn(size=(num_off_diag,), dtype=dtype, device=device) * 3.0

            # Construct upper triangular matrix U
            U_n = torch.zeros((d, d), dtype=dtype, device=device)
            triu_indices = torch.triu_indices(d, d, offset=0, device=device)
            
            diag_mask = (triu_indices[0] == triu_indices[1])
            off_diag_mask = ~diag_mask

            U_n[triu_indices[0][diag_mask], triu_indices[1][diag_mask]] = U_n_diag
            U_n[triu_indices[0][off_diag_mask], triu_indices[1][off_diag_mask]] = U_n_upper
            break
        else:
            raise RuntimeError(f"Could not sample a morphology within diag_range={diag_range}.")
        U_ns.append(U_n)
        
    return U_ns

def generate_true_param(
    d: int, 
    N: int, 
    initial_location: torch.Tensor, 
    initial_acceleration: torch.Tensor, 
    min_rot: float, 
    max_rot: float,
    device: torch.device = torch.device('cpu'), 
    velocity_model: str = "sparse",
    duration: float = 2.0,
    n_times: int = 128,
) -> dict[str, list[torch.tensor]]:
    """Generate a complete set of synthetic GMM parameters for testing.

    Parameters
    ----------
    velocity_model : {"sparse", "clustered"}, optional
        ``"sparse"`` rejection-samples velocities so vertical trajectories are separated;
        ``"clustered"`` uses plain independent draws (see ``generate_velocity_ensemble``).
        Estimates are always initialised with ``"clustered"``.

    Returns
    -------
    dict
        Keys: ``'alphas', 'U_skews', 'omegas', 'x0s', 'v0s', 'a0s'``.
    """
    if len(initial_location) != d:
        raise ValueError("initial_location must have length d.")
    if len(initial_acceleration) != d:
        raise ValueError("initial_acceleration must have length d.")

    # ---- Generate attenuation coefficients
    alphas = [
        torch.tensor(15., dtype=torch.float64, device=device) + 5 * n
        + torch.randn(1, dtype=torch.float64, device=device)
        for n in range(N)
    ]

    # ---- Generate morphology precision matrices – rejection-sample to enforce minimum anisotropy
    U_ns = generate_particle_morphology(N)

    # ---- Generate angular velocities
    num_rot_params = math.comb(d, 2)
    omegas = [
        max_rot - torch.rand(size=(num_rot_params,), dtype=torch.float64, device=device) * (max_rot - min_rot)
        for _ in range(N)
    ]

    # ---- Trajectory parameters
    x0s = [initial_location.to(torch.float64) for _ in range(N)]
    v0s = generate_velocity_ensemble(
        N, model=velocity_model, x0=x0s[0], a0=initial_acceleration.to(torch.float64),
        duration=duration, n_times=n_times,
    )
    a0s = [initial_acceleration.to(torch.float64) for _ in range(N)]

    return {"alphas": alphas, "U_skews": U_ns, "omegas": omegas,
            "x0s": x0s, "v0s": v0s, "a0s": a0s}

# ==========================================================================
# Helpers
# ==========================================================================

def set_random_seeds(seed=42):
    """Set random seeds for PyTorch and NumPy for reproducibility."""
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
    
    
def add_sinogram_noise(
    proj_data: list[torch.Tensor] | torch.Tensor,
    snr_db: float, 
    seed: int | None = None,
) -> list[torch.Tensor] | torch.Tensor:
    """Adds zero-mean Gaussian noise to sinogram projection data based on a target SNR."""
    if seed is not None:
        torch.manual_seed(seed)
    
    is_list = isinstance(proj_data, list)
    if is_list:
        data_tensor = torch.stack(proj_data)
    else:
        data_tensor = proj_data
    
    # Compute signal power (mean square of signal)
    signal_power = torch.mean(data_tensor ** 2, dim=1, keepdim=True)
    
    # Calculate required noise standard deviation per time
    noise_std = torch.sqrt(signal_power * (10.0 ** (-snr_db / 10.0)))
    
    # Broadcast noise scale across detector elements
    noise = torch.randn_like(data_tensor) * noise_std
    noisy_tensor = data_tensor + noise
    
    if is_list:
        return [noisy_tensor[i] for i in range(noisy_tensor.shape[0])]
    return noisy_tensor

def compute_dataset_identifiability(
    r_maxs_list: list[torch.Tensor],
    sigmas_list: list[torch.Tensor] | None = None,
    default_sigma: float = 0.05,
) -> dict[str, float]:
    """Computes peak separability metrics aggregated across all pairs and time steps.

    Returns:
        dict containing:
            - 'I_min': Global minimum separability (used for Regime A/B/C classification).
            - 'I_mean': Mean separability across all pairs and time steps.
            - 'overlap_fraction': Ratio of time frames with overlapping peaks (I < 1.0).
    """
    N = len(r_maxs_list)
    if N <= 1:
        return {"I_min": float('inf'), "I_mean": float('inf'), "overlap_fraction": 0.0}     # Single particle is trivially identifiable
    
    T = r_maxs_list[0].shape[0]
    
    # Extract modes into a single tensor: shape [N, T]
    mus = torch.stack([r_maxs_list[i][:, 1] for i in range(N)], dim=0)
    
    if sigmas_list is not None:
        sigmas = torch.stack([sigmas_list[i] for i in range(N)], dim=0)
    else:
        sigmas = torch.full_like(mus, default_sigma)

    # Vectorized pairwise upper-triangular indices
    i_idx, j_idx = torch.triu_indices(N, N, offset=1)
    
    # Pairwise separability matrix across all pairs and time steps: shape [N_pairs, T]
    delta_mu = torch.abs(mus[i_idx, :] - mus[j_idx, :])     # [N_pairs, T]
    sum_sigma = sigmas[i_idx, :] + sigmas[j_idx, :]         # [N_pairs, T]
    
    separability_matrix = delta_mu / sum_sigma              # [N_pairs, T]
    
    I_min = separability_matrix.min().item()
    I_mean = separability_matrix.mean().item()
    overlap_fraction = (separability_matrix < 1.0).float().mean().item()

    return {
        "I_min": I_min,
        "I_mean": I_mean,
        "overlap_fraction": overlap_fraction,
    }

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