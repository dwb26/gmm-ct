"""
Parameter and data generation utilities for GMM CT reconstruction.

This module contains functions for generating synthetic data and parameters
for testing and validation of the GMM reconstruction algorithm.
"""

import math
import warnings
import torch

def generate_true_param(d, K, initial_location, initial_velocity, initial_acceleration, min_rot, max_rot, device=None, sampling_dt=None, min_velocity_separation=0.5, min_diag_ratio=1.5):
    """
    Generate synthetic "true" parameters for GMM reconstruction testing.
    
    Creates a complete set of GMM parameters including attenuation coefficients,
    skewness matrices, rotation parameters, and projectile motion parameters.
    
    Parameters
    ----------
    d : int
        Dimensionality of the problem (typically 2)
    K : int
        Number of Gaussian components
    initial_location : torch.Tensor
        Initial location vector (d-dimensional)
    initial_velocity : torch.Tensor
        Initial velocity vector (d-dimensional)
    initial_acceleration : torch.Tensor
        Initial acceleration vector (d-dimensional)
    min_rot : float
        Minimum angular velocity for rotation
    max_rot : float
        Maximum angular velocity for rotation
    device : torch.device, optional
        Device to place tensors on (default: CPU)
    sampling_dt : float, optional
        Time interval between projections.  When provided, generated
        angular velocities are checked against the Nyquist-like
        aliasing condition for 2-D ellipses (π-symmetry): any ω
        for which ``|2πωΔt mod π| < ε`` would make the Gaussian
        appear non-rotating and is therefore rejected and re-sampled.
    min_velocity_separation : float, optional
        Minimum Euclidean distance between any two initial velocity vectors
        (in world-space units per second).  Since all Gaussians share the
        same initial position and acceleration, the separation between
        trajectories k and l grows as ``‖v0_k − v0_l‖ · t``, so this
        threshold directly controls the minimum trajectory divergence rate.
        Velocities are sampled via accept/reject: the first component is
        drawn freely; each subsequent candidate is accepted only when its
        distance to every already-accepted velocity exceeds this threshold.
        Default: 0.5.
    min_diag_ratio : float, optional
        Minimum ratio ``max(diag(U)) / min(diag(U))`` enforced on each
        generated U_skew matrix.  Near-isotropic Gaussians (ratio ≈ 1)
        have very weak rotation signatures in the projections, making ω
        unidentifiable in practice.  Candidates are rejection-sampled
        until the diagonal aspect ratio exceeds this threshold.
        Default: 1.5.
        
    Returns
    -------
    dict
        Dictionary containing:
        - alphas: List of attenuation coefficients
        - U_skews: List of skewness matrices
        - omegas: List of angular velocities
        - x0s: List of initial positions
        - v0s: List of initial velocities
        - a0s: List of initial accelerations
        
    Raises
    ------
    ValueError
        If initial_location, initial_velocity, or initial_acceleration don't match dimensionality d
    """
    if device is None:
        device = torch.device('cpu')
    if len(initial_location) != d:
        raise ValueError("Initial location must match the dimensionality d.")
    if len(initial_velocity) != d:
        raise ValueError("Initial velocity must match the dimensionality d.")
    if len(initial_acceleration) != d:
        raise ValueError("Initial acceleration must match the dimensionality d.")
    
    # Attenuation coefficients
    alphas = [
        torch.tensor(15., dtype=torch.float64, device=device) + 5*k + 
        torch.randn(1, dtype=torch.float64, device=device) 
        for k in range(K)
    ]

    # Skewness matrices — rejection-sample to enforce minimum anisotropy.
    # Near-isotropic Gaussians (diag ratio ≈ 1) have negligible rotation
    # signature in the projections, making ω unidentifiable.
    U_ks = []
    for _ in range(K):
        for _u_attempt in range(500):
            # Diagonal entries — must be strictly positive
            mean_diag_val = 7.5
            U_k_diag = torch.rand(size=(d,), dtype=torch.float64, device=device) * 18.0 + mean_diag_val
            U_k_diag = torch.abs(U_k_diag)

            # Enforce minimum diagonal aspect ratio
            diag_ratio = (U_k_diag.max() / U_k_diag.min()).item()
            if diag_ratio < min_diag_ratio:
                continue

            # Upper triangular entries — can take any value
            U_k_upper_triangle = 10 + torch.randn(
                size=((d - 1) * d // 2,), dtype=torch.float64, device=device
            )

            # Assemble upper-triangular matrix
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
                    U_k[i, j] = U_k_upper_triangle[upper_idx]
                    upper_idx += 1
            break
        else:
            warnings.warn(
                f"Could not generate a U_skew with diagonal ratio >= {min_diag_ratio} "
                "after 500 attempts. Accepting last sample.",
                RuntimeWarning, stacklevel=2,
            )
        U_ks.append(U_k)

    alias_buffer = 0.10  # fractional guard band (±10 % of the π period)
    def _is_aliased(omega_val):
        """Return True if omega would appear non-rotating at the given dt."""
        if sampling_dt is None:
            return False
        frac = abs(omega_val * 2 * sampling_dt) % 1  # fraction of a half-turn
        return frac < alias_buffer or frac > 1 - alias_buffer

    omegas = []
    for k in range(K):
        for _attempt in range(200):
            omega_k = (
                max_rot
                - torch.rand(size=(math.comb(d, 2),), dtype=torch.float64,
                             device=device)
                * (max_rot - min_rot)
            )
            if not any(_is_aliased(w.item()) for w in omega_k):
                break
        omegas.append(omega_k)
    
    # Projectile motion parameters
    # Initial locations (assumed known for all Gaussians)
    x0s = [initial_location.to(torch.float64) for _ in range(K)]

    # Initial velocities — accept/reject to enforce minimum pairwise separation.
    # Since all Gaussians share x0 and a0, trajectory separation grows as
    # ‖v0_k − v0_l‖ · t, so enforcing a minimum velocity gap guarantees
    # trajectories diverge at a physically realistic rate.
    def _sample_velocity():
        v_h = initial_velocity[0] + torch.rand(1, dtype=torch.float64, device=device).item() * 1.5
        v_v = (torch.rand(1, dtype=torch.float64, device=device).item() - 0.5) * 4.5
        return torch.tensor([v_h, v_v], dtype=torch.float64, device=device)

    _max_attempts = 500
    v0s = []
    generate = False
    if generate:
        for k in range(K):
            if k == 0:
                # First velocity: accept freely
                v0s.append(_sample_velocity())
                continue
            for attempt in range(_max_attempts):
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
                    f"min_velocity_separation={min_velocity_separation} after "
                    f"{_max_attempts} attempts. Accepting last candidate anyway. "
                    "Consider reducing min_velocity_separation or the number of components.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                v0s.append(candidate)
    else:
        v0s = [torch.tensor([1.0, 1+2.0], dtype=torch.float64, device=device),
               torch.tensor([1.5, 0+1.8], dtype=torch.float64, device=device), 
               torch.tensor([0.8, 1+1.5], dtype=torch.float64, device=device), 
               torch.tensor([0.75, 0+1.2], dtype=torch.float64, device=device), 
               torch.tensor([2.0, 1+2.0], dtype=torch.float64, device=device)]
    
    # Initial acceleration (assumed known for all Gaussians)
    a0s = [initial_acceleration.to(torch.float64) for _ in range(K)]
    
    return {
        "alphas": alphas, 
        "U_skews": U_ks, 
        "omegas": omegas, 
        "x0s": x0s, 
        "v0s": v0s, 
        "a0s": a0s
    }
