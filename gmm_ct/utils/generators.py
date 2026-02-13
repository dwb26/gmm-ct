"""
Parameter and data generation utilities for GMM CT reconstruction.

This module contains functions for generating synthetic data and parameters
for testing and validation of the GMM reconstruction algorithm.
"""

import math
import torch
import numpy as np

from ..config.defaults import GRAVITATIONAL_ACCELERATION


def generate_true_param(d, K, initial_location, initial_velocity, initial_acceleration, min_rot, max_rot, device=None):
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

    # Skewness matrices
    U_ks = []
    for _ in range(K):
        # Diagonal entries - must be strictly positive
        mean_diag_val = 7.5
        U_k_diag = torch.rand(size=(d,), dtype=torch.float64, device=device) * 18.0 + mean_diag_val
        U_k_diag = torch.abs(U_k_diag)
        
        # Upper triangular entries - these can take any value
        U_k_upper_triangle = 10 + torch.randn(size=((d - 1) * d // 2,), dtype=torch.float64, device=device)
        
        # Set the entries into the upper triangular U_skew matrix
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
        U_ks.append(U_k)

    # Rotation parameters
    omegas = [
        max_rot - torch.rand(size=(math.comb(d, 2),), dtype=torch.float64, device=device) * (max_rot - min_rot) 
        for _ in range(K)
    ]
    print("Generated angular velocities (omegas):")
    for k in range(K):
        print(f"Gaussian {k+1}: {omegas[k].cpu().numpy()}")
    
    # Projectile motion parameters
    # Initial locations (assumed known for all Gaussians)
    x0s = [initial_location.to(torch.float64) for _ in range(K)]

    # Initial velocities - Simple uniform distribution with increased vertical diversity
    v0s = []
    for _ in range(K):
        # Horizontal component: uniformly distributed, always positive
        v_horizontal = initial_velocity[0] + torch.rand(1, dtype=torch.float64, device=device).item() * 1.5
        
        # Vertical component: uniformly distributed over wider range for more diversity
        v_vertical = (torch.rand(1, dtype=torch.float64, device=device).item() - 0.5) * 4.5
        v0s.append(torch.tensor([v_horizontal, v_vertical], dtype=torch.float64, device=device))
    
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
