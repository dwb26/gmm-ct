"""
Helper utilities for GMM CT reconstruction.

This module contains miscellaneous helper functions including random seed management,
parameter export, and other utility operations.
"""

import torch
import numpy as np
from datetime import datetime


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across all random number generators.
    
    This ensures that results are reproducible across different runs by setting
    seeds for PyTorch, NumPy, and the global random number generator.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value (default: 42)
        
    Notes
    -----
    This function modifies global state in PyTorch and NumPy. It should be called
    at the start of your script before any random operations.
    """
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Update global random number generator
    rng = np.random.default_rng(seed)
    
    print(f"🎲 Random seeds set to {seed} for reproducibility")
    print("   ✅ PyTorch seed set")
    print("   ✅ NumPy seed set") 
    print("   ✅ Global random generator updated")
    
    return rng
    
    
def export_parameters(theta_dict, filename, title="GMM Parameters", theta_true=None, theta_init=None):
    """
    Export GMM parameters to a human-readable Markdown file.
    
    Creates a formatted Markdown document showing parameter values, optionally
    including initial guesses and errors compared to ground truth.
    
    Parameters
    ----------
    theta_dict : dict
        Dictionary containing the GMM parameters (as PyTorch tensors)
    filename : str
        Output filename for the .md file
    title : str, optional
        Main title for the Markdown document (default: "GMM Parameters")
    theta_true : dict, optional
        Dictionary with ground truth parameters for error computation
    theta_init : dict, optional
        Dictionary with initial guess parameters to display
        
    Notes
    -----
    The exported file includes:
    - Timestamp of export
    - Overall Gaussian errors (if theta_true provided)
    - Detailed parameter tables with values, initial guesses, and errors
    - Formatted matrices for U_skews parameters
    """
    with open(filename, 'w') as f:
        f.write(f"# {title}\n\n")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"*Exported on: {timestamp}*\n\n")

        # Table for overall Gaussian errors
        if theta_true:
            f.write("## Overall Gaussian Errors\n\n")
            f.write("| Gaussian Index | Absolute Error |\n")
            f.write("|----------------|----------------|\n")
            
            K = len(theta_dict.get('v0s', []))
            for i in range(K):
                error = 0
                
                # v0s error (L2 norm)
                if 'v0s' in theta_true and 'v0s' in theta_dict:
                    v0_true = theta_true['v0s'][i].detach().cpu().numpy()
                    v0_est = theta_dict['v0s'][i].detach().cpu().numpy()
                    error += np.linalg.norm(v0_est - v0_true)
                
                # omegas error (absolute difference)
                if 'omegas' in theta_true and 'omegas' in theta_dict:
                    omega_true = theta_true['omegas'][i].item()
                    omega_est = theta_dict['omegas'][i].item()
                    error += np.abs(omega_est - omega_true)
                
                # alphas error (absolute difference)
                if 'alphas' in theta_true and 'alphas' in theta_dict:
                    alpha_true = theta_true['alphas'][i].item()
                    alpha_est = theta_dict['alphas'][i].item()
                    error += np.abs(alpha_est - alpha_true)

                # U_skews error (Frobenius norm)
                if 'U_skews' in theta_true and 'U_skews' in theta_dict:
                    U_true = theta_true['U_skews'][i].detach().cpu().numpy()
                    U_est = theta_dict['U_skews'][i].detach().cpu().numpy()
                    error += np.linalg.norm(U_est - U_true)

                f.write(f"| {i+1:<14} | {error:.4f}         |\n")
            f.write("\n")

        for key, value in theta_dict.items():
            f.write(f"## `{key}`\n\n")

            if not isinstance(value, list) or not value:
                f.write(f"```\n{value}\n```\n\n")
                continue

            # Handle lists of tensors
            if isinstance(value[0], torch.Tensor):
                np_values = [v.detach().cpu().numpy() for v in value]
                
                # Prepare headers and data for init and error columns
                init_values = None
                if theta_init and key in theta_init:
                    init_values = [v.detach().cpu().numpy() for v in theta_init[key]]

                error_values = None
                if theta_true and key in theta_true:
                    true_values = [v.detach().cpu().numpy() for v in theta_true[key]]
                    if np_values and (np_values[0].ndim == 0 or np_values[0].size == 1):  # scalar
                        error_values = [np.abs(est - true) for est, true in zip(np_values, true_values)]
                    elif np_values:  # vector or matrix
                        error_values = [np.linalg.norm(est - true) for est, true in zip(np_values, true_values)]

                # Scalar parameters (e.g., alphas)
                if np_values and (np_values[0].ndim == 0 or np_values[0].size == 1):
                    header = "| Gaussian Index | Value "
                    separator = "|----------------|-------"
                    if init_values:
                        header += "| Initial Value "
                        separator += "|---------------"
                    if error_values:
                        header += "| Abs. Error "
                        separator += "|------------"
                    header += "|\n"
                    separator += "|\n"
                    
                    f.write(header)
                    f.write(separator)
                    for i, val in enumerate(np_values):
                        f.write(f"| {i+1:<14} | {val.item():.4f} ")
                        if init_values and i < len(init_values):
                            f.write(f"| {init_values[i].item():.4f}      ")
                        if error_values and i < len(error_values):
                            f.write(f"| {error_values[i].item():.4f}   ")
                        f.write("|\n")
                    f.write("\n")

                # Vector parameters (e.g., v0s, omegas)
                elif np_values and np_values[0].ndim == 1:
                    num_components = np_values[0].shape[0]
                    header = "| Gaussian Index | " + " | ".join([f"Component {j+1}" for j in range(num_components)])
                    separator = "|-" + "---------------|-" * num_components
                    if init_values:
                        header += " | Initial Value"
                        separator += "|---------------"
                    if error_values:
                        header += " | Abs. Error (L2)"
                        separator += "|-----------------"
                    header += "|\n"
                    separator += "|\n"
                    f.write(header)
                    f.write(separator)
                    for i, vec in enumerate(np_values):
                        row_data = " | ".join([f"{comp:.4f}" for comp in vec])
                        f.write(f"| {i+1:<14} | {row_data}")
                        if init_values and i < len(init_values):
                            init_row_data = " ".join([f"{comp:.2f}" for comp in init_values[i]])
                            f.write(f" | {init_row_data} ")
                        if error_values and i < len(error_values):
                            f.write(f" | {error_values[i]:.4f}          ")

                        f.write("|\n")
                    f.write("\n")

                # Matrix parameters (e.g., U_skews)
                elif np_values and np_values[0].ndim == 2:
                    for i, matrix in enumerate(np_values):
                        f.write(f"### Gaussian {i+1}\n")
                        if init_values and i < len(init_values):
                            f.write("#### Initial Value\n")
                            f.write("```\n")
                            f.write(np.array2string(init_values[i], precision=4, separator=', '))
                            f.write("\n```\n\n")

                        f.write("#### Estimated Value\n")
                        f.write("```\n")
                        f.write(np.array2string(matrix, precision=4, separator=', '))
                        f.write("\n```\n\n")
                        
                        if error_values and i < len(error_values):
                            f.write(f"**Absolute Error (Frobenius Norm):** {error_values[i]:.4f}\n\n")
    
    print(f"✓ Parameters successfully exported to Markdown: {filename}")
