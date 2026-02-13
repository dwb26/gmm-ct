
"""
    Main driver code for the 2-dimensional Gaussian mixture model application.
"""

from datetime import datetime
from pathlib import Path

from gmm_ct import GMM_reco, generate_true_param, construct_receivers, set_random_seeds, export_parameters
from gmm_ct.config.defaults import GRAVITATIONAL_ACCELERATION
from gmm_ct.visualization.publication import ( 
    plot_individual_gaussian_reconstruction,
    plot_temporal_gmm_comparison, 
    animate_temporal_gmm_comparison,
    reorder_theta_to_match_true
)
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

def compute_parameter_errors(theta_true, theta_est, N, detailed=False):
    """
    Compute relative L2 errors for each parameter type across all k Gaussians.
    
    Parameters:
        theta_true: True parameters dictionary
        theta_est: Estimated parameters dictionary
        N: Number of Gaussians
        detailed: If True, also return per-Gaussian errors
    
    Returns:
        dict: Relative errors for each parameter type
        dict (optional): Per-Gaussian errors if detailed=True
    """
    errors = {}
    per_gaussian = {} if detailed else None
    
    # Alphas (N scalar values)
    alphas_true = torch.stack([theta_true['alphas'][k] for k in range(N)])
    alphas_est = torch.stack([theta_est['alphas'][k] for k in range(N)])
    errors['alphas'] = (torch.norm(alphas_true - alphas_est) / torch.norm(alphas_true)).item()
    if detailed:
        per_gaussian['alphas'] = [(abs((theta_true['alphas'][k] - theta_est['alphas'][k]).item()) / 
                                   abs(theta_true['alphas'][k].item())) for k in range(N)]
    
    # x0s (N vectors of dimension d)
    x0s_true = torch.stack([theta_true['x0s'][k] for k in range(N)])
    x0s_est = torch.stack([theta_est['x0s'][k] for k in range(N)])
    errors['x0s'] = (torch.norm(x0s_true - x0s_est) / torch.norm(x0s_true)).item()
    
    # v0s (N vectors of dimension d)
    v0s_true = torch.stack([theta_true['v0s'][k] for k in range(N)])
    v0s_est = torch.stack([theta_est['v0s'][k] for k in range(N)])
    errors['v0s'] = (torch.norm(v0s_true - v0s_est) / torch.norm(v0s_true)).item()
    if detailed:
        per_gaussian['v0s'] = [(torch.norm(theta_true['v0s'][k] - theta_est['v0s'][k]) / 
                               torch.norm(theta_true['v0s'][k])).item() for k in range(N)]
    
    # U_skews (N matrices of dimension d x d)
    U_skews_true = torch.stack([theta_true['U_skews'][k].flatten() for k in range(N)])
    U_skews_est = torch.stack([theta_est['U_skews'][k].flatten() for k in range(N)])
    errors['U_skews'] = (torch.norm(U_skews_true - U_skews_est) / torch.norm(U_skews_true)).item()
    if detailed:
        per_gaussian['U_skews'] = [(torch.norm(theta_true['U_skews'][k] - theta_est['U_skews'][k], p='fro') / 
                                   torch.norm(theta_true['U_skews'][k], p='fro')).item() for k in range(N)]
    
    # omegas (N vectors)
    omegas_true = torch.stack([theta_true['omegas'][k] for k in range(N)])
    omegas_est = torch.stack([theta_est['omegas'][k] for k in range(N)])
    errors['omegas'] = (torch.norm(omegas_true - omegas_est) / torch.norm(omegas_true)).item()
    if detailed:
        per_gaussian['omegas'] = [(torch.norm(theta_true['omegas'][k] - theta_est['omegas'][k]) / 
                                  torch.norm(theta_true['omegas'][k])).item() for k in range(N)]
    
    if detailed:
        return errors, per_gaussian
    return errors


def compute_projection_error(proj_true, proj_est):
    """
    Compute relative L2 error for projections.
    
    Parameters:
        proj_true: True projections (list of tensors)
        proj_est: Estimated projections (list of tensors)
    
    Returns:
        float: Relative projection error
    """
    proj_true_flat = torch.cat([p.flatten() for p in proj_true])
    proj_est_flat = torch.cat([p.flatten() for p in proj_est])
    return (torch.norm(proj_true_flat - proj_est_flat) / torch.norm(proj_true_flat)).item()


def plot_error_analysis_table(param_errors_init, param_errors_final, proj_error_init, proj_error_final, output_path):
    """
    Create a publication-quality table showing error evolution from initialization to termination.
    
    Parameters:
        param_errors_init: Dictionary of initial parameter errors
        param_errors_final: Dictionary of final parameter errors
        proj_error_init: Initial projection error
        proj_error_final: Final projection error
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Parameter Type', 'Initial Error', 'Final Error', 'Improvement (%)', 'Error Reduction'])
    
    # Parameter labels
    param_labels = {
        'alphas': 'Amplitudes (α)',
        'x0s': 'Initial Positions (x₀)',
        'v0s': 'Initial Velocities (v₀)',
        'U_skews': 'Shape Matrices (U)',
        'omegas': 'Angular Velocities (ω)'
    }
    
    # Add parameter rows
    for param_key in ['alphas', 'x0s', 'v0s', 'U_skews', 'omegas']:
        init_err = param_errors_init[param_key]
        final_err = param_errors_final[param_key]
        improvement = 100 * (1 - final_err / init_err) if init_err > 0 else 0
        reduction = init_err / final_err if final_err > 0 else np.inf
        
        table_data.append([
            param_labels[param_key],
            f'{init_err:.4e}',
            f'{final_err:.4e}',
            f'{improvement:.1f}%',
            f'{reduction:.1f}×'
        ])
    
    # Add projection row
    proj_improvement = 100 * (1 - proj_error_final / proj_error_init) if proj_error_init > 0 else 0
    proj_reduction = proj_error_init / proj_error_final if proj_error_final > 0 else np.inf
    table_data.append([
        'Projections',
        f'{proj_error_init:.4e}',
        f'{proj_error_final:.4e}',
        f'{proj_improvement:.1f}%',
        f'{proj_reduction:.1f}×'
    ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', fontsize=13)
    
    # Style projection row (last row)
    for i in range(len(table_data[0])):
        cell = table[(len(table_data)-1, i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold')
    
    # Alternate row colors for parameters
    for i in range(1, len(table_data)-1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    ax.set_title('Error Analysis: Initialization vs Optimization', 
                fontweight='bold', fontsize=22, pad=15)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Error analysis table saved to: {output_path}")
    plt.close()


start_time = time()
starting_seed = 10
starting_seed = 40
# starting_seed = 99
# starting_seed = 90
N_simulations = 1
RANDOM_SEEDS = range(starting_seed, starting_seed + N_simulations)

for RANDOM_SEED in RANDOM_SEEDS:

    # Set seeds for reproducibility
    set_random_seeds(RANDOM_SEED)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    print(f"🚀 Generating data using random seed: {RANDOM_SEED}")


    # ------------- #
    'Hyperparameters'
    # ------------- #
    d = 2                                                                       # Dimensionality of the application
    # N = 10                                                                      # Number of Gaussians in the GMM
    N = 2                                                                       # Number of Gaussians in the GMM
    N_projs = 2**6 + 1                                                          # Number of projections
    t = torch.linspace(0., 2.0, N_projs, dtype=torch.float64, device=device)    # Considered time window (seconds)

    # Create experiment-specific output directory at project root
    project_root = Path(__file__).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = project_root / 'plots' / f"{timestamp}_seed{RANDOM_SEED}_N{N}"
    experiment_dir.mkdir(parents=True, exist_ok=True)


    # ------------------------------------- #
    'Source/receivers location specification'
    # ------------------------------------- #
    # Source construction
    n_sources = 1
    if n_sources == 1:
        sources = [torch.tensor([-1, -1], dtype=torch.float64, device=device)]
    else:
        raise NotImplementedError("Multiple sources not yet implemented.")
    
    # Receiver construction
    n_rcvrs = 128
    x1 = sources[0][0].item() + 5. # Depth of receivers
    x2_min = sources[0][1].item() - 2.; x2_max = sources[0][1].item() + 2. # Height range of receivers
    args = (n_rcvrs, x1, x2_min, x2_max)
    rcvrs = construct_receivers(device, args)


    # -------------------------------------------- #
    'Generate true parameters and their projections'
    # -------------------------------------------- #
    i_loc = torch.tensor([1., 1.], dtype=torch.float64, device=device)                              # Initial location (metres)
    v_loc = torch.tensor([.75, .5], dtype=torch.float64, device=device)                             # Initial velocity (metres/second)
    a_loc = torch.tensor([0., -GRAVITATIONAL_ACCELERATION], dtype=torch.float64, device=device)     # Initial acceleration (metres/second^2)
    # omega_min = -24.0; omega_max = omega_min + 4.0                                                  # Minimum and maximum angular velocities (rots/second)
    omega_min = -24.0; omega_max = omega_min + 8.0                                                  # Minimum and maximum angular velocities (rots/second)
    
    theta_true = generate_true_param(d, N, i_loc, v_loc, a_loc, omega_min, omega_max, device=device)
    
    # Extract known physical parameters (x0s and a0s are assumed known)
    x0s = theta_true['x0s']
    a0s = theta_true['a0s']
        
    GMM_true = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=experiment_dir)
    proj_data = GMM_true.generate_projections(t, theta_true)
    

    # ----------------------------------- #
    'Reconstruction - PyTorch optimization'
    # ----------------------------------- #
    GMM = GMM_reco(d, N, sources, rcvrs, x0s, a0s, omega_min, omega_max, device=device, output_dir=experiment_dir)
    soln_dict = GMM.fit(proj_data, t)
    theta_dict_init = GMM.theta_dict_init
    
    # Reorder estimated parameters to match true Gaussians (for consistent visualization)
    soln_dict, matching_indices = reorder_theta_to_match_true(theta_true, soln_dict, N)
    print(f"\n🎯 Matched estimated Gaussians to true Gaussians: {matching_indices}")
 
    # Export true and estimated parameters to files
    export_parameters(theta_true, experiment_dir / "true_parameters.md", title="Ground Truth Parameters")
    export_parameters(soln_dict, experiment_dir / "estimated_parameters.md", title="Estimated Parameters", theta_true=theta_true, theta_init=theta_dict_init)
    
    # Compare the fitted omegas to the true omegas
    print("\n🔍 Comparing true and estimated angular velocities (omegas):")
    for k in range(N):
        true_omega = theta_true['omegas'][k].item()
        est_omega = soln_dict['omegas'][k].item()
        error = abs(true_omega - est_omega)
        print(f"  Gaussian {k}: True ω = {true_omega:.4f}, Estimated ω = {est_omega:.4f}, Absolute Error = {error:.4e}")


    # ------------------- #
    'Error Analysis Table'
    # ------------------- #
    # print("\n" + "="*50)
    # print("Computing error analysis...")
    # print("="*50)
    
    # # Compute parameter errors at initialization and termination
    # param_errors_init, per_gauss_init = compute_parameter_errors(theta_true, theta_dict_init, N, detailed=True)
    # param_errors_final, per_gauss_final = compute_parameter_errors(theta_true, soln_dict, N, detailed=True)
    
    # # Compute projection errors
    # proj_init = GMM.generate_projections(t, theta_dict_init)
    # proj_final = GMM.generate_projections(t, soln_dict)
    # proj_error_init = compute_projection_error(proj_data, proj_init)
    # proj_error_final = compute_projection_error(proj_data, proj_final)
    
    # # Print summary to console
    # print("\n📊 Error Summary:")
    # print("\nParameter Errors (Relative L2):")
    # has_divergence = False
    # divergent_params = []
    # for param_key in ['alphas', 'x0s', 'v0s', 'U_skews', 'omegas']:
    #     init = param_errors_init[param_key]
    #     final = param_errors_final[param_key]
    #     improvement = 100 * (1 - final/init) if init > 0 else 0
        
    #     # Check if optimization made things significantly worse
    #     if final > init * 1.2:  # More than 20% worse
    #         status = " ⚠️ WORSE"
    #         has_divergence = True
    #         divergent_params.append(param_key)
    #     else:
    #         status = ""
        
    #     print(f"  {param_key:12s}: {init:.4e} → {final:.4e} ({improvement:+.1f}% improvement){status}")
    
    # print("\nProjection Error (Relative L2):")
    # proj_improvement = 100 * (1 - proj_error_final/proj_error_init) if proj_error_init > 0 else 0
    # print(f"  Projections : {proj_error_init:.4e} → {proj_error_final:.4e} ({proj_improvement:+.1f}% improvement)")
    
    # # If divergence detected, print per-Gaussian diagnostics
    # if has_divergence:
    #     print("\n" + "="*70)
    #     print("⚠️  DIVERGENCE DETECTED - Per-Gaussian Error Analysis")
    #     print("="*70)
    #     for param_key in divergent_params:
    #         if param_key == 'x0s':
    #             continue  # Skip x0s since they're fixed
    #         print(f"\n{param_key.upper()}:")
    #         init_errs = per_gauss_init[param_key]
    #         final_errs = per_gauss_final[param_key]
            
    #         # Sort by worst final error to highlight problematic Gaussians
    #         sorted_indices = sorted(range(N), key=lambda k: final_errs[k], reverse=True)
            
    #         for k in sorted_indices:
    #             init_err = init_errs[k]
    #             final_err = final_errs[k]
    #             change = 100 * (1 - final_err/init_err) if init_err > 0 else 0
                
    #             # Flag badly diverged Gaussians
    #             if final_err > init_err * 2.0:
    #                 flag = " ❌ DIVERGED"
    #             elif final_err > init_err * 1.2:
    #                 flag = " ⚠️ Worse"
    #             else:
    #                 flag = ""
                
    #             print(f"  Gaussian {k}: {init_err:.4e} → {final_err:.4e} ({change:+6.1f}%){flag}")
    #     print("="*70)
    
    # # Generate error analysis table
    # plot_error_analysis_table(param_errors_init, param_errors_final, 
    #                          proj_error_init, proj_error_final,
    #                          experiment_dir / "error_analysis.pdf")


    # ------ #
    'Plotting'
    # ------ #
    plot_individual_gaussian_reconstruction(theta_true, soln_dict, N, d, gaussian_indices=range(N), filename=experiment_dir / "individual_gaussian_reconstruction.pdf")
    plot_temporal_gmm_comparison(sources, rcvrs, theta_true, theta_dict_init, t, N, d, 
                                 time_indices=[17, 20, 22],
                                 filename=experiment_dir / "initial_temporal_gmm_comparison.pdf", title='Initial Guesses')
    plot_temporal_gmm_comparison(sources, rcvrs, theta_true, soln_dict, t, N, d, 
                                 time_indices=[17, 20, 22],
                                 filename=experiment_dir / "temporal_gmm_comparison.pdf")
    animate_temporal_gmm_comparison(sources, rcvrs, theta_true, soln_dict, t, N, d, filename=experiment_dir / "temporal_gmm_comparison.mp4")

print("Recovery results are ready.")
print(f"Total experiment time: {time() - start_time:.2f} seconds.")
print(f"Time per experiment: {(time() - start_time) / N_simulations:.2f} seconds.")