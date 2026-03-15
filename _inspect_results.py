import torch, numpy as np

data = torch.load('data/results/20260315_104536_seed7_N5/results.pt', weights_only=False)
theta_true = data['theta_true']
theta_est  = data['theta_est']

N = len(theta_true['omegas'])
print(f'N = {N}')

for k in range(N):
    print(f'\n--- Gaussian {k} ---')
    omega_t = theta_true['omegas'][k].item()
    omega_e = theta_est['omegas'][k].item()
    alpha_t = theta_true['alphas'][k].item()
    alpha_e = theta_est['alphas'][k].item()
    v0_t = theta_true['v0s'][k].detach().numpy()
    v0_e = theta_est['v0s'][k].detach().numpy()
    U_t  = theta_true['U_skews'][k].detach().numpy()
    U_e  = theta_est['U_skews'][k].detach().numpy()
    print(f'  omega  true={omega_t:.4f}  est={omega_e:.4f}  err={abs(omega_e-omega_t):.4f}')
    print(f'  alpha  true={alpha_t:.4f}  est={alpha_e:.4f}  err={abs(alpha_e-alpha_t):.4f}')
    print(f'  v0     true={v0_t}  est={v0_e}')
    print(f'  v0_err {np.abs(v0_e-v0_t)}')
    print(f'  U_true=\n{U_t}')
    print(f'  U_est=\n{U_e}')
    print(f'  U_err (abs)=\n{np.abs(U_e - U_t)}')

    # Decompose U into scale and rotation angle
    # U = D @ R  where D = diag, R = rotation matrix encoded in off-diag
    # For 2D upper-triangular: rotation angle from atan2(U[0,1], U[0,0])
    angle_t = np.degrees(np.arctan2(U_t[0,1], U_t[0,0]))
    angle_e = np.degrees(np.arctan2(U_e[0,1], U_e[0,0]))
    print(f'  implied angle true={angle_t:.2f} deg  est={angle_e:.2f} deg  err={abs(angle_e-angle_t):.2f} deg')
