"""
Post-hoc single-Gaussian polish.

For a saved results.pt, isolates one Gaussian by subtracting the estimated
contributions of all OTHER Gaussians from the observed sinogram (residual
sinogram), then runs a dense omega grid search followed by joint L-BFGS
refinement of (omega, U_skew, alpha) for that Gaussian alone.

Usage
-----
    python scripts/polish_single_gaussian.py <results_dir> [--gaussian-index K]

If --gaussian-index is omitted the script polishes the Gaussian with the
largest omega error (identified automatically via the Hungarian matching).

Example
-------
    python scripts/polish_single_gaussian.py \
        data/results/20260315_104536_seed7_N5
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchmin import minimize

from gmm_ct import GMM_reco
from gmm_ct.visualization.publication import reorder_theta_to_match_true


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dev(x, device):
    return x.to(device) if hasattr(x, 'to') else torch.tensor(x, device=device)


def _clone_dict(d):
    return {
        k: [v.clone().detach() for v in vals]
        for k, vals in d.items()
    }


def _build_model(data, device):
    cfg = data['config']
    sources   = [_dev(s, device) for s in data['sources']]
    receivers = [[_dev(r, device) for r in row] for row in data['receivers']]
    N         = cfg['N']
    theta_est = data['theta_est']
    x0s = [_dev(x, device) for x in theta_est['x0s']]
    a0s = [_dev(a, device) for a in theta_est['a0s']]
    model = GMM_reco(
        d=cfg['d'], N=N,
        sources=sources, receivers=receivers,
        x0s=x0s, a0s=a0s,
        omega_min=cfg.get('omega_min', 2.0),
        omega_max=cfg.get('omega_max', 6.0),
        device=device,
        output_dir=Path('/tmp/polish_dummy'),
    )
    return model, N


def _to_device_dict(theta, device):
    return {
        k: [_dev(v, device) for v in vals]
        for k, vals in theta.items()
    }


def _proj_to_tensor(proj_list):
    """Stack list-of-source projections into a 2-D tensor (T × R)."""
    return proj_list[0]  # single source


def identify_poorly_fitted_gaussians(model, theta_est, proj_obs, t, N, device):
    """
    For each Gaussian k, compute the residual sinogram
    (p_obs minus all other Gaussians' contributions) and measure how well
    p_k_est matches it.  Gaussians with high relative error are poorly fitted.

    Returns a list of dicts sorted by descending relative_err:
        [{'k': int, 'abs_err': float, 'rel_err': float, 'resid_norm': float}]
    """
    results = []
    for k in range(N):
        # Background: contributions from all Gaussians except k
        theta_bg = _clone_dict(theta_est)
        theta_bg['alphas'][k] = torch.zeros(1, dtype=torch.float64, device=device)
        with torch.no_grad():
            bg_proj = model.generate_projections(t, theta_bg)[0].double()

        resid = proj_obs.double() - bg_proj   # what Gaussian k should explain

        # k's own isolated projection
        theta_k_only = _clone_dict(theta_est)
        for j in range(N):
            if j != k:
                theta_k_only['alphas'][j] = torch.zeros(1, dtype=torch.float64, device=device)
        with torch.no_grad():
            proj_k = model.generate_projections(t, theta_k_only)[0].double()

        abs_err    = torch.norm(resid - proj_k).item()
        resid_norm = torch.norm(resid).item()
        rel_err    = abs_err / (resid_norm + 1e-10)
        results.append({'k': k, 'abs_err': abs_err, 'rel_err': rel_err,
                        'resid_norm': resid_norm})

    return sorted(results, key=lambda x: x['rel_err'], reverse=True)


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------

def polish_gaussian(results_path: Path, target_true_idx: int = None,
                    omega_step: float = 0.05, lbfgs_iters: int = 500,
                    device_str: str = None):
    """
    Load results.pt, identify the target Gaussian (worst omega error if
    target_true_idx is None), refine it via residual sinogram, save updated
    results.pt.
    """
    device = torch.device(
        device_str if device_str else
        ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Device: {device}")

    # --- Load data ---
    data = torch.load(results_path, map_location=device, weights_only=False)
    theta_true = _to_device_dict(data['theta_true'], device)
    theta_est  = _to_device_dict(data['theta_est'], device)
    proj_data_raw = data['proj_data']
    t = _dev(data['t'], device)
    N = data['config']['N']
    omega_min = data['config'].get('omega_min', 2.0)
    omega_max = data['config'].get('omega_max', 6.0)

    # proj_obs: (T, R) 
    if isinstance(proj_data_raw, (list, tuple)):
        proj_obs = _dev(proj_data_raw[0], device).double()
    else:
        proj_obs = _dev(proj_data_raw, device).double()

    # --- Apply reordering to identify labels ---
    theta_est_reordered, matching_indices = reorder_theta_to_match_true(
        theta_true, theta_est, N
    )
    # matching_indices[k_est] = k_true
    # inverse: est_for_true[k_true] = k_est
    est_for_true = [0] * N
    for k_est, k_true in enumerate(matching_indices):
        est_for_true[k_true] = k_est

    # --- Build model (just forward model — no peak detection needed) ---
    model, _ = _build_model(data, device)

    # --- Per-Gaussian projection discrepancy diagnostic ---
    print(f"\n  Per-Gaussian projection discrepancy (residual sinogram method):")
    diag = identify_poorly_fitted_gaussians(model, theta_est, proj_obs, t, N, device)
    for entry in diag:
        k_est = entry['k']
        k_true_labels = [kt for kt, ke in enumerate(est_for_true) if ke == k_est]
        lbl = f"rho_{k_true_labels[0]+1}" if k_true_labels else f"est_{k_est}"
        print(f"    {lbl} (est_idx={k_est}): rel_err={entry['rel_err']:.4f}  "
              f"abs_err={entry['abs_err']:.3e}  resid_norm={entry['resid_norm']:.3e}")

    # Identify which raw-est index to polish
    if target_true_idx is None:
        # Auto-select: Gaussian with worst projection discrepancy
        worst_k_est = diag[0]['k']
        target_true_labels = [kt for kt, ke in enumerate(est_for_true) if ke == worst_k_est]
        target_true_idx   = target_true_labels[0] if target_true_labels else 0
        print(f"\n  Auto-selected rho_{target_true_idx + 1} "
              f"(worst projection discrepancy: rel_err={diag[0]['rel_err']:.4f})")

    target_k_est = est_for_true[target_true_idx]
    print(f"  True index: {target_true_idx}  |  Raw-est index: {target_k_est}")
    print(f"  omega_true = {theta_true['omegas'][target_true_idx].item():.4f} Hz")
    print(f"  omega_est  = {theta_est['omegas'][target_k_est].item():.4f} Hz")
    print(f"  U_est      = {theta_est['U_skews'][target_k_est].cpu().numpy()}")

    # --- Compute residual sinogram ---
    theta_bg = _clone_dict(theta_est)
    theta_bg['alphas'][target_k_est] = torch.zeros(1, dtype=torch.float64, device=device)

    with torch.no_grad():
        proj_bg_list = model.generate_projections(t, theta_bg)
    proj_bg = _dev(proj_bg_list[0], device).double()

    proj_residual = proj_obs.double() - proj_bg  # (T, R): nearly pure rho_k signal
    print(f"\n  Residual sinogram range: [{proj_residual.min():.3f}, {proj_residual.max():.3f}]")

    # --- Dense omega grid search against residual ---
    print(f"\n  Grid search: omega in [{omega_min:.1f}, {omega_max:.1f}] "
          f"step={omega_step} Hz ...")

    omega_candidates = np.arange(omega_min, omega_max + omega_step * 0.5, omega_step)
    loss_func = nn.HuberLoss(delta=0.3)
    best_omega = theta_est['omegas'][target_k_est].item()
    best_loss  = float('inf')

    with torch.no_grad():
        for omega_val in omega_candidates:
            test_theta = _clone_dict(theta_est)
            test_theta['omegas'][target_k_est] = torch.tensor(
                [omega_val], dtype=torch.float64, device=device
            )
            # Zero out other alphas — project only rho_k
            test_theta_k_only = _clone_dict(test_theta)
            for j in range(N):
                if j != target_k_est:
                    test_theta_k_only['alphas'][j] = torch.zeros(
                        1, dtype=torch.float64, device=device
                    )
            proj_k_list = model.generate_projections(t, test_theta_k_only)
            proj_k = _dev(proj_k_list[0], device).double()
            loss_val = loss_func(proj_residual, proj_k).item()
            if loss_val < best_loss:
                best_loss  = loss_val
                best_omega = omega_val

    print(f"  Grid best omega: {best_omega:.4f} Hz  (loss={best_loss:.4e})")

    # --- Joint L-BFGS refinement: multi-start over U_skew initializations ---
    print(f"\n  Running multi-start joint L-BFGS (omega + U_skew + alpha) "
          f"for rho_{target_true_idx + 1} ...")

    d = data['config']['d']

    def _pack(alpha, U_skew, omega):
        EPS = 1e-8
        log_alpha = torch.log(alpha.clamp(min=EPS))
        diag_clamped = torch.diagonal(U_skew).clamp(min=EPS)
        U_packed = U_skew.clone()
        U_packed[torch.eye(d, dtype=torch.bool, device=device)] = torch.log(diag_clamped)
        triu_idx = torch.triu_indices(d, d, device=device)
        U_vals = U_packed[triu_idx[0], triu_idx[1]]
        omega_range = omega_max - omega_min
        p = ((omega - omega_min) / omega_range).clamp(1e-6, 1 - 1e-6)
        z_omega = torch.log(p / (1 - p))
        return torch.cat([log_alpha.reshape(-1), U_vals.reshape(-1), z_omega.reshape(-1)])

    def _unpack(x):
        idx = 0
        alpha = torch.exp(x[idx].clamp(-5, 5)).unsqueeze(0)
        idx += 1
        n_U = d * (d + 1) // 2
        U_vals = x[idx: idx + n_U]
        U = torch.zeros(d, d, dtype=torch.float64, device=device)
        triu_idx = torch.triu_indices(d, d, device=device)
        U[triu_idx[0], triu_idx[1]] = U_vals
        diag_mask = torch.eye(d, dtype=torch.bool, device=device)
        U[diag_mask] = torch.exp(U[diag_mask].clamp(-4, 4))
        idx += n_U
        z = x[idx]
        omega = (omega_min + (omega_max - omega_min) * torch.sigmoid(z)).unsqueeze(0)
        return alpha, U, omega

    def _loss_single(x):
        alpha, U, omega = _unpack(x)
        # Build a dict where only the target Gaussian has non-zero alpha.
        # Other entries are already detached (from _clone_dict); target entries
        # come directly from _unpack so they stay on the gradient tape.
        test_k_only = _clone_dict(theta_est)
        test_k_only['alphas'][target_k_est]  = alpha
        test_k_only['U_skews'][target_k_est] = U
        test_k_only['omegas'][target_k_est]  = omega
        for j in range(N):
            if j != target_k_est:
                test_k_only['alphas'][j] = torch.zeros(1, dtype=torch.float64, device=device)
        proj_k_list = model.generate_projections(t, test_k_only)
        proj_k = proj_k_list[0].double()
        if not torch.isfinite(proj_k).all():
            return torch.tensor(1e6, dtype=proj_k.dtype, device=device, requires_grad=True)
        return loss_func(proj_residual, proj_k)

    alpha0 = theta_est['alphas'][target_k_est].clone().detach()
    U_curr = theta_est['U_skews'][target_k_est].clone().detach()
    omega0 = torch.tensor([best_omega], dtype=torch.float64, device=device)

    # U_skew starting candidates:
    #  1. Current estimate (as-is)
    #  2. Off-diagonal sign-flipped  (correct for sign-flip failure mode)
    #  3. Isotropic (diagonal only, mean scale) — wide basin start
    mean_scale = float(U_curr.diagonal().abs().mean().item())
    U_iso = torch.eye(d, dtype=torch.float64, device=device) * mean_scale
    U_flip = U_curr.clone()
    triu_idx = torch.triu_indices(d, d, offset=1, device=device)
    U_flip[triu_idx[0], triu_idx[1]] = -U_flip[triu_idx[0], triu_idx[1]]

    starts = [('current',  U_curr),
              ('off-diag flipped', U_flip),
              ('isotropic', U_iso)]

    best_res_val = float('inf')
    alpha_refined = alpha0
    U_refined     = U_curr
    omega_refined = omega0

    for label, U_init in starts:
        x0 = _pack(alpha0, U_init, omega0).requires_grad_(True)
        res = minimize(
            _loss_single, x0=x0, method='l-bfgs',
            tol=1e-10, options={'gtol': 1e-10, 'max_iter': lbfgs_iters, 'disp': False},
        )
        val = res.fun.item()
        a_r, U_r, w_r = _unpack(res.x.detach())
        print(f"    start={label:20s}  loss={val:.4e}  "
              f"omega={w_r.item():.4f}  U_01={U_r[0,1].item():.3f}")
        if val < best_res_val:
            best_res_val = val
            alpha_refined = a_r
            U_refined     = U_r
            omega_refined = w_r

    # --- Pass 2: full-sinogram refinement with all other Gaussians fixed ---
    # Optimising against the residual gives good omega but weak U gradient
    # (small dynamic range).  Now fix the other 4 Gaussians' alpha/U/omega
    # and re-optimise rho_k against the full observed sinogram — larger signal.
    print(f"\n  Pass 2: full-sinogram refinement (others fixed) ...")
    x0_p2 = _pack(alpha_refined.detach(), U_refined.detach(),
                  omega_refined.detach()).requires_grad_(True)

    def _loss_full(x):
        alpha, U, omega = _unpack(x)
        test_full = _clone_dict(theta_est)
        test_full['alphas'][target_k_est]  = alpha
        test_full['U_skews'][target_k_est] = U
        test_full['omegas'][target_k_est]  = omega
        proj_list = model.generate_projections(t, test_full)
        proj_sim = proj_list[0].double()
        if not torch.isfinite(proj_sim).all():
            return torch.tensor(1e6, dtype=proj_sim.dtype, device=device, requires_grad=True)
        return loss_func(proj_obs.double(), proj_sim)

    best_p2_val = float('inf')
    for label, U_init in [('pass1_best', U_refined.detach()),
                           ('off-diag flipped', U_flip),
                           ('isotropic', U_iso)]:
        x0 = _pack(alpha_refined.detach(), U_init, omega_refined.detach()).requires_grad_(True)
        res2 = minimize(
            _loss_full, x0=x0, method='l-bfgs',
            tol=1e-10, options={'gtol': 1e-10, 'max_iter': lbfgs_iters, 'disp': False},
        )
        val2 = res2.fun.item()
        a2, U2, w2 = _unpack(res2.x.detach())
        print(f"    start={label:20s}  loss={val2:.4e}  "
              f"omega={w2.item():.4f}  U_01={U2[0,1].item():.3f}")
        if val2 < best_p2_val:
            best_p2_val = val2
            alpha_refined = a2
            U_refined     = U2
            omega_refined = w2

    # --- Pass 3: full joint re-optimisation of ALL Gaussians ---
    # ρ₂'s parameters are now in a better basin (correct ω sign, correct U sign).
    # Freeing all 5 Gaussians simultaneously against the full sinogram lets the
    # others adapt to the improved ρ₂ starting point and vice versa.
    print(f"\n  Pass 3: full 5-Gaussian joint re-optimisation ...")

    def _pack_all(th):
        """Pack all N Gaussians into a single flat vector."""
        parts = []
        EPS = 1e-8
        for k in range(N):
            a = th['alphas'][k]
            U = th['U_skews'][k]
            w = th['omegas'][k]
            parts.append(torch.log(a.clamp(min=EPS)).reshape(-1))
            diag_c = torch.diagonal(U).clamp(min=EPS)
            U_p = U.clone()
            U_p[torch.eye(d, dtype=torch.bool, device=device)] = torch.log(diag_c)
            triu = torch.triu_indices(d, d, device=device)
            parts.append(U_p[triu[0], triu[1]].reshape(-1))
            omega_range = omega_max - omega_min
            p = ((w - omega_min) / omega_range).clamp(1e-6, 1 - 1e-6)
            parts.append(torch.log(p / (1 - p)).reshape(-1))
        return torch.cat(parts)

    def _unpack_all(x):
        n_U = d * (d + 1) // 2
        per_k = 1 + n_U + 1     # log_alpha, U_vals, z_omega
        alphas, U_skews, omegas = [], [], []
        for k in range(N):
            base = k * per_k
            alpha = torch.exp(x[base].clamp(-5, 5)).unsqueeze(0)
            U_vals = x[base + 1: base + 1 + n_U]
            U = torch.zeros(d, d, dtype=torch.float64, device=device)
            triu = torch.triu_indices(d, d, device=device)
            U[triu[0], triu[1]] = U_vals
            diag_mask = torch.eye(d, dtype=torch.bool, device=device)
            U[diag_mask] = torch.exp(U[diag_mask].clamp(-4, 4))
            z = x[base + 1 + n_U]
            omega = (omega_min + (omega_max - omega_min) * torch.sigmoid(z)).unsqueeze(0)
            alphas.append(alpha)
            U_skews.append(U)
            omegas.append(omega)
        return alphas, U_skews, omegas

    def _loss_all(x):
        alphas, U_skews, omegas = _unpack_all(x)
        th = _clone_dict(theta_est)
        th['alphas']  = alphas
        th['U_skews'] = U_skews
        th['omegas']  = omegas
        proj_list = model.generate_projections(t, th)
        proj_sim = proj_list[0].double()
        if not torch.isfinite(proj_sim).all():
            return torch.tensor(1e6, dtype=proj_sim.dtype, device=device, requires_grad=True)
        return loss_func(proj_obs.double(), proj_sim)

    # Warm-start: use polished ρ₂, all others from original theta_est
    th_warm = _clone_dict(theta_est)
    th_warm['alphas'][target_k_est]  = alpha_refined.detach()
    th_warm['U_skews'][target_k_est] = U_refined.detach()
    th_warm['omegas'][target_k_est]  = omega_refined.detach()

    x0_all = _pack_all(th_warm).requires_grad_(True)
    res3 = minimize(
        _loss_all, x0=x0_all, method='l-bfgs',
        tol=1e-10, options={'gtol': 1e-10, 'max_iter': lbfgs_iters, 'disp': True},
    )
    alphas3, U_skews3, omegas3 = _unpack_all(res3.x.detach())

    print(f"\n  --- Full joint re-optimisation results ---")
    for k in range(N):
        k_true_label = [kt for kt, ke in enumerate(est_for_true) if ke == k]
        lbl = f"rho_{k_true_label[0]+1}" if k_true_label else f"est_{k}"
        print(f"  {lbl}: omega={omegas3[k].item():.4f}  "
              f"U=[[{U_skews3[k][0,0].item():.3f}, {U_skews3[k][0,1].item():.3f}], "
              f"[0, {U_skews3[k][1,1].item():.3f}]]")

    print(f"\n  rho_{target_true_idx+1} final:")
    print(f"    omega_refined = {omegas3[target_k_est].item():.4f} Hz "
          f"  (true={theta_true['omegas'][target_true_idx].item():.4f})")
    print(f"    U_refined =\n{U_skews3[target_k_est].detach().cpu().numpy()}")
    print(f"    U_true =\n{theta_true['U_skews'][target_true_idx].cpu().numpy()}")

    # Pick best across pass2 and pass3 for the target Gaussian
    # Initialise updated dict here so pass3 branch can also update others
    theta_est_updated = _clone_dict(theta_est)
    theta_est_updated['alphas'][target_k_est]  = alpha_refined.detach()
    theta_est_updated['U_skews'][target_k_est] = U_refined.detach()
    theta_est_updated['omegas'][target_k_est]  = omega_refined.detach()

    if res3.fun.item() < best_p2_val:
        alpha_refined = alphas3[target_k_est]
        U_refined     = U_skews3[target_k_est]
        omega_refined = omegas3[target_k_est]
        # Also update all others from pass3
        for k in range(N):
            theta_est_updated['alphas'][k]  = alphas3[k].detach()
            theta_est_updated['U_skews'][k] = U_skews3[k].detach()
            theta_est_updated['omegas'][k]  = omegas3[k].detach()
        print(f"  (Pass 3 was best: loss={res3.fun.item():.4e})")
    else:
        print(f"  (Pass 2 was best: loss={best_p2_val:.4e}, pass3={res3.fun.item():.4e})")

    # --- Re-save ---

    data_updated = dict(data)
    data_updated['theta_est'] = theta_est_updated

    # Save alongside original
    out_path = results_path.parent / 'results_polished.pt'
    torch.save(data_updated, out_path)
    print(f"\n  Saved polished results to: {out_path}")

    return theta_est_updated, target_true_idx, target_k_est


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-hoc single-Gaussian polish.')
    parser.add_argument('results_dir', type=str,
                        help='Path to results directory containing results.pt')
    parser.add_argument('--gaussian-index', type=int, default=None,
                        help='1-indexed true Gaussian to polish (default: auto-detect worst)')
    parser.add_argument('--omega-step', type=float, default=0.05,
                        help='Grid search step size in Hz (default: 0.05)')
    parser.add_argument('--lbfgs-iters', type=int, default=500,
                        help='Max L-BFGS iterations (default: 500)')
    args = parser.parse_args()

    results_dir  = Path(args.results_dir)
    results_path = results_dir / 'results.pt'

    if not results_path.exists():
        raise FileNotFoundError(f"No results.pt found in {results_dir}")

    # Convert 1-indexed user input to 0-indexed
    target_idx = (args.gaussian_index - 1) if args.gaussian_index is not None else None

    polish_gaussian(
        results_path,
        target_true_idx=target_idx,
        omega_step=args.omega_step,
        lbfgs_iters=args.lbfgs_iters,
    )
