"""
Diagnostic plotting functions for GMM-CT reconstruction.

Standalone plotting functions for inspecting trajectory estimation, peak
detection, and assignment results during the reconstruction pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np

# Default font sizes for diagnostic plots
_LABEL_FONTSIZE = 20
_TITLE_FONTSIZE = 22
_TICK_FONTSIZE = 16


def plot_trajectory_estimations(model, res):
    """
    Plot estimated maximizing receiver heights over time for each Gaussian.

    Parameters
    ----------
    model : GMM_reco
        The reconstruction model instance.
    res : OptimizeResult
        Trajectory optimization result.
    """
    r_maxs_list = model.map_velocities_to_maximising_receivers(
        model.map_from_tensor_to_dict(res.x),
    )

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    min_rcvr_height = model.receivers[0][-1][1]
    max_rcvr_height = model.receivers[0][0][1]

    for k in range(model.N):
        predicted_heights_k = r_maxs_list[k][:, 1]
        mask = (
            (predicted_heights_k >= min_rcvr_height)
            & (predicted_heights_k <= max_rcvr_height)
        )
        times_k = model.t_observable[mask]
        predicted_heights_k = predicted_heights_k[mask]
        plt.plot(
            times_k, predicted_heights_k.cpu().detach().numpy(),
            label=f'Cluster {k}', lw=1,
        )

        rcvrs_k = model.maximising_rcvrs[k]
        if len(rcvrs_k) == 0:
            continue

        rcvr_heights = torch.zeros(
            len(rcvrs_k), dtype=torch.float64, device=model.device,
        )
        for i in range(len(rcvrs_k)):
            rcvr_heights[i] = rcvrs_k[i][1]

        t_obs_k = model.t_obs_by_cluster[k]
        if isinstance(t_obs_k, list):
            t_obs_k = torch.tensor(t_obs_k, dtype=torch.float64, device=model.device)
        ax.scatter(
            t_obs_k.cpu(), rcvr_heights.cpu(),
            label=k, s=10, color='black',
        )

    plt.xlabel('Time', fontsize=_LABEL_FONTSIZE)
    plt.ylabel('Height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)

    filename = model.output_dir / f'trajectory_estimations_K{model.N}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_heights_by_assignment(model, true_data=False):
    """
    Plot receiver heights assigned to each Gaussian cluster over time.

    Parameters
    ----------
    model : GMM_reco
        The reconstruction model instance.
    true_data : bool
        If True, label as 'True Receiver Heights'.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for k, data_k in enumerate(model.assigned_curve_data):
        inds = [item[0] for item in data_k]
        heights = [item[1].item() for item in data_k]
        t_obs = model.t_observable[inds].cpu().numpy()
        ax.scatter(t_obs, heights, s=10)

    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)

    if true_data:
        ax.set_ylabel('True Receiver Heights', fontsize=_TITLE_FONTSIZE)
    else:
        ax.set_ylabel('Assigned Receiver Heights', fontsize=_LABEL_FONTSIZE)

    suffix = '_true_data' if true_data else ''
    filename = (
        model.output_dir / f'heights_by_assignment_K{model.N}{suffix}.png'
    )
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_raw_receiver_heights(model):
    """
    Plot raw, unassigned receiver heights where peaks were detected.

    Parameters
    ----------
    model : GMM_reco
        The reconstruction model instance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for time_val, heights in model.time_rcvr_heights_dict_non_empty.items():
        if heights:
            times = [time_val] * len(heights)
            height_vals = [h.item() for h in heights]
            ax.scatter(times, height_vals, s=10, color='black')

    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--')

    filename = model.output_dir / f'raw_receiver_heights_K{model.N}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================================
# Assignment quality panel (3-panel figure)
# ======================================================================

def plot_assignment_quality(model, res):
    """
    Three-panel figure showing the peak-to-trajectory assignment pipeline.

    Panel layout (left to right):

    1. **Raw peaks** – all detected peaks before any assignment (black scatter).
    2. **Hungarian assignment** – peaks color-coded by which Gaussian they were
       assigned to (one color per cluster).
    3. **Predicted trajectories + residuals** – predicted maximizing-receiver
       curves overlaid on the assigned peaks, with vertical lines showing
       the per-point residual.

    Parameters
    ----------
    model : GMM_reco
        The reconstruction model instance (after trajectory loss evaluation,
        so ``model.assigned_curve_data`` and friends are populated).
    res : OptimizeResult
        Trajectory optimisation result whose ``res.x`` encodes the v0 tensor.
    """
    colors = cm.rainbow(np.linspace(0, 1, model.N))

    # Recompute predicted trajectories from the optimisation result
    theta_dict = model.map_from_tensor_to_dict(res.x)
    r_maxs_list = model.map_velocities_to_maximising_receivers(theta_dict)

    min_rcvr_height = model.receivers[0][-1][1].item()
    max_rcvr_height = model.receivers[0][0][1].item()

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

    # ---- Panel 1: Raw detected peaks ----
    ax = axes[0]
    for time_val, heights in model.time_rcvr_heights_dict_non_empty.items():
        if heights:
            t_vals = [time_val] * len(heights)
            h_vals = [h.item() for h in heights]
            ax.scatter(t_vals, h_vals, s=14, color='black', zorder=5)
    ax.set_title('Detected Peaks (raw)', fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ---- Panel 2: Hungarian-assigned peaks ----
    ax = axes[1]
    for k, data_k in enumerate(model.assigned_curve_data):
        if not data_k:
            continue
        inds = [item[0] for item in data_k]
        heights = [item[1].item() for item in data_k]
        t_obs = model.t_observable[inds].cpu().numpy()
        ax.scatter(t_obs, heights, s=18, color=colors[k],
                   label=f'$\\rho_{{{k+1}}}$', zorder=5)
    ax.set_title('Hungarian Assignment', fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ---- Panel 3: Predicted curves + residuals ----
    ax = axes[2]
    for k in range(model.N):
        pred_h = r_maxs_list[k][:, 1].detach().cpu().numpy()
        t_all = model.t_observable.cpu().numpy()
        mask = (pred_h >= min_rcvr_height) & (pred_h <= max_rcvr_height)
        ax.plot(t_all[mask], pred_h[mask], color=colors[k], lw=2,
                label=f'$\\rho_{{{k+1}}}$ pred', zorder=3)

        # Scatter assigned points and draw residual bars
        data_k = model.assigned_curve_data[k]
        if not data_k:
            continue
        inds = [item[0] for item in data_k]
        obs_h = np.array([item[1].item() for item in data_k])
        t_obs = model.t_observable[inds].cpu().numpy()
        pred_h_k = r_maxs_list[k][inds, 1].detach().cpu().numpy()

        ax.scatter(t_obs, obs_h, s=18, color=colors[k], zorder=5)
        for ti, oh, ph in zip(t_obs, obs_h, pred_h_k):
            ax.plot([ti, ti], [oh, ph], color=colors[k], lw=0.8,
                    alpha=0.6, zorder=2)

    ax.set_title('Trajectories & Residuals', fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=12, framealpha=0.9, ncol=model.N)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Trajectory Optimisation — Assignment Quality',
                 fontsize=_TITLE_FONTSIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = model.output_dir / f'assignment_quality_K{model.N}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


# ======================================================================
# GMM + projections side-by-side figure
# ======================================================================

def plot_gmm_and_projections(model, res, n_gmm_times=8, theta_true=None):
    """
    Three-panel trajectory-optimisation narrative figure.

    1. **True GMM** – the ground-truth Gaussian mixture drawn at
       *n_gmm_times* snapshots via ``plot_gmm_snapshot`` (filled ellipses
       with 1σ/2σ/3σ confidence levels, matching the mp4 style), with
       ballistic trajectory paths and acquisition geometry overlaid.  If
       ``theta_true`` is not available, centroid markers are drawn instead.

    2. **Raw detected peaks** – scatter of all detected peak heights vs time,
       unassigned black dots, matching ``plot_raw_receiver_heights``.

    3. **Assignment + fitted trajectories** – predicted maximising-receiver
       curves (matching ``plot_trajectory_estimations``) overlaid on the
       Hungarian-assigned peaks colour-coded per Gaussian (matching
       ``plot_heights_by_assignment``), with thin vertical residual lines.

    Parameters
    ----------
    model : GMM_reco
        Reconstruction model instance.
    res : OptimizeResult
        Best trajectory optimisation result.
    n_gmm_times : int, optional
        Number of evenly-spaced GMM snapshots to draw (default 8).
    theta_true : dict or None, optional
        Ground-truth parameter dictionary.  When provided, ``plot_gmm_snapshot``
        is used to draw the true GMM; otherwise centroid markers are used.
    """
    if model.d != 2:
        print("plot_gmm_and_projections currently supports 2-D only — skipping.")
        return

    from ..visualization.publication import (
        plot_gmm_snapshot,
        plot_acquisition_geometry,
        plot_trajectories_single,
    )
    from matplotlib.patches import Patch

    # ── Parameter dict ────────────────────────────────────────────────────
    theta_dict = model.map_from_tensor_to_dict(res.x)
    full_theta = {'v0s': theta_dict['v0s']}
    for key in ('x0s', 'a0s', 'omegas', 'alphas', 'U_skews'):
        if key in theta_dict:
            full_theta[key] = theta_dict[key]
        elif hasattr(model, 'theta_dict_init') and key in model.theta_dict_init:
            full_theta[key] = model.theta_dict_init[key]
        elif hasattr(model, 'theta_fixed') and key in model.theta_fixed:
            full_theta[key] = model.theta_fixed[key]

    t_obs = model.t_observable
    t_full = model.t
    gauss_colors = cm.rainbow(np.linspace(0, 1, model.N))

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={'wspace': 0.30})

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1 — GMM snapshots (animation style)
    # ═══════════════════════════════════════════════════════════════════════
    ax = axes[0]

    plot_acquisition_geometry(ax, model.sources, model.receivers, model.d)

    n_total = len(t_obs)
    snap_indices = np.linspace(0, n_total - 1, min(n_gmm_times, n_total),
                               dtype=int)

    if theta_true is not None:
        # Draw the true GMM using the same animation-style filled ellipses
        plot_trajectories_single(ax, theta_true, t_full, model.N, gauss_colors)
        for t_idx in snap_indices:
            t_val = t_obs[t_idx].item()
            plot_gmm_snapshot(ax, theta_true, t_val, model.N, model.d,
                              gauss_colors, is_true=True, show_centroids=False)
    else:
        # Ground truth unavailable — fall back to centroid markers along
        # the estimated trajectories so the panel is still informative.
        plot_trajectories_single(ax, full_theta, t_full, model.N, gauss_colors)
        for k in range(model.N):
            x0 = full_theta['x0s'][k].detach().cpu().numpy()
            v0 = full_theta['v0s'][k].detach().cpu().numpy()
            a0 = full_theta['a0s'][k].detach().cpu().numpy()
            alpha_k = full_theta['alphas'][k].item()
            marker_size = np.clip(alpha_k * 3, 30, 200)
            for t_idx in snap_indices:
                t_val = t_obs[t_idx].item()
                mu = x0 + v0 * t_val + 0.5 * a0 * t_val ** 2
                ax.scatter(mu[0], mu[1], s=marker_size,
                           color=gauss_colors[k], edgecolors='black',
                           linewidths=0.6, alpha=0.55, zorder=10)

    leg_elems = [Patch(facecolor=gauss_colors[k], edgecolor='black',
                       label=f'$\\rho_{{{k+1}}}$')
                 for k in range(model.N)]
    ax.legend(handles=leg_elems, fontsize=13, framealpha=0.9, loc='upper left')

    src_x = model.sources[0][0].item()
    rcvr_x = model.receivers[0][0][0].item()
    rcvr_ys = np.array([r[1].item() for r in model.receivers[0]])
    ax.set_xlim(src_x - 0.4, rcvr_x + 0.4)
    ax.set_ylim(rcvr_ys.min() - 0.2, rcvr_ys.max() + 0.2)
    ax.set_title('True GMM in motion' if theta_true is not None else 'Estimated trajectories',
                 fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Depth', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--')

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2 — Detected peaks colour-coded by Gaussian assignment
    # ═══════════════════════════════════════════════════════════════════════
    ax = axes[1]

    for k, data_k in enumerate(model.assigned_curve_data):
        if not data_k:
            continue
        inds = [item[0] for item in data_k]
        heights = [item[1].item() for item in data_k]
        t_pts = model.t_observable[inds].cpu().numpy()
        ax.scatter(t_pts, heights, s=14, color=gauss_colors[k],
                   alpha=0.85, zorder=5, label=f'$\\rho_{{{k+1}}}$')

    ax.set_title('True detected peaks', fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Receiver height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=13, framealpha=0.9, ncol=max(1, model.N // 3))
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Trajectory optimisation — GMM & peak assignment',
                 fontsize=_TITLE_FONTSIZE + 2, fontweight='bold', y=1.01)
    plt.tight_layout()

    filename = model.output_dir / f'gmm_and_projections_K{model.N}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()


# ======================================================================
# Trajectory fitting 2-panel figure
# ======================================================================

def plot_trajectory_fitting(model, res):
    """
    Two-panel figure that mirrors the same style as
    ``plot_trajectory_estimations``.

    1. **Observed peaks (raw)** – all detected peak heights vs time in black,
       matching ``plot_raw_receiver_heights``.

    2. **Fitted trajectories** – predicted maximising-receiver curves (one
       coloured line per Gaussian, ``lw=1``) overlaid with a black scatter
       (``s=10``) of the observed best-receiver heights at each observation
       time, exactly as produced by ``plot_trajectory_estimations``.

    Parameters
    ----------
    model : GMM_reco
        Reconstruction model instance.
    res : OptimizeResult
        Best trajectory optimisation result.
    """
    import torch

    # ── Reconstruct theta ────────────────────────────────────────────────
    theta_dict = model.map_from_tensor_to_dict(res.x)
    full_theta = {'v0s': theta_dict['v0s']}
    for key in ('x0s', 'a0s', 'omegas', 'alphas', 'U_skews'):
        if key in theta_dict:
            full_theta[key] = theta_dict[key]
        elif hasattr(model, 'theta_dict_init') and key in model.theta_dict_init:
            full_theta[key] = model.theta_dict_init[key]
        elif hasattr(model, 'theta_fixed') and key in model.theta_fixed:
            full_theta[key] = model.theta_fixed[key]

    gauss_colors = cm.rainbow(np.linspace(0, 1, model.N))

    # Predicted maximising-receiver trajectories
    r_maxs_list = model.map_velocities_to_maximising_receivers(full_theta)
    t_all = model.t_observable.cpu().numpy()
    min_rcvr_h = model.receivers[0][-1][1].item()
    max_rcvr_h = model.receivers[0][0][1].item()

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={'wspace': 0.30})

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1 — Raw detected peaks (all black)
    # ═══════════════════════════════════════════════════════════════════════
    ax = axes[0]

    for time_val, heights in model.time_rcvr_heights_dict_non_empty.items():
        if heights:
            ax.scatter([time_val] * len(heights),
                       [h.item() for h in heights],
                       s=12, color='black', alpha=0.7, zorder=5)

    ax.set_title('Detected peaks (raw)', fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Receiver height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--')

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2 — Fitted trajectories (plot_trajectory_estimations style)
    # ═══════════════════════════════════════════════════════════════════════
    ax = axes[1]

    for k in range(model.N):
        # Coloured predicted curve (lw=1, clipped to receiver height range)
        pred_h = r_maxs_list[k][:, 1].detach().cpu().numpy()
        mask = (pred_h >= min_rcvr_h) & (pred_h <= max_rcvr_h)
        ax.plot(t_all[mask], pred_h[mask],
                color=gauss_colors[k], lw=1,
                label=f'$\\rho_{{{k+1}}}$', zorder=3)

        # Black scatter: maximising receivers heights vs observed times
        rcvrs_k = model.maximising_rcvrs[k]
        if not rcvrs_k:
            continue
        rcvr_heights = torch.zeros(len(rcvrs_k), dtype=torch.float64,
                                   device=model.device)
        for i in range(len(rcvrs_k)):
            rcvr_heights[i] = rcvrs_k[i][1]

        t_obs_k = model.t_obs_by_cluster[k]
        if isinstance(t_obs_k, list):
            t_obs_k = torch.tensor(t_obs_k, dtype=torch.float64,
                                   device=model.device)

        ax.scatter(t_obs_k.cpu(), rcvr_heights.cpu(),
                   s=10, color='black', zorder=5)

    ax.set_title('Fitted trajectories', fontsize=_TITLE_FONTSIZE)
    ax.set_xlabel('Time', fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel('Receiver height', fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_TICK_FONTSIZE)
    ax.legend(fontsize=13, framealpha=0.9, ncol=max(1, model.N // 3))
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Trajectory optimisation — fitting result',
                 fontsize=_TITLE_FONTSIZE + 2, fontweight='bold', y=1.01)
    plt.tight_layout()

    filename = model.output_dir / f'trajectory_fitting_K{model.N}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
