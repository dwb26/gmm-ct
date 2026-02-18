"""
Diagnostic plotting functions for GMM-CT reconstruction.

Standalone plotting functions for inspecting trajectory estimation, peak
detection, and assignment results during the reconstruction pipeline.
"""

import matplotlib.pyplot as plt
import torch

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

        rcvr_heights = torch.zeros(
            len(model.maximising_rcvrs[k]),
            dtype=torch.float64, device=model.device,
        )
        for i in range(len(model.maximising_rcvrs[k])):
            rcvr_heights[i] = model.maximising_rcvrs[k][i][1]
        ax.scatter(
            model.t_obs_by_cluster[k].cpu(), rcvr_heights.cpu(),
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
