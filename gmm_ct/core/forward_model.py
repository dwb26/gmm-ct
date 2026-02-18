"""
Forward model for X-ray CT projection simulation.

Provides the physics of X-ray projection through rotating Gaussian mixtures,
including projection computation, rotation matrix construction, and trajectory
functions.
"""

import torch


class ForwardModelMixin:
    """Mixin providing forward model projection computation for GMM_reco."""

    def generate_projections(self, t, theta_dict, loss_type=None):
        """
        Generate X-ray projection data for the Gaussian mixture model.

        Vectorized implementation that evaluates the full GMM at each time step.

        Parameters
        ----------
        t : torch.Tensor
            Sampled time vector.
        theta_dict : dict
            Parameter dictionary with keys
            'alphas', 'U_skews', 'omegas', 'x0s', 'v0s', 'a0s'.
        loss_type : str, optional
            If set, merges fixed parameters into theta_dict
            (used during optimization).

        Returns
        -------
        list of torch.Tensor
            Projections for each source at each time point.
        """
        rot_mat_funcs = self.construct_rotation_matrix_funcs()
        traj_funcs = self.construct_trajectory_funcs()
        projs = [
            torch.zeros(len(t), self.n_rcvrs, dtype=torch.float64, device=self.device)
            for _ in range(self.n_sources)
        ]

        # Merge fixed parameters when optimizing
        if loss_type is not None:
            complete_theta_dict = theta_dict.copy()
            for key, value in self.theta_fixed.items():
                if key not in complete_theta_dict:
                    complete_theta_dict[key] = value
            theta_dict = complete_theta_dict

        EPS = 1e-10

        for n_t, t_n in enumerate(t):
            rot_mat_of_t = rot_mat_funcs(t_n, theta_dict)
            traj_of_t = traj_funcs(t_n, theta_dict)

            for n_s, s in enumerate(self.sources):
                receivers_ns = self.receivers[n_s]
                r = torch.stack(receivers_ns)

                r_minus_s = r - s
                r_minus_s_hat = r_minus_s / torch.norm(r_minus_s, dim=1, keepdim=True)

                for k in range(self.N):
                    alpha_k = theta_dict['alphas'][k].squeeze()
                    U_k = theta_dict['U_skews'][k]
                    R_k_of_t = rot_mat_of_t[k]
                    mu_k_of_t = traj_of_t[k]
                    new_U_k = U_k @ R_k_of_t.mT

                    # Precompute matrix products
                    U_r_hat = new_U_k @ r_minus_s_hat.T
                    U_r = new_U_k @ r_minus_s.T
                    U_traj = new_U_k @ (s - mu_k_of_t).unsqueeze(1)

                    # Projection terms
                    norm_term = torch.norm(U_r_hat, dim=0)
                    quotient_term = self.sqrt_pi * alpha_k / (norm_term + EPS)

                    inner_prod_sq = torch.sum(U_r * U_traj, dim=0) ** 2
                    divisor = torch.norm(U_r, dim=0) ** 2 + EPS
                    subtractor = torch.norm(U_traj, dim=0) ** 2

                    exp_arg = inner_prod_sq / divisor - subtractor
                    exp_term = torch.exp(exp_arg)

                    projs[n_s][n_t] += quotient_term * exp_term

        return projs

    def construct_rotation_matrix_funcs(self):
        """
        Build a callable that returns per-Gaussian rotation matrices at time *t*.

        Rotations are indexed sequentially across axis pairs:
        (1,2), ..., (1,d), (2,3), ..., (d-1,d).

        Returns
        -------
        callable
            ``f(t, theta) -> list of (d, d)`` rotation matrices.
        """
        two_pi = 2 * torch.pi

        def all_rot_mat_funcs(t, theta):
            rot_matrices = []
            for k in range(self.N):
                omegas_k = theta['omegas'][k]
                kth_component_rot_mats = []
                for n_rots, omega in enumerate(omegas_k):
                    i, j = torch.combinations(
                        torch.arange(self.d, device=self.device), r=2
                    )[n_rots]

                    rot_mat = torch.eye(self.d, dtype=torch.float64, device=self.device)
                    rot_mat[i, i] = torch.cos(two_pi * omega * t)
                    rot_mat[i, j] = -torch.sin(two_pi * omega * t)
                    rot_mat[j, i] = torch.sin(two_pi * omega * t)
                    rot_mat[j, j] = torch.cos(two_pi * omega * t)

                    kth_component_rot_mats.append(rot_mat)

                kth_rot_mat = torch.eye(self.d, dtype=torch.float64, device=self.device)
                for rot_mat in kth_component_rot_mats:
                    kth_rot_mat = kth_rot_mat @ rot_mat
                rot_matrices.append(kth_rot_mat)

            return rot_matrices

        return all_rot_mat_funcs

    def construct_trajectory_funcs(self):
        """
        Build a callable that returns per-Gaussian trajectories at time *t*.

        Trajectory: ``mu_k(t) = x0_k + v0_k * t + 0.5 * a0_k * t**2``

        Returns
        -------
        callable
            ``f(t, theta) -> list`` of ``(d,)`` or ``(n_times, d)`` tensors.
        """

        def all_traj_funcs(t, theta):
            trajectories = []
            for k in range(self.N):
                x0 = theta['x0s'][k]
                v0 = theta['v0s'][k]
                a0 = theta['a0s'][k]

                if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
                    traj_k = x0 + v0 * t + 0.5 * a0 * t ** 2
                else:
                    t_reshaped = t.unsqueeze(1)
                    traj_k = x0 + v0 * t_reshaped + 0.5 * a0 * t_reshaped ** 2
                trajectories.append(traj_k)

            return trajectories

        return all_traj_funcs

    def process_projections(self, projections):
        """Flatten multi-source projections into a single tensor."""
        if self.n_sources == 1:
            return projections[0]
        return torch.cat([proj for proj in projections], dim=0)

    def _generate_peak_pattern_for_omega(
        self, alpha, U_skew, omega, x0, v0, a0, times, gaussian_idx
    ):
        """
        Generate predicted projection peak pattern for a given omega.

        Simulates projection peaks for a single Gaussian with a specific
        rotation rate, given its trajectory and morphology parameters.

        Parameters
        ----------
        alpha : float
            Peak height coefficient.
        U_skew : torch.Tensor, shape (d, d)
            Covariance structure matrix.
        omega : float
            Angular velocity (Hz) to test.
        x0, v0, a0 : torch.Tensor, shape (d,)
            Trajectory parameters.
        times : torch.Tensor, shape (n_times,)
            Time points where peaks were observed.
        gaussian_idx : int
            Index of Gaussian (for accessing correct sources/receivers).

        Returns
        -------
        torch.Tensor, shape (n_times,)
            Predicted peak values at each time point.
        """
        device = self.device
        sqrt_pi = torch.sqrt(torch.tensor(torch.pi, dtype=torch.float64, device=device))

        peak_values = []
        source = self.sources[0]
        receiver_line = self.receivers[0]

        for t_n in times:
            mu_t = x0 + v0 * t_n + 0.5 * a0 * t_n ** 2

            angle = 2 * torch.pi * omega * t_n
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            R_t = torch.stack([
                torch.stack([cos_a, -sin_a]),
                torch.stack([sin_a, cos_a]),
            ])

            U_rot = U_skew @ R_t.T

            projections = []
            for receiver in receiver_line:
                r_minus_s = receiver - source
                r_minus_s_hat = r_minus_s / torch.norm(r_minus_s)

                U_r_hat = U_rot @ r_minus_s_hat
                U_r = U_rot @ r_minus_s
                U_mu = U_rot @ (source - mu_t)

                norm_term = torch.norm(U_r_hat)
                quotient = sqrt_pi * alpha / (norm_term + 1e-10)

                inner_prod = torch.dot(U_r.squeeze(), U_mu)
                inner_prod_sq = inner_prod ** 2
                divisor = torch.norm(U_r) ** 2 + 1e-10
                subtractor = torch.norm(U_mu) ** 2

                exp_arg = inner_prod_sq / divisor - subtractor
                proj = quotient * torch.exp(exp_arg)
                projections.append(proj)

            peak_values.append(torch.max(torch.stack(projections)))

        return torch.stack(peak_values)
