"""Parameter initialization for GMM-CT reconstruction."""

import logging

import torch

from ..estimation.peak_analysis import PeakData

logger = logging.getLogger(__name__)


class InitializationMixin:
    """Mixin providing parameter initialization for GMM_reco."""

    def initialize_parameters(self, t, proj_data):
        """
        Initialize all GMM parameters for optimization.

        Initialization strategy:
          1. Isotropic Gaussians (U_skew = scaled identity) -- essential for
             trajectory fitting.
          2. Peak detection + random v0 -- starting point for trajectory
             optimization.
          3. Omega at midpoint of range -- simple starting guess (refined by
             optimization).
          4. Reasonable alpha values -- typical attenuation coefficients.

        Parameters
        ----------
        t : torch.Tensor
            Time vector.
        proj_data : torch.Tensor or list
            Observed projection data.

        Returns
        -------
        dict
            Complete parameter dictionary with all initialized values.
        """
        alphas = self.initialize_attenuation_coefficients()
        omegas = self.initialize_rotation_velocities()
        v0s = self.initialize_initial_velocities(t, proj_data)
        U_skews = self.initialize_U_skews(v0s)

        return {
            'alphas': alphas,
            'U_skews': U_skews,
            'omegas': omegas,
            'x0s': self.x0s,
            'v0s': v0s,
            'a0s': self.a0s,
        }

    def initialize_attenuation_coefficients(self):
        """
        Initialize attenuation coefficients to the midpoint of the prior range [10, 15].

        Returns
        -------
        list of torch.Tensor
            Alpha values, one per Gaussian.
        """
        return [
            torch.tensor([12.5], dtype=torch.float64, device=self.device)
            for _ in range(self.N)
        ]

    def initialize_initial_velocities(self, t, proj_data):
        """
        Detect peaks in projection data and initialize velocity parameters.

        Orchestrates:
          1. Peak detection via sliding 3-point window.
          2. Sequential assignment of peaks to Gaussians (bottom-to-top).
          3. Random v0 initialization for optimization.

        Parameters
        ----------
        t : torch.Tensor
            Time vector for observations.
        proj_data : torch.Tensor or list
            Observed projection data ``[n_times, n_receivers]``.

        Returns
        -------
        list of torch.Tensor
            Random initial velocities for each Gaussian.

        Side Effects
        ------------
        Creates ``self.peak_data`` with detection and assignment info.
        """
        self.peak_data = PeakData(self.N, self.device)

        proj_data_array = proj_data[0] if isinstance(proj_data, list) else proj_data
        receivers = self.receivers[0]

        self._detect_all_peaks(proj_data_array, receivers, t)

        self.peak_data.finalize_detections()
        self._create_legacy_aliases()

        return self._create_random_initial_velocities()

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def _detect_all_peaks(self, proj_data, receivers, t):
        """
        Detect peaks across all time points using a 3-point sliding window.

        Parameters
        ----------
        proj_data : torch.Tensor, shape (n_times, n_receivers)
        receivers : list of torch.Tensor
        t : torch.Tensor
        """
        for time_idx in range(len(t)):
            detected_heights = self._detect_peaks_at_single_time(
                proj_data[time_idx], receivers, t[time_idx], time_idx,
            )
            self.peak_data.add_time_detections(t[time_idx].item(), detected_heights)

    def _detect_peaks_at_single_time(self, projection, receivers, time_val, time_idx):
        """
        Detect peaks at a single time point using 3-point sliding window.

        A peak is detected when the center value exceeds both neighbours.
        Scans from bottom (high index) to top (low index).

        Parameters
        ----------
        projection : torch.Tensor, shape (n_receivers,)
        receivers : list of torch.Tensor
        time_val : float or torch.Tensor
        time_idx : int

        Returns
        -------
        list of float
            Receiver heights where peaks were detected.
        """
        detected_heights = []
        gaussian_idx = 0

        for offset in range(self.n_rcvrs - 2):
            idx_center = self.n_rcvrs - 2 - offset
            idx_lower = idx_center + 1
            idx_upper = idx_center - 1

            if (projection[idx_lower] < projection[idx_center]
                    and projection[idx_center] > projection[idx_upper]):
                receiver_pos = receivers[idx_center]

                self.peak_data.add_peak_detection(
                    time_idx, time_val, idx_center, receiver_pos,
                    projection[idx_center], gaussian_idx,
                )

                detected_heights.append(receiver_pos[1])
                gaussian_idx += 1

                if gaussian_idx >= self.N:
                    break

        return detected_heights

    # ------------------------------------------------------------------
    # Legacy aliases (backward compatibility for plotting)
    # ------------------------------------------------------------------

    def _create_legacy_aliases(self):
        """Create backward-compatible aliases for plotting methods."""
        self.t_obs_by_cluster = self.peak_data.times
        self.maximising_rcvrs = self.peak_data.receiver_positions
        self.maximising_inds = self.peak_data.receiver_indices
        self.peak_values = self.peak_data.peak_values
        self.observable_indices = self.peak_data.observable_indices
        self.time_rcvr_heights_dict_non_empty = self.peak_data.get_heights_dict_non_empty()
        self.sorted_list_of_heights_over_time = self.peak_data.get_heights_sorted_by_time()

    # ------------------------------------------------------------------
    # Random velocity initialization
    # ------------------------------------------------------------------

    def _create_random_initial_velocities(self):
        """
        Create random initial velocity estimates around [1, 1] with std=1.5.

        Returns
        -------
        list of torch.Tensor
            Initial velocity tensors with gradients enabled.
        """
        v0s = []
        for _ in range(self.N):
            v0 = torch.tensor([1.0, 1.0], dtype=torch.float64, device=self.device)
            v0 = v0 + 1.5 * torch.randn(2, dtype=torch.float64, device=self.device)
            v0.requires_grad_(True)
            v0s.append(v0)
        return v0s

    # ------------------------------------------------------------------
    # Covariance and rotation initialization
    # ------------------------------------------------------------------

    def initialize_U_skews(self, v0s):
        """
        Initialize anisotropic covariance structures aligned with initial velocities.

        Delegates to :meth:`initialize_anisotropic_U_skews` so that morphology
        is set once, before Stage 1 trajectory optimization begins.

        Parameters
        ----------
        v0s : list of torch.Tensor
            Initial velocity vectors, one per Gaussian.

        Returns
        -------
        list of torch.Tensor
            Anisotropic U_skew matrices, one per Gaussian.
        """
        return self.initialize_anisotropic_U_skews(v0s)

    def initialize_rotation_velocities(self):
        """
        Initialize rotation velocities to zero.

        A zero angular velocity is the simplest uninformative starting point;
        Stage 1.5a replaces these with grid-search estimates.

        Returns
        -------
        list of torch.Tensor
        """
        return [
            torch.zeros(size=(1,), dtype=torch.float64, device=self.device)
            for _ in range(self.N)
        ]

    def initialize_anisotropic_U_skews(self, v0s, eps=1.0):
        """
        Initialize anisotropic U_skew matrices aligned with velocity direction.

        Creates elongated Gaussians essential for omega estimation via DTW.
        Major axis along velocity (scale 15), minor axis perpendicular (scale 30),
        giving a 4:1 covariance ratio.  Upper off-diagonal entries are perturbed
        by small Gaussian noise ~ N(0, eps^2) to aid off-diagonal recovery.

        Parameters
        ----------
        v0s : list of torch.Tensor
            Optimized initial velocities from Phase 1, shape ``(2,)`` each.
        eps : float, optional
            Standard deviation of the Gaussian perturbation applied to the
            strictly upper-triangular entries.  Default: 1.0.

        Returns
        -------
        list of torch.Tensor
            Anisotropic U_skew matrices, shape ``(2, 2)`` each.
        """
        U_skews = []

        for k in range(self.N):
            v0_k = v0s[k]
            v_norm = torch.norm(v0_k)

            if v_norm < 1e-6:
                v_hat = torch.tensor([1.0, 0.0], dtype=torch.float64, device=self.device)
            else:
                v_hat = v0_k / v_norm

            v_perp = torch.tensor([-v_hat[1], v_hat[0]], dtype=torch.float64, device=self.device)

            # Small precision → large covariance → major axis
            major_direction_scale = 15.0
            # Large precision → small covariance → minor axis
            minor_direction_scale = 30.0

            U_skew_k = torch.stack([
                v_hat * major_direction_scale,
                v_perp * minor_direction_scale,
            ], dim=1).clone()

            # Perturb strictly upper-triangular entries
            if eps > 0:
                rows, cols = torch.triu_indices(self.d, self.d, offset=1, device=self.device)
                noise = eps * torch.randn(len(rows), dtype=torch.float64, device=self.device)
                U_skew_k[rows, cols] = U_skew_k[rows, cols] + noise

            U_skews.append(U_skew_k)

            ratio = (minor_direction_scale / major_direction_scale) ** 2
            logger.debug("  Gaussian %d: elongated along velocity (covariance ratio %.1f:1)",
                         k, ratio)

        return U_skews
