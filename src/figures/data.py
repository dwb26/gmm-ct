"""Load an experiment directory once into a plain-numpy ``Run`` and derive quantities from it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_KEYS = ("x0s", "v0s", "a0s", "omegas", "alphas", "U_skews")


def _to_numpy_theta(theta: dict | None) -> dict | None:
    if theta is None:
        return None
    out = {}
    for key in _KEYS:
        vals = [torch.as_tensor(v).detach().cpu().double().numpy() for v in theta[key]]
        out[key] = np.stack(vals)
    # One rotation rate per Gaussian in 2D
    out["omegas"] = out["omegas"].reshape(len(out["omegas"]))
    out["alphas"] = out["alphas"].reshape(len(out["alphas"]))
    return out


def _to_array(v) -> np.ndarray:
    """Flat float array from a tensor, array or list of (0-d) tensors/floats."""
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    elif isinstance(v, (list, tuple)):
        v = [x.detach().cpu().item() if hasattr(x, "detach") else x for x in v]
    return np.atleast_1d(np.asarray(v, dtype=float)).ravel()


@dataclass
class Run:
    """All inputs a figure needs. Detector axis is stored in ascending height."""

    exp_dir: Path
    N: int
    t: np.ndarray               # [T]
    proj: np.ndarray            # [T, R], column j <-> y[j] (ascending)
    y: np.ndarray               # [R] detector heights, ascending
    source: np.ndarray          # [2]
    detector_x: float
    theta_true: dict
    theta_est: dict | None
    theta_init: dict | None     # best trajectory initialisation
    snr_db: float
    seed: int
    detected_modes: dict | None = None  # {time: [heights]} saved by the reconstruction
    _modes: list | None = None

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def time_index(self, t_val: float) -> int:
        return int(np.argmin(np.abs(self.t - t_val)))

    def default_times(self, fractions=(0.3, 0.7)) -> tuple[float, float]:
        """Two times placed within the window where the objects are visible on the detector."""
        t0, t1 = self.active_window()
        return tuple(t0 + f * (t1 - t0) for f in fractions)

    def centers(self, theta: dict, t) -> np.ndarray:
        """Centre trajectories x0 + v0 t + a0 t^2 / 2, shape [N, len(t), 2]."""
        t = np.asarray(t, dtype=float)[None, :, None]
        return theta["x0s"][:, None] + theta["v0s"][:, None] * t + 0.5 * theta["a0s"][:, None] * t**2

    def covariance(self, theta: dict, k: int, t_val: float) -> np.ndarray:
        """Covariance R P^-1 R^T of Gaussian k, with P = U^T U and R the rotation by 2 pi omega t."""
        U = theta["U_skews"][k]
        ang = 2 * np.pi * theta["omegas"][k] * t_val
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        return R @ np.linalg.inv(U.T @ U) @ R.T

    def project(self, theta: dict) -> np.ndarray:
        """Noise-free projections [T, R] of ``theta`` (numpy port of ``GMM_reco.generate_projections``)."""
        EPS = 1e-10
        r_minus_s = np.stack([np.full_like(self.y, self.detector_x), self.y], axis=1) - self.source
        r_hat = r_minus_s / np.linalg.norm(r_minus_s, axis=1, keepdims=True)
        centers = self.centers(theta, self.t)
        out = np.zeros((len(self.t), len(self.y)))
        for n in range(self.N):
            ang = 2 * np.pi * theta["omegas"][n] * self.t
            c, s = np.cos(ang), np.sin(ang)
            R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)           # [T, 2, 2]
            U_t = theta["U_skews"][n] @ R.transpose(0, 2, 1)                          # [T, 2, 2]
            U_rhat = np.einsum("rd,ted->tre", r_hat, U_t)
            U_r = np.einsum("rd,ted->tre", r_minus_s, U_t)
            U_traj = np.einsum("td,ted->te", self.source - centers[n], U_t)[:, None]
            quotient = np.sqrt(np.pi) * theta["alphas"][n] / (np.linalg.norm(U_rhat, axis=-1) + EPS)
            exp_arg = (U_r * U_traj).sum(-1) ** 2 / ((U_r ** 2).sum(-1) + EPS) - (U_traj ** 2).sum(-1)
            out += quotient * np.exp(exp_arg)
        return out

    def mode_heights(self, theta: dict, t=None) -> np.ndarray:
        """Detector height hit by the ray from the source through each centre, shape [N, T]."""
        t = self.t if t is None else t
        c = self.centers(theta, t)
        s = self.source
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (self.detector_x - s[0]) / (c[..., 0] - s[0])
            return s[1] + lam * (c[..., 1] - s[1])

    def active_window(self, rel_threshold: float = 0.1) -> tuple[float, float]:
        """Times between the first and last frame whose peak intensity is non-negligible."""
        peak = self.proj.max(axis=1)
        active = np.flatnonzero(peak > rel_threshold * peak.max())
        return float(self.t[active[0]]), float(self.t[active[-1]])

    def detect_modes(self) -> list[np.ndarray]:
        """Per-time detected mode heights, the same ones the reconstruction fitted to.

        Uses the modes saved with the reconstruction when available, otherwise re-runs the
        model's own detector on the stored projections.
        """
        if self._modes is None:
            if self.detected_modes is not None:
                saved = {float(k): v for k, v in self.detected_modes.items()}
                keys = np.array(list(saved))
                self._modes = []
                for tv in self.t:
                    hit = np.flatnonzero(np.isclose(keys, tv, atol=1e-8))
                    self._modes.append(_to_array(saved[keys[hit[0]]]) if len(hit) else np.empty(0))
            else:
                self._modes = self._redetect_modes()
        return self._modes

    def _redetect_modes(self) -> list[np.ndarray]:
        import torch

        from ..model import GMM_reco

        n_r = len(self.y)
        # The model works with the detector ordered top to bottom
        rcv = torch.tensor(np.stack([np.full(n_r, self.detector_x), self.y[::-1]], axis=1))
        model = GMM_reco(
            d=2, N=self.N, sources=[torch.tensor(self.source)], receivers=[list(rcv)],
            x0s=[], a0s=[], omega_min=0.0, omega_max=0.0, exp_dir=None,
            save_diagnostics=False,
        )
        coords = rcv[:, 1]
        out = []
        for row in self.proj[:, ::-1]:
            params = model.fit_gmm_1d_fixed_N(torch.tensor(row.copy()), coords)
            params = params[params[:, 1] > 0.075]
            out.append(_to_array(params[:, 0]))
        return out


def load_run(exp_dir: str | Path) -> Run:
    exp_dir = Path(exp_dir)
    gt = torch.load(exp_dir / "ground_truth.pt", map_location="cpu", weights_only=False)
    pj = torch.load(exp_dir / "projections.pt", map_location="cpu", weights_only=False)
    rec_path = exp_dir / "reconstruction.pt"
    rec = torch.load(rec_path, map_location="cpu", weights_only=False) if rec_path.exists() else {}

    if gt["config"]["d"] != 2:
        raise NotImplementedError("Figures currently support d == 2 only.")

    receivers = np.stack([r.detach().cpu().double().numpy() for r in gt["receivers"][0]])
    # Stored projections have detector index 0 at the top; flip to ascending height
    proj = pj["projections"].detach().cpu().double().numpy()
    order = np.argsort(receivers[:, 1])

    return Run(
        exp_dir=exp_dir,
        N=gt["config"]["N"],
        t=pj["times"].detach().cpu().double().numpy(),
        proj=proj[:, order],
        y=receivers[order, 1],
        source=gt["sources"][0].detach().cpu().double().numpy(),
        detector_x=float(receivers[0, 0]),
        theta_true=_to_numpy_theta(gt["theta_true"]),
        theta_est=_to_numpy_theta(rec.get("theta_est")),
        theta_init=_to_numpy_theta(rec.get("theta_pre_stage_2")),
        snr_db=float(gt["config"]["snr_db"]),
        seed=gt["config"]["seed"],
        detected_modes=rec.get("detected_modes"),
    )
