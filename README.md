# GMM-CT: Gaussian Mixture Model CT Reconstruction

A Python package for reconstructing dynamic objects in CT imaging using Gaussian Mixture Models with motion estimation.

## Overview

GMM-CT recovers the internal structure and motion of objects undergoing projectile motion and rotation from limited CT projection data. Each object in the scene is modelled as a Gaussian component with anisotropic covariance, and the package jointly estimates:

- Attenuation coefficients ($\alpha$) and shape matrices ($U$)
- Initial velocities ($v_0$) and ballistic trajectory parameters
- Angular velocities ($\omega$) for in-plane rotation
- Multi-object assignment via the Hungarian algorithm

### Reconstruction Pipeline

The reconstruction proceeds in four stages:

| Stage | What is optimised | Method | Loss |
|---|---|---|---|
| **1. Trajectory** | Initial velocities $v_0$ | Multi-start L-BFGS (up to 1 500 iters) | L2 on peak receiver heights, with Hungarian assignment |
| **1.5. Velocity refinement** | $v_0$ (fine-tune) | L-BFGS root-finding on analytic derivative | Closed-form isotropic projection derivative |
| **2. Joint morphology** | $\alpha$, $U_{\text{skew}}$, $\omega$ | Multi-start L-BFGS (up to 1 000 iters) | Smooth L1 (Huber, $\beta{=}0.3$) on full projections |
| **3. Grid search** | $\omega$ | Brute-force grid ($\pm 3$ Hz, $0.1$ Hz steps) | Smooth L1 |
| **4. Final polish** | $\alpha$, $U_{\text{skew}}$, $\omega$ | L-BFGS (200 iters) | Smooth L1 |

**Physical model:** Each Gaussian follows a ballistic trajectory $\mu_k(t) = x_0 + v_0\,t + \tfrac{1}{2}\,a_0\,t^2$ with 2D rotation $R(2\pi\omega t)$. Projections are computed via a closed-form X-ray transform of the rotated Gaussian. Stage 1 decouples trajectory from rotation by using isotropic Gaussians; Stages 2–4 then recover the full anisotropic shape and rotation.

## Quick Start

### Installation

```bash
# Install in development mode
pip install -e .

# Or with dev tools (pytest, black, etc.)
pip install -e ".[dev]"
```

### Basic Usage

```python
from gmm_ct import (
    GMM_reco,
    generate_true_param,
    construct_receivers,
    set_random_seeds,
)
from gmm_ct.config.defaults import GRAVITATIONAL_ACCELERATION
import torch

# Reproducibility
set_random_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CT geometry
sources = [torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)]
receivers = construct_receivers(device, (128, 4.0, -3.0, 3.0))

# Generate synthetic ground-truth parameters
d, N = 2, 3
theta_true = generate_true_param(
    d, N,
    initial_location=[-8.0, 0.0],
    initial_velocity=[3.0, 2.0],
    initial_acceleration=[0.0, -GRAVITATIONAL_ACCELERATION],
    min_rot=-24.0, max_rot=-20.0,
    device=device
)

# Build model and fit
model = GMM_reco(
    d=d, N=N,
    sources=sources,
    receivers=receivers,
    x0s=theta_true['x0s'],
    a0s=theta_true['a0s'],
    omega_min=-24.0, omega_max=-20.0,
    device=device,
)

# Generate projections and reconstruct
time_points = torch.linspace(0, 0.5, 50, dtype=torch.float64, device=device)
proj_data = model.generate_projections(time_points, theta_true)
results = model.fit(proj_data, time_points)
```

See [examples/basic_reconstruction.py](examples/basic_reconstruction.py) for a complete working example, or use the split workflow:

```bash
# Run reconstruction → saves results.pt
python scripts/reconstruct.py --N 3 --seed 42

# Analyse saved results → error tables, plots, animations
python scripts/analyse.py data/results/<experiment_dir>/
```

## Project Structure

```
gmm-ct/
├── gmm_ct/                       # Main package
│   ├── __init__.py               # Public API re-exports
│   ├── cli.py                    # CLI entry point
│   ├── core/
│   │   ├── models.py             # GMM_reco class (reconstruction pipeline)
│   │   └── optimizer.py          # L-BFGS root-finding (velocity refinement)
│   ├── estimation/
│   │   ├── omega.py              # Omega estimation utilities
│   │   └── peak_analysis.py      # Peak detection (PeakData)
│   ├── utils/
│   │   ├── generators.py         # Synthetic parameter generation
│   │   ├── geometry.py           # Receiver construction
│   │   └── helpers.py            # Seeds, export, misc
│   ├── visualization/
│   │   ├── animations.py         # GMM & projection animations
│   │   └── publication.py        # Publication-quality figures
│   └── config/
│       └── defaults.py           # ReconstructionConfig dataclass
├── scripts/
│   ├── reconstruct.py            # Run experiment, save results
│   └── analyse.py                # Load results, compute errors, plot
├── tests/
│   ├── conftest.py               # Shared fixtures
│   └── unit/                     # 29 unit tests
├── experiments/
│   ├── demos/                    # Demo & verification scripts
│   ├── stability/                # Stability experiments
│   └── deprecated/               # Superseded scripts (reference only)
├── examples/
│   └── basic_reconstruction.py   # Self-contained end-to-end example
├── docs/
│   ├── guides/quickstart.md
│   └── research_notes/           # Algorithm design notes
├── pyproject.toml                # Build config, deps, tool settings
└── requirements.txt
```

## Key Modules

| Import path | Description |
|---|---|
| `gmm_ct.core.models.GMM_reco` | Main reconstruction class (4-stage pipeline) |
| `gmm_ct.core.optimizer.NewtonRaphsonLBFGS` | L-BFGS root-finder for velocity refinement |
| `gmm_ct.estimation.omega` | Omega estimation utilities |
| `gmm_ct.estimation.peak_analysis.PeakData` | Peak detection data container |
| `gmm_ct.utils.generators` | Synthetic data generation |
| `gmm_ct.utils.geometry` | CT geometry (receiver construction) |
| `gmm_ct.utils.helpers` | Random seeds, parameter export |
| `gmm_ct.config.defaults` | `ReconstructionConfig`, physical constants |
| `gmm_ct.visualization.animations` | Temporal animations |
| `gmm_ct.visualization.publication` | Publication-ready plots |

All commonly used symbols are also re-exported from `gmm_ct` directly:

```python
from gmm_ct import GMM_reco, ReconstructionConfig, generate_true_param, construct_receivers
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- dtaidistance >= 2.3.0
- pytorch-minimize >= 0.0.2

## Testing

```bash
# Run the full test suite
python -m pytest tests/unit/

# Run a specific test
python -m pytest tests/unit/test_dtw_omega.py -v
```

Note: some tests are compute-intensive (DTW smoothness, efficiency comparisons) and may take several minutes on CPU.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format code
black gmm_ct/

# Lint
flake8 gmm_ct/

# Type check
mypy gmm_ct/
```

## Documentation

- [Quick Start Guide](docs/guides/quickstart.md)
- [Research Notes](docs/research_notes/) — algorithm design decisions, experiment results
- [Examples](examples/) — runnable scripts

## License

MIT

## Author

**Daniel Burrows, Can Evren Yarman, Ozan Oktem**

## Citation

```bibtex
@software{gmm_ct,
  author  = {Burrows, Daniel and Yarman, Can Evren and Oktem, Ozan},
  title   = {GMM-CT: Gaussian Mixture Model CT Reconstruction},
  year    = {2026},
  url     = {https://github.com/dwb26/gmm-ct}
}
```
