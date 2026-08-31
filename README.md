# GMM-CT: Gaussian Mixture Model CT Reconstruction

A Python package for reconstructing dynamic objects in CT imaging using Gaussian Mixture Models with motion estimation.

![Example Results: 5-particle reconstruction](temporal_gmm_comparison.pdf)
*Three time samples with the unknown target state objects (left), the reconstructed estimates (right), and the two corresponding sinograms by which the reconstruction is informed (center).*

## Overview

GMM-CT recovers the morphology and motion of objects undergoing projectile motion and rotation from limited CT projection data. Each particle is modelled as a Gaussian component with anisotropic covariance, and the package jointly estimates:

- Attenuation coefficients ($\alpha$) and shape matrices ($U$)
- Initial velocities ($v_0$)
- Angular velocities ($\omega$)
- Multi-object assignment via the Hungarian algorithm

### Reconstruction Pipeline

The reconstruction proceeds in four stages:

| Stage | What is optimised | Method | Loss |
|---|---|---|---|
| **1. Trajectory** | Initial velocities $v_0$ | Multi-start L-BFGS (up to 1 500 iters) + Newton–Raphson refinement | L2 on peak receiver heights, Hungarian assignment |
| **1.5a. ω initialization** | Angular velocities $\omega$ | Per-Gaussian residual-sinogram grid search (200 candidates) | L2 on residual sinogram |
| **1.5b. α initialization** | Attenuation coefficients $\alpha$ | Non-negative least squares (NNLS) | Closed-form |
| **2. Joint optimization** | $\alpha$, $U_{\text{skew}}$, $\omega$ | Multi-start L-BFGS (up to 1 000 iters) | Smooth L1 (Huber) on full projections |

**Physical model:** Each Gaussian follows a trajectory $\mu_k(t) = x_0 + v_0\,t + \tfrac{1}{2}\,a_0\,t^2$ with in-plane rotation $R(2\pi\omega t)$. Projections are computed via a closed-form ray transform of the rotated Gaussian. Stage 1 decouples trajectory estimation from morphology by using isotropic Gaussians; Stages 1.5-2 then recover angular velocity, attenuation, and anisotropic shape.

## Quick Start

### Installation

```bash
# Install in development mode
pip install -e .

# Or with dev tools (pytest, black, etc.)
pip install -e ".[dev]"
```

### Basic Usage

See [docs/guides/quickstart](docs/guides/quickstart.md) for a minimum working example or
[examples/basic_reconstruction.py](examples/basic_reconstruction.py) for a complete working example.

### YAML + Command Line

For a cleaner separation of data generation and reconstruction, use the
YAML-driven CLI:

```bash
# 1. Generate synthetic projection data
gmm-ct simulate --config configs/simulate.yaml
# Prints: "Data saved to: data/simulated/<TIMESTAMP>_seed9_N5/"

# 2. Run reconstruction, passing the generated data path via --data
gmm-ct reconstruct --config configs/reconstruct.yaml \
    --data data/simulated/<TIMESTAMP>_seed9_N5/projections.pt
```

CLI flags can override config values:

```bash
gmm-ct simulate    --config configs/simulate.yaml --seed 99
gmm-ct reconstruct --config configs/reconstruct.yaml --device cuda \
    --data data/simulated/<TIMESTAMP>_seed9_N5/projections.pt
```

> **Note for reviewers**: `data/simulated/` is gitignored, so you will
> need to run `gmm-ct simulate` first. The simulate command prints the
> exact `--data` path to pass to `gmm-ct reconstruct`.


## Project Structure

```
gmm-ct/
├── gmm_ct/                       # Main package
│   ├── __init__.py               # Public API re-exports
│   ├── cli.py                    # CLI entry point (simulate / reconstruct)
│   ├── simulation.py             # Synthetic data generation runner
│   ├── reconstruct.py            # Reconstruction runner (loads data, runs fit)
│   ├── config.py                 # All config dataclasses + YAML loaders
│   ├── model.py                  # GMM_reco + PeakData (4-stage pipeline)
│   ├── utils.py                  # Geometry, parameter generation, helpers
│   └── visualization/
│       ├── animations.py         # GMM & projection animations
│       ├── publication.py        # Publication-quality figures
│       └── diagnostics.py        # Diagnostic plots (trajectory, peaks)
├── configs/
│   ├── simulate.yaml             # Example simulation config
│   └── reconstruct.yaml          # Example reconstruction config
├── scripts/
│   ├── reconstruct.py            # Standalone script (generate + reconstruct)
│   ├── run_experiments.py        # Batch experiment runner (N-sweep, seeds)
│   └── analyse.py                # Load results, compute errors, plot
├── experiments/
│   └── stability/                # N-scaling & stability experiment code
├── notebooks/                    # Exploratory Jupyter notebooks
├── examples/
│   └── basic_reconstruction.py   # Self-contained end-to-end example
├── docs/
│   └── guides/quickstart.md
├── data/
│   ├── simulated/                # Raw projection data (gitignored — generate with simulate)
│   └── results/                  # Reconstruction outputs (gitignored)
└── pyproject.toml                # Build config, deps, tool settings
```

## Key Modules

| Module | Contents |
|---|---|
| `gmm_ct.model` | `GMM_reco` (4-stage pipeline), `PeakData`, `NewtonRaphsonLBFGS` |
| `gmm_ct.config` | Config dataclasses, `load_reconstruct_config`, `load_simulate_config`, `GRAVITATIONAL_ACCELERATION` |
| `gmm_ct.utils` | `construct_receivers`, `generate_true_param`, `set_random_seeds`, `export_parameters` |
| `gmm_ct.simulation` | Synthetic data generation runner |
| `gmm_ct.reconstruct` | Reconstruction runner (data loading + fit) |
| `gmm_ct.visualization.animations` | Temporal animations |
| `gmm_ct.visualization.publication` | Publication-ready plots |
| `gmm_ct.visualization.diagnostics` | Diagnostic plots (trajectory, peak heights) |

All commonly used symbols are re-exported from `gmm_ct` directly:

```python
from gmm_ct import GMM_reco, generate_true_param, construct_receivers, GRAVITATIONAL_ACCELERATION
from gmm_ct import load_reconstruct_config, run_reconstruction  # YAML workflow
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- dtaidistance >= 2.3.0
- pytorch-minimize >= 0.0.2
- PyYAML >= 6.0

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
- [Examples](examples/) — runnable scripts

## License

MIT

## Authors

**Daniel Burrows, Can Evren Yarman, Ozan Oktem**

## Citation

A journal submission is in preparation. In the meantime, please cite this repository:

```bibtex
@software{gmm_ct,
  author  = {Burrows, Daniel and Yarman, Can Evren and Öktem, Ozan},
  title   = {GMM-CT: Gaussian Mixture Model CT Reconstruction},
  year    = {2026},
  url     = {https://github.com/dwb26/gmm-ct}
}
```
