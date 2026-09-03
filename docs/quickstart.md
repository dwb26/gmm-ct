# Quick Start Guide

## Installation

### From Source

```bash
git clone <repository-url>
cd gmm-ct
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Basic Reconstruction

There is no separate config object — `GMM_reco` is constructed directly.
See [examples/basic_reconstruction.py](../../examples/basic_reconstruction.py) for
a complete self-contained script.

### 1. Set Up Geometry

```python
import torch
from gmm_ct import construct_receivers, GRAVITATIONAL_ACCELERATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# X-ray source position
sources = [torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)]

# Receiver array: (n_receivers, x_coord, y_min, y_max)
receivers = construct_receivers(device, (128, 4.0, -3.0, 3.0))
```

### 2. Generate Synthetic Data

```python
from gmm_ct import generate_true_param

N = 3          # number of Gaussian components
D = 2          # spatial dimension
OMEGA_MIN, OMEGA_MAX = 2.0, 6.0

theta_true = generate_true_param(
    D, N,
    initial_location=torch.tensor([1.0, 1.0], dtype=torch.float64, device=device),
    initial_velocity=torch.tensor([0.75, 0.5], dtype=torch.float64, device=device),
    initial_acceleration=torch.tensor([0.0, -GRAVITATIONAL_ACCELERATION],
                                      dtype=torch.float64, device=device),
    min_rot=OMEGA_MIN,
    max_rot=OMEGA_MAX,
    device=device,
)
```

### 3. Simulate Projections

```python
from gmm_ct import GMM_reco

# Use known physics (x0s, a0s) to build a forward model and generate data
t = torch.linspace(0.0, 2.0, 65, dtype=torch.float64, device=device)

forward_model = GMM_reco(
    D, N, sources, receivers,
    x0s=theta_true['x0s'],
    a0s=theta_true['a0s'],
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    device=device,
    save_diagnostics=False,
)
proj_data = forward_model.generate_projections(t, theta_true)
```

### 4. Reconstruct

```python
model = GMM_reco(
    D, N, sources, receivers,
    x0s=theta_true['x0s'],   # initial positions — assumed known
    a0s=theta_true['a0s'],   # accelerations — assumed known
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    device=device,
    save_diagnostics=False,  # set True to write Stage 1 diagnostic plots
)
theta_estimated = model.fit(proj_data, t)
```

### 5. Visualize Results

```python
from gmm_ct.visualization.publication import (
    plot_temporal_gmm_comparison,
    reorder_theta_to_match_true,
    animate_temporal_gmm_comparison
)

# Match estimated Gaussians to ground truth by velocity for colour coding
theta_estimated, _ = reorder_theta_to_match_true(theta_true, theta_estimated, N)

# time_indices [8, 20, 35] → t ≈ 0.25 s, 0.625 s, 1.09 s
# (objects remain within the receiver y-range before gravity pulls them out at ~1.25 s)
plot_temporal_gmm_comparison(
    sources, receivers, theta_true, theta_estimated, t, N, D,
    time_indices=[8, 20, 35],
    filename='results/comparison.pdf',
)

anim = animate_temporal_gmm_comparison(
            sources, receivers, theta_true, theta_estimated, t, N, D,
            filename="results/temporal_gmm_comparison.mp4",
        )
```

## Next Steps

- Check out [examples/](../examples/) for complete working examples
- Use the [CLI workflow](../../README.md#yaml--command-line) for YAML-driven simulate → reconstruct
