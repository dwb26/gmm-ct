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

### 1. Set Up Geometry

```python
from gmm_ct import construct_receivers
import torch

# Define sources (X-ray sources)
sources = [torch.tensor([-20.0, 0.0], dtype=torch.float64)]

# Define receivers (detector array)
receivers = construct_receivers(
    device=None,  # or torch.device('cuda')
    (100, 20.0, -10.0, 10.0)  # (n_receivers, x, y_min, y_max)
)
```

### 2. Configure Reconstruction

```python
from gmm_ct import ReconstructionConfig

config = ReconstructionConfig(
    n_gaussians=3,
    omega_range=(0.0, 10.0),
    device='cuda',  # or 'cpu'
    max_iterations=500,
    verbose=True
)
```

### 3. Initialize Model

```python
from gmm_ct import GMM_reco

model = GMM_reco(
    d=2,  # 2D problem
    N=3,  # 3 Gaussians
    sources=sources,
    receivers=receivers,
    x0s=[torch.tensor([-8.0, 0.0])] * 3,  # Initial positions
    a0s=[torch.tensor([0.0, -9.81])] * 3,  # Accelerations (gravity)
    omega_min=config.omega_min,
    omega_max=config.omega_max,
    device=config.device
)
```

### 4. Fit to Data

```python
import numpy as np

# Load or generate projection data
proj_data = ...  # Shape: (n_sources, n_receivers, n_timepoints)
time_points = np.linspace(0, 2.0, 50)

# Fit model
theta_estimated = model.fit(proj_data, time_points)
```

### 5. Visualize Results

```python
from gmm_ct.visualization import plot_temporal_gmm_comparison

plot_temporal_gmm_comparison(
    sources=sources,
    receivers=receivers,
    theta_true=theta_true,  # if available
    theta_est=theta_estimated,
    t=time_points,
    output_path='results/comparison.png'
)
```

## Generate Synthetic Data

For testing, you can generate synthetic data:

```python
from gmm_ct import generate_true_param

theta_true = generate_true_param(
    d=2,
    K=3,
    initial_location=torch.tensor([-8.0, 0.0]),
    initial_velocity=torch.tensor([3.0, 0.0]),
    initial_acceleration=torch.tensor([0.0, -9.81]),
    min_rot=0.0,
    max_rot=10.0,
    device=device
)

# Generate projections from true parameters
proj_data = model.generate_projections(time_points, theta_true)
```

## Next Steps

- Check out [examples/](../examples/) for complete working examples
- Read the [API documentation](../api/) for detailed function references
- Explore [advanced usage](advanced_usage.md) for custom configurations
