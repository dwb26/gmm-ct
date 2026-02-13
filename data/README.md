# Data Directory

This directory contains data used for GMM-CT reconstruction.

## Structure

- `sample/` - Small sample datasets for testing and examples
- `results/` - Output directory for reconstruction results
  - `plots/` - Static plots and figures
  - `animations/` - Animation outputs

## Data Formats

### Projection Data

Projection data should be provided as NumPy arrays or PyTorch tensors with shape:
- `(n_sources, n_receivers, n_timepoints)` for single-source geometry
- `(n_timepoints, n_receivers)` for flattened format

### Time Points

Time points should be a 1D array of float values representing the acquisition times.

## Usage

```python
import numpy as np
from gmm_ct import GMM_reco

# Load your projection data
proj_data = np.load('sample/projections.npy')
time_points = np.load('sample/times.npy')

# Run reconstruction
model = GMM_reco(...)
results = model.fit(proj_data, time_points)
```

## Note

Large data files are gitignored. Only small sample datasets and READMEs are tracked.
