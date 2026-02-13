# GMM Reconstruction Stability Experiment

## Overview

This experiment evaluates the **long-term stability** and **accuracy** of the GMM_reco reconstruction algorithm as a function of model complexity (number of Gaussians N).

## What It Does

For each value of N (number of Gaussians):
1. Runs multiple simulations with different random seeds
2. Measures reconstruction accuracy in two spaces:
   - **Parameter Space**: How well we recover (α, μ, U, ω) for each Gaussian
   - **Projection Space**: How well reconstructed projections match true projections
3. Tracks computation time
4. Aggregates statistics across all repetitions

## Accuracy Metrics

Both metrics are expressed as **percentages** where:
- **100%** = Perfect reconstruction
- **0%** = Complete failure

### Parameter Space Accuracy
```python
For each Gaussian k:
  - Alpha error: |α_true - α_est| / α_true
  - Mu error: ||μ_true - μ_est|| / ||μ_true||
  - U error: ||U_true - U_est||_F / ||U_true||_F
  - Omega error: ||ω_true - ω_est|| / ||ω_true||

Accuracy = 100 × (1 - mean(all_relative_errors))
```

### Projection Space Accuracy
```python
Accuracy = 100 × (1 - ||P_true - P_est|| / ||P_true||)
```

## Usage

### Quick Start
```bash
cd src/gmm_ct
python run_stability_experiment.py
```

### Custom Configuration
Edit `run_stability_experiment.py` to modify:
```python
N_values = [3, 5, 7, 10, 15, 20]  # Which N values to test
N_simulations_per_N = 10           # Repetitions per N
base_seed = 100                     # Starting random seed
```

### Advanced Usage
Import and use programmatically:
```python
from stability_experiment import run_stability_experiment, plot_stability_results

# Run experiment
df, output_dir = run_stability_experiment(
    N_values=[3, 5, 10],
    N_simulations_per_N=5,
    base_seed=42
)

# Generate plots
plot_stability_results(df, output_dir)

# Analyze results
print(df.groupby('N')['param_accuracy'].describe())
```

## Output

The experiment generates:

### 1. CSV Data
`stability_results.csv` with columns:
- `N`: Number of Gaussians
- `seed`: Random seed used
- `param_accuracy`: Parameter space accuracy (%)
- `proj_accuracy`: Projection space accuracy (%)
- `computation_time`: Time taken (seconds)

### 2. Plots

**`stability_analysis.pdf/png`**
- Left: Accuracy vs N (both parameter and projection space)
- Right: Computation time vs N
- Includes mean ± std shaded regions

**`stability_boxplots.pdf`**
- Distribution of accuracies across all seeds for each N
- Separate boxplots for parameter and projection space

## Interpretation

Good reconstruction should show:
- ✅ High accuracy (>95%) across all N values
- ✅ Low variance (narrow error bands)
- ✅ Consistent performance across different random seeds
- ⚠️ Graceful degradation (if any) as N increases

Red flags:
- ❌ Large accuracy drops at certain N values
- ❌ High variance (wide error bands)
- ❌ Bimodal distributions in boxplots

## Integration with Main Code

The experiment reuses all infrastructure from `main.py`:
- Same data generation pipeline
- Same GMM_reco fitting procedure
- Same parameter matching/reordering
- Compatible device handling (CPU/GPU)

## Example Results

After running, you'll see output like:
```
📊 Summary Statistics:
   N  param_accuracy_mean  param_accuracy_std  proj_accuracy_mean  proj_accuracy_std
   3                98.45                0.23               99.12               0.15
   5                97.89                0.41               98.87               0.28
  10                96.34                0.67               98.23               0.45
  15                95.12                1.02               97.54               0.71
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib

All dependencies should already be installed if you can run `main.py`.

## Tips

1. **Start small**: Test with `N_values=[3, 5]` and `N_simulations_per_N=2` first
2. **Scale up**: Once working, increase to `N_simulations_per_N=20+` for publication
3. **Parallelization**: For large experiments, consider running different N values in parallel
4. **GPU**: Ensure GPU is available for faster computation (check device output)

## Troubleshooting

**"Out of memory"**: Reduce N_values or use smaller N_projs
**"Convergence failures"**: Check optimization settings in GMM_reco.fit()
**"Slow performance"**: Ensure GPU is being used (check device output)

---

Created as part of the GaussianMixtureCT project stability analysis.
