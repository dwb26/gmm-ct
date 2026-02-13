# Deprecated Experiments

The files in this directory are deprecated and no longer maintained.

## Files

- **`compare_stability_experiments.py`** — Stability experiment comparison script. Superseded by `experiments/stability/compare_stability_experiments.py`.
- **`single_projection_experiment.py`** — Single projection recovery experiment. Contains hardcoded paths to the old project location and a `sys.path` hack pointing to a nonexistent `src/` directory.
- **`__init__.py.bak`** — Former top-level `experiments/__init__.py`. No longer needed since `experiments/` is not an importable package.

These files were moved here during the migration to the modular `gmm_ct` package structure (Feb 2026). They may reference old import paths and will not run without modification.
