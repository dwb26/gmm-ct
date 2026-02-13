# 🎯 GMM-CT Quick Reference

## 📍 New Location
```
/Users/danburrows/Projects/gmm-ct/
```

## 🚀 Quick Start Commands

### 1. Open Workspace
```bash
cd /Users/danburrows/Projects/gmm-ct
code .
```

### 2. Validate Migration
```bash
python validate_migration.py
```

### 3. Install Package
```bash
pip install -e .
# or with dev tools:
pip install -e ".[dev]"
```

### 4. Test Installation
```bash
python -c "from gmm_ct.config.defaults import ReconstructionConfig; print('✅ OK')"
```

### 5. Run Tests
```bash
pytest tests/unit/ -v
```

## 📦 Import Cheat Sheet

### Core Functionality
```python
from gmm_ct import GMM_reco, ReconstructionConfig
from gmm_ct.utils import generate_true_param, construct_receivers
from gmm_ct.visualization import plot_temporal_gmm_comparison
```

### Within Package (relative)
```python
# In gmm_ct/core/models.py:
from ..utils.generators import generate_true_param
from ..config.defaults import GRAVITATIONAL_ACCELERATION
from .optimizer import NewtonRaphsonLBFGS
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview & quick start |
| `MIGRATION_COMPLETE.md` | Full migration report |
| `MIGRATION_GUIDE.md` | Import update instructions |
| `pyproject.toml` | Package configuration |
| `validate_migration.py` | Check migration status |

## 🔧 Common Tasks

### Add New Feature
```bash
# 1. Create module in appropriate directory
# 2. Add to relevant __init__.py
# 3. Write tests in tests/unit/
# 4. Update examples if needed
```

### Run Specific Tests
```bash
pytest tests/unit/test_models.py -v
pytest tests/unit/test_omega_estimation.py -v
```

### Generate Documentation
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Generate (once sphinx is set up)
cd docs && make html
```

### Format Code
```bash
black gmm_ct/
flake8 gmm_ct/
```

## 🐛 Troubleshooting

### ImportError: No module named 'methods'
**Fix**: Update imports in the file to use new structure
```python
# OLD: from methods import generate_true_param
# NEW: from gmm_ct.utils.generators import generate_true_param
```

### ImportError: No module named 'gmm_ct'
**Fix**: Install package or add to path
```bash
pip install -e .
```

### ModuleNotFoundError in tests
**Fix**: Ensure pytest runs from project root
```bash
cd /Users/danburrows/Projects/gmm-ct
pytest tests/unit/ -v
```

## 📊 Project Stats

- **Modules**: 12+ focused modules (was 3 large files)
- **Tests**: 30+ test files organized
- **Experiments**: 15+ research scripts
- **Documentation**: 6 research notes + guides
- **Lines Refactored**: ~3000+ lines reorganized

## 🎓 Architecture

```
Core Logic        → gmm_ct/core/
Estimation        → gmm_ct/estimation/  
Utilities         → gmm_ct/utils/
Visualization     → gmm_ct/visualization/
Configuration     → gmm_ct/config/
Tests            → tests/
Research         → experiments/
Examples         → examples/
Documentation    → docs/
```

## ✅ Migration Checklist

- [x] Directory structure created
- [x] Files migrated
- [x] Package configuration added
- [x] Documentation created
- [ ] **Imports updated** ← NEXT STEP
- [ ] Package installed
- [ ] Tests passing
- [ ] Examples working

## 🎯 Your Next Action

**Open the workspace and start fixing imports:**

```bash
cd /Users/danburrows/Projects/gmm-ct
code .
```

Then follow `MIGRATION_GUIDE.md` to update imports!

---
*Migration completed: 2026-02-11*
