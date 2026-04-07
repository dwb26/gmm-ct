# Changelog

All notable changes to the GMM-CT project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-03

### Changed
- Replaced all `print()` statements in library code with Python `logging`
- Organized imports (alphabetized, grouped by stdlib / third-party / local)
- Added `logging.NullHandler()` in `__init__.py` (library best practice)
- Updated `pyproject.toml` URLs to point to actual GitHub repository

### Fixed
- `generators.py`: removed dead `generate = False` branch with hard-coded velocities;
  velocity sampling now always uses the accept/reject loop
- `scripts/reconstruct.py`: un-commented `d = 2` so the script runs without `NameError`
- `reconstruct.py`: removed duplicate `plot_projection_modes` import

### Removed
- `_inspect_results.py` (ad-hoc inspection script)
- `QUICK_REFERENCE.md` (migration-era cruft with hard-coded paths)
- `requirements.txt` (redundant with `pyproject.toml`)
- Commented-out animation-frame saving block in `reconstruct.py`

## [0.1.0] - 2026-02-11

### Added
- Initial restructured release
- Core GMM reconstruction algorithm with motion estimation
- FFT-based omega estimation
- DTW-based trajectory alignment
- Hungarian algorithm for peak assignment
- Comprehensive visualization tools
- Publication-quality plotting utilities
- Configuration management system
- Proper package structure with submodules
- Test suite organization
- Documentation framework
- Example scripts

### Changed
- Restructured codebase from flat to modular hierarchy
- Split large monolithic files into focused modules
- Reorganized imports for better clarity
- Improved configuration with dataclass-based system

### Improved
- Code organization and maintainability
- Documentation and docstrings
- Testing structure
- Separation of concerns (core/utils/visualization/estimation)

## [0.0.1] - Pre-release

### Initial Development
- Prototype implementation
- Research and experimental code
- Various stability experiments
- DTW and soft-DTW testing
- Omega estimation methods
