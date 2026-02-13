"""Pytest configuration and fixtures for GMM-CT tests."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def device():
    """Get the default device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create a temporary output directory for test results."""
    return tmp_path_factory.mktemp("test_output")


@pytest.fixture
def simple_geometry(device):
    """Create a simple CT geometry for testing."""
    sources = [torch.tensor([-20.0, 0.0], dtype=torch.float64, device=device)]
    
    # Create receivers
    n_receivers = 50
    x1 = 20.0
    x2_min, x2_max = -10.0, 10.0
    x2 = torch.linspace(x2_min, x2_max, n_receivers, dtype=torch.float64, device=device)
    x2 = torch.flip(x2, dims=[0])
    receivers = [[torch.tensor([x1, x2_val], dtype=torch.float64, device=device) 
                  for x2_val in x2]]
    
    return {
        'sources': sources,
        'receivers': receivers,
        'n_receivers': n_receivers,
    }


@pytest.fixture
def simple_gmm_params(device):
    """Create simple GMM parameters for testing."""
    return {
        'n_gaussians': 2,
        'x0s': [torch.tensor([-8.0, 0.0], dtype=torch.float64, device=device)] * 2,
        'v0s': [
            torch.tensor([3.0, 2.0], dtype=torch.float64, device=device),
            torch.tensor([3.5, 1.5], dtype=torch.float64, device=device),
        ],
        'a0s': [torch.tensor([0.0, -9.81], dtype=torch.float64, device=device)] * 2,
        'omega_min': 0.0,
        'omega_max': 10.0,
    }


@pytest.fixture
def sample_projection_data(simple_geometry, device):
    """Generate sample projection data for testing."""
    n_receivers = simple_geometry['n_receivers']
    n_timepoints = 20
    
    # Create synthetic projection data
    proj_data = torch.randn(1, n_receivers, n_timepoints, dtype=torch.float64, device=device)
    time_points = torch.linspace(0, 2.0, n_timepoints, dtype=torch.float64, device=device)
    
    return {
        'proj_data': proj_data,
        'time_points': time_points,
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup not needed for random seeds
