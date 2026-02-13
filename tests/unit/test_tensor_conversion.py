#!/usr/bin/env python
"""Quick test to verify tensor conversion works correctly for joint_with_v0 mode."""

import torch
import pytest
from gmm_ct import GMM_reco
from gmm_ct.utils.geometry import construct_receivers


@pytest.fixture
def gmm_instance():
    """Create a minimal GMM instance with required positional args."""
    device = torch.device('cpu')
    d = 2
    N = 2
    sources = [torch.tensor([-20.0, 0.0], dtype=torch.float64, device=device)]
    receivers = construct_receivers(device, (50, 20.0, -10.0, 10.0))
    x0s = [torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)] * N
    a0s = [torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)] * N
    return GMM_reco(d, N, sources, receivers, x0s, a0s, -24.0, -20.0, device=device)


@pytest.fixture
def test_soln_dict():
    """Create a test solution dictionary with all parameters."""
    return {
        'v0s': [torch.tensor([2.0, -0.5], dtype=torch.float64),
                torch.tensor([1.5, -1.0], dtype=torch.float64)],
        'alphas': [torch.tensor([1.5], dtype=torch.float64),
                   torch.tensor([1.2], dtype=torch.float64)],
        'U_skews': [torch.tensor([[10.0, 2.0], [0.0, 15.0]], dtype=torch.float64),
                    torch.tensor([[12.0, -1.0], [0.0, 18.0]], dtype=torch.float64)],
        'omegas': [torch.tensor([-20.0], dtype=torch.float64),
                   torch.tensor([-18.0], dtype=torch.float64)],
        'x0s': [torch.tensor([1.0, 0.5], dtype=torch.float64),
                torch.tensor([0.0, -0.5], dtype=torch.float64)],
        'a0s': [torch.tensor([0.0, 0.0], dtype=torch.float64),
                torch.tensor([0.0, 0.0], dtype=torch.float64)]
    }


def test_dict_to_tensor_shape(gmm_instance, test_soln_dict):
    """Test dict -> tensor conversion produces correct shape."""
    tensor = gmm_instance.map_from_dict_to_tensor(test_soln_dict, mode='joint_with_v0')
    assert tensor.shape[0] == 2
    assert tensor.shape[1] == 7  # v0(2) + alpha(1) + U_skew(3) + omega(1)


def test_tensor_roundtrip(gmm_instance, test_soln_dict):
    """Test dict -> tensor -> dict roundtrip preserves all parameter keys."""
    tensor = gmm_instance.map_from_dict_to_tensor(test_soln_dict, mode='joint_with_v0')
    reconstructed = gmm_instance.map_from_tensor_to_dict(tensor, mode='joint_with_v0')
    assert 'v0s' in reconstructed
    assert 'alphas' in reconstructed
    assert 'U_skews' in reconstructed
    assert 'omegas' in reconstructed


def test_tensor_roundtrip_values(gmm_instance, test_soln_dict):
    """Test dict -> tensor -> dict roundtrip preserves values."""
    tensor = gmm_instance.map_from_dict_to_tensor(test_soln_dict, mode='joint_with_v0')
    reconstructed = gmm_instance.map_from_tensor_to_dict(tensor, mode='joint_with_v0')

    v0_error = torch.norm(reconstructed['v0s'][0] - test_soln_dict['v0s'][0])
    alpha_error = torch.norm(reconstructed['alphas'][0] - test_soln_dict['alphas'][0])
    omega_error = torch.norm(reconstructed['omegas'][0] - test_soln_dict['omegas'][0])

    assert v0_error < 1e-4
    assert alpha_error < 1e-4
    assert omega_error < 1e-4
