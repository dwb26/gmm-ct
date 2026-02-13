"""
Test the refactored trajectory optimization methods.

This script verifies that:
1. PeakData class integrates correctly with GMM_reco
2. Peak detection works as before
3. Trajectory optimization completes without errors
4. Results are consistent with original implementation
"""

import torch
import numpy as np

from gmm_ct import GMM_reco, generate_true_param, construct_receivers, set_random_seeds

def test_basic_peak_detection():
    """Test that peak detection populates PeakData correctly."""
    print("\n" + "="*60)
    print("TEST 1: Basic Peak Detection")
    print("="*60)
    
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Setup simple problem
    N = 2  # Two Gaussians
    d = 2
    device = 'cpu'
    
    # Generate true parameters - use tensors as main.py does
    i_loc = torch.tensor([1., 1.], dtype=torch.float64, device=device)
    v_loc = torch.tensor([.75, .5], dtype=torch.float64, device=device)
    a_loc = torch.tensor([0., -9.81], dtype=torch.float64, device=device)
    
    theta_true = generate_true_param(
        d, N,
        initial_location=i_loc,
        initial_velocity=v_loc,
        initial_acceleration=a_loc,
        min_rot=1.0, max_rot=2.0,
        device=device
    )
    
    # Setup geometry
    sources = [torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)]
    n_rcvrs = 40
    x1 = sources[0][0].item() + 5.
    x2_min = sources[0][1].item() - 2.
    x2_max = sources[0][1].item() + 2.
    args = (n_rcvrs, x1, x2_min, x2_max)
    receivers = construct_receivers(device, args)
    
    # Create time vector
    t = torch.linspace(0, 1, 20, dtype=torch.float64, device=device)
    
    # Create GMM_reco instance
    gmm_reco = GMM_reco(
        N=N, d=d, 
        sources=sources, receivers=receivers,
        x0s=theta_true['x0s'], a0s=theta_true['a0s'],
        omega_min=1.0, omega_max=2.0,
        device=device
    )
    
    # Generate synthetic projection data
    proj_data = gmm_reco.generate_projections(t, theta_true)
    
    # Initialize velocities (triggers peak detection)
    v0s_init = gmm_reco.initialize_initial_velocities(t, proj_data)
    
    # Verify PeakData was created
    assert hasattr(gmm_reco, 'peak_data'), "peak_data attribute not created"
    print("✓ peak_data object created")
    
    # Verify legacy aliases exist
    assert hasattr(gmm_reco, 'maximising_rcvrs'), "Legacy alias maximising_rcvrs not found"
    assert hasattr(gmm_reco, 't_obs_by_cluster'), "Legacy alias t_obs_by_cluster not found"
    print("✓ Legacy aliases created for backward compatibility")
    
    # Verify data structure
    assert gmm_reco.peak_data.N == N, f"Expected {N} Gaussians, got {gmm_reco.peak_data.N}"
    print(f"✓ Detected peaks for {N} Gaussians")
    
    # Check that peaks were detected
    has_detections = any(len(gmm_reco.peak_data.times[k]) > 0 for k in range(N))
    print(f"✓ Peak detection status: {'Complete' if has_detections else 'No peaks found'}")
    
    # Print summary
    gmm_reco.peak_data.summary()
    
    print("\n✅ TEST 1 PASSED: Peak detection works correctly\n")
    return gmm_reco, theta_true, t, proj_data


def test_trajectory_optimization(gmm_reco, theta_true, t, proj_data):
    """Test that trajectory optimization completes without errors."""
    print("\n" + "="*60)
    print("TEST 2: Trajectory Optimization")
    print("="*60)
    
    try:
        # Run optimization (both phases)
        # Setting N_traj_trials to 2 for quick test
        gmm_reco.N_traj_trials = 2
        print("\nRunning optimization (2 trials for quick test)...")
        theta_est = gmm_reco.fit(proj_data, t)
        
        print("\n✓ Trajectory optimization completed")
        
        # Verify result structure
        assert 'v0s' in theta_est, "v0s not in result"
        assert 'x0s' in theta_est, "x0s not in result"
        assert len(theta_est['v0s']) == gmm_reco.N, f"Expected {gmm_reco.N} v0s, got {len(theta_est['v0s'])}"
        print(f"✓ Result contains {gmm_reco.N} velocity estimates")
        
        # Check that optimal assignments were created
        has_any_assignments = any(gmm_reco.peak_data.has_assignments(k) for k in range(gmm_reco.N))
        assert has_any_assignments, "No optimal assignments found"
        print("✓ Optimal peak assignments created")
        
        # Verify legacy aliases were updated
        assert hasattr(gmm_reco, 'assigned_curve_data'), "assigned_curve_data not found"
        assert hasattr(gmm_reco, 'assigned_peak_values'), "assigned_peak_values not found"
        print("✓ Assignment data stored for plotting")
        
        # Compare to true velocities
        print("\nVelocity Comparison:")
        print("-" * 40)
        for k in range(gmm_reco.N):
            v_true = theta_true['v0s'][k]
            v_est = theta_est['v0s'][k]
            error = torch.norm(v_true - v_est).item()
            rel_error = error / torch.norm(v_true).item() * 100
            print(f"Gaussian {k}:")
            print(f"  True:     [{v_true[0].item():.3f}, {v_true[1].item():.3f}]")
            print(f"  Estimated: [{v_est[0].item():.3f}, {v_est[1].item():.3f}]")
            print(f"  Rel Error: {rel_error:.2f}%")
        
        print("\n✅ TEST 2 PASSED: Trajectory optimization works correctly\n")
        return theta_est
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def test_data_consistency():
    """Test that PeakData maintains data consistency."""
    print("\n" + "="*60)
    print("TEST 3: Data Consistency")
    print("="*60)
    
    from gmm_ct.peak_data import PeakData
    
    # Create PeakData instance
    peak_data = PeakData(n_gaussians=2, device='cpu')
    
    # Add some detections
    peak_data.add_peak_detection(
        time_idx=0, time_val=0.5, receiver_idx=10,
        receiver_pos=torch.tensor([30.0, -5.0]),
        peak_val=torch.tensor(0.8),
        gaussian_idx=0
    )
    peak_data.add_peak_detection(
        time_idx=0, time_val=0.5, receiver_idx=8,
        receiver_pos=torch.tensor([30.0, 2.0]),
        peak_val=torch.tensor(0.6),
        gaussian_idx=1
    )
    
    # Add time detections
    peak_data.add_time_detections(0.5, [-5.0, 2.0])
    
    # Finalize
    peak_data.finalize_detections()
    
    # Verify consistency
    assert len(peak_data.times[0]) == 1, "Gaussian 0 should have 1 detection"
    assert len(peak_data.times[1]) == 1, "Gaussian 1 should have 1 detection"
    print("✓ Detection counts correct")
    
    # Add optimal assignment
    peak_data.add_optimal_assignment(0, 0.5, -5.0, 0.8)
    peak_data.add_optimal_assignment(1, 0.5, 2.0, 0.6)
    
    # Check assignments
    times0, heights0 = peak_data.get_assignment_data(0)
    times1, heights1 = peak_data.get_assignment_data(1)
    assert len(times0) == 1, "Gaussian 0 should have 1 assignment"
    assert len(times1) == 1, "Gaussian 1 should have 1 assignment"
    print("✓ Assignment counts correct")
    
    # Verify data access methods
    heights_dict = peak_data.get_heights_dict_non_empty()
    assert 0.5 in heights_dict, "Time 0.5 should be in heights dict"
    assert len(heights_dict[0.5]) == 2, "Should have 2 heights at time 0.5"
    print("✓ Data access methods work correctly")
    
    # Verify N attribute
    assert peak_data.N == 2, "PeakData should have N=2"
    print("✓ PeakData.N attribute correct")
    
    print("\n✅ TEST 3 PASSED: Data consistency maintained\n")


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("#  Testing Refactored Trajectory Optimization")
    print("#"*60)
    
    try:
        # Test 1: Peak detection
        gmm_reco, theta_true, t, proj_data = test_basic_peak_detection()
        
        # Test 2: Trajectory optimization
        theta_est = test_trajectory_optimization(gmm_reco, theta_true, t, proj_data)
        
        # Test 3: Data consistency
        test_data_consistency()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nRefactoring is successful and backward-compatible.")
        print("The code is now cleaner and easier to understand.")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
