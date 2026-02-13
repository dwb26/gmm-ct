"""
Test script to verify rotation model implementation.

This tests whether the rotation formula θ(t) = 2π·ω·t produces the expected
angular displacement at specific times.
"""

from gmm_ct import GMM_reco, construct_receivers
import torch

def main():
    # Setup basic configuration
    device = torch.device('cpu')
    N = 2  # Number of Gaussians
    d = 2  # 2D problem
    
    # Construct minimal GMM_reco instance
    # construct_receivers expects (n_rcvrs, x1, x2_min, x2_max)
    receivers = construct_receivers(device, (21, 3.0, -2.0, 2.0))
    sources = [torch.tensor([-3.0, 0.0], dtype=torch.float64, device=device)]
    
    # Dummy parameters
    x0s = [torch.tensor([0.0, 0.0], dtype=torch.float64, device=device) for _ in range(N)]
    a0s = [torch.tensor([0.0, -9.81], dtype=torch.float64, device=device) for _ in range(N)]
    
    gmm_reco = GMM_reco(
        d=d,
        N=N,
        sources=sources,
        receivers=receivers,
        x0s=x0s,
        a0s=a0s,
        omega_min=-20.0,
        omega_max=20.0,
        device=device,
        output_dir=Path('./test_output')
    )
    
    # Test with omega = -18.5239 Hz (claimed true value)
    omega_test = -18.5239
    
    # Test at two times:
    # 1. Theoretical quarter-turn time: 1/(4×18.5239) = 0.0135 sec
    # 2. Observed quarter-turn time: 0.125 sec
    test_times = [0.0135, 0.125]
    
    print("\nThis test will show:")
    print("  - At t=0.0135 sec (theoretical quarter-turn): should get θ ≈ 90°")
    print("  - At t=0.125 sec (observed quarter-turn): should get θ ≈ 832°")
    print("\nIf the second case shows ~90° instead, there's a bug in the rotation model.")
    
    gmm_reco.test_rotation_period(omega=omega_test, test_times=test_times)

if __name__ == '__main__':
    main()
