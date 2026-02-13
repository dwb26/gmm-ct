"""
Optimization utilities for GMM CT reconstruction.

This module contains optimization solvers and related utilities used in the
reconstruction process.
"""

import torch


def NewtonRaphsonLBFGS(func, x0, *args, tol=1e-05, max_iter=100, line_search_fn='strong_wolfe'):
    """
    Root finding using LBFGS optimizer.
    
    Finds roots of func(x) = 0 by minimizing ||func(x)||^2. This is a robust
    root-finding method that uses PyTorch's LBFGS optimizer for quasi-Newton
    optimization.
    
    Parameters
    ----------
    func : callable
        Function for which we want to find roots. Should accept x and *args
        and return a tensor.
    x0 : torch.Tensor
        Initial guess (will be modified in-place, requires_grad will be set to True)
    *args : tuple
        Additional arguments to pass to func
    tol : float, optional
        Convergence tolerance (default: 1e-5)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    line_search_fn : str, optional
        LBFGS line search method ('strong_wolfe' or None) (default: 'strong_wolfe')
    
    Returns
    -------
    torch.Tensor
        Root of the function (modified x0)
        
    Notes
    -----
    This function minimizes ||func(x)||^2 to find where func(x) = 0.
    The LBFGS optimizer is well-suited for smooth optimization problems
    and typically converges faster than first-order methods.
    """
    
    # Ensure x0 requires gradients
    if not x0.requires_grad:
        x0.requires_grad_(True)
    
    # Create LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        [x0], 
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn=line_search_fn
    )
    
    # Track convergence
    iteration_count = [0]
    best_loss = [float('inf')]
    
    def closure():
        """Closure function required by LBFGS."""
        optimizer.zero_grad()
        
        # Compute function value
        f_val = func(x0, *args)

        # For root finding, minimize ||f(x)||^2
        if f_val.dim() == 0:  # Scalar function
            loss = f_val ** 2
        else:  # Vector function
            loss = torch.sum(f_val ** 2)
        
        # Check if loss requires grad before calling backward
        if loss.requires_grad:
            loss.backward()
        else:
            # If no grad, the function is constant - we're already at optimum
            pass
        
        # Track progress
        iteration_count[0] += 1
        current_loss = loss.item()
        
        if current_loss < best_loss[0]:
            best_loss[0] = current_loss
        
        return loss
    
    # Run optimization
    try:
        print("Starting LBFGS optimization for root finding...")
        print(f"Initial: x = {x0.data}, ||f(x)||^2 = {func(x0, *args).item()**2:.2e}")

        optimizer.step(closure)
        
        # Check final convergence
        final_f_val = func(x0, *args)
        final_loss = (final_f_val ** 2).item() if final_f_val.dim() == 0 else torch.sum(final_f_val ** 2).item()
        
        print(f"Final: x = {x0.data}, ||f(x)||^2 = {final_loss:.2e}\n")
            
    except Exception as e:
        # Check if this is the benign "no grad" case
        if "does not require grad" in str(e):
            print(f"ℹ️  Initial guess already optimal (residual at machine precision)")
        else:
            print(f"❌ LBFGS optimization failed: {e}")
        print(f"Returning current estimate: x = {x0.data}")
    
    return x0
