"""
Numerical solvers for GMM-CT reconstruction.

Contains root-finding and optimization solvers used during the
reconstruction pipeline.
"""

import torch


def NewtonRaphsonLBFGS(func, x0, *args, tol=1e-05, max_iter=100,
                       line_search_fn='strong_wolfe'):
    """
    Root finding using L-BFGS optimizer.

    Finds roots of ``func(x) = 0`` by minimizing ``||func(x)||^2``.
    Uses PyTorch's L-BFGS optimizer for quasi-Newton optimization.

    Parameters
    ----------
    func : callable
        Function for which we want to find roots.  Should accept ``x``
        and ``*args`` and return a tensor.
    x0 : torch.Tensor
        Initial guess (modified in-place; ``requires_grad`` will be set
        to ``True``).
    *args : tuple
        Additional arguments passed to *func*.
    tol : float, optional
        Convergence tolerance (default: 1e-5).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    line_search_fn : str, optional
        L-BFGS line search method (default: ``'strong_wolfe'``).

    Returns
    -------
    torch.Tensor
        Root of the function (modified ``x0``).
    """
    if not x0.requires_grad:
        x0.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x0],
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        line_search_fn=line_search_fn,
    )

    iteration_count = [0]
    best_loss = [float('inf')]

    def closure():
        """Closure function required by L-BFGS."""
        optimizer.zero_grad()
        f_val = func(x0, *args)

        if f_val.dim() == 0:
            loss = f_val ** 2
        else:
            loss = torch.sum(f_val ** 2)

        if loss.requires_grad:
            loss.backward()

        iteration_count[0] += 1
        current_loss = loss.item()
        if current_loss < best_loss[0]:
            best_loss[0] = current_loss

        return loss

    try:
        print("Starting LBFGS optimization for root finding...")
        print(f"Initial: x = {x0.data}, ||f(x)||^2 = {func(x0, *args).item()**2:.2e}")

        optimizer.step(closure)

        final_f_val = func(x0, *args)
        final_loss = (
            (final_f_val ** 2).item()
            if final_f_val.dim() == 0
            else torch.sum(final_f_val ** 2).item()
        )
        print(f"Final: x = {x0.data}, ||f(x)||^2 = {final_loss:.2e}\n")

    except Exception as e:
        if "does not require grad" in str(e):
            print("ℹ️  Initial guess already optimal (residual at machine precision)")
        else:
            print(f"❌ LBFGS optimization failed: {e}")
        print(f"Returning current estimate: x = {x0.data}")

    return x0
