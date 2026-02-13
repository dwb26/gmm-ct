"""
Geometric utilities for CT imaging setup.

This module contains functions for constructing receiver geometries,
rotation matrices, and other geometric transformations used in CT reconstruction.
"""

import torch


def construct_receivers(device=None, *args):
    """
    Construct receivers based on specified coordinates.
    
    Currently supports flat, cone beam geometries for 2D applications.
    
    Parameters
    ----------
    device : torch.device, optional
        Device to place tensors on (default: CPU)
    *args : tuple
        Receiver configuration as (n_rcvrs, x1, x2_min, x2_max) where:
        - n_rcvrs: Number of receivers
        - x1: X-coordinate of the receiver line
        - x2_min: Minimum Y-coordinate
        - x2_max: Maximum Y-coordinate
    
    Returns
    -------
    list
        List of receiver tensors, each representing a source's receiver array.
        Each receiver is a d-dimensional tensor.
        
    Notes
    -----
    The receivers are constructed on a vertical line at x1, spanning from
    x2_min to x2_max. The coordinates are flipped to maintain conventional
    CT geometry orientation.
    """
    if device is None:
        device = torch.device('cpu')
        
    n_rcvrs, x1, x2_min, x2_max = args[0]
    x2 = torch.linspace(x2_min, x2_max, n_rcvrs, dtype=torch.float64, device=device)
    x2 = torch.flip(x2, dims=[0])
    rcvrs = [[
        torch.tensor([x1, x2_val], dtype=torch.float64, device=device) 
        for x2_val in x2
    ]]    
    return rcvrs
