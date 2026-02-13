"""
Data structure for organizing peak detection and assignment results.

This module consolidates all peak-related data that was previously scattered
across multiple attributes in the GMM_reco class.
"""

import torch
import numpy as np


class PeakData:
    """
    Consolidated storage for peak detection and assignment data.
    
    This replaces the scattered attributes:
    - self.maximising_rcvrs, self.t_obs_by_cluster, self.maximising_inds
    - self.peak_values, self.observable_indices
    - self.time_rcvr_heights_dict_non_empty, self.sorted_list_of_heights_over_time
    - self.assigned_curve_data, self.assigned_peak_values
    
    Attributes
    ----------
    Raw Detection Data (per time point):
        observable_times : torch.Tensor, shape (n_observable,)
            Times where peaks were detected
        observable_indices : list of int
            Indices into full time array for observable times
        receiver_heights_by_time : dict
            {time: [heights]} - detected peak heights at each time
    
    Initial Sequential Assignment (per Gaussian):
        times : list of torch.Tensor
            Times where peaks detected for each Gaussian
        receiver_positions : list of list of torch.Tensor
            Receiver positions for detected peaks
        receiver_indices : list of list of int
            Receiver indices for detected peaks
        peak_values : list of list of float
            Projection values at detected peaks
    
    Final Optimal Assignment (after Hungarian algorithm):
        assigned_times : list of list of float
            Optimally assigned times for each Gaussian
        assigned_heights : list of list of float
            Optimally assigned receiver heights for each Gaussian
        assigned_values : list of list of float
            Peak values for optimal assignments
    """
    
    def __init__(self, n_gaussians, device):
        """
        Initialize empty peak data structure.
        
        Parameters
        ----------
        n_gaussians : int
            Number of Gaussians in the model
        device : torch.device
            Device for tensor operations
        """
        self.N = n_gaussians
        self.device = device
        
        # Raw detection data (per time point)
        self.observable_indices = []
        self.receiver_heights_by_time = {}
        
        # Initial sequential assignment (per Gaussian)
        # These are populated during peak detection
        self.times = [[] for _ in range(n_gaussians)]
        self.receiver_positions = [[] for _ in range(n_gaussians)]
        self.receiver_indices = [[] for _ in range(n_gaussians)]
        self.peak_values = [[] for _ in range(n_gaussians)]
        
        # Final optimal assignment (per Gaussian)
        # These are populated after Hungarian algorithm
        self.assigned_times = [[] for _ in range(n_gaussians)]
        self.assigned_heights = [[] for _ in range(n_gaussians)]
        self.assigned_values = [[] for _ in range(n_gaussians)]
    
    def add_peak_detection(self, time_idx, time_val, receiver_idx, receiver_pos, 
                          peak_val, gaussian_idx):
        """
        Record a detected peak during sequential bottom-to-top scan.
        
        Parameters
        ----------
        time_idx : int
            Index into full time array
        time_val : float or torch.Tensor
            Time value
        receiver_idx : int
            Index of receiver where peak detected
        receiver_pos : torch.Tensor, shape (2,)
            Position [x, y] of receiver
        peak_val : float or torch.Tensor
            Projection value at peak
        gaussian_idx : int
            Which Gaussian (cluster) this peak is assigned to
        """
        self.times[gaussian_idx].append(time_val)
        self.receiver_positions[gaussian_idx].append(receiver_pos)
        self.receiver_indices[gaussian_idx].append(receiver_idx)
        self.peak_values[gaussian_idx].append(peak_val)
        
        # Track observable time indices (using first Gaussian as reference)
        if gaussian_idx == 0 and time_idx not in self.observable_indices:
            self.observable_indices.append(time_idx)
    
    def add_time_detections(self, time_val, detected_heights):
        """
        Record all peak heights detected at a single time point.
        
        Parameters
        ----------
        time_val : float
            Time value
        detected_heights : list of float or torch.Tensor
            Heights where peaks detected (one per Gaussian ideally)
        """
        if len(detected_heights) > 0:
            self.receiver_heights_by_time[time_val] = detected_heights
    
    def finalize_detections(self):
        """
        Convert accumulated lists to tensors after detection phase complete.
        
        This should be called once after all peaks have been detected but before
        optimization begins.
        """
        for k in range(self.N):
            if len(self.times[k]) > 0:
                self.times[k] = torch.tensor(
                    self.times[k], 
                    dtype=torch.float64, 
                    device=self.device
                )
    
    def add_optimal_assignment(self, gaussian_idx, time_idx, height, value):
        """
        Record an optimal assignment from Hungarian algorithm.
        
        Parameters
        ----------
        gaussian_idx : int
            Which Gaussian this peak is assigned to
        time_idx : int
            Index into observable times
        height : float or torch.Tensor
            Receiver height
        value : float or torch.Tensor
            Peak value
        """
        self.assigned_times[gaussian_idx].append(time_idx)
        self.assigned_heights[gaussian_idx].append(height)
        self.assigned_values[gaussian_idx].append(value)
    
    def get_observable_times(self, full_time_array):
        """
        Extract observable times from full time array.
        
        Parameters
        ----------
        full_time_array : torch.Tensor
            Complete time vector
        
        Returns
        -------
        torch.Tensor
            Times where peaks were detected
        """
        return full_time_array[self.observable_indices]
    
    def get_heights_sorted_by_time(self):
        """
        Get detected heights sorted at each time point.
        
        Returns
        -------
        list of list
            Sorted heights at each time
        """
        return [sorted(heights) for heights in self.receiver_heights_by_time.values()]
    
    def get_heights_dict_non_empty(self):
        """
        Get dictionary of time -> heights, filtering empty times.
        
        Returns
        -------
        dict
            {time: [heights]} for times with detections
        """
        return {t: h for t, h in self.receiver_heights_by_time.items() if len(h) > 0}
    
    def get_assignment_data(self, gaussian_idx):
        """
        Get optimal assignment data for a specific Gaussian.
        
        Parameters
        ----------
        gaussian_idx : int
            Which Gaussian
        
        Returns
        -------
        tuple
            (time_indices, heights) for this Gaussian's assignments
        """
        return (
            self.assigned_times[gaussian_idx],
            self.assigned_heights[gaussian_idx]
        )
    
    def get_all_assignment_data(self):
        """
        Get optimal assignment data for all Gaussians.
        
        Returns
        -------
        list of tuple
            List of (times, heights) tuples, one per Gaussian
        """
        return [
            (self.assigned_times[k], self.assigned_heights[k])
            for k in range(self.N)
        ]
    
    def has_assignments(self, gaussian_idx):
        """Check if Gaussian has any optimal assignments."""
        return len(self.assigned_times[gaussian_idx]) > 0
    
    def summary(self):
        """Print summary statistics of detected and assigned peaks."""
        print("\n" + "="*60)
        print("PEAK DATA SUMMARY")
        print("="*60)
        print(f"Observable time points: {len(self.observable_indices)}")
        print(f"Time points with detections: {len(self.receiver_heights_by_time)}")
        
        print("\nInitial Sequential Detections:")
        for k in range(self.N):
            n_peaks = len(self.times[k]) if isinstance(self.times[k], list) else len(self.times[k])
            print(f"  Gaussian {k}: {n_peaks} peaks")
        
        print("\nFinal Optimal Assignments:")
        for k in range(self.N):
            print(f"  Gaussian {k}: {len(self.assigned_times[k])} assignments")
        print("="*60 + "\n")
