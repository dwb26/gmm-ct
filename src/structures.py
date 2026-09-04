"""Data structures for peak detection and optimal assignment tracking."""

from typing import Dict, List
import torch

class PeakData:
    """Store peak-detection and trajectory-assignment data.
    
    Attributes
    ----------
    observable_indices : list of int
        Indices into the full time array where peaks were found.
    receiver_heights_by_time : dict
        ``{time_val: [heights]}`` – detected peak heights at each time.
    times, receiver_positions, receiver_indices, peak_values : list of list
        Per-Gaussian sequential detection results (used by diagnostic plots).
    assigned_times, assigned_heights, assigned_values : list of list
        Per-Gaussian optimal assignments (set by Hungarian / nearest-neighbour).
    """
    
    def __init__(self, n_gaussians: int, device: torch.device):
        self.N = n_gaussians
        self.device = device
        
        self.observable_indices: List[int] = []
        self.receiver_heights_by_time: Dict[float, List[float]] = {}
        
        # Per-Gaussian detection traces (for diagnostic plots)
        self.times = [[] for _ in range(n_gaussians)]
        self.receiver_positions = [[] for _ in range(n_gaussians)]
        self.receiver_indices = [[] for _ in range(n_gaussians)]
        self.peak_values = [[] for _ in range(n_gaussians)]
        
        # Refine assignments (for Newton-Raphson)
        self.assigned_times = [[] for _ in range(n_gaussians)]
        self.assigned_heights = [[] for _ in range(n_gaussians)]
        self.assigned_values = [[] for _ in range(n_gaussians)]
        
    def add_peak_detection(
        self,
        time_idx: int,
        time_val: float,
        receiver_idx: int,
        receiver_pos: float,
        peak_val: float,
        gaussian_idx: int,
    ) -> None:
        self.times[gaussian_idx].append(time_val)
        self.receiver_positions[gaussian_idx].append(receiver_pos)
        self.receiver_indices[gaussian_idx].append(receiver_idx)
        self.peak_values[gaussian_idx].append(peak_val)
        
        if gaussian_idx == 0 and time_idx not in self.observable_indices:
            self.observable_indices.append(time_idx)
            
    def add_time_detections(
        self, 
        time_val: float,
        detected_heights: List[float]
    ) -> None:
        if detected_heights:
            self.receiver_heights_by_time[time_val] = detected_heights
    
    def finalize_detections(self) -> None:
        """Convert accumulated per-Gaussian lists to PyTorch tensors."""
        for k in range(self.N):
            vals = self.times[k]
            self.times[k] = (
                torch.tensor(vals, dtype=torch.float64, device=self.device)
                if vals else torch.tensor([], dtype=torch.float64, device=self.device)
            )
            
    def add_optimal_assignment(
        self, 
        gaussian_idx: int, 
        time_val: float, 
        height: float, 
        value: float
    ) -> None:
        self.assigned_times[gaussian_idx].append(time_val)
        self.assigned_heights[gaussian_idx].append(height)
        self.assigned_values[gaussian_idx].append(value)
        
    def get_assignment_data(
        self, 
        gaussian_idx: int
    ) -> tuple[List[List[float]], List[List[float]]]:
        return self.assigned_times[gaussian_idx], self.assigned_heights[gaussian_idx]
    
    
    
# ==========================================================================
# PeakData – container for peak detection and assignment results
# ==========================================================================

# class PeakData:
#     """Store peak-detection and trajectory-assignment data.

#     Attributes
#     ----------
#     observable_indices : list of int
#         Indices into the full time array where peaks were found.
#     receiver_heights_by_time : dict
#         ``{time_val: [heights]}`` – detected peak heights at each time.
#     times, receiver_positions, receiver_indices, peak_values : list of list
#         Per-Gaussian sequential detection results (used by diagnostic plots).
#     assigned_times, assigned_heights, assigned_values : list of list
#         Per-Gaussian optimal assignments (set by Hungarian / nearest-neighbour).
#     """

#     def __init__(self, n_gaussians: int, device: torch.device):
#         self.N = n_gaussians
#         self.device = device

#         # Raw detection (per time point)
#         self.observable_indices: list = []
#         self.receiver_heights_by_time: dict = {}

#         # Sequential assignment (per Gaussian) – used by diagnostic plots
#         self.times = [[] for _ in range(n_gaussians)]
#         self.receiver_positions = [[] for _ in range(n_gaussians)]
#         self.receiver_indices = [[] for _ in range(n_gaussians)]
#         self.peak_values = [[] for _ in range(n_gaussians)]

#         # Optimal assignment (per Gaussian) – used by Newton-Raphson refinement
#         self.assigned_times = [[] for _ in range(n_gaussians)]
#         self.assigned_heights = [[] for _ in range(n_gaussians)]
#         self.assigned_values = [[] for _ in range(n_gaussians)]

#     def add_peak_detection(self, time_idx, time_val, receiver_idx, receiver_pos,
#                            peak_val, gaussian_idx):
#         """Record one detected peak from the sequential bottom-to-top scan."""
#         self.times[gaussian_idx].append(time_val)
#         self.receiver_positions[gaussian_idx].append(receiver_pos)
#         self.receiver_indices[gaussian_idx].append(receiver_idx)
#         self.peak_values[gaussian_idx].append(peak_val)

#         if gaussian_idx == 0 and time_idx not in self.observable_indices:
#             self.observable_indices.append(time_idx)

#     def add_time_detections(self, time_val, detected_heights):
#         """Record all peak heights found at one time point."""
#         if detected_heights:
#             self.receiver_heights_by_time[time_val] = detected_heights

#     def finalize_detections(self):
#         """Convert accumulated per-Gaussian lists to tensors."""
#         for k in range(self.N):
#             vals = self.times[k]
#             self.times[k] = (
#                 torch.tensor(vals, dtype=torch.float64, device=self.device)
#                 if vals
#                 else torch.tensor([], dtype=torch.float64, device=self.device)
#             )

#     def add_optimal_assignment(self, gaussian_idx, time_val, height, value):
#         """Record one peak-to-trajectory assignment from Hungarian or NN."""
#         self.assigned_times[gaussian_idx].append(time_val)
#         self.assigned_heights[gaussian_idx].append(height)
#         self.assigned_values[gaussian_idx].append(value)

#     def get_assignment_data(self, gaussian_idx):
#         """Return ``(times, heights)`` for the optimal assignment of Gaussian k."""
#         return self.assigned_times[gaussian_idx], self.assigned_heights[gaussian_idx]

#     def get_heights_dict_non_empty(self):
#         """Return ``{time: heights}`` filtered to times with detections."""
#         return {t: h for t, h in self.receiver_heights_by_time.items() if h}

#     def get_heights_sorted_by_time(self):
#         """Return detected heights sorted bottom-to-top at each time point."""
#         return [sorted(h) for h in self.receiver_heights_by_time.values()]