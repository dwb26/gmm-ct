"""
FFT-based omega estimation module for GMM reconstruction.

This module provides spectral analysis tools to directly estimate angular velocities
from peak value time series, eliminating the need for expensive multi-start optimization.

Key Principle:
--------------
An anisotropic (elliptical) Gaussian rotating with angular velocity ω creates periodic
oscillations in the X-ray projection peak values. Due to the 180° symmetry of ellipses,
the oscillation frequency is 2ω (two peaks per full rotation).

FFT decomposes the time-domain signal into frequency components:
  - LOW frequencies (< 1 Hz): Trajectory effects (slow baseline drift)
  - HIGH frequencies (~ 2ω): Rotation effects (fast oscillations)

By detrending (removing trajectory baseline) and applying FFT, we can directly
measure the dominant oscillation frequency and extract ω = frequency_peak / 2.

Advantages:
-----------
1. Direct estimation - no search required (O(n log n) FFT)
2. Robust to trajectory effects - frequencies are well-separated
3. Handles multiple Gaussians - each creates independent frequency peak
4. Eliminates multi-start requirement - provides good initialization
5. Computationally cheap - replaces expensive optimization with signal processing

Workflow Integration:
--------------------
Phase 1: Fit trajectory (x₀, v₀, a₀) for each Gaussian
Phase 1.5 (NEW): Extract peak values → FFT → Estimate ω for each Gaussian
Phase 2: Fit morphology (U_skew, α) with ω FIXED
Phase 3 (optional): Fine-tune ω + morphology together from good initialization

Author: GitHub Copilot
Date: 31 January 2026
"""

import torch
import numpy as np
from scipy.signal import detrend as scipy_detrend, windows
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks


def estimate_omega_from_peak_values(peak_values, t, method='detrend_hann', 
                                   expected_harmonics=4, min_omega=0.1, max_omega=10.0):
    """
    Estimate angular velocity ω from peak value time series using FFT.
    
    This is the main entry point for omega estimation. It applies spectral analysis
    to extract the rotation frequency from observed projection peak values.
    
    Parameters
    ----------
    peak_values : torch.Tensor or np.ndarray, shape (n_times,)
        Peak projection values over time (max across receivers at each time point)
    
    t : torch.Tensor or np.ndarray, shape (n_times,)
        Time vector corresponding to peak_values
    
    method : str, default='detrend_hann'
        Preprocessing method before FFT:
        - 'detrend_hann': Linear detrend + Hann window (recommended)
        - 'detrend': Linear detrend only
        - 'raw': No preprocessing (not recommended with motion)
    
    expected_harmonics : int, default=4
        Number of harmonics to check for validation
    
    min_omega : float, default=0.1
        Minimum physically plausible omega (Hz) - used to filter spurious peaks
    
    max_omega : float, default=10.0
        Maximum physically plausible omega (Hz) - used to filter spurious peaks
    
    Returns
    -------
    omega_estimate : float
        Estimated angular velocity in Hz (rotations per second)
    
    confidence : float
        Confidence score [0, 1] based on harmonic matching
        (higher = more confident, > 0.5 indicates clear peak)
    
    spectrum_info : dict
        Diagnostic information containing:
        - 'dominant_freq': The dominant frequency found (should be ~ 2ω)
        - 'freqs': Array of all positive frequencies
        - 'power': Power spectrum at each frequency
        - 'candidates': List of candidate omega values tested
        - 'method': Preprocessing method used
    
    Notes
    -----
    • The ellipse has 180° symmetry, so oscillation frequency = 2 × rotation frequency
    • Therefore: omega = dominant_frequency / 2
    • Works best when ω ≥ 1.5 Hz (rotation faster than trajectory changes)
    • For slower rotations, increase observation duration to improve resolution
    
    Examples
    --------
    >>> peak_values = torch.max(projections, dim=1)[0]  # Extract peak values
    >>> omega_est, confidence, info = estimate_omega_from_peak_values(peak_values, t)
    >>> print(f"Estimated ω = {omega_est:.3f} Hz (confidence: {confidence:.2f})")
    """
    
    # Convert to numpy if needed
    if isinstance(peak_values, torch.Tensor):
        peak_values = peak_values.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    
    # Validate inputs
    if len(peak_values) != len(t):
        raise ValueError(f"peak_values and t must have same length (got {len(peak_values)} vs {len(t)})")
    
    if len(peak_values) < 50:
        print(f"⚠ Warning: Only {len(peak_values)} time points. FFT works best with > 100 points.")
    
    # Preprocess signal
    signal = _preprocess_signal(peak_values, method)
    
    # Compute FFT with zero-padding for better frequency resolution
    dt = (t[-1] - t[0]) / (len(t) - 1)  # Time step
    n_fft = 2 ** int(np.ceil(np.log2(4 * len(signal))))  # Zero-pad to next power of 2
    
    spectrum = fft(signal, n=n_fft)
    freqs = fftfreq(n_fft, dt)
    
    # Extract positive frequencies only
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    power = np.abs(spectrum[pos_mask]) ** 2
    power_norm = power / np.max(power)
    
    # Find peaks in power spectrum
    peaks_idx, properties = find_peaks(power_norm, prominence=0.05, distance=5)
    
    if len(peaks_idx) == 0:
        print("⚠ Warning: No clear peaks found in spectrum. Results may be unreliable.")
        # Return midpoint of search range as fallback
        omega_fallback = (min_omega + max_omega) / 2
        return omega_fallback, 0.0, {
            'dominant_freq': None,
            'freqs': freqs_pos,
            'power': power,
            'candidates': [],
            'method': method,
            'message': 'No peaks found'
        }
    
    # Sort peaks by power
    peak_powers = power[peaks_idx]
    sorted_idx = np.argsort(peak_powers)[::-1]
    peaks_sorted = peaks_idx[sorted_idx]
    
    dominant_freq = freqs_pos[peaks_sorted[0]]
    
    # Test candidate omega values
    # The dominant peak should be at 2ω due to ellipse 180° symmetry
    # But we test multiple hypotheses in case of harmonics
    candidates = []
    
    for harmonic_factor in [2, 4, 6, 1, 3]:  # Prioritize even harmonics (ellipse symmetry)
        omega_candidate = dominant_freq / harmonic_factor
        
        # Check if this omega is in plausible range
        if not (min_omega <= omega_candidate <= max_omega):
            continue
        
        # Check if we see expected harmonics at multiples of this omega
        # For ellipse, expect peaks at 2ω, 4ω, 6ω, ...
        expected_freqs = [omega_candidate * m for m in [2, 4, 6, 8]]
        
        matches = 0
        for exp_freq in expected_freqs:
            # Look for peak near this frequency (5% tolerance)
            tolerance = 0.05 * exp_freq
            nearby_peaks = np.abs(freqs_pos[peaks_sorted[:10]] - exp_freq) < tolerance
            if np.any(nearby_peaks):
                matches += 1
        
        score = matches / len(expected_freqs)
        
        candidates.append({
            'omega': omega_candidate,
            'harmonic_factor': harmonic_factor,
            'matches': matches,
            'score': score
        })
    
    # Select best candidate
    if len(candidates) == 0:
        print(f"⚠ Warning: No candidates in plausible range [{min_omega}, {max_omega}] Hz")
        omega_fallback = (min_omega + max_omega) / 2
        return omega_fallback, 0.0, {
            'dominant_freq': dominant_freq,
            'freqs': freqs_pos,
            'power': power,
            'candidates': [],
            'method': method,
            'message': 'No candidates in range'
        }
    
    best_candidate = max(candidates, key=lambda x: x['score'])
    omega_estimate = best_candidate['omega']
    confidence = best_candidate['score']
    
    return omega_estimate, confidence, {
        'dominant_freq': dominant_freq,
        'freqs': freqs_pos,
        'power': power,
        'power_norm': power_norm,
        'peaks': peaks_sorted[:10],
        'candidates': candidates,
        'best_candidate': best_candidate,
        'method': method,
        'dt': dt,
        'n_fft': n_fft
    }


def _preprocess_signal(signal, method):
    """
    Preprocess signal before FFT to remove trajectory effects and reduce leakage.
    
    Parameters
    ----------
    signal : np.ndarray
        Raw peak value time series
    
    method : str
        Preprocessing method ('detrend_hann', 'detrend', or 'raw')
    
    Returns
    -------
    processed_signal : np.ndarray
        Preprocessed signal ready for FFT
    """
    
    if method == 'detrend_hann':
        # Remove linear trend (trajectory baseline)
        signal_detrended = scipy_detrend(signal, type='linear')
        # Apply Hann window to reduce spectral leakage
        window = windows.hann(len(signal_detrended))
        return signal_detrended * window
    
    elif method == 'detrend':
        # Remove linear trend only
        return scipy_detrend(signal, type='linear')
    
    elif method == 'raw':
        # No preprocessing (mean-centering only)
        return signal - np.mean(signal)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'detrend_hann', 'detrend', or 'raw'")


def estimate_omega_for_all_gaussians(projections_per_gaussian, t, method='detrend_hann',
                                     min_omega=0.1, max_omega=10.0):
    """
    Estimate omega for each Gaussian in a multi-Gaussian system.
    
    This function processes multiple Gaussians independently, extracting peak values
    and estimating omega for each one using FFT.
    
    Parameters
    ----------
    projections_per_gaussian : list of torch.Tensor
        List of length N, where each element has shape (n_times, n_receivers)
        projections_per_gaussian[i] contains the projection contribution from Gaussian i
    
    t : torch.Tensor or np.ndarray, shape (n_times,)
        Time vector
    
    method : str, default='detrend_hann'
        Preprocessing method (see estimate_omega_from_peak_values)
    
    min_omega : float, default=0.1
        Minimum plausible omega
    
    max_omega : float, default=10.0
        Maximum plausible omega
    
    Returns
    -------
    omega_estimates : list of float
        Estimated angular velocity for each Gaussian
    
    confidences : list of float
        Confidence score for each estimate
    
    spectrum_infos : list of dict
        Diagnostic information for each Gaussian
    
    Examples
    --------
    >>> # After Phase 1, you have trajectory for each Gaussian
    >>> # Generate synthetic projections for each Gaussian separately
    >>> omegas, confs, infos = estimate_omega_for_all_gaussians(proj_list, t)
    >>> for i, (omega, conf) in enumerate(zip(omegas, confs)):
    >>>     print(f"Gaussian {i}: ω = {omega:.3f} Hz (confidence: {conf:.2f})")
    """
    
    N = len(projections_per_gaussian)
    
    omega_estimates = []
    confidences = []
    spectrum_infos = []
    
    print(f"\n{'='*60}")
    print(f"FFT-based Omega Estimation for {N} Gaussian(s)")
    print(f"{'='*60}")
    
    for i, proj_i in enumerate(projections_per_gaussian):
        print(f"\n--- Gaussian {i+1}/{N} ---")
        
        # Extract peak values (max across receivers at each time)
        if isinstance(proj_i, torch.Tensor):
            peak_values_i = torch.max(proj_i, dim=1)[0]
        else:
            peak_values_i = np.max(proj_i, axis=1)
        
        # Estimate omega using FFT
        omega_i, conf_i, info_i = estimate_omega_from_peak_values(
            peak_values_i, t, method=method,
            min_omega=min_omega, max_omega=max_omega
        )
        
        omega_estimates.append(omega_i)
        confidences.append(conf_i)
        spectrum_infos.append(info_i)
        
        # Print result
        if conf_i > 0.5:
            quality = "✓✓ Excellent" if conf_i > 0.7 else "✓ Good"
        else:
            quality = "⚠ Low confidence"
        
        print(f"  Estimated ω = {omega_i:.4f} Hz")
        print(f"  Confidence = {conf_i:.2f} {quality}")
        if info_i.get('dominant_freq') is not None:
            print(f"  Dominant frequency = {info_i['dominant_freq']:.4f} Hz")
            print(f"  Interpretation: {info_i['best_candidate']['harmonic_factor']}×ω harmonic")
    
    print(f"\n{'='*60}")
    print("FFT-based estimation complete!")
    print(f"{'='*60}\n")
    
    return omega_estimates, confidences, spectrum_infos


# Utility function for visualization (optional)
def plot_omega_estimation_diagnostics(peak_values, t, omega_true=None, 
                                     output_path='omega_diagnostics.png'):
    """
    Create diagnostic plot showing time domain signal and frequency spectrum.
    
    Useful for validating omega estimation and understanding the signal characteristics.
    
    Parameters
    ----------
    peak_values : array-like
        Peak values over time
    t : array-like
        Time vector
    omega_true : float, optional
        True omega value (if known) for comparison
    output_path : str
        Path to save the diagnostic plot
    """
    import matplotlib.pyplot as plt
    
    omega_est, conf, info = estimate_omega_from_peak_values(peak_values, t)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time domain
    ax1 = axes[0]
    if isinstance(peak_values, torch.Tensor):
        peak_values_np = peak_values.cpu().numpy()
    else:
        peak_values_np = peak_values
    if isinstance(t, torch.Tensor):
        t_np = t.cpu().numpy()
    else:
        t_np = t
    
    ax1.plot(t_np, peak_values_np, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Peak Value')
    ax1.set_title('Time Domain: Peak Values')
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain
    ax2 = axes[1]
    freqs = info['freqs']
    power_norm = info['power_norm']
    
    ax2.plot(freqs, power_norm, 'b-', linewidth=1)
    
    # Mark dominant peak
    dom_freq = info['dominant_freq']
    ax2.axvline(dom_freq, color='red', linestyle='--', linewidth=2, 
               label=f'Dominant: {dom_freq:.2f} Hz')
    
    # Mark estimated omega harmonics
    for k in [2, 4, 6]:
        est_freq = k * omega_est
        if est_freq < freqs[-1]:
            ax2.axvline(est_freq, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Mark true omega harmonics if provided
    if omega_true is not None:
        for k in [2, 4, 6]:
            true_freq = k * omega_true
            if true_freq < freqs[-1]:
                ax2.axvline(true_freq, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    
    ax2.set_xlim(0, min(10 * omega_est, freqs[-1]))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Normalized Power')
    title_str = f'Frequency Domain: ω_est = {omega_est:.3f} Hz (conf={conf:.2f})'
    if omega_true is not None:
        error = abs(omega_est - omega_true) / omega_true * 100
        title_str += f' | True ω = {omega_true:.3f} Hz (error {error:.1f}%)'
    ax2.set_title(title_str)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plot saved to: {output_path}")
    plt.close()
