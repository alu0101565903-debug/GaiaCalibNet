"""
Spectra Utilities for Resolution Degradation and Interpolation.

This module provides physical and mathematical tools to process astronomical 
spectra, including Gaussian convolution for instrument profile simulation 
and cubic spline interpolation onto a common wavelength grid.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple

def gaussian_kernel_convolution(wavelengths: np.ndarray, flux: np.ndarray, fwhm_nm: float) -> np.ndarray:
    """
    Degrade the resolution of a spectrum using a Gaussian kernel convolution.
    
    This simulates an instrument's Line Spread Function (LSF).
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelengths (in nanometers or Angstroms).
    flux : np.ndarray
        Array of flux values matching the wavelengths.
    fwhm_nm : float
        Full Width at Half Maximum (FWHM) of the target Gaussian kernel 
        in the same units as wavelengths.
        
    Returns
    -------
    np.ndarray
        The convolved (degraded) flux array.
    """
    # Calculate standard deviation (sigma) from FWHM
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    convolved_flux = np.zeros_like(flux)
    
    # Apply Gaussian kernel smoothing over each wavelength point
    for i, lambda_core in enumerate(wavelengths):
        # Weighting factor based on Gaussian distribution
        weights = np.exp(-0.5 * ((wavelengths - lambda_core) / sigma) ** 2)
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            convolved_flux[i] = np.sum(flux * weights) / weights_sum
        else:
            convolved_flux[i] = flux[i]
            
    return convolved_flux

def interpolate_to_grid(wavelengths: np.ndarray, flux: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate a spectrum onto a standard, uniformly sampled wavelength grid.
    
    Uses a Cubic Spline interpolation method to maintain smoothness and preserve
    spectral features without introducing linear artifacts.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Original non-uniform wavelength array.
    flux : np.ndarray
        Original flux values.
    target_grid : np.ndarray
        The standardized wavelength grid (Lambda) for the neural network.
        
    Returns
    -------
    np.ndarray
        Interpolated flux values on the target_grid.
    """
    # Ensure sorted order for CubicSpline
    sort_indices = np.argsort(wavelengths)
    w_sorted = wavelengths[sort_indices]
    f_sorted = flux[sort_indices]
    
    # Initialize and evaluate the Cubic Spline
    spline = CubicSpline(w_sorted, f_sorted, extrapolate=False)
    interpolated_flux = spline(target_grid)
    
    # Handle out-of-bounds values by replacing NaNs with a floor value (0.0)
    interpolated_flux = np.nan_to_num(interpolated_flux, nan=0.0)
    
    return interpolated_flux