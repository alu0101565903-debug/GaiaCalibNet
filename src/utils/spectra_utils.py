"""
Utility functions for CALSPEC spectral processing.

This module provides helpers to load CALSPEC FITS spectra, degrade spectral
resolution to mimic Gaia observations, normalize spectra, and simulate Gaia
BP/RP observations on a common wavelength grid.
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid


def load_calspec_fits(fits_path: str) -> tuple:
    """
    Reads a raw CALSPEC FITS file and converts units to standard nanometers.
    
    Parameters:
    - fits_path (str): Path to the local FITS file.
    Returns: tuple (wavelength array in nm, flux density array)
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wave_length_angstroms = np.array(data["WAVELENGTH"], dtype=float)
        flux = np.array(data["FLUX"], dtype=float)
        
        # Convert Angstroms to Nanometers (1 nm = 10 Angstroms)
        wave_length_nm = wave_length_angstroms / 10.0
        
    return wave_length_nm, flux

def degrade_spectral_resolution(wave_nm: np.ndarray, flux: np.ndarray, sigma_nm: float = 8.0) -> tuple:
    """
    Downsamples high-resolution spectra to emulated Gaia resolution via LSF (Eq. 1).
    
    Parameters:
    - wave_nm (np.ndarray): Source wavelengths.
    - flux (np.ndarray): Source flux profile.
    - sigma_nm (float): Line Spread Function (LSF) smoothing scale.
    Returns:
        tuple : (degraded wavelength grid, degraded flux profile)
    """
    
    fine_grid = np.linspace(wave_nm.min(), wave_nm.max(), len(wave_nm) * 2)
    interp_func = interp1d(wave_nm, flux, kind='linear', fill_value="extrapolate")
    fine_flux = interp_func(fine_grid)
    
    pixel_spacing = fine_grid[1] - fine_grid[0]
    sigma_pixels = sigma_nm / pixel_spacing
    
    degraded_fine_flux = gaussian_filter1d(fine_flux, sigma=sigma_pixels)
    
    return fine_grid, degraded_fine_flux

def normalize_spectrum(flux_array: np.ndarray, wave_length_grid: np.ndarray) -> np.ndarray:
    """
    Normalizes the continuous spectrum using area-under-the-curve (Section 4.5).
    Parameters:
    - flux_array (np.ndarray): The flux values to normalize.
    - wave_length_grid (np.ndarray): The wavelength grid.
    Returns:
    - np.ndarray : The normalized flux values.
    """
    total_area = trapezoid(flux_array, wave_length_grid)
    if total_area == 0:
        return flux_array
    return flux_array / total_area

def mock_load_gaia_observed_spectrum(calspec_wave: np.ndarray, calspec_flux_degraded: np.ndarray, wave_length_grid: np.ndarray) -> np.ndarray:
    """
    A template loader function for Gaia BP/RP observed sampled spectra.
    Simulates instrumental systematic bias and random noise (Section 5.4).
    Parameters:
    - calspec_wave (np.ndarray): The wavelength array from degraded CALSPEC.
    - calspec_flux_degraded (np.ndarray): The degraded flux array from CALSPEC
    - wave_length_grid (np.ndarray): The common wavelength grid for alignment.
    Returns:
    - observed_gaia_flux (np.ndarray) : Simulated Gaia observed flux array aligned to the common grid.
    """
    interp_func = interp1d(calspec_wave, calspec_flux_degraded, kind="cubic", fill_value="extrapolate")
    clean_sampled_flux = interp_func(wave_length_grid)
    
    # Simulating instrumental systematic bias and random noise (Section 5.4)
    systematic_bias = 1.0 + 0.05 * np.sin(wave_length_grid / 50.0)
    random_noise = np.random.normal(0, 0.01 * np.mean(clean_sampled_flux), size=wave_length_grid.shape)
    
    observed_gaia_flux = (clean_sampled_flux * systematic_bias) + random_noise
    return observed_gaia_flux