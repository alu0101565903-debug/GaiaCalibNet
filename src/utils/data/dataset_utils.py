"""
Module: dataset_utils.py

This module provides functions to load CALSPEC FITS spectra, convert them into a
consistent physical units grid, degrade spectral resolution to resemble Gaia
BP/RP observations, and simulate Gaia-like instrumental noise for model training.
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def load_calspec_fits(fits_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a CALSPEC FITS file and return wavelength and flux arrays.

    The CALSPEC FITS format stores wavelength values in Angstroms. This helper
    reads the FITS binary table, converts the wavelength axis to nanometers,
    and returns arrays suitable for downstream processing.

    Parameters:
    - fits_path (str): Path to the CALSPEC FITS file.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Wavelengths in nanometers and flux values.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wave_length_angstroms = np.array(data["WAVELENGTH"], dtype=float)
        flux = np.array(data["FLUX"], dtype=float)

        # Convert Angstroms to nanometers (1 nm = 10 Angstroms) to match Gaia units.
        wave_length_nm = wave_length_angstroms / 10.0

    return wave_length_nm, flux


def degrade_spectral_resolution(
    wave_nm: np.ndarray,
    flux: np.ndarray,
    sigma_nm: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Degrade a high-resolution spectrum using a Gaussian line spread function.

    This function interpolates the original high-resolution spectrum onto a finer
    grid and applies Gaussian smoothing to approximate the broader instrument
    response expected from Gaia BP/RP observations.

    Parameters:
    - wave_nm (np.ndarray): Original wavelength grid in nanometers.
    - flux (np.ndarray): Original flux values.
    - sigma_nm (float): Smoothing width in nanometers.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Degraded wavelength grid and flux values.
    """
    # Create a finer grid to avoid aliasing artifacts during convolution.
    fine_grid = np.linspace(wave_nm.min(), wave_nm.max(), len(wave_nm) * 2)
    interp_func = interp1d(wave_nm, flux, kind='linear', fill_value="extrapolate")
    fine_flux = interp_func(fine_grid)

    pixel_spacing = fine_grid[1] - fine_grid[0]
    sigma_pixels = sigma_nm / pixel_spacing

    # Gaussian smoothing approximates the instrument's line spread function.
    degraded_fine_flux = gaussian_filter1d(fine_flux, sigma=sigma_pixels)

    return fine_grid, degraded_fine_flux


def mock_load_gaia_observed_spectrum(
    calspec_wave: np.ndarray,
    calspec_flux_degraded: np.ndarray,
    wave_length_grid: np.ndarray,
    gain_factor: float = 1e13,
    sigma_ron: float = 5e-15,
) -> np.ndarray:
    """Generate a mock Gaia BP/RP-like observed spectrum with instrumental noise.

    This helper interpolates the degraded CALSPEC spectrum onto the shared
    wavelength grid, applies a wavelength-dependent systematic bias, and adds
    noise from both photon statistics and read-out noise.

    Parameters:
        calspec_wave (np.ndarray): Wavelength array of the degraded CALSPEC spectrum.
        calspec_flux_degraded (np.ndarray): Degraded flux values.
        wave_length_grid (np.ndarray): Target wavelength grid (300 channels).
        gain_factor (float): Mock system gain G (electrons per flux unit).
        sigma_ron (float): Read-out noise standard deviation.

    Returns:
        np.ndarray: Simulated Gaia-observed flux values with stochastic variance.
    """
    # Interpolate on the common training grid using smooth cubic splines.
    interp_func = interp1d(calspec_wave, calspec_flux_degraded, kind="cubic", fill_value="extrapolate")
    clean_sampled_flux = interp_func(wave_length_grid)
    
    # 2. Simulating instrumental systematic bias (e.g., optical prism imperfections)
    systematic_bias = 1.0 + 0.05 * np.sin(wave_length_grid / 50.0)
    biased_flux = clean_sampled_flux * systematic_bias
    
    # 3. Physical Noise Coupling (Equation 50 from Appendix C.3.2)
    # Variance = Poisson Limit (flux / G) + Read-Out Noise limit (sigma_RON^2)
    variance_i = (np.abs(biased_flux) / gain_factor) + (sigma_ron**2)
    sigma_i = np.sqrt(variance_i)
    
    # 4. Stochastic extraction (Equation 51)
    random_noise = np.random.normal(loc=0.0, scale=sigma_i, size=wave_length_grid.shape)
    
    observed_gaia_flux = biased_flux + random_noise
    return observed_gaia_flux