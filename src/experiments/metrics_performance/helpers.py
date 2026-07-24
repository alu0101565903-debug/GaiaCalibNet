"""
Module: helpers.py
Path: src/experiments/metrics/helpers.py

Purpose:
    Provides shared helper utilities for physical spectrum simulation and 
    domain-specific metrics evaluation. Centralizes data generation to enforce 
    the DRY (Don't Repeat Yourself) principle across all experiment scripts.
"""

import numpy as np
from src.utils.metrics.alternative_metrics import calculate_mape, calculate_band_errors, calculate_pseudo_chi_square

def get_wavelength_grid(n_channels: int = 300) -> np.ndarray:
    """
    Generates the standard wavelength grid representing the Gaia spectrophotometer.

    Args:
    - n_channels (int): Number of spectral channels (pixels). Defaults to 300.

    Returns:
    - np.ndarray: 1-D array containing wavelength values in nanometers [nm].
    """
    return np.linspace(330, 1050, n_channels)


def simulate_spectra(n_samples: int = 1, n_channels: int = 300, seed: int = 42):
    """
    Simulates physical target spectra (CALSPEC-like) and three model predictions.

    Models simulated:
    - Target (Ground Truth): Smooth stellar continuum modeled as a Gaussian peak.
    - Model A: Severe systematic failure/divergence at the blue edge (< 360 nm).
    - Model B: Instrumental uniform white noise distributed across all pixels.
    - Model PINN: Physically consistent reconstruction with minimal Gaussian noise.

    Args:
    - n_samples (int): Number of stellar spectra to simulate. Defaults to 1 (1D vectors).
    - n_channels (int): Number of wavelength bins. Defaults to 300.
    - seed (int): Random seed for reproducible noise generation.

    Returns:
    - tuple: (wavelengths, y_true, y_pred_A, y_pred_B, y_pred_pinn)
            - wavelengths (1D array): Wavelength grid in nm.
            - y_true (1D/2D array): Ground-truth target spectra.
            - y_pred_A (1D/2D array): Model A predictions (edge failure).
            - y_pred_B (1D/2D array): Model B predictions (noisy baseline).
            - y_pred_pinn (1D/2D array): PINN predictions (high-fidelity).
    """
    rng = np.random.RandomState(seed)
    wavelengths = get_wavelength_grid(n_channels)
    
    # Base physics: synthetic CALSPEC stellar profile (Gaussian continuum with a floor)
    base_spectrum = np.exp(-0.5 * ((wavelengths - 600) / 110) ** 2) + 0.04
    
    if n_samples == 1:
        y_true = base_spectrum
        
        # Model A: Matches the center perfectly but introduces a severe flux jump at the blue edge
        y_pred_A = np.copy(y_true)
        y_pred_A[wavelengths < 360] += 0.167
        
        # Model B: Adds substantial uniform white noise mimicking a poor instrumental fit
        y_pred_B = y_true + rng.normal(0, 0.037, size=n_channels)
        
        # PINN Model: Emulates a robust physics-guided fit with negligible residual noise
        y_pred_pinn = y_true + rng.normal(0, 0.008, size=n_channels)
    else:
        # Tiling to simulate a batch of multiple observations (2D arrays)
        y_true = np.tile(base_spectrum, (n_samples, 1))
        
        # Model A batch
        y_pred_A = np.copy(y_true)
        y_pred_A[:, wavelengths < 360] += 0.167
        
        # Model B batch
        y_pred_B = y_true + rng.normal(0, 0.037, size=(n_samples, n_channels))
        
        # PINN Model batch
        y_pred_pinn = y_true + rng.normal(0, 0.008, size=(n_samples, n_channels))
        
    return wavelengths, y_true, y_pred_A, y_pred_B, y_pred_pinn


def evaluate_predictions(wavelengths: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes both global statistical metrics and domain-specific spectral diagnostics.

    This function supports both 1D single-spectrum vectors and 2D batch matrices.
    Astronomical metrics are calculated per-sample and averaged across the batch.

    Metrics evaluated:
    - Mean Squared Error (MSE): Traditional global statistical penalty.
    - Coefficient of Determination ($R^2$): Global variance explained.
    - Mean Absolute Percentage Error (MAPE): Sensitive to errors in low-flux regions.
    - Pseudo Chi-Square ($\chi^2$): Physics-based diagnostic weighted by flux.
    - Blue Band Error: Mean absolute error computed strictly below 500 nm.

    Args:
    - wavelengths (np.ndarray): 1D wavelength grid in nm.
    - y_true (np.ndarray): Ground-truth spectra (1D or 2D).
    - y_pred (np.ndarray): Predicted reconstruction spectra (same shape as y_true).

    Returns:
    - dict: Evaluated average metrics containing keys:
              ["mse", "r2", "pseudo_chi2", "mape", "blue_err"]
    """
    # Enforce 2D arrays to treat single vectors and batches uniformly
    y_t_2d = np.atleast_2d(y_true)
    y_p_2d = np.atleast_2d(y_pred)
    
    # 1. Compute global statistical metrics
    mse = np.mean((y_t_2d - y_p_2d) ** 2)
    ss_res = np.sum((y_t_2d - y_p_2d) ** 2)
    ss_tot = np.sum((y_t_2d - np.mean(y_t_2d)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    
    # 2. Compute astronomical and band-specific diagnostics per spectrum
    batch_pseudo_chi2 = []
    batch_mape = []
    batch_blue_err = []
    
    for yt, yp in zip(y_t_2d, y_p_2d):
        batch_pseudo_chi2.append(calculate_pseudo_chi_square(yt, yp))
        batch_mape.append(calculate_mape(yt, yp))
        
        # Calculate local band errors to identify edge-distortion issues
        bands = calculate_band_errors(wavelengths, yt, yp)
        batch_blue_err.append(bands["blue"])
        
    return {
        "mse": float(mse),
        "r2": float(r2),
        "pseudo_chi2": float(np.mean(batch_pseudo_chi2)),
        "mape": float(np.mean(batch_mape)),
        "blue_err": float(np.mean(batch_blue_err))
    }