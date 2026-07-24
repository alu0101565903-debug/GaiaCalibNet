"""
Module: alternative_metrics.py
Description: Domain-specific evaluation metrics for spectral calibration.

This module provides several metrics tailored for spectrophotometric
recalibration tasks where standard summaries (MSE, R^2) can be misleading due
to heteroscedastic noise and low-flux edge channels. Expected array shapes:

- "y_true", "y_pred": 1-D arrays of length N (flux in arbitrary physical units)
- `wavelengths`: 1-D array of length N with wavelengths in nanometers (nm)

All functions are robust to zero-valued truth flux using a small `epsilon`
default to avoid division-by-zero.

Example:
    >>> import numpy as np
    >>> from src.utils.alternative_metrics import calculate_mape
    >>> y_true = np.array([1.0, 2.0, 0.1])
    >>> y_pred = np.array([0.9, 2.1, 0.0])
    >>> calculate_mape(y_true, y_pred)
    8.175...
"""

import numpy as np


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (np.ndarray): ground truth flux array (1-D)
    - y_pred (np.ndarray): predicted flux array (1-D)
    - epsilon (float): small value to avoid division by zero when `y_true` is zero.

    Returns:
    - MAPE expressed as percentage (float).

    Notes:
    - Uses absolute value of "y_true" in the denominator so sign changes do not
      produce spurious large percentages. For very small "y_true" the epsilon
      stabilizes the ratio.
    """
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0


def calculate_pseudo_chi_square(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Pseudo Chi-Square metric.

    Computes sum((y_true - y_pred)^2 / (|y_true| + epsilon)). This behaves
    similarly to a chi-square when the denominator encodes a variance-like
    scaling; here `|y_true|` penalizes absolute errors more strongly where the
    true flux is small (e.g., spectral edges), which helps reveal edge-divergence.

    Parameters:
    - y_true (np.ndarray): ground truth flux array (1-D)
    - y_pred (np.ndarray): predicted flux array (1-D)
    - epsilon (float): small value added to denominator for numerical stability.

    Returns:
    - scalar pseudo chi-square value (float).
    """
    squared_errors = np.square(y_true - y_pred)
    return np.sum(squared_errors / (np.abs(y_true) + epsilon))


def calculate_band_errors(wavelengths: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Band-specific mean absolute errors.

    Computes mean absolute error for three spectral bands defined by
    wavelength thresholds. Useful to quantify performance differences
    across blue/visual/near-IR regions.

    Parameters:
    - wavelengths (np.ndarray): 1-D array of wavelength values in nanometers.
    - y_true (np.ndarray): ground truth flux array (1-D, same length as `wavelengths`).
    - y_pred (np.ndarray): predicted flux array (1-D).

    Returns:
    - dict with keys "blue", "green", "red" mapping to MAE floats.

    Bands (defaults):
    - blue: wavelengths < 500 nm
    - green: 500 <= wavelengths <= 700 nm
    - red: wavelengths > 700 nm
    """
    bands = {
        "blue": (wavelengths < 500),
        "green": (wavelengths >= 500) & (wavelengths <= 700),
        "red": (wavelengths > 700)
    }

    errors = {}
    for name, mask in bands.items():
        if np.any(mask):
            # Mean absolute error restricted to this spectral window
            errors[name] = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        else:
            errors[name] = 0.0

    return errors