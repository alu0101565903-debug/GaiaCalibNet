"""
Module: alternative_metrics.py
Report Section: 

Domain-Specific Astrophysical Metrics for Spectrophotometric Recalibration.

This module provides a comprehensive suite of evaluation metrics tailored for 
astrophysical continuum recalibration models (e.g., Gaia BP/RP Physics-Informed 
Neural Networks). Standard statistical metrics (such as unweighted MAPE) often suffer 
from edge-effect singularities or near-zero flux divisions near stellar absorption lines 
and spectral band edges. 

The metrics in this module address these challenges by introducing:
- Peak-Normalized MAE (PN-MAE / NMAE) and NSTD percentage using global continuum baselines.
- True Reduced Chi-Square ($\chi^2_\nu$) accounting for both instrumental uncertainties 
  and systematic reference floors (e.g., CALSPEC ground-truth errors).
- Automated spectral band decomposition (Blue, Green, Red) with unit autodetection (Å vs. nm).
- Spectral Cosine Similarity to evaluate continuum shape alignment independently of flux scale.
"""

from typing import Dict
import numpy as np


def calculate_mape(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    epsilon: float = 1e-8
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE) with zero-division safeguard.

    Calculates the relative absolute deviation normalized element-wise by the ground-truth 
    flux. An epsilon term is added to prevent division singularities in low-flux regions.
  
    Parameters
    - y_true (np.ndarray): Ground-truth reference flux values.
    - y_pred (np.ndarray): Model-predicted flux values.
    - epsilon (float, default=1e-8): Small offset constant to prevent division by zero near continuum drop-offs.

    Returns
    - float: Mean absolute percentage error as a percentage (%).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0)


def calculate_peak_normalized_mae(y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    Compute Peak-Normalized Mean Absolute Error (PN-MAE / NMAE).

    Normalizes the overall mean absolute flux error by the target star's maximum 
    spectral peak flux. This avoids the numerical instability of point-by-point division 
    in low-signal spectral wings.

    Parameters
    - y_true (np.ndarray): Ground-truth reference flux array.
    - y_pred (np.ndarray): Model-predicted recalibrated flux array.

    Returns
    - float: Peak-normalized MAE expressed as a percentage (%). Returns `0.0` if peak flux < 1e-12.
    """
    peak_flux = np.max(np.abs(y_true))
    if peak_flux < 1e-12:
        return 0.0
    
    mae_absolute = np.mean(np.abs(y_true - y_pred))
    return float((mae_absolute / peak_flux) * 100.0)


def calculate_reduced_chi_square(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_err: np.ndarray, 
    n_params: int = 25,
    sys_err_frac: float = 0.01
) -> float:
    """
    Compute Observational Reduced Chi-Square ($\chi^2_\nu$) with systematic noise floor.

    Evaluates goodness-of-fit by combining instrumental measurement variance ($\sigma_{\\text{inst}}^2$) 
    and systematic uncertainty ($\sigma_{\\text{sys}}^2 = (f_{\\text{sys}} \\cdot y_{\\text{true}})^2$) in quadrature:

    .. math::
        \\sigma_{\\text{tot}, i}^2 = \\sigma_{\\text{inst}, i}^2 + (f_{\\text{sys}} \\cdot y_{\\text{true}, i})^2

    .. math::
        \\chi^2_{\\nu} = \\frac{1}{\\nu} \\sum_{i=1}^{N} \\frac{(y_{\\text{true}, i} - y_{\\text{pred}, i})^2}{\\sigma_{\\text{tot}, i}^2}

    where $\\nu = N - n_{\\text{params}}$ is the effective degrees of freedom.

    Parameters
    - y_true (np.ndarray): Ground-truth reference flux values.
    - y_pred (np.ndarray): Model-predicted recalibrated flux values.
    - y_err (np.ndarray): Instrumental 1-sigma flux uncertainty array.
    - n_params (int, default=25): Estimated number of free parameters / loss of degrees of freedom.
    - sys_err_frac (float, default=0.01): Systematic noise floor fraction added in quadrature (e.g., 0.01 = 1% CALSPEC intrinsic error).

    Returns
    - float: Reduced chi-square statistic ($\chi^2_\nu$).
    """
    # Sum of variances in quadrature
    instrumental_variance = np.square(y_err)
    systematic_variance = np.square(sys_err_frac * y_true)
    
    # Total variance is the sum in quadrature
    total_variance = instrumental_variance + systematic_variance
    
    # Prevent division by zero
    valid_variance = np.where(total_variance <= 0, 1e-16, total_variance)
    
    chi_square = np.sum(np.square(y_true - y_pred) / valid_variance)
    
    num_points = y_true.size
    degrees_of_freedom = num_points - n_params
    
    if degrees_of_freedom <= 0:
        degrees_of_freedom = max(1, num_points - 1)
        
    return float(chi_square / degrees_of_freedom)


def calculate_band_errors(
    wavelengths: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Decompose absolute and relative error metrics across spectral passbands.

    Automatically detects grid wavelength units (Ångströms vs. Nanometers) and 
    segments spectra into Blue, Green, and Red passbands.

    Passband Thresholds:
    - Ångströms ($\text{\AA}$): Blue (< 5000), Green (5000 - 7000), Red (> 7000)
    - Nanometers ($\text{nm}$): Blue (< 500), Green (500 - 700), Red (> 700)

    Parameters:
    - wavelengths (np.ndarray): Wavelength grid array.
    - y_true (np.ndarray): Ground truth flux values.
    - y_pred (np.ndarray): Predicted flux values.

    Returns:
    - dict: Dictionary containing absolute MAE and relative % errors per band.
    """
    is_angstrom = np.max(wavelengths) > 2000.0
    blue_limit = 5000.0 if is_angstrom else 500.0
    red_limit = 7000.0 if is_angstrom else 700.0

    bands = {
        "blue": (wavelengths < blue_limit),
        "green": (wavelengths >= blue_limit) & (wavelengths <= red_limit),
        "red": (wavelengths > red_limit),
    }

    peak_flux = np.max(np.abs(y_true))
    
    errors = {}
    for name, mask in bands.items():
        if np.any(mask):
            y_t_band = y_true[..., mask]
            y_p_band = y_pred[..., mask]
            
            mae_band = np.mean(np.abs(y_t_band - y_p_band))
            rel_error = (mae_band / peak_flux) * 100.0 if peak_flux > 1e-12 else 0.0
            
            errors[f"{name}_mae"] = float(mae_band)
            errors[f"{name}_rel_error_percent"] = float(rel_error)
        else:
            errors[f"{name}_mae"] = 0.0
            errors[f"{name}_rel_error_percent"] = 0.0

    return errors

def calculate_band_bias(
    wavelengths: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Band-Specific Photometric Bias Decomposition (MBE & Bias_norm %).

    Slices the spectrum into blue, green, and red wavelength bands (automatically 
    detecting Å vs. nm units) and computes:
    - Absolute Mean Bias Error (MBE): mean(y_pred - y_true)
    - Normalized Photometric Bias (Bias_norm %): sum(y_pred - y_true) / sum(y_true) * 100

    Parameters:
    - wavelengths (np.ndarray): Wavelength grid array.
    - y_true (np.ndarray): Ground truth flux values.
    - y_pred (np.ndarray): Predicted flux values.

    Returns:
    - Dict[str, float]: Dictionary containing `{band}_mbe` and `{band}_bias_norm_percent` for each band.
    """
    is_angstrom = np.max(wavelengths) > 2000.0
    blue_limit = 5000.0 if is_angstrom else 500.0
    red_limit = 7000.0 if is_angstrom else 700.0

    bands = {
        "blue": (wavelengths < blue_limit),
        "green": (wavelengths >= blue_limit) & (wavelengths <= red_limit),
        "red": (wavelengths > red_limit),
    }

    bias_results = {}
    for name, mask in bands.items():
        if np.any(mask):
            y_t_band = y_true[..., mask]
            y_p_band = y_pred[..., mask]
            
            mbe_band = np.mean(y_p_band - y_t_band)
            sum_true_band = np.sum(y_t_band)
            
            if np.abs(sum_true_band) > 1e-12:
                bias_norm = (np.sum(y_p_band - y_t_band) / sum_true_band) * 100.0
            else:
                bias_norm = 0.0
            
            bias_results[f"{name}_mbe"] = float(mbe_band)
            bias_results[f"{name}_bias_norm_percent"] = float(bias_norm)
        else:
            bias_results[f"{name}_mbe"] = 0.0
            bias_results[f"{name}_bias_norm_percent"] = 0.0

    return bias_results


def calculate_std_residual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard Deviation of Flux Residuals (\sigma_{res}).

    Measures the statistical dispersion and random variance of predictions 
    around the true ground truth spectrum (precision indicator).

    Parameters:
    - y_true (np.ndarray): Ground truth flux array.
    - y_pred (np.ndarray): Predicted flux array.

    Returns:
    - float: Standard deviation of (y_pred - y_true) in flux units.
    """
    return float(np.std(y_pred - y_true))


def calculate_normalized_std_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Peak-Normalized Standard Deviation Percentage (NSTD %).

    Expresses the dispersion of flux residuals as a percentage relative to 
    the star's global peak flux baseline.

    Parameters:
    - y_true (np.ndarray): Ground truth flux array.
    - y_pred (np.ndarray): Predicted flux array.

    Returns:
    - float: Standard deviation percentage relative to peak flux (%).
    """
    peak_flux = np.max(np.abs(y_true))
    if peak_flux < 1e-12:
        return 0.0
    std_val = np.std(y_pred - y_true)
    return float((std_val / peak_flux) * 100.0)


def calculate_mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Bias Error (MBE).

    Quantifies global systematic flux offsets without taking absolute values.

    Parameters:
    - y_true (np.ndarray): Ground truth flux array.
    - y_pred (np.ndarray): Predicted flux array.

    Returns:
    - float: Mean directional bias in flux units.
    """
    return float(np.mean(y_pred - y_true))


def calculate_normalized_bias_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Bias Error (NMBE %).

    Expresses systematic model bias as a percentage relative to the peak flux.

    Parameters:
    - y_true (np.ndarray): Ground truth flux array.
    - y_pred (np.ndarray): Predicted flux array.

    Returns:
    - float: Percentage bias relative to peak flux (%).
    """
    peak_flux = np.max(np.abs(y_true))
    if peak_flux < 1e-12:
        return 0.0
    bias = np.mean(y_pred - y_true)
    return float((bias / peak_flux) * 100.0)


def calculate_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spectral Cosine Similarity.

    Measures pure continuum shape alignment independent of absolute flux scale.

    Parameters:
    - y_true (np.ndarray): Ground truth flux spectrum.
    - y_pred (np.ndarray): Predicted flux spectrum.

    Returns:
    - float: Similarity score between 0.0 and 1.0.
    """
    y_t_flat = y_true.ravel()
    y_p_flat = y_pred.ravel()
    
    dot_product = np.dot(y_t_flat, y_p_flat)
    norm_true = np.linalg.norm(y_t_flat)
    norm_pred = np.linalg.norm(y_p_flat)
    
    if norm_true == 0 or norm_pred == 0:
        return 0.0
        
    return float(dot_product / (norm_true * norm_pred))