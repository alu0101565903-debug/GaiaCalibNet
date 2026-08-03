"""
Module: alternative_metrics.py
Description: Domain-specific astrophysical evaluation metrics for 
             spectral continuum recalibration models (e.g., Gaia BP/RP PINN).

Key Features:
- Peak-Normalized MAE to avoid edge-effect singularities.
- True Reduced Chi-Square incorporating both instrumental and systematic noise.
- Automated spectral band slicing with automatic unit detection (Å vs. nm).
- Consistent normalization baseline across global and band-specific metrics.
"""

import numpy as np


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Parameters:
        y_true (np.ndarray): Ground truth flux values.
        y_pred (np.ndarray): Predicted flux values.
        epsilon (float): Small offset to prevent division by zero near continuum drop-offs.

    Returns:
        float: Mean absolute percentage error.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0)


def calculate_peak_normalized_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Peak-Normalized Mean Absolute Error (PN-MAE).

    Normalizes the overall mean absolute flux error by the star's global peak flux.

    Parameters:
        y_true (np.ndarray): Ground truth flux values.
        y_pred (np.ndarray): Model predicted flux values.

    Returns:
        float: Peak-normalized MAE expressed as a percentage (%).
    """
    peak_flux = np.max(np.abs(y_true))
    print(f"Peak flux: {peak_flux}")
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
    Observational Reduced Chi-Square (\chi^2_\nu) with Systematic Error.

    Evaluates model fit quality against total variance (instrumental + systematic).
    Adding a systematic error floor prevents the metric from artificially inflating
    when the instrumental noise is unreasonably small compared to the intrinsic 
    uncertainty of the physical ground truth (e.g., CALSPEC's ~1% calibration error).

    Parameters:
        y_true (np.ndarray): Ground truth reference flux.
        y_pred (np.ndarray): Predicted recalibrated flux.
        y_err (np.ndarray): Instrumental 1-sigma uncertainty array.
        n_params (int): Loss of degrees of freedom.
        sys_err_frac (float): Systematic error fraction to add in quadrature 
                              (default is 0.01, representing a 1% uncertainty).

    Returns:
        float: Reduced chi-square statistic.
    """
    # Calculate variances
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
) -> dict:
    """
    Band-Specific Error Decomposition.

    Automatically detects wavelength units and computes both absolute MAE and 
    peak-normalized relative percentage MAE using the same global peak flux baseline.

    Parameters:
        wavelengths (np.ndarray): Wavelength grid array.
        y_true (np.ndarray): Ground truth flux values.
        y_pred (np.ndarray): Predicted flux values.

    Returns:
        dict: Dictionary containing absolute MAE and relative % errors per band.
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


def calculate_mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Bias Error (MBE).

    Quantifies global systematic flux offsets without taking absolute values.

    Parameters:
        y_true (np.ndarray): Ground truth flux array.
        y_pred (np.ndarray): Predicted flux array.

    Returns:
        float: Mean directional bias in flux units.
    """
    return float(np.mean(y_pred - y_true))


def calculate_normalized_bias_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Bias Error (NMBE %).

    Expresses systematic model bias as a percentage relative to the peak flux.

    Parameters:
        y_true (np.ndarray): Ground truth flux array.
        y_pred (np.ndarray): Predicted flux array.

    Returns:
        float: Percentage bias relative to peak flux (%).
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
        y_true (np.ndarray): Ground truth flux spectrum.
        y_pred (np.ndarray): Predicted flux spectrum.

    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    y_t_flat = y_true.ravel()
    y_p_flat = y_pred.ravel()
    
    dot_product = np.dot(y_t_flat, y_p_flat)
    norm_true = np.linalg.norm(y_t_flat)
    norm_pred = np.linalg.norm(y_p_flat)
    
    if norm_true == 0 or norm_pred == 0:
        return 0.0
        
    return float(dot_product / (norm_true * norm_pred))