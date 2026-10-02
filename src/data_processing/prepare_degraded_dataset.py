"""Prepare degraded Gaia XP and CALSPEC spectra for model training.

High-resolution CALSPEC spectra are convolved with an empirical,
wavelength-dependent Gaia BP/RP line-spread function (LSF) and aligned with
Gaia spectra on a shared wavelength grid.

Features
--------
- Converts CALSPEC wavelengths and fluxes to the pipeline units.
- Retrieves Gaia XP spectra in batches with retry handling.
- Produces absolute and normalized arrays for model training and evaluation.
"""

import os
import time
import warnings
from typing import Any, Dict, List, Tuple, Union

import gaiaxpy
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray
from scipy.integrate import simpson
from scipy.interpolate import interp1d

from src.utils.processing.pinn_data import compute_calibration_ratio, stable_flux_denominator

# Suppress runtime warnings for cleaner terminal output
warnings.filterwarnings("ignore")

try:
    from src.data_processing.gaia_xp_lsf_convolution import _degrade_to_gaia_xp as c_module_correct_lsf

    print("[✓] Custom LSF convolution module successfully imported.")
except ImportError:
    print("[!] Warning: Custom module not found in system path.")
    print("    Falling back to empirical Gaia BP/RP LSF convolution (Montegriffo et al. 2023).")

    def c_module_correct_lsf(x: NDArray[np.float64], y: NDArray[np.float64], x_eval: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Simulate the Gaia BP/RP Line Spread Function (LSF) using empirical resolution curves.

        Uses the empirical spectral resolution R(lambda) table from Montegriffo et al. (2023).

        Parameters
        ----------
        x : numpy.ndarray
            Original high-resolution wavelength array in nanometers.
        y : numpy.ndarray
            Original high-resolution flux array in W / (m^2 nm).
        x_eval : numpy.ndarray
            Evaluation wavelength grid array in nanometers.

        Returns
        -------
        y_degraded : numpy.ndarray
            Degraded flux array evaluated on the `x_eval` grid.
        """
        # Gaia BP/RP spectral-resolution anchors from Montegriffo et al. (2023).
        montegriffo_waves = np.array(
            [
                330, 350, 400, 450, 500, 550, 600, 630,        # Blue Photometer (BP)
                640, 700, 750, 800, 850, 900, 950, 1000, 1050  # Red Photometer (RP)
            ],
            dtype=np.float64,
        )

        montegriffo_R = np.array(
            [
                100, 115, 100, 80, 60, 45, 35, 30,  # BP resolution curve
                90, 80, 70, 60, 50, 45, 40, 35, 30   # RP resolution curve
            ],
            dtype=np.float64,
        )

        get_R = interp1d(montegriffo_waves, montegriffo_R, kind="linear", fill_value="extrapolate")
        y_degraded = np.zeros_like(x_eval, dtype=np.float64)

        # Convolve each wavelength with a wavelength-dependent Gaussian kernel.
        R_calspec = 800.0  # Intrinsic resolution of CALSPEC spectra

        for i, w_eval in enumerate(x_eval):
            R_w = get_R(w_eval)

            # Subtract resolutions in quadrature to isolate instrumental broadening.
            sigma_gaia = w_eval / (2.355 * R_w)
            sigma_initial = w_eval / (2.355 * R_calspec)

            # Guard against empirical resolution exceeding the CALSPEC resolution.
            sigma_kernel_sq = sigma_gaia**2 - sigma_initial**2
            sigma_w = np.sqrt(sigma_kernel_sq) if sigma_kernel_sq > 0 else 1e-5

            kernel = np.exp(-0.5 * ((x - w_eval) / sigma_w) ** 2)

            # Normalize the convolution kernel.
            kernel_area = simpson(y=kernel, x=x)
            if kernel_area > 0:
                kernel = kernel / kernel_area

            y_degraded[i] = simpson(y=y * kernel, x=x)

        return y_degraded


def process_calspec_target(fits_path: str, wave_grid_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Process a single CALSPEC FITS file by converting units, applying empirical LSF 
    convolution, and interpolating onto the target Gaia wavelength grid via cubic spline.

    Parameters
    ----------
    fits_path : str
        Path to the CALSPEC FITS file.
    wave_grid_nm : numpy.ndarray
        Target wavelength evaluation grid in nanometers.

    Returns
    -------
    y_aligned : numpy.ndarray
        Processed, convolved, and grid-aligned CALSPEC flux array in W / (m^2 nm).

    Raises
    ------
    ValueError
        If key spectral data tables are missing, data coverage is insufficient, 
        or output contains unexpected NaN values.
    """
    # Extract spectral data from the FITS file.
    with fits.open(fits_path) as hdul:
        data = None
        for ext in hdul:
            if hasattr(ext, "columns") and "WAVELENGTH" in ext.columns.names:
                data = ext.data
                break

        if data is None:
            raise ValueError(f"Could not find WAVELENGTH/FLUX columns in {fits_path}")

        raw_wave = np.array(data["WAVELENGTH"], dtype=np.float64)
        raw_flux = np.array(data["FLUX"], dtype=np.float64)

    # Convert CALSPEC wavelengths and fluxes to nm and W/m^2/nm.
    wave_nm = raw_wave / 10.0
    flux_wm2nm = raw_flux * 1e-2

    # Extrapolate linearly to 1200 nm on a dense grid.
    if wave_nm[-1] < 1200.0:
        step = np.median(np.diff(wave_nm))
        ext_wave = np.arange(wave_nm[-1] + step, 1200.0 + step, step)

        # Compute slope using the last 10 points for smooth extrapolation transition
        n_points = min(10, len(wave_nm))
        dy = flux_wm2nm[-1] - flux_wm2nm[-n_points]
        dx = wave_nm[-1] - wave_nm[-n_points]
        slope = dy / (dx + 1e-8)

        # Linearly extrapolate and clamp flux to prevent negative values
        ext_flux = flux_wm2nm[-1] + slope * (ext_wave - wave_nm[-1])
        ext_flux = np.clip(ext_flux, 0.0, None)
        
        wave_nm = np.concatenate([wave_nm, ext_wave])
        flux_wm2nm = np.concatenate([flux_wm2nm, ext_flux])

    # Restrict the spectrum to the convolution and interpolation range.
    mask = (wave_nm >= wave_grid_nm.min() - 15.0) & (wave_nm <= 1200.0)
    xr_nm = wave_nm[mask]
    yr_flux = flux_wm2nm[mask]

    if len(xr_nm) < 10:
        raise ValueError(f"Insufficient data within range [{wave_grid_nm.min()}, {wave_grid_nm.max()}] nm")

    # Apply the Gaia LSF convolution.
    res_flux = c_module_correct_lsf(xr_nm, yr_flux, xr_nm)

    # Interpolate directly onto the target wavelength grid.
    interp_func = interp1d(
        xr_nm, 
        res_flux, 
        kind="cubic", 
        bounds_error=False, 
        fill_value=np.nan
    )

    y_aligned = interp_func(wave_grid_nm).astype(np.float32)

    if np.isnan(y_aligned).any():
        raise ValueError("CALSPEC spectrum does not fully cover the requested wavelength grid.")

    return y_aligned


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert catalog attributes to floating point values with fallback defaults.

    Parameters
    ----------
    value : Any
        Input catalog value (e.g., string, float, integer, NaN).
    default : float, optional
        Fallback default value if conversion fails. Default is 0.0.

    Returns
    -------
    val : float
        Converted float value or default fallback.
    """
    if pd.isna(value):
        return default
    val_str = str(value).strip()
    if val_str in ["--", "-", "nan", ""]:
        return default
    try:
        return float(val_str)
    except ValueError:
        return default

def calculate_c2_c3(flux: NDArray[np.float64], wave: NDArray[np.float64]) -> Tuple[float, float]:
    """
    Calculate C2 and C3 pseudo-color indices based on normalized XP spectral slopes.

    Computes spectral slopes across specific wavelength windows following Huang et al. (2024).

    Parameters
    ----------
    flux : numpy.ndarray
        Spectral flux density array.
    wave : numpy.ndarray
        Wavelength grid array in nanometers.

    Returns
    -------
    c2_val : float
        Calculated C2 spectral slope index scaled by 100.
    c3_val : float
        Calculated C3 spectral slope index scaled by 100.
    """
    # Normalization using mean flux between 460 and 490 nm
    norm_mask = (wave >= 460.0) & (wave <= 490.0)
    norm_factor = np.nanmean(flux[norm_mask])

    if norm_factor <= 0 or np.isnan(norm_factor):
        return 0.0, 0.0

    norm_flux = flux / norm_factor

    # Masks for C2 and C3 wavelength ranges
    c2_mask = (wave >= 390.0) & (wave <= 476.0)
    c3_mask = (wave >= 478.0) & (wave <= 600.0)

    # Compute slopes via linear regression (degree 1) and scale by 100
    c2_val, c3_val = 0.0, 0.0
    
    if np.sum(c2_mask) > 1:
        c2_slope, _ = np.polyfit(wave[c2_mask], norm_flux[c2_mask], 1)
        c2_val = c2_slope * 100.0
        
    if np.sum(c3_mask) > 1:
        c3_slope, _ = np.polyfit(wave[c3_mask], norm_flux[c3_mask], 1)
        c3_val = c3_slope * 100.0

    return float(c2_val), float(c3_val)

def create_dataset(
    csv_path: str = "data/processed/training_catalog.csv",
    calspec_dir: str = "data/raw/calspec_fits",
    output_dir: str = "data/processed/nn_data",
    wave_step_nm: float = 2.0,
    wave_min_nm: float = 336.0,
    wave_max_nm: float = 1020.0,
) -> None:
    """
    Generate consolidated (X, Y) spectral datasets by batch-querying GaiaXPy 
    and processing CALSPEC spectra.

    Parameters
    ----------
    csv_path : str, optional
        Path to the cross-matched catalog CSV file.
    calspec_dir : str, optional
        Directory containing CALSPEC FITS files.
    output_dir : str, optional
        Destination directory to export processed NumPy array files (.npy).
    wave_step_nm : float, optional
        Grid wavelength step size in nanometers. Default is 2.0.
    wave_min_nm : float, optional
        Minimum wavelength bound in nanometers. Default is 336.0.
    wave_max_nm : float, optional
        Maximum wavelength bound in nanometers. Default is 1020.0.

    Returns
    -------
    None
    """
    print("=" * 60)
    print("EXECUTING PIPELINE: GENERATING (X, Y) SPECTRAL DATASET")
    print("=" * 60)

    abs_csv_path = os.path.abspath(csv_path)
    abs_calspec_dir = os.path.abspath(calspec_dir)
    abs_output_dir = os.path.abspath(output_dir)

    if not os.path.exists(abs_csv_path):
        print(f"[!] Critical Error: Catalog missing: {abs_csv_path}")
        return

    # Standardized wavelength grid for both Gaia and CALSPEC spectra
    wave_length_grid = np.arange(wave_min_nm, wave_max_nm + wave_step_nm, wave_step_nm, dtype=np.float32)
    df_catalog = pd.read_csv(abs_csv_path, dtype={"gaia_source_id": str})

    valid_gaia_ids = []
    for _, row in df_catalog.iterrows():
        raw_id = row.get("gaia_source_id", "0")
        if pd.notna(raw_id) and str(raw_id).strip().lower() != "nan":
            id_str = str(raw_id).strip()
            if "e" in id_str.lower():
                raise ValueError(
                    f"Precision loss detected in the CSV value {id_str}. "
                    "Store gaia_source_id as plain text."
                )
            
            clean_id = int(id_str.split(".")[0])
            valid_gaia_ids.append(clean_id)

    chunk_size = 5
    gaia_dict = {}
    total_chunks = (len(valid_gaia_ids) + chunk_size - 1) // chunk_size

    print(f"-> Querying GaiaXPy for {len(valid_gaia_ids)} sources in {total_chunks} chunks...")

    for i in range(0, len(valid_gaia_ids), chunk_size):
        chunk = valid_gaia_ids[i : i + chunk_size]
        current_chunk = (i // chunk_size) + 1
        print(f"   Downloading chunk {current_chunk}/{total_chunks} ({len(chunk)} sources)...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch sampled spectra directly on wave_length_grid
                calibrated_spectra, _ = gaiaxpy.calibrate(chunk, sampling=wave_length_grid)
                if calibrated_spectra is not None and not calibrated_spectra.empty:
                    chunk_dict = calibrated_spectra.set_index("source_id").to_dict(orient="index")
                    gaia_dict.update(chunk_dict)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      [!] Connection error. Retrying... (Attempt {attempt+2}/{max_retries})")
                    time.sleep(5)
                else:
                    print(f"      [X] Failed chunk {current_chunk} after {max_retries} attempts: {e}")

    if not gaia_dict:
        print("[!] Critical Error: Could not retrieve any Gaia data from servers.")
        return

    print(f"-> Gaia data downloaded successfully. Extracted {len(gaia_dict)} spectra.")

    available_fits = [f.lower() for f in os.listdir(abs_calspec_dir)] if os.path.exists(abs_calspec_dir) else []
    real_fits_names = {f.lower(): f for f in os.listdir(abs_calspec_dir)} if available_fits else {}

    X_list, Y_list, X_error_list = [], [], []
    G_mags_list, bp_rp_list, ruwe_list = [], [], []
    c2_list, c3_list = [], []
    processed_stars, processed_ids = [], []

    for idx, row in df_catalog.iterrows():
        star_name = str(row.get("calspec_name", row.get("Name"))).strip()
        fits_filename = str(row.get("fits_file", row.get("File_Source"))).strip()

        raw_id = row.get("gaia_source_id", "0")
        if pd.isna(raw_id) or str(raw_id).strip().lower() == "nan":
            continue

        id_str = str(raw_id).strip()
        gaia_id = int(id_str.split(".")[0])

        if gaia_id not in gaia_dict:
            print(f"  [X] {star_name:<15} -> Missing Gaia spectrum in batch download.")
            continue

        matched_filename = None
        fits_lower = fits_filename.lower()
        if fits_lower in real_fits_names:
            matched_filename = real_fits_names[fits_lower]
        else:
            for f_lower, real_name in real_fits_names.items():
                prefix = star_name.lower()
                if f_lower.startswith(prefix) and f_lower.endswith((".fits", ".fit")):
                    # Extract the suffix following the star name.
                    remainder = f_lower[len(prefix):]
                    # Accept an exact FITS extension or a suffix separated by an underscore.
                    if remainder == ".fits" or remainder == ".fit" or remainder.startswith("_"):
                        matched_filename = real_name
                        break

        if not matched_filename:
            print(f"  [X] {star_name:<15} -> FITS file not found ('{fits_filename}')")
            continue

        fits_full_path = os.path.join(abs_calspec_dir, matched_filename)

        try:
            x_input = np.array(gaia_dict[gaia_id]["flux"], dtype=np.float32)
            x_error = np.array(gaia_dict[gaia_id]["flux_error"], dtype=np.float32)

            if np.isnan(x_input).any() or np.isnan(x_error).any():
                raise ValueError("Gaia returned NaN values for this spectrum.")

            c2_val, c3_val = calculate_c2_c3(x_input, wave_length_grid)

            # Process ground truth spectrum Y (CALSPEC) via LSF convolution & interpolation
            y_target = process_calspec_target(fits_full_path, wave_length_grid)

            # Validate the flux scale.
            median_x = np.nanmedian(x_input)
            median_y = np.nanmedian(y_target)

            if median_x > 0 and median_y > 0:
                ratio = median_x / median_y
                if ratio > 100.0 or ratio < 0.01:
                    raise ValueError(f"Flux scale mismatch! Gaia med: {median_x:.2e}, CALSPEC med: {median_y:.2e}")
            else:
                raise ValueError(f"Invalid median flux. Gaia: {median_x:.2e}, CALSPEC: {median_y:.2e}")

            y_target = y_target.astype(np.float32)

            X_list.append(x_input)
            Y_list.append(y_target)
            X_error_list.append(x_error)
            G_mags_list.append(safe_float(row.get("phot_g_mean_mag"), 15.0))
            bp_rp_list.append(safe_float(row.get("bp_rp"), 0.75))
            ruwe_value = safe_float(row.get("ruwe"), np.nan)
            ruwe_list.append(ruwe_value if np.isfinite(ruwe_value) and ruwe_value > 0 else np.nan)
            c2_list.append(c2_val)
            c3_list.append(c3_val)
            processed_stars.append(star_name)
            processed_ids.append(str(gaia_id))

            print(f"  [✓] {star_name:<15} -> Successfully processed")

        except Exception as e:
            print(f"  [!] {star_name:<15} -> Failed processing: {e}")
            continue

    if not processed_stars:
        print("\n[!] Critical Error: Zero stars were successfully processed.")
        return

    # Convert Python lists to NumPy arrays.
    X_np = np.array(X_list, dtype=np.float32)
    Y_np = np.array(Y_list, dtype=np.float32)
    X_err_np = np.array(X_error_list, dtype=np.float32)

    # Prepare normalized arrays for neural network training.
    # Scale each input spectrum by its maximum flux.
    scales = np.max(X_np, axis=1, keepdims=True) + 1e-30
    X_norm_np = X_np / scales

    # Compute the target spectral correction ratio.
    Y_ratio_np = compute_calibration_ratio(X_np, Y_np)
    
    # Propagate fractional uncertainties for the heteroscedastic loss.
    Errors_norm_np = X_err_np / stable_flux_denominator(X_np)

    os.makedirs(abs_output_dir, exist_ok=True)
    
    # Save absolute arrays for metric evaluation.
    np.save(os.path.join(abs_output_dir, "X.npy"), X_np)
    np.save(os.path.join(abs_output_dir, "Y.npy"), Y_np)
    np.save(os.path.join(abs_output_dir, "X_error.npy"), X_err_np)

    # Save normalized arrays for neural network training.
    np.save(os.path.join(abs_output_dir, "X_norm.npy"), X_norm_np)
    np.save(os.path.join(abs_output_dir, "Y_ratio.npy"), Y_ratio_np)
    np.save(os.path.join(abs_output_dir, "Errors_norm.npy"), Errors_norm_np)

    # Save stellar parameters and metadata arrays.
    np.save(os.path.join(abs_output_dir, "G_mags.npy"), np.array(G_mags_list, dtype=np.float32))
    np.save(os.path.join(abs_output_dir, "bp_rp.npy"), np.array(bp_rp_list, dtype=np.float32))
    ruwe_np = np.array(ruwe_list, dtype=np.float32)
    np.save(os.path.join(abs_output_dir, "ruwe.npy"), ruwe_np)
    np.save(os.path.join(abs_output_dir, "ruwe_missing.npy"), np.isnan(ruwe_np))
    np.save(os.path.join(abs_output_dir, "c2.npy"), np.array(c2_list, dtype=np.float32))
    np.save(os.path.join(abs_output_dir, "c3.npy"), np.array(c3_list, dtype=np.float32))
    np.save(os.path.join(abs_output_dir, "star_names.npy"), np.array(processed_stars))
    np.save(os.path.join(abs_output_dir, "gaia_source_ids.npy"), np.array(processed_ids))
    np.save(os.path.join(abs_output_dir, "wave_grid.npy"), wave_length_grid)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    create_dataset()