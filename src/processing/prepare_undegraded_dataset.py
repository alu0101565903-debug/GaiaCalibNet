"""
Module: prepare_undegraded_dataset.py
Report Section: Chapter 5 Data Processing & Feature Engineering

Undegraded Spectral Dataset Generator for Super-Resolution PINNs.

This script constructs the full regression dataset (X and Y matrices) for training 
and evaluating Physics-Informed Neural Network (PINN) models aiming to reconstruct 
high-resolution physical absorption lines. It maps CALSPEC target spectra to Gaia DR3 
source IDs, converts wavelengths from Ångströms to nanometers, and interpolates the 
data onto a unified wavelength grid WITHOUT applying instrumental Line Spread 
Function (LSF) degradation to the target (Y) spectra.

The resulting matrices, along with associated physical stellar parameters, 
are saved as single-precision (float32) NumPy arrays to optimize GPU memory.
"""

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

# Attempt to import GaiaXPy, required for fetching Gaia BP/RP spectra
try:
    from gaiaxpy import calibrate
except ImportError:
    calibrate = None


# =====================================================================
# DATA ACQUISITION HELPERS
# =====================================================================

def fetch_real_gaia_spectrum(
    source_id: int, 
    wave_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch and calibrate real Gaia BP/RP spectra using the official GaiaXPy tool.

    Queries the Gaia archive for a specific source ID, calibrates raw continuous 
    coefficients into absolute flux, and samples it onto the specified wavelength grid.

    Parameters
    ----------
    source_id : int
        The unique Gaia DR3 Source Identifier.
    wave_grid : numpy.ndarray
        The target wavelength grid array in nanometers.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing:
        - flux (numpy.ndarray): Calibrated absolute flux array (float32).
        - flux_err (numpy.ndarray): Associated flux uncertainty array (float32).

    Raises
    ------
    ImportError
        If the `gaiaxpy` package is not installed in the current environment.
    ValueError
        If the returned spectrum is empty or contains non-physical NaN values.
    RuntimeError
        If the Gaia API query or calibration routine fails.
    """
    if calibrate is None:
        raise ImportError(
            "The 'gaiaxpy' library is required to query the Gaia API. "
            "Install it with: pip install gaiaxpy"
        )

    try:
        # Execute official query and calibration onto the uniform wavelength grid
        calibrated_spectra, _ = calibrate(
            [int(source_id)], 
            sampling=wave_grid, 
            truncation=False, 
            verbose=False
        )
        flux = np.array(calibrated_spectra["flux"].iloc[0], dtype=np.float32)
        flux_err = np.array(calibrated_spectra["flux_error"].iloc[0], dtype=np.float32)

        # Validate spectrum integrity
        if np.any(np.isnan(flux)) or len(flux) == 0:
            raise ValueError(f"The returned spectrum for Gaia Source ID {source_id} contains NaN values.")

        return flux, flux_err

    except Exception as e:
        # Prevent contaminated or empty rows in the X matrix
        raise RuntimeError(
            f"Real Gaia spectrum was not recovered for Source ID {source_id}: {e}"
        ) from e


def load_calspec_pristine(
    file_path: Union[str, Path], 
    wave_grid: np.ndarray
) -> np.ndarray:
    """
    Load and interpolate a CALSPEC reference spectrum onto a defined grid 
    without applying instrumental degradation (LSF convolution).

    Provides the PINN with a pristine ground-truth target (Y) containing 
    its original high-resolution absorption lines, encouraging the network to 
    learn super-resolution deconvolution.

    Parameters
    ----------
    file_path : str or Path
        Path to the CALSPEC spectrum file (.fits, .npy, or .csv).
    wave_grid : numpy.ndarray
        The target wavelength grid array in nanometers.

    Returns
    -------
    numpy.ndarray
        The interpolated, pristine CALSPEC flux array (float32).
    """
    str_path = str(file_path)

    # Parse file format based on extension
    if str_path.endswith(".npy"):
        data = np.load(str_path, allow_pickle=True).item()
        # Wavelengths in CALSPEC arrays are typically in Angstroms
        raw_wave = data["wave"] / 10.0 if np.max(data["wave"]) > 2000.0 else data["wave"]
        raw_flux = data["flux"]

    elif str_path.endswith((".fits", ".fit")):
        from astropy.io import fits
        with fits.open(str_path) as hdul:
            data = None
            for ext in hdul:
                if hasattr(ext, "columns") and "WAVELENGTH" in ext.columns.names:
                    data = ext.data
                    break
            if data is None:
                data = hdul[1].data
            
            # Convert Angstroms to nanometers (1 nm = 10 Å)
            raw_wave = np.array(data["WAVELENGTH"], dtype=np.float64) / 10.0
            raw_flux = np.array(data["FLUX"], dtype=np.float64)
    else:
        # CSV fallback
        df_spec = pd.read_csv(str_path)
        raw_wave = df_spec["wavelength"].values
        if np.max(raw_wave) > 2000.0:
            raw_wave /= 10.0
        raw_flux = df_spec["flux"].values

    # Perform direct linear interpolation onto the uniform machine-learning grid
    y_pristine = np.interp(wave_grid, raw_wave, raw_flux)
    return y_pristine.astype(np.float32)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def create_dataset(
    csv_path: Union[str, Path], 
    calspec_dir: Union[str, Path], 
    output_dir: Union[str, Path],
    n_channels: int = 300, 
    wave_min_nm: float = 340.0, 
    wave_max_nm: float = 1000.0
) -> None:
    """
    Generate the synchronized Undegraded Spectral Dataset for PINN training.

    Iterates through a cross-matched catalog, extracts pristine CALSPEC targets, 
    queries real Gaia spectra (using cached arrays if available), and bundles all 
    associated physical stellar parameters into synchronized NumPy matrices.

    Parameters
    ----------
    csv_path : str or Path
        Path to the cross-matched index catalog CSV file.
    calspec_dir : str or Path
        Directory containing the raw high-resolution CALSPEC files.
    output_dir : str or Path
        Target directory to save the final compiled NumPy matrices.
    n_channels : int, optional
        Number of discrete wavelength bins for the uniform grid. Default is 300.
    wave_min_nm : float, optional
        Lower boundary of the wavelength grid in nanometers. Default is 340.0.
    wave_max_nm : float, optional
        Upper boundary of the wavelength grid in nanometers. Default is 1000.0.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If input catalog path does not exist.
    RuntimeError
        If zero target stars are successfully processed.
    """
    print("=" * 60)
    print("EXECUTING PIPELINE: GENERATING (X, Y) SPECTRAL DATASET")
    print("=" * 60)

    csv_file = Path(csv_path).resolve()
    calspec_path = Path(calspec_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Dynamic uniform wavelength grid generation
    wave_grid = np.linspace(wave_min_nm, wave_max_nm, n_channels, dtype=np.float32)

    # 2. Validate and load catalog
    if not csv_file.exists():
        raise FileNotFoundError(f"Input catalog not found at {csv_file}")

    df = pd.read_csv(csv_file)

    # 3. Cache Optimization: Reuse pre-fetched Gaia X matrices if available
    mode = out_path.name
    degraded_x_cache = Path(f"data/processed/nn/{mode}/X.npy")
    degraded_x_err_cache = Path(f"data/processed/nn/{mode}/X_error.npy")

    cached_X = np.load(degraded_x_cache) if degraded_x_cache.exists() else None
    cached_X_err = np.load(degraded_x_err_cache) if degraded_x_err_cache.exists() else None

    if cached_X is not None:
        print(f"[INFO] Reusing X and X_error matrices from cache: {degraded_x_cache.parent}")

    # Synchronized dataset accumulators
    X_list, Y_list, X_error_list = [], [], []
    G_mags_list, bp_rp_list, ruwe_list = [], [], []
    processed_stars, processed_ids = [], []

    print(f"[INFO] Processing {len(df)} stars for the UNDEGRADED dataset (Mode: {mode})...")

    # 4. Iterate over catalog targets
    for idx, row in df.iterrows():
        source_id = row["gaia_source_id"]
        filename = row["fits_file"]
        star_name = str(row["calspec_name"])
        fits_full_path = calspec_path / filename

        try:
            # Extract stellar parameters
            if pd.isna(row["phot_g_mean_mag"]) or pd.isna(row["bp_rp"]) or pd.isna(row["ruwe"]):
                raise ValueError("One or more stellar parameters (phot_g_mean_mag, bp_rp, ruwe) are NaN.")

            g_val = float(row["phot_g_mean_mag"])
            bp_rp_val = float(row["bp_rp"])
            ruwe_val = float(row["ruwe"])

            # --- Target (Y): CALSPEC Pristine (Undegraded) ---
            y_pristine = load_calspec_pristine(fits_full_path, wave_grid)

            # --- Input (X): Real Gaia Spectrum ---
            if cached_X is not None and cached_X_err is not None and idx < len(cached_X):
                x_gaia = cached_X[idx].astype(np.float32)
                x_gaia_err = cached_X_err[idx].astype(np.float32)
            else:
                x_gaia, x_gaia_err = fetch_real_gaia_spectrum(source_id, wave_grid)

            X_list.append(x_gaia)
            Y_list.append(y_pristine)
            X_error_list.append(x_gaia_err)

            G_mags_list.append(g_val)
            bp_rp_list.append(bp_rp_val)
            ruwe_list.append(ruwe_val)

            processed_stars.append(star_name)
            processed_ids.append(int(source_id))

        except Exception as err:
            print(f"[WARN] Skipping star {idx} (Gaia: {source_id}) -> {err}")
            continue

    if len(X_list) == 0:
        raise RuntimeError("[CRITICAL ERROR] No star could be processed successfully.")

    # 5. Finalize single-precision arrays for memory optimization
    X_final = np.array(X_list, dtype=np.float32)
    Y_final = np.array(Y_list, dtype=np.float32)
    X_err_final = np.array(X_error_list, dtype=np.float32)

    G_final = np.array(G_mags_list, dtype=np.float32)
    BP_RP_final = np.array(bp_rp_list, dtype=np.float32)
    RUWE_final = np.array(ruwe_list, dtype=np.float32)

    stars_final = np.array(processed_stars, dtype=str)
    ids_final = np.array(processed_ids, dtype=np.int64)

    # 6. Export matrices to disk
    np.save(out_path / "X.npy", X_final)
    np.save(out_path / "Y.npy", Y_final)
    np.save(out_path / "X_error.npy", X_err_final)
    np.save(out_path / "G_mags.npy", G_final)
    np.save(out_path / "bp_rp.npy", BP_RP_final)
    np.save(out_path / "ruwe.npy", RUWE_final)
    np.save(out_path / "processed_stars.npy", stars_final)
    np.save(out_path / "processed_ids.npy", ids_final)
    np.save(out_path / "wave_grid.npy", wave_grid)

    print("\n[✓] Undegraded Dataset generated successfully:")
    print(f"    - Matrix X (Real Gaia inputs):              {X_final.shape}")
    print(f"    - Matrix Y (CALSPEC pristine, high-res):    {Y_final.shape}")
    print(f"    - Matrix X_error (Gaia uncertainties):      {X_err_final.shape}")
    print(f"    - Parameter G Magnitude:                    {G_final.shape}")
    print(f"    - Parameter BP-RP Color:                    {BP_RP_final.shape}")
    print(f"    - Parameter RUWE (Astrometry):              {RUWE_final.shape}")
    print(f"    - Star Names (Metadata):                    {stars_final.shape}")
    print(f"    - Gaia Source IDs (Metadata):               {ids_final.shape}")
    print(f"\n[✓] Results saved to: {out_path}\n")


if __name__ == "__main__":
    catalog_mode = "solar"
    create_dataset(
        csv_path=f"data/processed/catalogs_by_color/{catalog_mode}_catalog.csv",
        calspec_dir="data/raw/calspec_fits",
        output_dir=f"data/processed/nn_undegraded/{catalog_mode}",
        n_channels=300,
        wave_min_nm=340.0,
        wave_max_nm=1000.0,
    )