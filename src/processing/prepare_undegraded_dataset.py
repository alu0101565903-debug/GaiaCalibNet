"""
Module: prepare_undegraded_dataset.py
Report Section: Chapter 5 Data Processing & Feature Engineering

Undegraded Spectral Dataset Generator for Super-Resolution PINNs.

This script constructs the full regression dataset (X and Y matrices) for training 
and evaluating Physics-Informed Neural Network (PINN) models aiming to reconstruct 
high-resolution physical lines. It maps CALSPEC target spectra to Gaia DR3 
source IDs, converts wavelengths from Angstroms to nanometers, and interpolates the 
data onto a unified wavelength grid WITHOUT applying instrumental Line Spread 
Function (LSF) degradation to the target (Y) spectra.

The resulting matrices, along with associated physical stellar parameters, 
are saved as single-precision (float32) NumPy arrays to optimize GPU memory.
"""

import os
from pathlib import Path
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

def fetch_real_gaia_spectrum(source_id: int, wave_grid: np.ndarray) -> np.ndarray:
    """
    Fetches and calibrates real Gaia BP/RP spectra using the official GaiaXPy tool.

    Queries the Gaia archive for a specific source ID, calibrates the raw continuous 
    coefficients into absolute flux, and samples it onto the specified wavelength grid.

    Parameters
    ----------
    source_id : int
        The unique Gaia DR3 Source Identifier.
    wave_grid : np.ndarray
        The target wavelength grid array in nanometers.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - flux: The calibrated absolute flux array (float32).
        - flux_err: The associated flux uncertainty array (float32).

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
        # Wrap and raise the exception to prevent contaminated or empty rows in the X matrix
        raise RuntimeError(
            f" Real Gaia spectrum was not recovered for Source ID {source_id}: {str(e)}"
        ) from e


def load_calspec_pristine(file_path: str, wave_grid: np.ndarray) -> np.ndarray:
    """
    Loads and interpolates a CALSPEC reference spectrum onto a defined grid
    without applying any instrumental degradation (LSF convolution).

    This provides the PINN with a "pristine" ground truth target (Y) containing 
    its original high-resolution absorption lines, encouraging the network to 
    learn super-resolution deconvolution.

    Parameters
    ----------
    file_path : str
        Path to the CALSPEC spectrum file (.fits, .npy, or .csv).
    wave_grid : np.ndarray
        The target wavelength grid array in nanometers.

    Returns
    -------
    np.ndarray
        The interpolated, pristine CALSPEC flux array (float32).
    """
    # Parse file format based on extension
    if file_path.endswith(".npy"):
        data = np.load(file_path, allow_pickle=True).item()
        # Convert Angstroms to nanometers (1 nm = 10 A)
        raw_wave = data["wave"]
        raw_flux = data["flux"]

    elif file_path.endswith(".fits"):   
        from astropy.io import fits
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            # Convert Angstroms to nanometers (1 nm = 10 A)
            raw_wave = data['WAVELENGTH'] / 10.0
            raw_flux = data['FLUX']
    else:
        # Assume CSV format as a universal fallback
        df_spec = pd.read_csv(file_path)
        raw_wave = df_spec["wavelength"].values
        raw_flux = df_spec["flux"].values

    # Perform direct linear interpolation onto the uniform machine-learning grid.
    # Note: No Gaussian filtering or instrumental LSF is applied here.
    y_pristine = np.interp(wave_grid, raw_wave, raw_flux)
    return y_pristine.astype(np.float32)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def create_dataset(
        csv_path: str, 
        calspec_dir: str, 
        output_dir: str,
        n_channels: int, 
        wave_min_nm: float, 
        wave_max_nm: float
    ) -> None:
    """
    Generates the synchronized Undegraded Spectral Dataset for PINN training.

    Iterates through a cross-matched catalog, extracts pristine CALSPEC targets, 
    queries real Gaia spectra (using cache if available), and bundles all 
    associated physical stellar parameters into synchronized NumPy matrices.

    Parameters
    ----------
    csv_path : str
        Path to the cross-matched index catalog CSV file.
    calspec_dir : str
        Directory containing the raw high-resolution CALSPEC files.
    output_dir : str
        Target directory to save the final compiled NumPy matrices.
    n_channels : int
        Number of discrete wavelength bins for the uniform grid.
    wave_min_nm : float
        Lower boundary of the wavelength grid (in nanometers).
    wave_max_nm : float
        Upper boundary of the wavelength grid (in nanometers).
    """
    print("=" * 60)
    print("EXECUTING PIPELINE: GENERATING (X, Y) SPECTRAL DATASET")
    print("=" * 60)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate the uniform spatial resolution grid dynamically
    wave_grid = np.linspace(wave_min_nm, wave_max_nm, n_channels)
    
    # 2. Validate and load the cross-matched catalog
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input catalog not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 3. Cache Optimization: Try to reuse existing real X matrices
    # Querying the Gaia archive takes time. We assume the equivalent degraded path 
    # uses the same naming structure, allowing us to borrow the X (input) data.
    mode = Path(output_dir).name
    degraded_x_cache = Path(f"data/processed/nn/{mode}/X.npy")
    degraded_x_err_cache = Path(f"data/processed/nn/{mode}/X_error.npy")
    
    cached_X = np.load(degraded_x_cache) if degraded_x_cache.exists() else None
    cached_X_err = np.load(degraded_x_err_cache) if degraded_x_err_cache.exists() else None
    
    if cached_X is not None:
        print(f"[INFO] Reusing X and X_error matrices from cache: {degraded_x_cache.parent}")

    # Accumulator lists for synchronized dataset generation
    X_list, Y_list, X_error_list = [], [], []
    G_mags_list, bp_rp_list, ruwe_list = [], [], []
    processed_stars, processed_ids = [], []
    
    print(f"[INFO] Processing {len(df)} stars for the UNDEGRADED dataset (Mode: {mode})...")

    # 4. Iterate over catalog targets
    for idx, row in df.iterrows():
        source_id = row["gaia_source_id"]
        
        # Build the path to the original CALSPEC FITS file using the 'fits_file' column
        filename = row["fits_file"]
        star_name = str(row["calspec_name"])
        calspec_path = os.path.join(calspec_dir, filename)


        try:
            # --- Extract stellar parameters directly by column name ---
            if pd.isna(row["phot_g_mean_mag"]) or pd.isna(row["bp_rp"]) or pd.isna(row["ruwe"]):
                raise ValueError("One or more stellar parameters (phot_g_mean_mag, bp_rp, ruwe) are NaN.")

            g_val = float(row["phot_g_mean_mag"])
            bp_rp_val = float(row["bp_rp"])
            ruwe_val = float(row["ruwe"])

            # --- Target (Y): CALSPEC Pristine (Undegraded) ---
            y_pristine = load_calspec_pristine(calspec_path, wave_grid)

            # --- Input (X): Real Gaia Spectrum ---
            # Use cached matrices if available to drastically reduce network overhead
            if cached_X is not None and cached_X_err is not None and idx < len(cached_X):
                x_gaia = cached_X[idx].astype(np.float32)
                x_gaia_err = cached_X_err[idx].astype(np.float32)
            else:
                x_gaia, x_gaia_err = fetch_real_gaia_spectrum(source_id, wave_grid)

            # If the extraction succeeds, append all synchronized data to lists
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

    # Guard against completely failed dataset generation
    if len(X_list) == 0:
        raise RuntimeError("[CRITICAL ERROR] No star could be processed successfully.")

    # 5. Finalize and cast matrices to memory-efficient float32 arrays
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

    # Output detailed shape summaries
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
    # Example execution parameters
    mode = "solar"
    create_dataset(
        csv_path = f"data/processed/catalogs_by_color/{mode}_catalog.csv",
        calspec_dir = "data/raw/calspec_fits",
        output_dir = f"data/processed/nn_undegraded/{mode}",
        n_channels = 300,
        wave_min_nm = 340.0,
        wave_max_nm = 1000.0
    )