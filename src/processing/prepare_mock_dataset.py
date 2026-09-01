"""
Module: prepare_mock_dataset.py
Report Section: 5.0 Mock Dataset Preparation

This module constructs a spectral regression dataset for training and evaluating 
neural network architectures. It matches CALSPEC FITS reference spectra with Gaia 
source identifiers, degrades high-resolution spectra to simulate Gaia BP/RP instrument 
resolution, interpolates fluxes onto a homogeneous wavelength grid, and exports physical 
flux matrices as NumPy binaries (.npy).
"""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.utils.data.dataset_utils import (
    degrade_spectral_resolution,
    load_calspec_fits,
    mock_load_gaia_observed_spectrum,
)


def build_and_save_dataset(
    csv_metadata_path: Union[str, Path],
    calspec_dir: Union[str, Path],
    output_dir: Union[str, Path],
    wave_length_grid: np.ndarray,
) -> None:
    """
    Build and export the (X, Y) regression dataset from CALSPEC and Gaia catalog data.

    Reads the cross-match metadata table, locates corresponding CALSPEC FITS files, 
    degrades high-resolution spectra to match empirical LSF conditions, interpolates 
    fluxes onto a uniform evaluation grid, generates mock Gaia observations, and 
    saves feature and ground-truth matrices.

    Parameters
    ----------
    csv_metadata_path : str or Path
        Path to the cross-match CSV metadata catalog.
    calspec_dir : str or Path
        Directory containing raw CALSPEC FITS spectral files.
    output_dir : str or Path
        Destination directory where output NumPy binary files (.npy) will be stored.
    wave_length_grid : numpy.ndarray
        Homogeneous wavelength grid in nanometers for spectral interpolation.

    Returns
    -------
    None

    Notes
    -----
    This function preserves absolute physical flux scales (W / m^2 / nm) to allow 
    custom scaling transformations during network training.
    """
    print("=" * 60)
    print("[5.0] EXECUTING DATASET GENERATION PIPELINE")
    print("=" * 60)

    csv_path = Path(csv_metadata_path).resolve()
    calspec_path = Path(calspec_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not csv_path.exists():
        print(f"[!] ERROR: Crossmatch CSV file not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    processed_stars: List[str] = []

    print(f"Starting pipeline processing for {len(df)} target stars...")
    print(f"Searching FITS data inside: {calspec_path}")

    # Inspect directory contents for physical file matching
    if not calspec_path.exists():
        print(f"  [!] CRITICAL: Directory does not exist physically: {calspec_path}")
        files_in_dir: List[str] = []
    else:
        files_in_dir = [f.name for f in calspec_path.iterdir() if f.is_file()]

    print(f"  [i] Total files physically detected in directory: {len(files_in_dir)}")
    if len(files_in_dir) > 0:
        print(f"  [i] Sample files in directory (first 5): {files_in_dir[:5]}")
    else:
        print("  [!] WARNING: Directory is empty or unreadable.")
    print("-" * 60)

    for _, row in df.iterrows():
        star_name = str(row["calspec_name"]).strip()
        fits_filename = str(row["fits_file"]).strip()
        gaia_id = str(row.get("gaia_source_id", "0"))

        matched_filename = None

        # 1. Direct filename match
        if fits_filename in files_in_dir:
            matched_filename = fits_filename
        else:
            # 2. Case-insensitive and prefix match fallback
            star_name_lower = star_name.lower()
            fits_filename_lower = fits_filename.lower()

            for f in files_in_dir:
                f_lower = f.lower()
                if f_lower == fits_filename_lower or (
                    f_lower.startswith(star_name_lower)
                    and (f_lower.endswith(".fits") or f_lower.endswith(".fit"))
                ):
                    matched_filename = f
                    break

        if not matched_filename:
            print(f"  [!] Missing FITS for {star_name}: Searched '{fits_filename}' (Match failed)")
            continue

        fits_file_path = calspec_path / matched_filename

        try:
            # Load HST/CALSPEC high-resolution reference spectrum
            calspec_wave, calspec_flux = load_calspec_fits(str(fits_file_path))

            # Degrade spectral resolution via Gaussian LSF convolution (sigma = 8.0 nm)
            deg_wave, deg_flux = degrade_spectral_resolution(calspec_wave, calspec_flux, sigma_nm=8.0)

            # Cubic spline interpolation onto target wavelength grid
            y_interp_func = interp1d(
                deg_wave, 
                deg_flux, 
                kind="cubic", 
                fill_value="extrapolate"
            )
            y_flux_aligned = y_interp_func(wave_length_grid)

            # Generate synthetic low-resolution Gaia observation (X)
            x_flux_raw = mock_load_gaia_observed_spectrum(deg_wave, deg_flux, wave_length_grid)

            X_list.append(x_flux_raw)
            Y_list.append(y_flux_aligned)
            processed_stars.append(star_name)

            print(f"  [✓] Successfully processed: {star_name} -> {matched_filename}")

        except Exception as e:
            print(f"  [!] Failed processing {star_name} ({matched_filename}): {e}")
            continue

    if not processed_stars:
        print("\n[!] ERROR: No valid target spectra were processed.")
        print(f"    Verify catalog filenames against directory contents in: {calspec_path}")
        return

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "X.npy", X)
    np.save(output_path / "Y.npy", Y)
    np.save(output_path / "star_names.npy", np.array(processed_stars))

    print("\n" + "=" * 40)
    print("        PIPELINE COMPILED SUMMARY")
    print("=" * 40)
    print(f"Total processed stars successfully saved: {len(processed_stars)}")
    print(f"Features Array Matrix Shape (X)         : {X.shape}")
    print(f"Targets Array Matrix Shape (Y)          : {Y.shape}")
    print(f"Output files stored inside location     : {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Alignment with Section 3.2 data architecture
    default_csv_path = "data/processed/crossmatch_catalog.csv"
    default_calspec_dir = "data/raw/calspec_fits/"
    default_output_dir = "data/processed/"

    # Homogeneous wavelength grid (300 channels between 330.0 nm and 1050.0 nm)
    n_channels = 300
    grid = np.linspace(330.0, 1050.0, n_channels, dtype=np.float32)

    build_and_save_dataset(default_csv_path, default_calspec_dir, default_output_dir, grid)