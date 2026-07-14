"""
Module: prepare_dataset.py
Report Section: Chapter 4 & 5 Data Preparation, Spectral Interpolation, and Realignment

This script implements the continuous data preparation pipeline:
1. Loads the cross-matched catalog.
2. Dynamically locates CALSPEC FITS files using a case-insensitive scan.
3. Degrades CALSPEC resolution via LSF Gaussian convolution (Eq. 1).
4. Interpolates both Gaia and CALSPEC spectra onto a common grid Lambda (Eq. 2).
5. Normalizes continuous fluxes via trapezoidal integration to ensure numerical stability.
6. Saves the final regression matrices X and Y as NumPy binaries.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.processing.download_gaia import load_real_gaia_spectrum
from src.utils.spectra_utils import load_calspec_fits, degrade_spectral_resolution, normalize_spectrum


def build_and_save_dataset(csv_metadata_path: str, calspec_dir: str, output_dir: str, wave_length_grid: np.ndarray):
    """
    Main pipeline execution loop that processes all stars and stores X and Y arrays.
    Parameters:
    - csv_metadata_path (str): Path to the crossmatch CSV metadata file.
    - calspec_dir (str): Directory containing CALSPEC FITS files.
    - output_dir (str): Directory where processed X and Y arrays will be saved.
    - wave_length_grid (np.ndarray): The common wavelength grid for interpolation.
    """
    print("=" * 60)
    print("[5.0] EXECUTING DATASET GENERATION PIPELINE")
    print("=" * 60)

    abs_csv_path = os.path.abspath(csv_metadata_path)
    abs_calspec_dir = os.path.abspath(calspec_dir)
    abs_output_dir = os.path.abspath(output_dir)

    if not os.path.exists(abs_csv_path):
        print(f"[!] ERROR: Crossmatch CSV file not found at: {abs_csv_path}")
        return

    df = pd.read_csv(abs_csv_path)
    
    X_list = []
    Y_list = []
    processed_stars = []
    
    print(f"Starting pipeline processing for {len(df)} stars...")
    print(f"Searching FITS data inside: {abs_calspec_dir}")
    
    # DIAGNOSTIC BLOCK 
    if not os.path.exists(abs_calspec_dir):
        print(f"  [!] CRITICAL: The directory does not exist physically: {abs_calspec_dir}")
        files_in_dir = []
    else:
        files_in_dir = [f for f in os.listdir(abs_calspec_dir) if os.path.isfile(os.path.join(abs_calspec_dir, f))]
    
    print(f"  [i] Total files physically detected in directory: {len(files_in_dir)}")
    if len(files_in_dir) > 0:
        print(f"  [i] Sample files in directory (first 5): {files_in_dir[:5]}")
    else:
        print("  [!] WARNING: The folder is completely empty or cannot be read by WSL/Linux.")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        star_name = str(row["calspec_name"]).strip()
        fits_filename = str(row["fits_file"]).strip()
        gaia_id = str(row["gaia_source_id"]) if "gaia_source_id" in row else "0"
        
        matched_filename = None
        
        # 1. Exact Match Check
        if fits_filename in files_in_dir:
            matched_filename = fits_filename
        else:
            # 2. Case-Insensitive & Pattern Fallback Check
            star_name_lower = star_name.lower()
            fits_filename_lower = fits_filename.lower()
            
            for f in files_in_dir:
                f_lower = f.lower()
                # Check match against lowercase filename or star prefix
                if f_lower == fits_filename_lower or (f_lower.startswith(star_name_lower) and (f_lower.endswith(".fits") or f_lower.endswith(".fit"))):
                    matched_filename = f
                    break
                    
        if not matched_filename:
            print(f"  [!] Missing FITS for {star_name}: Searched '{fits_filename}' (Case-Insensitive match failed)")
            continue
            
        fits_path = os.path.join(abs_calspec_dir, matched_filename)
                
        try:
            # 1. Load high-resolution absolute truth from HST/CALSPEC
            calspec_wave, calspec_flux = load_calspec_fits(fits_path)
            
            # 2. Degrade resolution via LSF Convolution to emulate Gaia conditions (Eq. 1)
            deg_wave, deg_flux = degrade_spectral_resolution(calspec_wave, calspec_flux, sigma_nm=8.0)
            
            # 3. Align the Ground Truth (Y) onto the common homogenous grid Lambda (Eq. 2)
            y_interp_func = interp1d(deg_wave, deg_flux, kind='cubic', fill_value="extrapolate")
            y_flux_aligned = y_interp_func(wave_length_grid)
            
            # 4. Load/Fetch the corresponding Gaia BP/RP spectrum (X)
            #x_flux_raw = mock_load_gaia_observed_spectrum(deg_wave, deg_flux)

            x_flux_raw = load_real_gaia_spectrum(gaia_id, wave_length_grid)

            # Seguro vital: Si Gaia no tiene el espectro, saltamos esta estrella y no la metemos en X_list
            if x_flux_raw is None:
                print(f"  [>>] Skipping {star_name} due to missing real Gaia data.")
                continue

            
            # 5. Apply continuous normalization scaling (Section 4.5)
            x_normalized = normalize_spectrum(x_flux_raw, wave_length_grid)
            y_normalized = normalize_spectrum(y_flux_aligned, wave_length_grid)
            
            X_list.append(x_normalized)
            Y_list.append(y_normalized)
            processed_stars.append(star_name)
            print(f"  [✓] Successfully processed: {star_name} -> {matched_filename}")
            
        except Exception as e:
            print(f"  [!] Failed processing {star_name} ({matched_filename}): {str(e)}")
            continue

    if len(processed_stars) == 0:
        print("\n[!] ERROR: No valid stars were processed.")
        print(f"Please verify why the filenames above do not match your files in: {abs_calspec_dir}")
        return

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    os.makedirs(abs_output_dir, exist_ok=True)
    
    np.save(os.path.join(abs_output_dir, "X.npy"), X)
    np.save(os.path.join(abs_output_dir, "Y.npy"), Y)
    np.save(os.path.join(abs_output_dir, "star_names.npy"), np.array(processed_stars))
    
    print("\n" + "=" * 40)
    print("       PIPELINE COMPILED SUMMARY")
    print("=" * 40)
    print(f"Total processed stars successfully saved: {len(processed_stars)}")
    print(f"Features Array Matrix Shape (X)         : {X.shape}")
    print(f"Targets Array Matrix Shape (Y)          : {Y.shape}")
    print(f"Output files stored inside location     : {abs_output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Path configuration synchronized with the project routing layout (Section 3.2)
    csv_path = "data/processed/crossmatch_catalog.csv"
    calspec_dir = "data/raw/calspec_fits/" 
    processed_data_dir = "data/processed/"

    # Define globally the homogeneous wavelength grid (Lambda) in nanometers (Section 5.3)
    wave_length_grid = np.linspace(330.0, 1050.0, 300)[:280]
    
    build_and_save_dataset(csv_path, calspec_dir, processed_data_dir, wave_length_grid)