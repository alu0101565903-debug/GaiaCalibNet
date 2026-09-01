"""
Module: extract_features.py
Report Section: Chapter 5 Data Processing & Feature Engineering

This script isolates 5 key scalar astrometric and photometric features from the Gaia DR3 
crossmatch catalog (parallax, pmra, pmdec, phot_g_mean_mag, and bp_rp). It handles 
missing-value imputation via column medians, performs a data quality audit, and exports 
the final feature matrix as a float32 NumPy binary (.npy) file for PyTorch model training.
"""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


def extract_gaia_features(
    csv_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Extract and preprocess Gaia scalar features for feature-based model training.

    Reads the crossmatch catalog, validates required Gaia astrometric and photometric 
    columns, coerces values to numerical format, imputes missing entries using 
    feature-wise medians, and exports a float32 NumPy feature matrix.

    Parameters
    ----------
    csv_path : str or Path
        Path to the input processed crossmatch catalog CSV file.
    output_path : str or Path
        Target file path to save the output NumPy binary array (.npy).

    Returns
    -------
    None

    Notes
    -----
    Target feature vector layout:
    [parallax, pmra, pmdec, phot_g_mean_mag, bp_rp]
    """
    print("=" * 60)
    print("[5.1] EXTRACTING GAIA SCALAR ASTROMETRIC & PHOTOMETRIC FEATURES")
    print("=" * 60)

    csv_file = Path(csv_path)
    output_file = Path(output_path)

    if not csv_file.exists():
        print(f"[!] ERROR: Not able to find catalog at {csv_path}")
        return

    # Read catalog and convert common missing-value representations to NaN
    df = pd.read_csv(csv_file, na_values=["--", "-", "NA", "N/A", ""])

    # Select the scalar features used by the feature-based model branch
    feature_cols: List[str] = ["parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"]

    # Verify column presence
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"[!] ERROR: Missing expected columns in {csv_path}: {missing_cols}")
        return

    print("-> Cohering columns to numeric vectors and scanning for missing data...")

    # Convert feature columns to numeric vectors, mapping unparseable values to NaN
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Data quality audit
    nan_counts = df[feature_cols].isna().sum()
    total_nans = int(nan_counts.sum())

    if total_nans > 0:
        print("\n[i] Missing Values Detected (imputing with feature median):")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"    - {col}: {count} missing values")
    else:
        print("[✓] Matrix is completely dense. No NaNs detected.")

    # Perform column-wise median imputation
    medians = df[feature_cols].median()
    X_features = df[feature_cols].fillna(medians).values.astype(np.float32)

    # Export array to disk
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, X_features)

    print("\n" + "=" * 40)
    print("        FEATURE EXTRACTION SUMMARY")
    print("=" * 40)
    print(f"Isolated Features Vector Layout: {feature_cols}")
    print(f"Matrix Dimension Saved (X)      : {X_features.shape}")
    print(f"Memory Footprint                : {X_features.nbytes / 1024:.2f} KB")
    print(f"Data Type                       : {X_features.dtype}")
    print(f"Output saved at location        : {output_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Alignment with pipeline data management structure
    input_csv = "data/processed/crossmatch_catalog.csv"
    output_npy = "data/processed/X_features.npy"

    extract_gaia_features(input_csv, output_npy)