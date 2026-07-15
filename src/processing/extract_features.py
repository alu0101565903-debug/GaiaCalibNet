"""
Module: extract_features.py
Report Section: Chapter 5 Data Processing & Feature Engineering

This script isolates the 5 key scalar features from the Gaia DR3 crossmatch
catalog (parallax, pmra, pmdec, phot_g_mean_mag, and bp_rp). It handles
missing-value imputation, performs a simple data quality audit, and exports the
final feature matrix as a float32 NumPy binary.
"""

import os
import numpy as np
import pandas as pd

def extract_gaia_features(csv_path: str, output_path: str):
    """Extract and preprocess Gaia scalar features for feature-based training.

    This function reads the processed crossmatch catalog, validates the
    required Gaia feature columns, coerces values to numeric form, imputes any
    missing entries with the median value per feature, and writes the result as
    a float32 NumPy array.
    """
    print("=" * 60)
    print("[5.1] EXTRACTING GAIA SCALAR ASTROMETRIC & PHOTOMETRIC FEATURES")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"[!] ERROR: Not able to find {csv_path}")
        return

    # Read the crossmatch catalog, converting common missing-value markers to NaN.
    df = pd.read_csv(csv_path, na_values=["--", "-", "NA", "N/A", ""])

    # Select the scalar features used by the feature-based model branch.
    feature_cols = ["parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"]

    # Ensure all required metadata columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[!] ERROR: Missing expected columns in {csv_path}: {missing_cols}")
        return

    print("-> Cohering columns to numeric vectors and scanning for missing data...")

    # Convert all feature columns to numeric values, setting invalid entries to NaN.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Data quality audit: report how many NaNs remain after coercion.
    nan_counts = df[feature_cols].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print("\n[i] Missing Values Detected (will be imputed with feature median):")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"    - {col}: {count} missing values")
    else:
        print("[✓] Matrix is completely dense. No NaNs detected.")

    # Impute missing values using the median of each feature column.
    medians = df[feature_cols].median()

    # Enforce float32 precision for efficient PyTorch compatibility.
    X_features = df[feature_cols].fillna(medians).values.astype(np.float32)

    # Create the target directory if it does not exist.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, X_features)
    
    print("\n" + "=" * 40)
    print("       FEATURE EXTRACTION SUMMARY")
    print("=" * 40)
    print(f"Isolated Features Vector Layout: {feature_cols}")
    print(f"Matrix Dimension Saved (X)      : {X_features.shape}")
    print(f"Memory Footprint                : {X_features.nbytes / 1024:.2f} KB")
    print(f"Data Type                       : {X_features.dtype}")
    print(f"Output saved at location        : {output_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Path routing aligned with Section 3.2 data management architecture
    input_csv = "data/processed/crossmatch_catalog.csv"
    output_npy = "data/processed/X_features.npy"
    
    extract_gaia_features(input_csv, output_npy)