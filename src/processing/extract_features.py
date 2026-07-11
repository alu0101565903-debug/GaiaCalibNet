"""
Module: extract_features.py
Report Section: Chapter 5 Data Processing & Feature Engineering

This script isolates the 5 key scalar features from the Gaia DR3 archive 
(parallax, pmra, pmdec, phot_g_mean_mag, and bp_rp) for the alternative 
feature-based training branch, handles missing value imputation, and exports 
the matrix as a NumPy binary file.
"""

import os
import numpy as np
import pandas as pd

def extract_gaia_features(csv_path: str, output_path: str):
    """
    Extracts, cleans, and imputes the 5 scalar astrometric and photometric 
    features from the processed crossmatch catalog.

    Parameters:
    - csv_path (str): Path to the processed crossmatch catalog CSV.
    - output_path (str): Path where the final feature array binary matrix will be saved.
    """
    print("=" * 60)
    print("[5.1] EXTRACTING GAIA SCALAR ASTROMETRIC & PHOTOMETRIC FEATURES")
    print("=" * 60)
    
    if not os.path.exists(csv_path):
        print(f"[!] ERROR: Not able to find {csv_path}")
        return
        
    # Read CSV treating common non-numeric markers as NaN
    df = pd.read_csv(csv_path, na_values=["--", "-", "NA", "N/A", ""])

    # Standard feature selection matching report specifications
    feature_cols = ["parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"]

    # Ensure all required metadata columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"[!] ERROR: Missing expected columns in {csv_path}: {missing_cols}")
        return

    print("-> Cohere columns to numeric vectors and executing missing data scanning...")

    # Coerce columns to numeric (turn any remaining non-numeric strings into NaN)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Data imputation: handle missing values by filling them with the dataset median
    medians = df[feature_cols].median()
    X_features = df[feature_cols].fillna(medians).values

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.save(output_path, X_features)
    
    print("\n" + "=" * 40)
    print("     FEATURE EXTRACTION SUMMARY")
    print("=" * 40)
    print(f"Isolated Features Vector Layout: {feature_cols}")
    print(f"Matrix Dimension Saved (X)      : {X_features.shape}")
    print(f"Output saved at location        : {output_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Path routing aligned with Section 3.2 data management architecture
    input_csv = "data/processed/crossmatch_catalog.csv"
    output_npy = "data/processed/X_features.npy"
    
    extract_gaia_features(input_csv, output_npy)