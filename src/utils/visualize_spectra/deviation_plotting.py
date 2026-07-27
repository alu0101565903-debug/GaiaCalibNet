"""
Module: deviation_plotting.py

Module for visualizing spectral deviations between Gaia and CALSPEC data.
Generates standardized residual (Chi) plots and organizes them into subdirectories
based on the color group (BP-RP index) of the stars.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.visualize_spectra.catalogs_management import _load_catalog

def generate_individual_deviations_by_color(
    output_dir: str | Path,
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    x_path: str | Path = "data/processed/X.npy",
    y_path: str | Path = "data/processed/Y.npy",
    x_error_path: str | Path = "data/processed/X_error.npy",
    color_column: str = "bp_rp"
):
    """
    Generates individual plots of the Standardized Deviation (Chi) for each star.
    
    The standardized deviation is calculated as:
    Deviation = (Gaia_Flux - CALSPEC_Flux) / Gaia_Error
    
    The resulting plots are saved in separate subdirectories based on the star's 
    color group (Blue, Solar, or Red) to facilitate comparative analysis.

    Args:
    - output_dir (str | Path): Base directory where the plot folders will be saved.
    - catalog_path (str | Path): Path to the crossmatched catalog CSV file.
    - x_path (str | Path): Path to the Gaia input features matrix (X.npy).
    - y_path (str | Path): Path to the CALSPEC ground truth matrix (Y.npy).
    - x_error_path (str | Path): Path to the Gaia flux errors matrix (X_error.npy).
    - color_column (str): Name of the column in the catalog containing the color index.
    """
    base_out_path = Path(output_dir)
    
    print("=" * 60)
    print(f"[*] Extracting catalog and matrices for Deviation Plots...")

    # 1. Load the catalog and numpy array matrices into memory
    df = _load_catalog(catalog_path)

    # Validate the existence of the color index column
    if color_column not in df.columns:
        raise ValueError(f"Column '{color_column}' not found in catalog. Available columns: {df.columns.tolist()}")

    # Load the pre-processed spectral data matrices
    X = np.load(x_path)               # Gaia observed flux
    Y = np.load(y_path)               # CALSPEC reference flux
    X_error = np.load(x_error_path)   # Gaia observational error

    # Define the unified wavelength grid (e.g., from 330 nm to 1050 nm)
    wavelengths = np.linspace(330, 1050, X.shape[1])

    # 2. Categorize stars by BP-RP color index
    bins = [-np.inf, 0.5, 1.5, np.inf]
    labels = ["Blue (BP-RP < 0.5)", "Solar (0.5 <= BP-RP <= 1.5)", "Red (BP-RP > 1.5)"]

    # Assign each star to a color group
    df["Color_Group"] = pd.cut(df[color_column], bins=bins, labels=labels)
    
    print(f"[*] Grouping stars by {color_column}...")

    # 3. Iterate over each color classification group
    for group_name in labels:
        group_df = df[df["Color_Group"] == group_name]
        num_stars = len(group_df)

        # Skip the group if no stars fall into this color category
        if num_stars == 0:
            continue
            
        print(f"    [+] Processing '{group_name}' ({num_stars} stars)...")

        # Create a safe directory name (e.g., "Blue", "Solar", "Red") and ensure it exists
        safe_folder_name = group_name.split(" ")[0]
        group_dir = base_out_path / safe_folder_name
        group_dir.mkdir(parents=True, exist_ok=True)

        # Extract row indices to correctly map catalog entries to the NumPy matrices
        indices = group_df.index.values

        # Dynamically identify the correct column names for star identifiers
        name_col = "calspec_name" if "calspec_name" in group_df.columns else "Name"
        id_col = "source_id" if "source_id" in group_df.columns else "gaia_source_id"
        
        calspec_names = group_df[name_col].astype(str).tolist()
        gaia_ids = group_df[id_col].astype(str).str.strip().tolist()

        # Process and plot each star within the current color group
        for i, idx in enumerate(indices):
            # Extract individual star data using its matrix index
            x_gaia = X[idx]
            y_calspec = Y[idx]
            
            x_err = X_error[idx] 
            
            star_name = calspec_names[i]
            gaia_id = gaia_ids[i]

            # Prevent division-by-zero errors by clipping the minimum error to a tiny positive float
            safe_error = np.clip(x_err, a_min=1e-20, a_max=None)

            # Calculate the standardized residual (Chi)
            deviation = (x_gaia - y_calspec) / safe_error

            # Initialize the plot figure
            fig, ax = plt.subplots(figsize=(10, 4))
            
            ax.set_title(f"Standardized Residuals (Chi) | CALSPEC: {star_name} | Gaia ID: {gaia_id}", fontsize=12, fontweight="bold")

            # Plot the calculated deviation across the wavelength grid
            ax.plot(wavelengths, deviation, color="#1f77b4", linewidth=1.5, label=r"(Gaia - CALSPEC) / $\sigma$")

            # Add horizontal reference lines (0 = perfect match, +/- 3 sigma = statistical thresholds)
            ax.axhline(0.0, color="black", linestyle="-", linewidth=1.5)
            ax.axhline(3.0, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label=r"+3$\sigma$ limit")
            ax.axhline(-3.0, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label=r"-3$\sigma$ limit")

            # Shade the optimal +/- 1 sigma expected region (green band)
            ax.fill_between(wavelengths, -1, 1, color="green", alpha=0.1, label=r"±1$\sigma$ (Expected)")
            
            ax.set_xlabel("Wavelength λ [nm]", fontsize=11)
            ax.set_ylabel(r"Deviation [$\sigma$]", fontsize=11)

            # Dynamically set y-limits based on percentiles to prevent extreme outliers from distorting the view
            d_min, d_max = np.nanpercentile(deviation, [1, 99])
            ax.set_ylim(min(-5, d_min - 1), max(5, d_max + 1))
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", frameon=True, fontsize=9)

            # Sanitize the star name for safe file writing (replaces spaces and slashes)
            clean_star_name = star_name.replace("/", "_").replace(" ", "_")
            filename = f"deviation_{clean_star_name}_{gaia_id}.png"
            save_path = group_dir / filename

            # Save the figure to disk and close it to free up system memory
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            
    print("=" * 60)
    print(f"[*] Task completed! All deviation plots saved inside {base_out_path}/")

if __name__ == "__main__":
    generate_individual_deviations_by_color(
        output_dir="data/processed/deviations_by_color"
    )