"""
Module: color_plotting.py

Description: Generates visualization plots for Gaia vs. CALSPEC flux ratios. 
The module groups stars by their color index (BP-RP) and outputs individual 
diagnostic plots to isolated subdirectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.visualize_spectra.catalogs_management import _load_catalog

def generate_individual_ratios_by_color(
    output_dir: str | Path,
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    x_path: str | Path = "data/processed/X.npy",
    y_path: str | Path = "data/processed/Y.npy",
    color_column: str = "bp_rp"
):
    """
    Generates individual plots of the Gaia/CALSPEC flux ratio.
    
    The ratio is defined as: Ratio = Gaia_Flux / CALSPEC_Flux.
    Plots are saved in separate subdirectories based on the star's color group.
    The title and filename automatically include the CALSPEC name and Gaia ID.
    Dynamic Y-axis limits are applied to prevent clipping high-frequency infrared noise.

    Args:
    - output_dir (str | Path): Base directory where the plot folders will be saved.
    - catalog_path (str | Path): Path to the crossmatched catalog CSV file.
    - x_path (str | Path): Path to the Gaia input features matrix (X.npy).
    - y_path (str | Path): Path to the CALSPEC ground truth matrix (Y.npy).
    - color_column (str): Catalog column name corresponding to the color index (e.g., "bp_rp").
        
    Raises:
    - ValueError: If the specified `color_column` does not exist in the loaded catalog.
    """
    base_out_path = Path(output_dir)
    
    print("=" * 60)
    print(f"[*] Extracting catalog and matrices for Individual Ratios...")
    
    # 1. Efficient loading: Load heavy matrices into memory only once
    df = _load_catalog(catalog_path)

    # Validate the existence of the color index column
    if color_column not in df.columns:
        raise ValueError(f"Column '{color_column}' not found in catalog. Available columns: {df.columns.tolist()}")

    # Load the pre-processed spectral data matrices
    X = np.load(x_path)  # Gaia observed flux
    Y = np.load(y_path)  # CALSPEC reference flux

    # Define the unified wavelength grid for the x-axis
    wavelengths = np.linspace(330, 1050, X.shape[1])
    
    # 2. Define the color groups (Adjust the bins according to your astrophysical needs)
    bins = [-np.inf, 0.5, 1.5, np.inf]
    labels = ["Blue (BP-RP < 0.5)", "Solar (0.5 <= BP-RP <= 1.5)", "Red (BP-RP > 1.5)"]
    
    # Create a new temporary column with the color category
    df["Color_Group"] = pd.cut(df[color_column], bins=bins, labels=labels)
    
    print(f"[*] Grouping stars by {color_column}...")
    
    # 3. Iterate over each color classification group
    for group_name in labels:
        group_df = df[df["Color_Group"] == group_name]
        num_stars = len(group_df)

        # Skip processing if no stars fall into the current bin
        if num_stars == 0:
            print(f"    [-] Skipping '{group_name}': No stars found in this range.")
            continue
            
        print(f"    [+] Processing '{group_name}' ({num_stars} stars)...")
        
        # Create an isolated subfolder for the group (using the first word, e.g., "Blue")
        safe_folder_name = group_name.split(" ")[0]
        group_dir = base_out_path / safe_folder_name
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract dataframe indices to perfectly map metadata with the unindexed X and Y matrices
        indices = group_df.index.values
        
        # Dynamically determine the correct column names for celestial identifiers
        name_col = "calspec_name" if "calspec_name" in group_df.columns else "Name"
        id_col = "source_id" if "source_id" in group_df.columns else "gaia_source_id"
        
        calspec_names = group_df[name_col].astype(str).tolist()
        gaia_ids = group_df[id_col].astype(str).str.strip().tolist()
        
        # Generate and render an individual visualization for every star in the group
        for i, idx in enumerate(indices):
            # Extract target matrices arrays via row index
            x_gaia = X[idx]
            y_calspec = Y[idx]

            star_name = calspec_names[i]
            gaia_id = gaia_ids[i]
            
            # Safe ratio calculation to avoid division by zero in the spectrum edges
            safe_calspec = np.clip(y_calspec, a_min=1e-15, a_max=None)
            ratio = x_gaia / safe_calspec
            
            # Initialize a figure with standard aspect proportions
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Updated title including both CALSPEC name and Gaia ID
            ax.set_title(f"Gaia / CALSPEC Flux Ratio | CALSPEC: {star_name} | Gaia ID: {gaia_id}", fontsize=12, fontweight="bold")

            # Plot the flux ratio across the wavelength grid
            ax.plot(wavelengths, ratio, color="#d62728", linewidth=1.5, label=f"CALSPEC: {star_name}")

            # Add visual guidelines for a perfect match (1.0) and acceptable variance (±5%)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.fill_between(wavelengths, 0.95, 1.05, color="gray", alpha=0.2, label="±5% Error")

            ax.set_xlabel("Wavelength λ [nm]", fontsize=11)
            ax.set_ylabel("Ratio", fontsize=11)
            
            # --- DYNAMIC Y-AXIS LIMITS LOGIC ---
            # 1. Get the absolute minimum and maximum values of the entire spectrum
            r_min = np.nanmin(ratio)
            r_max = np.nanmax(ratio)
            
            # 2. Calculate a small margin (5% of the range) so the peaks 
            # don't touch the very edge of the bounding box
            margin = (r_max - r_min) * 0.05
            
            # 3. Keep the rule: if the spectrum is clean, keep the plot 
            # from 0.7 to 1.3. If it has massive peaks, expand to fit them.
            lower_bound = min(0.7, r_min - margin)
            upper_bound = max(1.3, r_max + margin)
            
            # 4. (Optional but recommended) Safety net in case an isolated 
            # pixel causes a division by zero error making the ratio skyrocket.
            lower_bound = max(-0.5, lower_bound)
            upper_bound = min(4.0, upper_bound)
            
            ax.set_ylim(lower_bound, upper_bound)
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", frameon=True, fontsize=9)
            
            # Save image with a clean filename (now includes the Gaia ID to avoid overwrites)
            clean_star_name = star_name.replace("/", "_").replace(" ", "_")
            filename = f"ratio_{clean_star_name}_{gaia_id}.png"
            save_path = group_dir / filename

            # Flush and close the plot to optimize memory usage across large datasets
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            
    print("=" * 60)
    print(f"[*] Task completed! All individual plots saved inside {base_out_path}/")

if __name__ == "__main__":
    # Example execution
    generate_individual_ratios_by_color(
        output_dir="results/plots/ratios_by_color",
        catalog_path="data/processed/crossmatch_catalog.csv",
        color_column="bp_rp"
    )