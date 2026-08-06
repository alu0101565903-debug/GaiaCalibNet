"""
Module: plots_individual_by_color.py

Description: Unified visualization pipeline for individual spectral diagnostic metrics 
(Gaia/CALSPEC Flux Ratios or Standardized Chi Residuals) grouped into subdirectories 
by star color index (BP-RP). Replaces redundant modules color_plotting.py and deviation_plotting.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.visualize_spectra.catalogs_management import (
    _load_catalog,
    get_clean_filename,
    get_wavelength_grid,
    resolve_catalog_columns,
)
from src.utils.data.catalogs_by_color import group_catalog_by_color


def generate_individual_plots_by_color(
    mode: str = "ratio",  # Options: "ratio" or "chi"
    output_dir: str | Path = "data/processed/by_color_individual_plots",
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    x_path: str | Path = "data/processed/X.npy",
    y_path: str | Path = "data/processed/Y.npy",
    x_error_path: str | Path = "data/processed/X_error.npy",
    color_column: str = "bp_rp"
):
    """
    Generates individual diagnostic plots for each star grouped by color index.

    Parameters:
    - mode (str): 
        - "ratio": Plots Gaia / CALSPEC flux ratio.
        - "chi"  : Plots (Gaia - CALSPEC) / Gaia_Error standardized residual.
    - output_dir (str | Path): Base output directory.
    - catalog_path (str | Path): Crossmatch CSV catalog path.
    - x_path (str | Path): Gaia observed flux matrix path (X.npy).
    - y_path (str | Path): CALSPEC reference flux matrix path (Y.npy).
    - x_error_path (str | Path): Gaia flux error matrix path (X_error.npy, required if mode="chi").
    - color_column (str): Catalog column representing the color index (default: "bp_rp").
    """
    if mode not in ["ratio", "chi"]:
        raise ValueError("The 'mode' parameter must be either 'ratio' or 'chi'.")

    base_out_path = Path(output_dir) / mode
    
    print("=" * 60)
    print(f"[*] Generating individual plots in mode: '{mode.upper()}'...")

    # 1. Load catalog and data arrays into memory
    df = _load_catalog(catalog_path)
    X = np.load(x_path)
    Y = np.load(y_path)
    X_error = np.load(x_error_path) if mode == "chi" else None

    # Resolve shared dimensions and column mappings using centralized helpers
    wavelengths = get_wavelength_grid(X.shape[1])
    id_col, name_col = resolve_catalog_columns(df)
    df, labels = group_catalog_by_color(df, name_column=name_col)

    # 2. Iterate through each color classification group
    for group_name in labels:
        group_df = df[df["color_category"] == group_name]
        num_stars = len(group_df)

        if num_stars == 0:
            print(f"    [-] Skipping '{group_name}': No stars found in this range.")
            continue

        print(f"    [+] Processing '{group_name}' ({num_stars} stars)...")

        safe_folder_name = group_name.split(" ")[0]
        group_dir = base_out_path / safe_folder_name
        group_dir.mkdir(parents=True, exist_ok=True)

        indices = group_df.index.values
        calspec_names = group_df[name_col].astype(str).tolist()
        gaia_ids = group_df[id_col].astype(str).str.strip().tolist()

        # 3. Generate individual plots per star
        for i, idx in enumerate(indices):
            x_gaia = X[idx]
            y_calspec = Y[idx]
            star_name = calspec_names[i]
            gaia_id = gaia_ids[i]

            fig, ax = plt.subplots(figsize=(10, 4))

            # --- DYNAMIC PLOTTING LOGIC PER MODE ---
            if mode == "ratio":
                safe_calspec = np.clip(y_calspec, a_min=1e-15, a_max=None)
                metric = x_gaia / safe_calspec

                ax.set_title(f"Gaia / CALSPEC Flux Ratio | CALSPEC: {star_name} | Gaia ID: {gaia_id}", fontsize=12, fontweight="bold")
                ax.plot(wavelengths, metric, color="#d62728", linewidth=1.5, label=f"CALSPEC: {star_name}")
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
                ax.fill_between(wavelengths, 0.95, 1.05, color="gray", alpha=0.2, label="±5% Error")
                ax.set_ylabel("Ratio", fontsize=11)

                # Dynamic Y-axis limits
                r_min, r_max = np.nanmin(metric), np.nanmax(metric)
                margin = (r_max - r_min) * 0.05
                ax.set_ylim(max(-0.5, min(0.7, r_min - margin)), min(4.0, max(1.3, r_max + margin)))

            elif mode == "chi":
                x_err = X_error[idx]
                safe_error = np.clip(x_err, a_min=1e-20, a_max=None)
                metric = (x_gaia - y_calspec) / safe_error

                ax.set_title(f"Standardized Residuals (Chi) | CALSPEC: {star_name} | Gaia ID: {gaia_id}", fontsize=12, fontweight="bold")
                ax.plot(wavelengths, metric, color="#1f77b4", linewidth=1.5, label=r"(Gaia - CALSPEC) / $\sigma$")
                ax.axhline(0.0, color="black", linestyle="-", linewidth=1.5)
                ax.axhline(3.0, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label=r"+3$\sigma$ limit")
                ax.axhline(-3.0, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label=r"-3$\sigma$ limit")
                ax.fill_between(wavelengths, -1, 1, color="green", alpha=0.1, label=r"±1$\sigma$ (Expected)")
                ax.set_ylabel(r"Deviation [$\sigma$]", fontsize=11)

                # Dynamic Y-axis limits based on percentiles
                d_min, d_max = np.nanpercentile(metric, [1, 99])
                ax.set_ylim(min(-5, d_min - 1), max(5, d_max + 1))

            ax.set_xlabel("Wavelength λ [nm]", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", frameon=True, fontsize=9)

            clean_star_name = get_clean_filename(star_name)
            filename = f"{mode}_{clean_star_name}_{gaia_id}.png"
            plt.savefig(group_dir / filename, dpi=200, bbox_inches="tight")
            plt.close(fig)

    print("=" * 60)
    print(f"[*] Task completed! All individual plots saved inside {base_out_path}/")


if __name__ == "__main__":
    # Generate individual plots for ratios
    generate_individual_plots_by_color(mode="ratio")

    # Generate individual plots for standardized chi residuals
    generate_individual_plots_by_color(mode="chi")