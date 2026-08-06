"""
Module: stacked_by_color.py

Generates vertically stacked subplot figures grouped by color classification, 
supporting both Ratio and Standardized Residual (Chi) metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.visualize_spectra.catalogs_management import (
    _load_catalog,
    get_clean_filename,
    get_wavelength_grid,
    resolve_catalog_columns,
)
from src.utils.data.catalogs_by_color import group_catalog_by_color

def generate_stacked_plots_by_color(
    mode: str = "ratio",  # Options: "ratio" or "chi"
    output_dir: str | Path = "data/processed/stacked_plots",
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    x_path: str | Path = "data/processed/X.npy",
    y_path: str | Path = "data/processed/Y.npy",
    x_error_path: str | Path = "data/processed/X_error.npy",
    color_column: str = "bp_rp"
):
    if mode not in ["ratio", "chi"]:
        raise ValueError("The 'mode' parameter must be either 'ratio' or 'chi'.")

    base_out_path = Path(output_dir) / mode
    base_out_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"[*] Generating stacked panels in mode: '{mode.upper()}'...")

    # 1. Load catalog, metrics, and helper attributes
    df = _load_catalog(catalog_path)
    X = np.load(x_path)
    Y = np.load(y_path)
    X_error = np.load(x_error_path) if mode == "chi" else None

    wavelengths = get_wavelength_grid(X.shape[1])
    id_col, name_col = resolve_catalog_columns(df)
    df, labels = group_catalog_by_color(df, name_column=name_col)

    # 2. Iterate through color groups
    for group_name in labels:
        group_df = df[df["color_category"] == group_name]
        num_stars = len(group_df)

        if num_stars == 0:
            continue

        safe_folder_name = group_name.split(" ")[0]
        print(f"    [+] Generating stacked panel for '{group_name}' ({num_stars} stars)...")

        indices = group_df.index.values
        calspec_names = group_df[name_col].astype(str).tolist()
        gaia_ids = group_df[id_col].astype(str).str.strip().tolist()

        fig_height = max(4.0, 1.3 * num_stars)
        fig, axes = plt.subplots(
            nrows=num_stars, 
            ncols=1, 
            figsize=(10, fig_height), 
            sharex=True,
            gridspec_kw={"hspace": 0.08}
        )

        if num_stars == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            x_gaia, y_calspec = X[idx], Y[idx]
            star_name, gaia_id = calspec_names[i], gaia_ids[i]

            if mode == "ratio":
                safe_calspec = np.clip(y_calspec, a_min=1e-15, a_max=None)
                metric = x_gaia / safe_calspec
                line_color, ylabel = "#d62728", "Ratio"
                
                ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.fill_between(wavelengths, 0.95, 1.05, color="gray", alpha=0.25)
                r_min, r_max = np.nanmin(metric), np.nanmax(metric)
                margin = (r_max - r_min) * 0.05
                ax.set_ylim(max(-0.2, min(0.7, r_min - margin)), min(3.5, max(1.3, r_max + margin)))

            elif mode == "chi":
                safe_error = np.clip(X_error[idx], a_min=1e-20, a_max=None)
                metric = (x_gaia - y_calspec) / safe_error
                line_color, ylabel = "#1f77b4", r"Dev [$\sigma$]"
                
                ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
                ax.fill_between(wavelengths, -1.0, 1.0, color="green", alpha=0.15)
                ax.axhline(3.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.axhline(-3.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
                d_min, d_max = np.nanpercentile(metric, [1, 99])
                ax.set_ylim(min(-5.0, d_min - 1), max(5.0, d_max + 1))

            ax.plot(wavelengths, metric, color=line_color, linewidth=1.2)
            ax.text(
                0.98, 0.85, f"{star_name} | Gaia: {gaia_id}", 
                transform=ax.transAxes, fontsize=8, fontweight="bold", 
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none")
            )
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.set_ylabel(ylabel, fontsize=8)

        axes[-1].set_xlabel("Wavelength λ [nm]", fontsize=10, fontweight="bold")
        main_title = "Gaia / CALSPEC Flux Ratios" if mode == "ratio" else r"Standardized Residuals ($\chi$)"
        fig.suptitle(f"{main_title} — Group: {group_name}", fontsize=12, fontweight="bold", y=1.001)

        save_path = base_out_path / f"stacked_{mode}_{safe_folder_name}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.close(fig)

    print("=" * 60)
    print(f"[*] Task completed! Files saved to: {base_out_path}/")

if __name__ == "__main__":
    generate_stacked_plots_by_color("ratio")
    generate_stacked_plots_by_color("chi")