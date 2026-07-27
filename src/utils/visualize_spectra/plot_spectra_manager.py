"""
Module: plot_spectra_manager.py

Description: Orchestrates spectral visualization across the crossmatched catalog.
Provides a unified manager interface supporting multiple modes: local crossmatched 
comparisons, dual-panel ratio analysis, local processed Gaia spectra, 
direct ESA archive Gaia queries, and raw CALSPEC FITS files.
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.utils.visualize_spectra.catalogs_management import (
    get_gaia_ids,
    resolve_star_info,
    plot_calspec_spectrum,
    plot_gaia_spectrum,
    extract_fits_data
)
from src.utils.visualize_spectra.plot_raw_gaia import plot_raw_gaia_server_spectrum
from src.utils.visualize_spectra.plot_raw_calspec import plot_raw_fits_spectrum

def process_catalog_spectra(
    output_dir: str | Path, 
    mode: str = "crossmatch",
    fits_dir: str | Path = "data/raw/calspec_fits"
) -> None:
    """
    Unified loop: Iterates through the catalog to plot spectra based on the selected mode.
    
    Parameters
    - output_dir (str or Path): Directory where generated plots will be saved.
    - mode (str):
        "processed_crossmatch"          : Combined Gaia vs CALSPEC plots (local .npy data).
        "processed_crossmatch_quotient" : Combined Gaia vs CALSPEC plots with ratio panel.
        "processed_gaia"                : Individual Gaia plots (local .npy data).
        "raw_gaia"                      : Individual raw Gaia plots downloaded directly from ESA.
        "raw_calspec"                   : Individual raw CALSPEC plots generated from local FITS files.
    - fits_dir (str or Path, optional): Directory where raw CALSPEC FITS files are stored (used in "raw_calspec" mode).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fits_path_dir = Path(fits_dir)
    
    print("=" * 60)
    print(f"[1/3] Extracting catalog IDs for processing mode: '{mode}'...")

    # Extract target list and initialize statistics counters
    gaia_id_list = get_gaia_ids()
    total_stars = len(gaia_id_list)
    stats = {"rendered": 0, "failed": 0}
    
    print(f"   [+] Found {total_stars} targets in catalog.")
    print("\n[2/3] Generating spectral plots...")

    for i, gaia_id in enumerate(gaia_id_list):
        # Resolve target crossmatch identifiers
        try:
            _, _, calspec_name = resolve_star_info(gaia_id)
        except ValueError:
            calspec_name = "Unknown"

        # Sanitize star name for file naming compatibility
        clean_calspec = str(calspec_name).strip().replace("/", "_").replace(" ", "_")

        # -------------------------------------------------------------
        # MODE 1: Crossmatch (Processed Gaia + Processed CALSPEC)
        # -------------------------------------------------------------
        if mode == "processed_crossmatch":
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_gaia_spectrum(identifier=gaia_id, ax=ax)
            plot_calspec_spectrum(identifier=gaia_id, ax=ax)
            
            ax.set_title(f"Spectral Comparison | CALSPEC: {clean_calspec} | Gaia: {gaia_id}", fontsize=12, pad=12, fontweight="bold")
            ax.set_xlabel("Wavelength λ [nm]", fontsize=11)
            ax.set_ylabel("Absolute Flux [erg / s / cm² / Å]", fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="best", frameon=True, fontsize=10)
            
            save_path = out_path / f"spectrum_{clean_calspec}_Gaia_{gaia_id}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            stats["rendered"] += 1

        # -------------------------------------------------------------
        # MODE 2: Crossmatch with Quotient (Processed Gaia + Processed CALSPEC + Ratio Panel)
        # -------------------------------------------------------------
        elif mode == "processed_crossmatch_quotient":
            # Create a figure with 2 subplots. The top (absolute) is twice as tall as the bottom (ratio)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            
            # --- Upper Panel: Absolute Spectral Flux ---
            plot_gaia_spectrum(identifier=gaia_id, ax=ax1)
            plot_calspec_spectrum(identifier=gaia_id, ax=ax1)
            
            ax1.set_title(f"Spectral Analysis | CALSPEC: {clean_calspec} | Gaia: {gaia_id}", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Absolute Flux", fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle="--")
            ax1.legend(loc="best", frameon=True, fontsize=10)
            
            # --- Lower Panel: Gaia / CALSPEC Ratio ---
            lines = ax1.get_lines()
            if len(lines) >= 2:
                # Retrieve plotted spectral data from artist elements
                gaia_line = next((l for l in lines if "Gaia" in str(l.get_label())), lines[0])
                calspec_line = next((l for l in lines if "CALSPEC" in str(l.get_label())), lines[1])
                
                wavelengths = gaia_line.get_xdata()
                y_gaia = gaia_line.get_ydata()
                y_calspec = calspec_line.get_ydata()
                
                # Prevent Division-by-Zero: Apply hard floor to CALSPEC reference array
                safe_calspec = np.clip(y_calspec, a_min=1e-15, a_max=None)
                ratio = y_gaia / safe_calspec
                
                ax2.plot(wavelengths, ratio, color="#d62728", linewidth=1.5, label="Gaia / CALSPEC")
                ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
                ax2.fill_between(wavelengths, 0.95, 1.05, color="gray", alpha=0.2, label="±5% Error")
                
                ax2.set_xlabel("Wavelength λ [nm]", fontsize=11)
                ax2.set_ylabel("Ratio", fontsize=11)
                ax2.grid(True, alpha=0.3, linestyle="--")
                ax2.legend(loc="best", frameon=True, fontsize=9)
                
                # Dynamic Y-axis scaling logic to absorb potential high-frequency IR noise
                r_min, r_max = np.nanmin(ratio), np.nanmax(ratio)
                margin = (r_max - r_min) * 0.05
                lower_bound = max(-0.5, min(0.7, r_min - margin))
                upper_bound = min(4.0, max(1.3, r_max + margin))
                ax2.set_ylim(lower_bound, upper_bound)
            
            plt.tight_layout()
            
            save_path = out_path / f"combined_{clean_calspec}_{gaia_id}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            stats["rendered"] += 1

        # -------------------------------------------------------------
        # MODE 3: Individual Gaia (Local processed .npy data)
        # -------------------------------------------------------------
        elif mode == "processed_gaia":
            plot_gaia_spectrum(identifier=gaia_id, output_dir=out_path)
            stats["rendered"] += 1

        # -------------------------------------------------------------
        # MODE 4: Gaia Server (Raw ESA Gaia DR3 data)
        # -------------------------------------------------------------
        elif mode == "raw_gaia":
            success = plot_raw_gaia_server_spectrum(gaia_id=gaia_id, output_dir=out_path)
            if success:
                stats["rendered"] += 1
                time.sleep(1.0)  # Polite pause for ESA server
            else:
                stats["failed"] += 1

        # -------------------------------------------------------------
        # MODE 5: Raw CALSPEC (Local FITS files)
        # -------------------------------------------------------------
        elif mode == "raw_calspec":
            clean_name = str(calspec_name).strip()
            fits_file = fits_path_dir / f"{clean_name.lower()}_mod_001.fits" 

            # Fallback file discovery using pattern matching if standard filename is missing
            if not fits_file.exists():
                matching_files = list(fits_path_dir.glob(f"*{clean_name}*.fits"))
                fits_file = matching_files[0] if matching_files else fits_file

            if fits_file.exists() and plot_raw_fits_spectrum(clean_name, fits_file, out_path):
                stats["rendered"] += 1
            else:
                stats["failed"] += 1

        # Print periodic progress updates
        if (i + 1) % 10 == 0 or (i + 1) == total_stars:
            print(f"   [*] Processed {i + 1}/{total_stars} targets (Latest: {clean_calspec})")

    print(f"\n{'='*40}\n      SUMMARY OF SPECTRAL PLOTS GENERATION\n{'='*40}")
    print(f"Generated plots : {stats['rendered']}")
    print(f"Omitted / Failed: {stats['failed']}")
    print(f"\n[3/3] Task completed! Files saved to: {out_path}/")


if __name__ == "__main__":
    # Example execution in raw CALSPEC mode
    process_catalog_spectra(
        output_dir="data/raw/gaia_spectrum", 
        mode="raw_gaia"
    )