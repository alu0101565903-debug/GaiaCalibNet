"""
Module: plot_spectra_manager.py

Unified execution manager for spectral plotting modes. Reuses shared catalog 
helpers and plot engines to avoid code duplication.
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
    get_clean_filename
)
from src.utils.visualize_spectra.plot_raw_gaia import plot_raw_gaia_server_spectrum
from src.utils.visualize_spectra.plot_raw_calspec import plot_raw_fits_spectrum


def process_catalog_spectra(
    output_dir: str | Path, 
    mode: str = "processed_crossmatch",
    catalog_path: str = "data/processed/crossmatch_catalog_v2.csv",
    fits_dir: str | Path = "data/raw/calspec_fits",
    x_path: str = "data/processed/nn/all/X.npy",
    y_path: str = "data/processed/nn/all/Y.npy"
) -> None:
    """
    Generate spectral plots according to the selected plotting mode and catalog configuration.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory to save generated plot images.
    mode : str, optional
        Plotting mode ('processed_crossmatch', 'processed_crossmatch_ratio', 
        'processed_gaia', 'raw_gaia', 'raw_calspec').
    catalog_path : str, optional
        Path to the CSV catalog file.
    fits_dir : str or pathlib.Path, optional
        Directory containing raw CALSPEC FITS files.
    x_path : str, optional
        Path to input Gaia spectra array (.npy).
    y_path : str, optional
        Path to target CALSPEC spectra array (.npy).

    Returns
    -------
    None
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fits_path_dir = Path(fits_dir)
    
    print("=" * 60)
    print(f"[1/3] Extracting catalog IDs for processing mode: '{mode}'...")

    gaia_id_list = get_gaia_ids(catalog_path)
    total_stars = len(gaia_id_list)
    stats = {"rendered": 0, "failed": 0}
    
    print(f"   [+] Found {total_stars} targets in catalog.")
    print("\n[2/3] Generating spectral plots...")

    for i, gaia_id in enumerate(gaia_id_list):
        try:
            _, _, calspec_name = resolve_star_info(gaia_id)
        except ValueError:
            calspec_name = "Unknown"

        clean_calspec = get_clean_filename(calspec_name)

        try:
            # MODE 1: Processed Crossmatch
            if mode == "processed_crossmatch":
                fig, ax = plt.subplots(figsize=(10, 5))
                # Pass 'catalog_path' so the index matches the .npy subset
                plot_gaia_spectrum(identifier=gaia_id, x_path=x_path, catalog_path=catalog_path, ax=ax)
                plot_calspec_spectrum(identifier=gaia_id, y_path=y_path, catalog_path=catalog_path, ax=ax)
                
                ax.set_title(f"Spectral Comparison | CALSPEC: {clean_calspec} | Gaia: {gaia_id}", fontsize=12, pad=12, fontweight="bold")
                ax.set_xlabel("Wavelength λ [nm]", fontsize=11)
                ax.set_ylabel("Absolute Flux [erg / s / cm² / Å]", fontsize=11)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.legend(loc="best", frameon=True, fontsize=10)
                
                plt.savefig(out_path / f"spectrum_{clean_calspec}_Gaia_{gaia_id}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
                stats["rendered"] += 1

            # MODE 2: Processed Crossmatch with Ratio
            elif mode == "processed_crossmatch_ratio":
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
                
                plot_gaia_spectrum(identifier=gaia_id, x_path=x_path, catalog_path=catalog_path, ax=ax1)
                plot_calspec_spectrum(identifier=gaia_id, y_path=y_path, catalog_path=catalog_path, ax=ax1)
                
                ax1.set_title(f"Spectral Analysis | CALSPEC: {clean_calspec} | Gaia: {gaia_id}", fontsize=12, fontweight="bold")
                ax1.set_ylabel("Absolute Flux", fontsize=11)
                ax1.grid(True, alpha=0.3, linestyle="--")
                ax1.legend(loc="best", frameon=True, fontsize=10)
                
                lines = ax1.get_lines()
                if len(lines) >= 2:
                    gaia_line = next((l for l in lines if "Gaia" in str(l.get_label())), lines[0])
                    calspec_line = next((l for l in lines if "CALSPEC" in str(l.get_label())), lines[1])
                    
                    wavelengths = gaia_line.get_xdata()
                    safe_calspec = np.clip(calspec_line.get_ydata(), a_min=1e-15, a_max=None)
                    ratio = gaia_line.get_ydata() / safe_calspec
                    
                    ax2.plot(wavelengths, ratio, color="#d62728", linewidth=1.5, label="Gaia / CALSPEC")
                    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
                    ax2.fill_between(wavelengths, 0.95, 1.05, color="gray", alpha=0.2, label="±5% Error")
                    
                    ax2.set_xlabel("Wavelength λ [nm]", fontsize=11)
                    ax2.set_ylabel("Ratio", fontsize=11)
                    ax2.grid(True, alpha=0.3, linestyle="--")
                    ax2.legend(loc="best", frameon=True, fontsize=9)
                    
                    r_min, r_max = np.nanmin(ratio), np.nanmax(ratio)
                    margin = (r_max - r_min) * 0.05
                    ax2.set_ylim(max(-0.5, min(0.7, r_min - margin)), min(4.0, max(1.3, r_max + margin)))
                
                plt.tight_layout()
                plt.savefig(out_path / f"combined_{clean_calspec}_{gaia_id}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
                stats["rendered"] += 1

            # MODE 3: Individual Processed Gaia
            elif mode == "processed_gaia":
                plot_gaia_spectrum(identifier=gaia_id, x_path=x_path, catalog_path=catalog_path, output_dir=out_path)
                stats["rendered"] += 1

            # MODE 4: ESA Gaia Archive Query
            elif mode == "raw_gaia":
                if plot_raw_gaia_server_spectrum(gaia_id=gaia_id, output_dir=out_path):
                    stats["rendered"] += 1
                    time.sleep(1.0)
                else:
                    stats["failed"] += 1

            # MODE 5: Raw CALSPEC FITS
            elif mode == "raw_calspec":
                clean_name = str(calspec_name).strip()
                matching_files = list(fits_path_dir.glob(f"*{clean_name}*.fits"))
                
                if matching_files and plot_raw_fits_spectrum(clean_name, matching_files[0], out_path):
                    stats["rendered"] += 1
                else:
                    stats["failed"] += 1

        except Exception as e:
            print(f"   [!] Error processing Gaia ID {gaia_id}: {e}")
            stats["failed"] += 1
            plt.close('all')

        if (i + 1) % 10 == 0 or (i + 1) == total_stars:
            print(f"   [*] Processed {i + 1}/{total_stars} targets (Latest: {clean_calspec})")

    print(f"\n{'='*40}\n       SUMMARY OF SPECTRAL PLOTS GENERATION\n{'='*40}")
    print(f"Generated plots : {stats['rendered']}")
    print(f"Omitted / Failed: {stats['failed']}")
    print(f"\n[3/3] Task completed! Files saved to: {out_path}/")


if __name__ == "__main__":
    process_catalog_spectra(
        output_dir="data/processed/color_gaia_vs_calspec_spectra/solar_spectra_plots", 
        mode="processed_crossmatch",
        catalog_path="data/processed/catalogs_by_color/solar_catalog.csv",
        x_path="data/processed/nn/solar/X.npy",
        y_path="data/processed/nn/solar/Y.npy"
    )