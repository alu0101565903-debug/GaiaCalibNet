"""
Module: plot_spectra_manager.py

Description:
    Unified manager module to generate spectral plots across the catalog in
    different modes: local crossmatch, individual Gaia, ESA server, or raw CALSPEC FITS.
"""

import time
from pathlib import Path
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
    ----------
    output_dir : str or Path
        Directory where generated plots will be saved.
    mode : str
        "crossmatch"  : Combined Gaia vs CALSPEC plots (local .npy data).
        "gaia"        : Individual Gaia plots (local .npy data).
        "server"      : Individual raw Gaia plots downloaded directly from ESA.
        "raw_calspec" : Individual raw CALSPEC plots generated from local FITS files.
    fits_dir : str or Path, optional
        Directory where raw CALSPEC FITS files are stored (used in "raw_calspec" mode).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fits_path_dir = Path(fits_dir)
    
    print("=" * 60)
    print(f"[1/3] Extracting catalog IDs for processing mode: '{mode}'...")
    
    gaia_id_list = get_gaia_ids()
    total_stars = len(gaia_id_list)
    stats = {"rendered": 0, "failed": 0}
    
    print(f"   [+] Found {total_stars} targets in catalog.")
    print("\n[2/3] Generating spectral plots...")

    for i, gaia_id in enumerate(gaia_id_list):
        try:
            _, _, calspec_name = resolve_star_info(gaia_id)
        except ValueError:
            calspec_name = "Unknown"

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
            ax.legend(loc="upper right", frameon=True, fontsize=10)
            
            save_path = out_path / f"spectrum_{clean_calspec}_Gaia_{gaia_id}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            stats["rendered"] += 1

        # -------------------------------------------------------------
        # MODE 2: Individual Gaia (Local processed .npy data)
        # -------------------------------------------------------------
        elif mode == "processed_gaia":
            plot_gaia_spectrum(identifier=gaia_id, output_dir=out_path)
            stats["rendered"] += 1

        # -------------------------------------------------------------
        # MODE 3: Gaia Server (Raw ESA Gaia DR3 data)
        # -------------------------------------------------------------
        elif mode == "raw_gaia":
            success = plot_raw_gaia_server_spectrum(gaia_id=gaia_id, output_dir=out_path)
            if success:
                stats["rendered"] += 1
                time.sleep(1.0)  # Polite pause for ESA server
            else:
                stats["failed"] += 1

        # -------------------------------------------------------------
        # MODE 4: Raw CALSPEC (Local FITS files)
        # -------------------------------------------------------------
        elif mode == "raw_calspec":
            clean_name = str(calspec_name).strip()
            fits_file = fits_path_dir / f"{clean_name.lower()}_mod_001.fits" 
            
            if not fits_file.exists():
                matching_files = list(fits_path_dir.glob(f"*{clean_name}*.fits"))
                fits_file = matching_files[0] if matching_files else fits_file

            if fits_file.exists() and plot_raw_fits_spectrum(clean_name, fits_file, out_path):
                stats["rendered"] += 1
            else:
                stats["failed"] += 1

        # Progress tracking
        if (i + 1) % 10 == 0 or (i + 1) == total_stars:
            print(f"   [*] Processed {i + 1}/{total_stars} targets (Latest: {clean_calspec})")

    print(f"\n{'='*40}\n      SUMMARY OF SPECTRAL PLOTS GENERATION\n{'='*40}")
    print(f"Generated plots : {stats['rendered']}")
    print(f"Omitted / Failed: {stats['failed']}")
    print(f"\n[3/3] Task completed! Files saved to: {out_path}/")


if __name__ == "__main__":
    # Example execution in raw CALSPEC mode
    process_catalog_spectra(
        output_dir="data/raw/calspec_spectrum", 
        mode="raw_calspec"
    )