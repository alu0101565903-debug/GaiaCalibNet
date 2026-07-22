"""
Module: plot_calspec_spectra.py
Report Section: 4.2 CALSPEC Spectral Plot Generation

This module processes CALSPEC catalog metadata, loads the corresponding FITS files,
and generates standardized spectral plots in the 3000–11000 Å wavelength range.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from src.processing.download_calspec import download_missing_fits  


def plot_calspec_spectra(csv_path: str, fits_dir_str: str, plot_dir_str: str) -> None:
    """
    Generate spectral plots in the 3000 - 11000 Å range using downloaded FITS files.

    Parameters
    csv_path (str): Path to the CALSPEC CSV catalog file containing target metadata.
    fits_dir_str (str): Directory path where CALSPEC FITS files are stored.
    plot_dir_str (str): Output directory path where generated spectrum PNG plots will be saved.
    """
    # Convert string paths to Path objects
    csv_file = Path(csv_path)
    fits_dir = Path(fits_dir_str)
    plot_dir = Path(plot_dir_str)

    print("=" * 60)
    print("[4.2] GENERATING CALSPEC SPECTRAL PLOTS")
    print("=" * 60)

    # Check if input catalog exists
    if not csv_file.exists():
        print(f"[!] Error: Catalog not found at '{csv_path}'")
        return

    # Ensure output plot directory exists
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_file)

    stats = {"rendered": 0, "failed": 0}

    # Iterate over each catalog target
    for _, row in df.iterrows():
        star_name = str(row["calspec_name"]).strip()
        fits_file = str(row["fits_file"]).strip()
        fits_path = fits_dir / fits_file

        # Validate local existence of FITS file
        if not fits_path.exists():
            print(f"   [!] FITS not found for {star_name} ({fits_file}). Omitting.")
            stats["failed"] += 1
            continue

        try:
            # Extract wavelength and flux columns from FITS binary table
            with fits.open(fits_path) as hdul:
                data = hdul[1].data
                cols = [c.name.upper() for c in hdul[1].columns]
                wave_col = "WAVELENGTH" if "WAVELENGTH" in cols else cols[0]
                flux_col = "FLUX" if "FLUX" in cols else cols[1]

                wavelengths = data[wave_col]
                fluxes = data[flux_col]

            # Target wavelength range bounds in Angstroms
            #WAVE_MIN, WAVE_MAX = 3000.0, 11000.0

            # Definimos la ventana científica relevante
            WAVE_MIN_LIMIT = 3000.0
            WAVE_MAX_LIMIT = 11000.0

            # Ajuste dinámico inteligente: se adapta al FITS pero dentro de la ventana útil
            WAVE_MIN = max(WAVE_MIN_LIMIT, wavelengths.min())
            WAVE_MAX = min(WAVE_MAX_LIMIT, wavelengths.max())

            # Mask data to target spectral window (3000 - 11000 Å)
            mask = (wavelengths >= WAVE_MIN) & (wavelengths <= WAVE_MAX)
            if not mask.any():
                print(f"   [!] No data in range 3000-11000 Å for {star_name}.")
                stats["failed"] += 1
                continue

            # Compute dynamic y-axis display boundaries
            y_max = fluxes[mask].max() * 1.15
            y_min = max(0.0, fluxes[mask].min() * 0.9)

            # Build spectral plot figure
            plt.figure(figsize=(10, 5))
            plt.plot(wavelengths[mask], fluxes[mask], color="#1f77b4", linewidth=1.2, label="CALSPEC Reference")

            plt.xlim(WAVE_MIN, WAVE_MAX)
            plt.ylim(y_min, y_max)

            plt.title(f"CALSPEC Spectrum: {star_name}", fontsize=13, fontweight="bold")
            plt.xlabel("Wavelength [Å]", fontsize=11)
            plt.ylabel("Flux [erg / s / cm² / Å]", fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(loc="upper right")
            plt.tight_layout()

            # Save plot to disk
            output_plot = plot_dir / f"spectrum_{star_name}.png"
            plt.savefig(output_plot, dpi=300)
            plt.close()

            stats["rendered"] += 1
            print(f"   [✓] Gráfico generado: {output_plot.name}")

        except Exception as e:
            print(f"   [!] Error procesando {star_name}: {e}")
            stats["failed"] += 1

    # Display processing execution summary
    print("\n" + "=" * 40)
    print("      GRAPHS GENERATION SUMMARY")
    print("=" * 40)
    print(f"Generated graphs : {stats['rendered']}")
    print(f"Omited or error  : {stats['failed']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Define relative paths for local execution
    catalog_csv = "data/raw/calspec_catalog.csv"
    spectra_dir = "data/raw/calspec_fits"
    plots_output = "data/raw/calspec_spectrum"

    # 1. Execute concurrent FITS download
    download_missing_fits(catalog_csv, spectra_dir)

    # 2. Generate spectral plots from downloaded FITS files
    plot_calspec_spectra(catalog_csv, spectra_dir, plots_output)