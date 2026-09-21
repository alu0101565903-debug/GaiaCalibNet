"""
Module: plot_raw_calspec.py

Report Section: 4.2 CALSPEC Spectral Plot Generation

Description:
    Processes CALSPEC catalog metadata, loads the corresponding raw FITS files,
    and generates standardized spectral plots directly from the raw data.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing.download_calspec import download_missing_fits
from src.utils.visualize_spectra.catalogs_management import extract_fits_data


def plot_raw_fits_spectrum(star_name: str, fits_path: Path, output_dir: Path) -> bool:
    """
    Generates and saves the spectral plot for a raw CALSPEC FITS file.

    Parameters
    - star_name (str): The CALSPEC target name.
    - fits_path (Path): Path to the raw CALSPEC FITS file.
    - output_dir (Path): Directory where the generated PNG plot will be saved.

    Returns
    - bool: True if the plot was generated successfully, False otherwise.
    """
    try:
        # Extract wavelength and flux using the utility function
        raw_wavelengths, fluxes = extract_fits_data(fits_path)

        # Convert wavelength from Ångströms (Å) to Nanometers (nm) for pipeline consistency
        wavelengths = raw_wavelengths / 10.0
        
        # Verify if valid data was extracted within the target range
        if len(wavelengths) == 0:
            print(f"   [!] No data in range 3000-11000 Å for {star_name}.")
            return False

        # Canvas setup and spectral trace plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavelengths, fluxes, color="#1f77b4", linewidth=1.2, label="Raw CALSPEC (FITS)")

        # Set dynamic axes limits based on extracted data
        ax.set_xlim(wavelengths.min(), wavelengths.max())
        ax.set_ylim(max(0.0, fluxes.min() * 0.9), fluxes.max() * 1.15)
        
        # Apply standard formatting
        ax.set_title(f"CALSPEC Spectrum: {star_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Wavelength [nm]", fontsize=11)
        ax.set_ylabel("Flux [erg / s / cm² / Å]", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        
        # Save plot to disk
        plt.tight_layout()
        plt.savefig(output_dir / f"spectrum_{star_name}.png", dpi=300)
        plt.close(fig)

        return True

    except Exception as e:
        print(f"   [!] Error plotting {star_name}: {e}")
        return False


def process_all_calspec_plots(csv_path: Path, fits_dir: Path, plot_dir: Path) -> None:
    """
    Iterates through the CALSPEC catalog to plot all valid FITS files.

    Parameters
    - csv_path (Path): Path to the CSV catalog containing CALSPEC metadata.
    - fits_dir (Path): Directory containing the downloaded raw FITS files.
    - plot_dir (Path): Directory to save the generated plots.
    """
    print("=" * 60 + "\n[4.2] GENERATING RAW CALSPEC SPECTRAL PLOTS\n" + "=" * 60)

    # Validate catalog existence before executing batch processing
    if not csv_path.exists():
        print(f"[!] Error: Catalog not found at '{csv_path}'")
        return

    # Ensure output destination folder exists
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata catalog
    df = pd.read_csv(csv_path)
    stats = {"rendered": 0, "failed": 0}

    # Iterate over the catalog to process each FITS file
    for _, row in df.iterrows():
        star_name = str(row["calspec_name"]).strip()
        fits_file = str(row["fits_file"]).strip()
        fits_path = fits_dir / fits_file

        # Check file existence and attempt plot rendering
        if not fits_path.exists() or not plot_raw_fits_spectrum(star_name, fits_path, plot_dir):
            print(f"   [!] Failed or omitted: {star_name}")
            stats["failed"] += 1
        else:
            print(f"   [✓] Plot generated for: {star_name}")
            stats["rendered"] += 1

    # Print execution summary
    print(f"\n{'='*40}\n      GRAPHS GENERATION SUMMARY\n{'='*40}")
    print(f"Generated graphs : {stats['rendered']}\nOmitted or error : {stats['failed']}\n{'='*60}\n")


if __name__ == "__main__":
    # Define standard relative file paths
    catalog_csv = Path("data/raw/calspec_catalog.csv")
    spectra_dir = Path("data/raw/calspec_fits")
    plots_output = Path("data/raw/calspec_spectrum")

    # 1. Download any missing FITS files
    download_missing_fits(str(catalog_csv), str(spectra_dir))

    #plot_raw_fits_spectrum("GD153", spectra_dir / "gd153_mod_002.fits", plots_output)  # Example single plot
    
    # 2. Process and plot all valid spectra
    process_all_calspec_plots(catalog_csv, spectra_dir, plots_output)