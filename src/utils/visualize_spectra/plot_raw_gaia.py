"""
Module extension: Fetch and plot raw Gaia spectra directly from ESA Archive.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia


def plot_raw_gaia_server_spectrum(
    gaia_id: str | int, 
    output_dir: str | Path = None, 
    ax=None
) -> bool:
    """
    Downloads and plots the official raw sampled BP/RP spectrum 
    directly from the ESA Gaia DR3 archive using DataLink.

    Parameters
    ----------
    gaia_id : str or int
        The unique Gaia Source ID (e.g., 5853498713190525696).
    output_dir : str or Path, optional
        Directory where the plot PNG will be saved if ax is None.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on (for subplots or combined comparisons).

    Returns
    -------
    bool
        True if retrieval and plotting succeeded, False otherwise.
    """
    clean_id = int(str(gaia_id).strip())

    print(f"   [+] Fetching raw spectrum for Gaia ID {clean_id} from ESA Gaia archive...")

    try:
        # 1. Query DataLink to retrieve the XP Sampled spectrum from Gaia DR3
        datalink = Gaia.load_data(
            ids=[clean_id],
            data_release="Gaia DR3",
            retrieval_type="XP_SAMPLED",
            format="csv",
        )

        if not datalink or len(datalink) == 0:
            print(f"   [!] No spectral data found on ESA server for Gaia ID {clean_id}.")
            return False

        # 2. Extract spectrum table from DataLink product dictionary
        first_key = list(datalink.keys())[0]
        spectrum_df = datalink[first_key][0].to_pandas()

        wavelengths = spectrum_df["wavelength"]  # Wavelengths in nm directly from ESA
        fluxes = spectrum_df["flux"]              # Flux in W / (m² · nm)

        # 3. Handle plot axis creation
        save_and_close = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            save_and_close = True

        # 4. Draw curve (with error margins if available in the ESA table)
        if "flux_error" in spectrum_df.columns:
            ax.errorbar(
                wavelengths,
                fluxes,
                yerr=spectrum_df["flux_error"],
                fmt="-",
                color="#264ab7",
                linewidth=1.2,
                alpha=0.8,
                ecolor="#a8bba2",
                elinewidth=0.6,
                label="Raw Gaia DR3 (ESA Server)",
            )
        else:
            ax.plot(
                wavelengths,
                fluxes,
                color="#264ab7",
                linewidth=1.5,
                label="Raw Gaia DR3 (ESA Server)",
            )

        # 5. Apply standard formatting
        ax.set_title(f"Raw ESA Gaia DR3 Spectrum | Source ID: {clean_id}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Wavelength [nm]", fontsize=11)
        ax.set_ylabel("Flux [W / m² / nm]", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", frameon=True)

        # 6. Save plot if running in standalone mode
        if save_and_close:
            plt.tight_layout()
            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_path / f"gaia_server_spectrum_{clean_id}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        return True

    except Exception as e:
        print(f"   [!] Error retrieving spectrum from Gaia server: {e}")
        return False


if __name__ == "__main__":
    # Example usage with a real Gaia DR3 ID
    sample_gaia_id = 5853498713190525696
    plot_raw_gaia_server_spectrum(
        gaia_id=sample_gaia_id, 
        output_dir="data/raw/gaia_spectrum"
    )