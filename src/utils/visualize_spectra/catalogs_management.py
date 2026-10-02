"""Resolve catalog entries and plot Gaia and CALSPEC spectra."""


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import functools
import warnings
from astropy.utils.exceptions import AstropyWarning


# Internal helpers.

def get_clean_filename(name: str) -> str:
    """Convert a name to a filesystem-friendly filename component.

    Parameters
    ----------
    name : str
        Value to clean.

    Returns
    -------
    str
        Filename component with spaces and slashes replaced by underscores.
    """
    return str(name).strip().replace("/", "_").replace(" ", "_")

def get_wavelength_grid(num_channels: int) -> np.ndarray:
    """Build the default wavelength grid.

    Parameters
    ----------
    num_channels : int
        Number of wavelength samples.

    Returns
    -------
    numpy.ndarray
        Linearly spaced wavelength grid from 330 to 1050 nm.
    """
    return np.linspace(330, 1050, num_channels)

@functools.lru_cache(maxsize=4)
def _load_numpy_matrix(matrix_path: str | Path) -> np.ndarray:
    """Load and cache a NumPy matrix.

    Parameters
    ----------
    matrix_path : str or pathlib.Path
        Path to the NumPy matrix file.

    Returns
    -------
    numpy.ndarray
        Loaded matrix.
    """
    return np.load(Path(matrix_path))

def _load_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Load a catalog after validating that its file exists.

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        Path to the catalog CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded catalog.

    Raises
    ------
    FileNotFoundError
        If the catalog file does not exist.
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found at: {path}")
    return pd.read_csv(path)

def resolve_catalog_columns(df: pd.DataFrame):
    """Resolve the source identifier and target name columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog containing source identifiers and target names.

    Returns
    -------
    tuple of str
        Identifier column name followed by target name column name.
    """
    name_col = "calspec_name" if "calspec_name" in df.columns else "Name"
    id_col = "source_id" if "source_id" in df.columns else "gaia_source_id"
    return id_col, name_col

# def group_catalog_by_color(df: pd.DataFrame, color_column: str = "bp_rp"):
    # """Apply the standard BP-RP color bins to the catalog."""
    #bins = [-np.inf, 0.5, 1.5, np.inf]
    #labels = ["Blue (BP-RP < 0.5)", "Solar (0.5 <= BP-RP < 1.5)", "Red (BP-RP >= 1.5)"]
    #df_result = df.copy()
    #df_result["Color_Group"] = pd.cut(df[color_column], bins=bins, labels=labels)
    #return df_results, labels

def extract_fits_data(fits_path: Path):
    """Extract wavelength and flux arrays from a FITS file.

    Parameters
    ----------
    fits_path : pathlib.Path
        Path to the FITS file.

    Returns
    -------
    tuple of numpy.ndarray
        Wavelength and flux arrays restricted to 3000-11000 Angstroms.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)

        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            cols = [c.name.upper() for c in hdul[1].columns]
            
            wavelengths = data["WAVELENGTH"] if "WAVELENGTH" in cols else data[cols[0]]
            fluxes = data["FLUX"] if "FLUX" in cols else data[cols[1]]
            
        # Restrict the arrays to the target wavelength range.
        wave_min = max(3000.0, wavelengths.min())
        wave_max = min(11000.0, wavelengths.max())
        mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
    
    return wavelengths[mask], fluxes[mask]

def _plot_spectrum_base(
    identifier,
    matrix_path: str | Path,
    label: str,
    color: str,
    title_prefix: str,
    file_prefix: str,
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    output_dir: str | Path = None,
    ax=None,
    linewidth: float = 1.8,
    alpha: float = 0.8,
):
    """Plot one spectrum loaded from a processed NumPy matrix."""
    star_idx, gaia_id, calspec_name = resolve_star_info(identifier, catalog_path=catalog_path)

    data = _load_numpy_matrix(str(matrix_path))
    if star_idx >= len(data):
        raise IndexError(f"Index {star_idx} out of bounds for matrix at {matrix_path}.")

    spectrum = data[star_idx]
    wavelengths = np.linspace(330, 1050, len(spectrum))

    save_and_close = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        save_and_close = True

    ax.plot(wavelengths, spectrum, color=color, linewidth=linewidth, alpha=alpha, label=label)

    if save_and_close:
        ax.set_title(f"{title_prefix} | Star: {calspec_name} | Gaia ID: {gaia_id}", fontweight="bold")
        ax.set_xlabel("Wavelength λ [nm]")
        ax.set_ylabel("Absolute Flux [erg / s / cm² / Å]")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            clean_name = calspec_name.replace("/", "_").replace(" ", "_")
            filename = f"{file_prefix}_{clean_name}_{gaia_id}.png"
            plt.savefig(out_path / filename, dpi=200, bbox_inches="tight")

        plt.close(fig)

# Identifier extraction and resolution.

def get_gaia_ids(catalog_path: str | Path = "data/processed/crossmatch_catalog.csv"):
    """Return Gaia source identifiers from a catalog.

    Parameters
    ----------
    catalog_path : str or pathlib.Path, optional
        Path to the catalog CSV file.

    Returns
    -------
    list of str
        Trimmed Gaia source identifiers.
    """
    df = _load_catalog(catalog_path)
    id_col = "source_id" if "source_id" in df.columns else "gaia_source_id"
    return df[id_col].astype(str).str.strip().tolist()

def get_calspec_names(catalog_path: str | Path = "data/processed/crossmatch_catalog.csv"):
    """Return CALSPEC target names from a catalog.

    Parameters
    ----------
    catalog_path : str or pathlib.Path, optional
        Path to the catalog CSV file.

    Returns
    -------
    list of str
        Trimmed CALSPEC target names.
    """
    df = _load_catalog(catalog_path)
    name_col = "calspec_name" if "calspec_name" in df.columns else "Name"
    return df[name_col].astype(str).str.strip().tolist()

def resolve_star_info(identifier, catalog_path: str | Path = "data/processed/crossmatch_catalog.csv"):
    """
    Resolve a target by Gaia source identifier or CALSPEC name.

    Parameters
    ----------
    identifier : str or int
        Gaia source identifier or CALSPEC target name.
    catalog_path : str or pathlib.Path, optional
        Path to the catalog CSV file.

    Returns
    -------
    tuple
        Matrix index, Gaia source identifier, and CALSPEC target name.

    Raises
    ------
    ValueError
        If the target is not present in the catalog.
    """
    df = _load_catalog(catalog_path)
    id_col = "source_id" if "source_id" in df.columns else "gaia_source_id"
    name_col = "calspec_name" if "calspec_name" in df.columns else "Name"

    df["clean_id"] = df[id_col].astype(str).str.strip().str.lower()
    df["clean_name"] = df[name_col].astype(str).str.strip().str.lower()

    target = str(identifier).strip().lower()
    match = df[(df["clean_id"] == target) | (df["clean_name"] == target)]

    if match.empty:
        raise ValueError(f"Identifier '{identifier}' not found in catalog.")

    star_idx = match.index[0]
    row = match.iloc[0]

    return star_idx, str(row[id_col]), str(row[name_col])

# Public plotting functions.

def plot_gaia_spectrum(
    identifier,
    x_path: str | Path = "data/processed/X.npy",
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    output_dir: str | Path = None,
    ax=None,
):
    """Plot the Gaia spectrum for a catalog target."""
    _plot_spectrum_base(
        identifier=identifier,
        matrix_path=x_path,
        label="Gaia BP/RP Spectrum",
        color="#264ab7",
        title_prefix="Gaia Spectrum",
        file_prefix="gaia_spectrum",
        catalog_path=catalog_path,
        output_dir=output_dir,
        ax=ax,
    )

def plot_calspec_spectrum(
    identifier,
    y_path: str | Path = "data/processed/Y.npy",
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    output_dir: str | Path = None,
    ax=None,
):
    """Plot the CALSPEC reference spectrum for a catalog target."""
    _plot_spectrum_base(
        identifier=identifier,
        matrix_path=y_path,
        label="CALSPEC Reference",
        color="black",
        title_prefix="CALSPEC Spectrum",
        file_prefix="calspec_spectrum",
        catalog_path=catalog_path,
        output_dir=output_dir,
        ax=ax,
    )