"""
Module: catalogs_management.py

Description:
    Core utility functions for catalog crossmatching, ID resolution, 
    and standalone/modular plotting of Gaia and CALSPEC spectra.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
import functools
import warnings
from astropy.utils.exceptions import AstropyWarning


# ============================
# 1. INTERNAL HELPERS
# ============================

def get_clean_filename(name: str) -> str:
    """Standardizes string formatting for safe file saving."""
    return str(name).strip().replace("/", "_").replace(" ", "_")

def get_wavelength_grid(num_channels: int) -> np.ndarray:
    """Centralizes the wavelength definition."""
    return np.linspace(330, 1050, num_channels)

@functools.lru_cache(maxsize=4)
def _load_numpy_matrix(matrix_path: str | Path) -> np.ndarray:
    """Internal cached helper to avoid re-reading binary matrix files in loops."""
    return np.load(Path(matrix_path))

def _load_catalog(catalog_path: str | Path) -> pd.DataFrame:
    """Internal helper to validate and load the CSV catalog."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found at: {path}")
    return pd.read_csv(path)

def resolve_catalog_columns(df: pd.DataFrame):
    """Returns the correct column names for ID and Name."""
    name_col = "calspec_name" if "calspec_name" in df.columns else "Name"
    id_col = "source_id" if "source_id" in df.columns else "gaia_source_id"
    return id_col, name_col

#def group_catalog_by_color(df: pd.DataFrame, color_column: str = "bp_rp"):
    #"""Applies the standard BP-RP color bins to the catalog."""
    #bins = [-np.inf, 0.5, 1.5, np.inf]
    #labels = ["Blue (BP-RP < 0.5)", "Solar (0.5 <= BP-RP <= 1.5)", "Red (BP-RP > 1.5)"]
    #df["Color_Group"] = pd.cut(df[color_column], bins=bins, labels=labels)
    #return df, labels

def group_catalog_by_color(df: pd.DataFrame, color_column: str = "bp_rp") -> tuple[pd.DataFrame, list[str]]:
    """
    Categorizes catalog stars into color groups, handling saturated stars (G < 6.0) 
    and White Dwarfs, returning a modified copy and the label list.
    """
    df_out = df.copy()

    bins = [-np.inf, 0.5, 1.5, np.inf]
    labels = ["Blue (BP-RP < 0.5)", "Solar (0.5 <= BP-RP <= 1.5)", "Red (BP-RP > 1.5)"]
    
    # 1. Asignación inicial estándar por fotometría BP-RP
    df_out["Color_Group"] = pd.cut(df_out[color_column], bins=bins, labels=labels).astype(str)

    # 2. Corrección: Enanas Blancas (WD/GD/HZ/Feige/G191) siempre en grupo Azul
    if "calspec_name" in df_out.columns:
        wd_mask = df_out["calspec_name"].astype(str).str.lower().str.startswith(("wd", "gd", "hz", "feige", "g191"))
        df_out.loc[wd_mask, "Color_Group"] = labels[0]

    # 3. Corrección: Estrellas brillantes/saturadas con fotometría BP-RP falsa (ej. eta1dor -> Solar)
    if "phot_g_mean_mag" in df_out.columns and "calspec_name" in df_out.columns:
        saturated_bright = (df_out["phot_g_mean_mag"] < 6.0) & (df_out["calspec_name"].astype(str).str.lower() == "eta1dor")
        df_out.loc[saturated_bright, "Color_Group"] = labels[1]

    return df_out, labels

def extract_fits_data(fits_path: Path):
    """Extracts and filters wavelength and flux from a FITS file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)

        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            cols = [c.name.upper() for c in hdul[1].columns]
            
            wavelengths = data["WAVELENGTH"] if "WAVELENGTH" in cols else data[cols[0]]
            fluxes = data["FLUX"] if "FLUX" in cols else data[cols[1]]
            
        # Mask data to target spectral window (3000.0 - 11000.0)
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
    """Internal engine to plot spectra from processed matrix files (.npy)."""
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

# =======================================
# 2. ID EXTRACTION AND RESOLUTION
# =======================================

def get_gaia_ids(catalog_path: str | Path = "data/processed/crossmatch_catalog.csv"):
    """Extracts all Gaia Source IDs from the catalog."""
    df = _load_catalog(catalog_path)
    id_col = "source_id" if "source_id" in df.columns else "gaia_source_id"
    return df[id_col].astype(str).str.strip().tolist()

def get_calspec_names(catalog_path: str | Path = "data/processed/crossmatch_catalog.csv"):
    """Extracts all CALSPEC names from the catalog."""
    df = _load_catalog(catalog_path)
    name_col = "calspec_name" if "calspec_name" in df.columns else "Name"
    return df[name_col].astype(str).str.strip().tolist()

def resolve_star_info(identifier, catalog_path: str | Path = "data/processed/crossmatch_catalog.csv"):
    """
    Finds a star in the catalog using either its Gaia ID or CALSPEC name.
    Returns: tuple (matrix_index, gaia_id, calspec_name)
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

# ===============================
# 3. PUBLIC PLOTTING FUNCTIONS
# ===============================

def plot_gaia_spectrum(
    identifier,
    x_path: str | Path = "data/processed/X.npy",
    catalog_path: str | Path = "data/processed/crossmatch_catalog.csv",
    output_dir: str | Path = None,
    ax=None,
):
    """Plots the raw Gaia spectrum for a given Gaia ID or CALSPEC name."""
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
    """Plots the CALSPEC target spectrum for a given Gaia ID or CALSPEC name."""
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