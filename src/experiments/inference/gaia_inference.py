"""Apply the trained neural correction to a Gaia DR3 source by its ID.

GaiaXPy reconstructs the XP spectrum on the training wavelength grid. The
neural ensemble then applies the learned correction; no reference spectrum
or local spectral file is required.
"""

from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from src.utils.processing.pinn_inference import load_ensemble_from_bundle
from src.utils.processing.stellar_loader import transform_stellar_parameters


# Match the output directory configured in src/train/mains/main_pinn.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results/models/full_pinn_training/all_samples"


def _download_gaia_data(source_id: str, wave_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the requested XP spectrum and [BP-RP, G, RUWE] features."""
    # Import archive clients only when downloading, keeping module imports offline.
    from astroquery.gaia import Gaia
    from gaiaxpy import calibrate

    query = f"""
        SELECT source_id, bp_rp, phot_g_mean_mag, ruwe, has_xp_continuous
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
    """
    catalog = Gaia.launch_job(query, dump_to_file=False).get_results()
    if len(catalog) != 1 or str(catalog["source_id"][0]) != source_id:
        raise ValueError(f"Gaia DR3 source {source_id} was not found.")
    row = catalog[0]
    if np.ma.is_masked(row["has_xp_continuous"]) or not row["has_xp_continuous"]:
        raise ValueError(f"Gaia DR3 source {source_id} has no published continuous XP spectrum.")

    features = np.array([
        np.nan if np.ma.is_masked(row[name]) else row[name]
        for name in ("bp_rp", "phot_g_mean_mag", "ruwe")
    ], dtype=np.float32)
    if not np.isfinite(features[:2]).all():
        raise ValueError(f"Gaia DR3 source {source_id} is missing BP-RP or G magnitude.")

    spectra, sampling = calibrate([source_id], sampling=wave_grid, save_file=False)
    if spectra.empty:
        raise ValueError(f"No XP spectrum was returned for Gaia DR3 source {source_id}.")
    matches = spectra.loc[spectra["source_id"].astype(str) == source_id]
    if len(matches) != 1:
        raise ValueError(f"No unique XP spectrum was returned for Gaia DR3 source {source_id}.")
    if np.shape(sampling) != wave_grid.shape or not np.allclose(sampling, wave_grid):
        raise ValueError("GaiaXPy returned a wavelength grid different from the model grid.")
    flux = np.asarray(np.ma.filled(np.ma.asarray(matches.iloc[0]["flux"]), np.nan), dtype=np.float32)
    if flux.shape != wave_grid.shape or not np.isfinite(flux).all():
        raise ValueError("The Gaia spectrum must contain finite fluxes across the entire model grid.")
    return flux, features


def calibrate_gaia_spectrum(
    gaia_source_id: str | int,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Download and correct one Gaia DR3 spectrum with the trained neural net.

    Parameters
    ----------
    gaia_source_id : str or int
        Gaia DR3 source identifier. Use an integer or decimal string, not a
        floating-point number, to preserve all identifier digits.
    model_dir : str or pathlib.Path, optional
        Directory containing ``model_bundle.pt``. Defaults to the
        ``all_samples`` training run, resolved from the repository root.
        Run ``python -m src.train.mains.main_pinn`` once to create this bundle.
        The bundle supplies the network, wavelength grid, feature statistics,
        and RUWE imputation median. No separate normalization files are needed.
    device : str or torch.device, optional
        Inference device. Defaults to CUDA when available, otherwise CPU.

    Returns
    -------
    wavelength : numpy.ndarray
        Wavelengths in nanometers on the model's grid, with shape ``(n_wave,)``.
    corrected_flux : numpy.ndarray
        Neural-network-corrected flux density in W/m^2/nm, with the same shape.

    Raises
    ------
    FileNotFoundError
        The selected model bundle does not exist.
    ValueError
        The identifier is invalid, the source has no usable XP spectrum or
        stellar features, or the model metadata are incompatible.

    Notes
    -----
    Requires network access to the Gaia Archive. Missing RUWE is imputed using
    the saved training median. Missing BP-RP or G cannot be imputed. This
    function returns arrays and does not save spectra, plots, or model files.
    Archive and network errors propagate to the caller.

    Examples
    --------
    >>> wave, flux = calibrate_gaia_spectrum("2462426800883134336")
    """
    if isinstance(gaia_source_id, (bool, np.bool_)) or not isinstance(gaia_source_id, (str, Integral)):
        raise ValueError("gaia_source_id must be an integer or a decimal string.")
    source_id = str(gaia_source_id).strip()
    if not source_id.isascii() or not source_id.isdecimal() or not 0 < int(source_id) < 2**63:
        raise ValueError("gaia_source_id must be a positive signed 64-bit integer.")
    source_id = str(int(source_id))

    bundle_path = Path(model_dir).expanduser() / "model_bundle.pt"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}. "
            "Run 'python -m src.train.mains.main_pinn' to train the all-samples "
            "model, or set model_dir to another trained model directory."
        )
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensemble, metadata = load_ensemble_from_bundle(bundle_path, device)
    wave_grid = np.asarray(metadata["wave_grid"], dtype=np.float32)
    mean_p = np.asarray(metadata.get("stellar_mean"), dtype=np.float32)
    std_p = np.asarray(metadata.get("stellar_std"), dtype=np.float32)
    num_params = metadata["num_params"]
    if num_params not in (2, 3) or mean_p.shape != (num_params,) or std_p.shape != mean_p.shape:
        raise ValueError("The model bundle must contain matching BP-RP/G[/RUWE] feature statistics.")
    if wave_grid.ndim != 1 or wave_grid.size < 2 or not np.isfinite(wave_grid).all() or np.any(np.diff(wave_grid) <= 0):
        raise ValueError("The model wavelength grid must be finite and strictly increasing.")

    flux, features = _download_gaia_data(source_id, wave_grid)
    params = transform_stellar_parameters(
        features[:num_params], mean_p, std_p, metadata.get("ruwe_imputation_median")
    )
    flux_tensor = torch.as_tensor(flux, dtype=torch.float32, device=device).unsqueeze(0)
    params_tensor = torch.as_tensor(params, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        ratio = ensemble(flux_tensor, params_tensor).squeeze(0).cpu().numpy()
    if ratio.shape != wave_grid.shape:
        raise ValueError("The neural correction must have one value per wavelength.")
    corrected_flux = flux * ratio
    if corrected_flux.shape != wave_grid.shape or not np.isfinite(corrected_flux).all():
        raise ValueError("The neural model returned an invalid corrected spectrum.")
    return wave_grid, corrected_flux


if __name__ == "__main__":
    # Choose the Gaia DR3 source to correct.
    source_id = "2462426800883134336"
    wavelength, corrected_flux = calibrate_gaia_spectrum(source_id)
    print(f"Corrected Gaia DR3 {source_id}: {len(wavelength)} samples from {wavelength[0]:g} to {wavelength[-1]:g} nm.")
