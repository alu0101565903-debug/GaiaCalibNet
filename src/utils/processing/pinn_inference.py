"""Shared model loading and single-spectrum PINN inference helpers."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

from src.models.pinn_architecture import BlueWiggleNet, DirectConvPINN
from src.models.pinn_export_models import EnsembleExportModel
from src.utils.processing.stellar_loader import transform_stellar_parameters


def load_ensemble_from_bundle(
    model_bundle_path: str | Path,
    device: torch.device,
) -> tuple[EnsembleExportModel, dict]:
    """Build an ensemble from a self-contained training bundle."""
    bundle = torch.load(model_bundle_path, map_location=device, weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Unsupported model file format: {model_bundle_path}")

    if "base_model" not in bundle or "blue_model" not in bundle:
        raise KeyError(
            "The inference file must be a self-contained model_bundle.pt with "
            "base_model and blue_model state dictionaries."
        )

    metadata = bundle
    base_state = bundle["base_model"]
    blue_state = bundle["blue_model"]

    wave_grid = np.asarray(metadata.get("wave_grid"), dtype=np.float32)
    pca_basis = metadata.get("pca_basis")
    pca_mean = metadata.get("pca_mean")
    num_params = metadata.get("num_params")
    input_dim = metadata.get("input_dim", len(wave_grid))
    hidden_dim = metadata.get("hidden_dim")
    if hidden_dim is None:
        first_hidden_weight = base_state.get("mlp.0.weight")
        hidden_dim = int(first_hidden_weight.shape[0]) if first_hidden_weight is not None else 128

    if pca_basis is None or pca_mean is None or num_params is None or wave_grid.size == 0:
        raise KeyError(
            "The model bundle must contain wave_grid, pca_basis, pca_mean, and num_params."
        )

    blue_mask = np.asarray(metadata.get("blue_mask", wave_grid < 450.0), dtype=bool)
    base_model = DirectConvPINN(
        input_dim=input_dim,
        pca_basis=pca_basis,
        pca_mean=pca_mean,
        num_params=num_params,
        hidden_dim=int(hidden_dim),
    ).to(device)
    blue_model = BlueWiggleNet(
        input_dim=int(np.sum(blue_mask)),
        num_params=num_params,
    ).to(device)
    ensemble = EnsembleExportModel(
        base_model=base_model,
        blue_model=blue_model,
        blue_mask=blue_mask,
        wave_grid=wave_grid,
    ).to(device)

    base_model.load_state_dict(base_state)
    blue_model.load_state_dict(blue_state)

    ensemble.eval()
    return ensemble, metadata


def calibrate_spectrum(
    wave_raw: np.ndarray,
    flux_raw: np.ndarray,
    bp_rp: float,
    g_mag: float,
    ruwe: float | None,
    model_bundle_path: str | Path,
    mean_p: np.ndarray,
    std_p: np.ndarray,
    device: str | torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate a spectrum; pass None/NaN for RUWE to use the saved training median."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ensemble, metadata = load_ensemble_from_bundle(model_bundle_path, device)
    wave_grid = np.asarray(metadata["wave_grid"], dtype=np.float32)
    interp = interp1d(
        wave_raw,
        flux_raw,
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    flux_interp = interp(wave_grid).astype(np.float32)

    params_raw = np.array([bp_rp, g_mag, ruwe], dtype=np.float32)
    params_std = transform_stellar_parameters(
        params_raw, mean_p, std_p, metadata.get("ruwe_imputation_median")
    )
    x_tensor = torch.tensor(flux_interp, dtype=torch.float32, device=device).unsqueeze(0)
    params_tensor = torch.tensor(params_std, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        ratio = ensemble(x_tensor, params_tensor).squeeze(0).cpu().numpy()

    return wave_grid, flux_interp, flux_interp * ratio
