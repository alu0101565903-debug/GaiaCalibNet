"""Shared dataset, PCA, and empirical-baseline helpers for PINN workflows."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from src.utils.processing.stellar_loader import load_raw_stellar_parameters, load_stellar_parameters


PinnDataset = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]



def stable_flux_denominator(
    X: np.ndarray,
    relative_floor: float = 1e-3,
) -> np.ndarray:
    """Floor low/negative input flux at a small fraction of each spectrum's peak."""
    X = np.asarray(X, dtype=np.result_type(np.asarray(X).dtype, np.float32))
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D flux array, got shape {X.shape}.")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive.")

    peak = np.max(np.abs(X), axis=1, keepdims=True)
    floor = np.maximum(peak * relative_floor, np.finfo(X.dtype).tiny)
    return np.maximum(X, floor)


def compute_calibration_ratio(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute the bounded calibration ratio shared by preparation and PCA fitting."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have matching shapes; got {X.shape} and {Y.shape}.")
    ratio = Y / stable_flux_denominator(X)
    return np.clip(ratio, 0.1, 10.0)


def load_pinn_dataset(data_dir: str | Path, *, standardize_params: bool = True) -> PinnDataset:
    """Load NN arrays; use standardize_params=False before splitting into folds."""
    data_path = Path(data_dir)
    X = np.load(data_path / "X.npy").astype(np.float32)
    Y = np.load(data_path / "Y.npy").astype(np.float32)
    X_error = np.load(data_path / "X_error.npy").astype(np.float32)
    params = (load_stellar_parameters(data_path) if standardize_params
              else load_raw_stellar_parameters(data_path))
    wave_grid = np.load(data_path / "wave_grid.npy").astype(np.float32)
    g_mags = np.load(data_path / "G_mags.npy").astype(np.float32)
    source_ids = np.load(data_path / "gaia_source_ids.npy")
    return X, Y, X_error, params, wave_grid, g_mags, source_ids


def fit_log_ratio_pca(
    X: np.ndarray,
    Y: np.ndarray,
    num_components: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on the provided smoothed log calibration-ratio field."""
    ratio = compute_calibration_ratio(X, Y)
    ratio_log_smooth = gaussian_filter1d(np.log(ratio), sigma=0.5, axis=1)

    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(ratio_log_smooth)
    return pca.components_, pca.mean_


def apply_huang_correction(
    X: np.ndarray,
    X_error: np.ndarray,
    source_ids: np.ndarray,
    g_mags: np.ndarray,
) -> np.ndarray:
    """Apply the Huang et al. correction to a batch of Gaia spectra."""
    from GaiaXPcorrection.GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1

    huang_input = pd.DataFrame(
        {
            "source_id": np.asarray(source_ids).flatten(),
            "flux": list(X),
            "flux_error": list(X_error),
        }
    )
    corrected = Gaia_Correction_V1.correction_df(
        huang_input,
        pd.Series(np.asarray(g_mags).flatten()),
        have_error=True,
        Truncation=True,
    )
    return np.vstack(corrected["flux_cor"].values)
