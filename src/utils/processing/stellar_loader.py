"""Load stellar features and fit RUWE imputation/scaling on training stars only."""

from pathlib import Path
from typing import Union

import numpy as np


def load_raw_stellar_parameters(data_dir: Union[str, Path]) -> np.ndarray:
    """Read [BP-RP, G, RUWE], preserving missing RUWE as NaN without fitting."""
    data_path = Path(data_dir)
    params = np.column_stack([
        np.load(data_path / f"{name}.npy").reshape(-1)
        for name in ("bp_rp", "G_mags", "ruwe")
    ]).astype(np.float32)
    mask_path = data_path / "ruwe_missing.npy"
    if mask_path.exists():
        missing = np.load(mask_path)
        if missing.dtype != np.bool_ or missing.shape != (len(params),):
            raise ValueError("ruwe_missing.npy must be a boolean mask aligned with the stars.")
        params[missing, 2] = np.nan
    return params


def _validate_stellar_parameters(params: np.ndarray) -> np.ndarray:
    """Allow absent RUWE, but reject invalid observed values or other features."""
    values = np.array(params, dtype=np.float32, copy=True)
    if values.ndim not in (1, 2) or values.shape[-1] not in (2, 3):
        raise ValueError("Expected [BP-RP, G] or [BP-RP, G, RUWE] features.")
    if not np.isfinite(values[..., :2]).all():
        raise ValueError("BP-RP and G must be finite.")
    if values.shape[-1] == 3:
        ruwe = values[..., 2]
        observed = ruwe[~np.isnan(ruwe)]
        if not np.isfinite(observed).all() or np.any(observed <= 0):
            raise ValueError("Observed RUWE must be finite and positive; use NaN for missing RUWE.")
    return values


def transform_stellar_parameters(
    params: np.ndarray, mean: np.ndarray, std: np.ndarray,
    ruwe_imputation_median: float | None = None,
) -> np.ndarray:
    """Impute absent RUWE using the saved training median, then standardize.

    Models without an imputation median can still transform fully observed
    parameters. They cannot infer a replacement for a missing RUWE value.
    """
    values = _validate_stellar_parameters(params)
    mean, std = np.asarray(mean), np.asarray(std)
    if values.shape[-1:] != mean.shape or std.shape != mean.shape:
        raise ValueError("Parameters and normalization statistics must have matching feature dimensions.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError("Statistics must be finite, with nonnegative standard deviations.")
    if ruwe_imputation_median is not None:
        if not np.isfinite(ruwe_imputation_median) or ruwe_imputation_median <= 0:
            raise ValueError("The saved RUWE imputation median must be finite and positive.")
    if values.shape[-1] == 3 and np.isnan(values[..., 2]).any():
        if ruwe_imputation_median is None:
            raise ValueError("Missing RUWE requires a saved training imputation median; retrain legacy models first.")
        values[..., 2] = np.where(np.isnan(values[..., 2]), ruwe_imputation_median, values[..., 2])
    return ((values - mean) / (std + 1e-8)).astype(np.float32)


def standardize_stellar_parameters(
    params: np.ndarray, train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Fit RUWE median and mean/std on training rows, then transform all rows.

    Returns (normalized, mean, std, ruwe_imputation_median). The median is None
    for the two-feature ablation. Mean/std include imputed training values;
    the median itself only uses observed training RUWE. No files are written.
    """
    values = _validate_stellar_parameters(params)
    train_indices = np.asarray(train_indices)
    if values.ndim != 2:
        raise ValueError("Stellar parameters must be a 2D array.")
    if train_indices.ndim != 1 or not np.issubdtype(train_indices.dtype, np.integer):
        raise ValueError("train_indices must be a 1D integer array.")
    if train_indices.size == 0 or np.any(train_indices < 0) or np.any(train_indices >= len(values)):
        raise ValueError("Training indices must be nonempty and within the dataset.")

    ruwe_median = None
    if values.shape[1] == 3:
        observed = values[train_indices, 2]
        observed = observed[~np.isnan(observed)]
        if observed.size == 0:
            raise ValueError("Cannot impute RUWE: no observed RUWE in the training stars.")
        ruwe_median = float(np.median(observed))
        values[:, 2] = np.where(np.isnan(values[:, 2]), ruwe_median, values[:, 2])

    train_params = values[train_indices]
    mean = train_params.mean(axis=0)
    std = train_params.std(axis=0)
    normalized = transform_stellar_parameters(params, mean, std, ruwe_median)
    return normalized, mean, std, ruwe_median


def load_stellar_parameters(data_dir: Union[str, Path]) -> np.ndarray:
    """Fit and save imputation/scaling on all loaded stars.

    Only use this convenience function when all supplied stars are training
    data. CV and held-out evaluation must load raw parameters and call
    standardize_stellar_parameters with their training indices instead.
    """
    data_path = Path(data_dir).resolve()
    params = load_raw_stellar_parameters(data_path)
    normalized, mean, std, ruwe_median = standardize_stellar_parameters(
        params, np.arange(len(params))
    )
    np.save(data_path / "stellar_mean_p.npy", mean)
    np.save(data_path / "stellar_std_p.npy", std)
    np.save(data_path / "ruwe_imputation_median.npy", ruwe_median)
    return normalized
