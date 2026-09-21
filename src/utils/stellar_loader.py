"""
Module: stellar_params_loader.py
Report Section: 4.2 Data Preprocessing & Feature Engineering

This module provides data loader utility functions for auxiliary stellar physical 
parameters (e.g., BP-RP color index, G magnitude, RUWE astrometric indicator), 
applying standard z-score normalization and persisting normalization statistics.

Functions:
    - load_stellar_parameters: Loads, standardizes, and saves mean/std parameters.
"""

from pathlib import Path
from typing import Union

import numpy as np


def load_stellar_parameters(data_dir: Union[str, Path]) -> np.ndarray:
    """
    Load and standard-scale auxiliary stellar parameters from binary files.

    Reads `bp_rp.npy`, `G_mags.npy`, and `ruwe.npy` from the target directory, 
    computes z-score normalization for each feature, persists the normalization 
    statistics (`mean` and `std`) to `.npy` arrays, and returns the stacked features.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing `bp_rp.npy`, `G_mags.npy`, and `ruwe.npy`.

    Returns
    -------
    numpy.ndarray
        Column-stacked float32 array of shape (n_samples, 3) containing 
        standardized stellar features [bp_rp_std, g_mag_std, ruwe_std].

    Raises
    ------
    FileNotFoundError
        If any of the required parameter binary files are missing from `data_dir`.
    """
    data_path = Path(data_dir).resolve()

    # Load individual feature arrays and reshape to column vectors
    bp_rp = np.load(data_path / "bp_rp.npy").reshape(-1, 1)
    g_mag = np.load(data_path / "G_mags.npy").reshape(-1, 1)
    ruwe = np.load(data_path / "ruwe.npy").reshape(-1, 1)

    # Calculate individual feature z-score normalizations
    bp_rp_std = (bp_rp - np.mean(bp_rp)) / (np.std(bp_rp) + 1e-8)
    g_mag_std = (g_mag - np.mean(g_mag)) / (np.std(g_mag) + 1e-8)
    ruwe_std = (ruwe - np.mean(ruwe)) / (np.std(ruwe) + 1e-8)

    # Persist normalization statistics for reproducibility and downstream inference
    mean_p = np.array([np.mean(bp_rp), np.mean(g_mag), np.mean(ruwe)], dtype=np.float32)
    std_p = np.array([np.std(bp_rp), np.std(g_mag), np.std(ruwe)], dtype=np.float32)
    
    np.save(data_path / "stellar_mean_p.npy", mean_p)
    np.save(data_path / "stellar_std_p.npy", std_p)

    return np.hstack([bp_rp_std, g_mag_std, ruwe_std]).astype(np.float32)