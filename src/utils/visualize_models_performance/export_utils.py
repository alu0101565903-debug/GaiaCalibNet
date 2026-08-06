"""
Module: export_utils.py

Utility functions for exporting model predictions and classical baselines 
for further analysis and visualization.

This module provides tools to take a trained PyTorch model, perform inference 
on a given dataset, compute a baseline for comparison, and package all inputs, 
predictions, and references into a single compressed NumPy archive (.npz). 
This decoupling allows the evaluation and plotting scripts to run independently 
of the PyTorch training environment.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch


def export_test_predictions(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    Y_tensor: torch.Tensor,
    X_err_tensor: torch.Tensor,
    output_filepath: str,
    params_tensor: Optional[torch.Tensor] = None,
    wave: Optional[np.ndarray] = None,
    G_mags_tensor: Optional[torch.Tensor] = None,
    X_huang_tensor: Optional[torch.Tensor] = None,
) -> None:
    """
    Run model inference and export predictions, references, and auxiliary arrays to disk.

    The exported `.npz` archive is intended for downstream evaluation and plotting
    routines, decoupling training from the analysis pipeline.

    Args:
        model: Trained PyTorch model that produces either a predicted correction factor
            R(lambda) or PCA coefficients that can be reconstructed to R(lambda).
        X_tensor: Input Gaia BP/RP spectra with shape (num_samples, num_channels).
        Y_tensor: Reference CALSPEC spectra with shape (num_samples, num_channels).
        X_err_tensor: Input flux uncertainties with shape (num_samples, num_channels).
        output_filepath: Destination `.npz` filename for the exported results.
        params_tensor: Optional physical parameters per sample.
        wave: Optional wavelength grid. If omitted, a default 330-1050 nm grid is generated.
        G_mags_tensor: Optional Gaia G magnitudes for each sample.
        X_huang_tensor: Optional baseline predictions from Huang et al. 2024.

    Returns:
        None
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        X_device = X_tensor.to(device)
        params_device = params_tensor.to(device) if params_tensor is not None else None

        if params_device is not None:
            out = model(X_device, params_device)
        else:
            out = model(X_device)

        if hasattr(model, "reconstruct") and callable(getattr(model, "reconstruct")):
            R_pred = model.reconstruct(out)
        else:
            R_pred = out

        X_cal_pinn = X_device * R_pred

    X_np = X_tensor.detach().cpu().numpy()
    Y_np = Y_tensor.detach().cpu().numpy()
    X_err_np = X_err_tensor.detach().cpu().numpy()
    X_cal_pinn_np = X_cal_pinn.detach().cpu().numpy()
    R_pred_np = R_pred.detach().cpu().numpy()
    params_np = params_tensor.detach().cpu().numpy() if params_tensor is not None else None

    if G_mags_tensor is not None:
        G_mags_np = G_mags_tensor.detach().cpu().numpy()
    else:
        G_mags_np = np.zeros((X_np.shape[0],), dtype=np.float32)

    if X_huang_tensor is not None:
        X_huang_np = X_huang_tensor.detach().cpu().numpy()
    else:
        X_huang_np = np.zeros_like(X_np, dtype=np.float32)

    num_samples, num_channels = X_np.shape
    if wave is None:
        wave = np.linspace(330, 1050, num_channels, dtype=np.float32)

    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_dict: Dict[str, np.ndarray] = {
        "wave": wave,
        "X_orig": X_np,
        "Y_ref": Y_np,
        "X_err": X_err_np,
        "X_pinn": X_cal_pinn_np,
        "R_pred": R_pred_np,
        "G_mags": G_mags_np,
        "X_huang": X_huang_np,
    }

    if params_np is not None:
        save_dict["params"] = params_np

    np.savez_compressed(output_filepath, **save_dict)
    print(f"[✓] Inference data exported to: {output_filepath}")
