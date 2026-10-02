"""Export model predictions and baselines for evaluation and visualization.

The module packages inputs, predictions, references, and metadata in compressed
NumPy archives so evaluation tools can run independently of model training.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch


def save_prediction_archive(
    output_filepath: str,
    wave: np.ndarray,
    X_orig: np.ndarray,
    Y_ref: np.ndarray,
    X_pinn: np.ndarray,
    X_err: Optional[np.ndarray] = None,
    params: Optional[np.ndarray] = None,
    G_mags: Optional[np.ndarray] = None,
    X_huang: Optional[np.ndarray] = None,
    R_pred: Optional[np.ndarray] = None,
) -> None:
    """Save predictions and metadata in the shared compressed archive format.

    Parameters
    ----------
    output_filepath : str
        Destination path for the compressed NumPy archive.
    wave : numpy.ndarray
        Wavelength grid in nanometers.
    X_orig : numpy.ndarray
        Original Gaia spectra.
    Y_ref : numpy.ndarray
        Reference spectra.
    X_pinn : numpy.ndarray
        PINN-calibrated spectra.
    X_err : numpy.ndarray, optional
        Input flux uncertainties.
    params : numpy.ndarray, optional
        Stellar feature values.
    G_mags : numpy.ndarray, optional
        Gaia G-band magnitudes.
    X_huang : numpy.ndarray, optional
        Baseline spectra from Huang et al. (2024).
    R_pred : numpy.ndarray, optional
        Predicted calibration ratios.

    Returns
    -------
    None
    """
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_samples = X_orig.shape[0]
    save_dict: Dict[str, np.ndarray] = {
        "wave": wave,
        "X_orig": X_orig,
        "Y_ref": Y_ref,
        "X_pinn": X_pinn,
        "X_err": X_err if X_err is not None else np.full_like(X_orig, 1e-4, dtype=np.float32),
        "G_mags": G_mags if G_mags is not None else np.zeros(num_samples, dtype=np.float32),
    }
    if params is not None:
        save_dict["params"] = params
    if X_huang is not None:
        save_dict["X_huang"] = X_huang
    if R_pred is not None:
        save_dict["R_pred"] = R_pred

    np.savez_compressed(output_filepath, **save_dict)


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

    Parameters
    ----------
    model : torch.nn.Module
        Model that predicts a correction ratio or PCA coefficients for that ratio.
    X_tensor : torch.Tensor
        Input Gaia BP/RP spectra with shape ``(num_samples, num_channels)``.
    Y_tensor : torch.Tensor
        Reference CALSPEC spectra with shape ``(num_samples, num_channels)``.
    X_err_tensor : torch.Tensor
        Input flux uncertainties with shape ``(num_samples, num_channels)``.
    output_filepath : str
        Destination path for the exported ``.npz`` file.
    params_tensor : torch.Tensor, optional
        Physical parameters for each sample.
    wave : numpy.ndarray, optional
        Wavelength grid. If omitted, a 330-1050 nm grid is generated.
    G_mags_tensor : torch.Tensor, optional
        Gaia G magnitudes for each sample.
    X_huang_tensor : torch.Tensor, optional
        Baseline predictions from Huang et al. (2024).

    Returns
    -------
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

    save_prediction_archive(
        output_filepath=output_filepath,
        wave=wave,
        X_orig=X_np,
        Y_ref=Y_np,
        X_pinn=X_cal_pinn_np,
        X_err=X_err_np,
        params=params_np,
        G_mags=G_mags_np,
        X_huang=X_huang_np,
        R_pred=R_pred_np,
    )
    print(f"[✓] Inference data exported to: {output_filepath}")
