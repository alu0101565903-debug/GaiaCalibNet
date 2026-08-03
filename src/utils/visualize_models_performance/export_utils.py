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
import numpy as np
import torch

def export_test_predictions(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    Y_tensor: torch.Tensor,
    X_err_tensor: torch.Tensor,
    output_filepath: str,
    wave: np.ndarray = None,
    G_mags_tensor: torch.Tensor = None,
    X_huang_tensor: torch.Tensor = None
):
    """
    Performs universal model inference and exports predictions and references to a .npz file.

    This function automatically detects the output format of the given PyTorch model:
      - Coefficient-based models with a .reconstruct() method (e.g., PCA, Chebyshev).
      - Models predicting the correction factor R(lambda) directly.
      - Models predicting the fully calibrated spectrum directly.
      
    Args:
    - model (torch.nn.Module): The trained neural network model (e.g., SpectralPINN) 
        used to predict the calibration factor.
    - X_tensor (torch.Tensor): The uncalibrated input spectra (Gaia BP/RP). 
        Expected shape: (num_samples, num_channels).
    - Y_tensor (torch.Tensor): The reference ground-truth spectra (CALSPEC). 
        Expected shape: (num_samples, num_channels).
    - X_err_tensor (torch.Tensor): The flux errors associated with the input spectra, 
        used later for chi-squared metrics. Expected shape: (num_samples, num_channels).
    - output_filepath (str): The full destination path (including filename and .npz 
        extension) where the results will be saved. Parent directories are created 
        automatically if they do not exist.
    - wave (np.ndarray, optional): The wavelength array corresponding to the spectral 
        channels. If None, a default linear space from 330nm to 1050nm is generated. 
        Defaults to None.
    - G_mags_tensor (torch.Tensor, optional): The Gaia G magnitudes for each spectrum.
        Required for downstream visualization baselines (e.g., Huang et al. 2024).
    - X_huang_tensor (torch.Tensor, optional): The Huang et al. 2024 baseline spectra.
        Required for downstream visualization comparisons.

    Returns:
    - None: The function does not return any Python objects. It outputs an archive 
        containing the arrays: "wave", "X_orig", "Y_ref", "X_err", "X_pinn", "G_mags", "X_huang".
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Model inference on GPU/CPU
    with torch.no_grad():
        X_device = X_tensor.to(device)
        R_pred = model(X_device)
        X_cal_pinn = X_device * R_pred

    # 2. Direct conversion to NumPy
    X_np = X_tensor.detach().cpu().numpy()
    Y_np = Y_tensor.detach().cpu().numpy()
    X_err_np = X_err_tensor.detach().cpu().numpy()
    X_cal_pinn_np = X_cal_pinn.detach().cpu().numpy()

    if G_mags_tensor is not None:
        G_mags_np = G_mags_tensor.detach().cpu().numpy()
    else:
        G_mags_np = np.zeros(X_np.shape[0], dtype=np.float32)

    if X_huang_tensor is not None:
        X_huang_np = X_huang_tensor.detach().cpu().numpy()
    else:
        X_huang_np = np.zeros_like(X_np, dtype=np.float32)
        
    num_samples, num_channels = X_np.shape
    if wave is None:
        wave = np.linspace(330, 1050, num_channels)

    # 3. Safe directory creation
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        output_filepath,
        wave=wave,
        X_orig=X_np,
        Y_ref=Y_np,
        X_err=X_err_np,
        X_pinn=X_cal_pinn_np,
        G_mags=G_mags_np,
        X_huang=X_huang_np
    )
    print(f"[✓] Inference data exported to: {output_filepath}")
