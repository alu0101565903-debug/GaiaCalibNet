"""
Module: export_utils.py

Utility functions for exporting model predictions and classical baselines 
for further analysis and visualization.

This module provides tools to take a trained PyTorch model, perform inference 
on a given dataset, compute a classical polynomial baseline (Chebyshev) for 
comparison, and package all inputs, predictions, and references into a 
single compressed NumPy archive (.npz). This decoupling allows the evaluation 
and plotting scripts to run independently of the PyTorch training environment.
"""

import os
import numpy as np
import torch
from numpy.polynomial.chebyshev import Chebyshev


def export_test_predictions(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    Y_tensor: torch.Tensor,
    X_err_tensor: torch.Tensor,
    output_filepath: str,
    poly_deg: int = 5,
    wave: np.ndarray = None
):
    """
    Performs model inference and calculates the classic polynomial baseline (Chebyshev),
    saving the complete set of results in a compressed .npz file.

    This function executes a forward pass of the provided PINN model in evaluation 
    mode to generate the calibrated spectra. In parallel, it computes a standard 
    Chebyshev polynomial fit for each individual sample to serve as a traditional 
    performance baseline. All tensors are detached from the computational graph, 
    moved to the CPU, converted to NumPy arrays, and saved to disk.

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
    - poly_deg (int, optional): The degree of the Chebyshev polynomial used for the 
        baseline calculation. Defaults to 5.
    - wave (np.ndarray, optional): The wavelength array corresponding to the spectral 
        channels. If None, a default linear space from 330nm to 1050nm is generated. 
        Defaults to None.

    Returns:
    - None: The function does not return any Python objects. It outputs an archive 
        containing the arrays: "wave", "X_orig", "Y_ref", "X_err", "X_pinn", 
        "X_baseline", and "poly_deg".
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Inferencia del modelo en GPU/CPU
    with torch.no_grad():
        X_device = X_tensor.to(device)
        R_pred = model(X_device)
        X_cal_pinn = X_device * R_pred

    # 2. Direct conversion to NumPy
    X_np = X_tensor.detach().cpu().numpy()
    Y_np = Y_tensor.detach().cpu().numpy()
    X_err_np = X_err_tensor.detach().cpu().numpy()
    X_cal_pinn_np = X_cal_pinn.detach().cpu().numpy()
    
    num_samples, num_channels = X_np.shape
    if wave is None:
        wave = np.linspace(330, 1050, num_channels)
    x_cheb_domain = np.linspace(-1, 1, num_channels)

    # 3. Calculation of the classic Polynomial Baseline per sample
    X_cal_baseline_np = np.zeros_like(X_np)
    for i in range(num_samples):
        ratio_ideal = Y_np[i] / (X_np[i] + 1e-15)
        cheb_fit = Chebyshev.fit(x_cheb_domain, ratio_ideal, deg=poly_deg)
        R_baseline_i = cheb_fit(x_cheb_domain)
        X_cal_baseline_np[i] = X_np[i] * R_baseline_i

    # 4. Safe directory creation
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
        X_baseline=X_cal_baseline_np,
        poly_deg=poly_deg
    )
    print(f"[✓] Inference data exported to: {output_filepath}")
