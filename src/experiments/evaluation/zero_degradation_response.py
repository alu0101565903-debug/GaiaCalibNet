"""
Module: zero_degradation_response.py

Zero-Degradation Response Test for PCA-Based PINN Architectures.

This script evaluates the physical response of trained `pca_nn` and
`pca_params_nn` architectures when they are fed pristine, undegraded CALSPEC
reference spectra directly as input. The goal is to assess whether the trained
models behave like an identity recalibration filter in the presence of perfect
input spectra, rather than introducing spurious multiplicative corrections.

Usage example:
    python -m src.experiments.evaluation.zero_degradation_response
"""

import os
from typing import Dict, Optional

import joblib
import numpy as np
import torch

from src.models.pinn_architectures import PCASpectralPINN, PCAParamsSpectralPINN
from src.utils.metrics.alternative_metrics import (
    calculate_peak_normalized_mae,
    calculate_normalized_bias_percentage,
    calculate_cosine_similarity,
    calculate_band_errors,
)


def load_stellar_parameters(params_dir: str) -> np.ndarray:
    """Load and standardize auxiliary stellar parameters from a directory."""
    bp_rp_path = os.path.join(params_dir, "bp_rp.npy")
    g_mag_path = os.path.join(params_dir, "G_mags.npy")
    if not os.path.exists(g_mag_path):
        g_mag_path = os.path.join(params_dir, "G_mag.npy")
    ruwe_path = os.path.join(params_dir, "ruwe.npy")

    for path in [bp_rp_path, g_mag_path, ruwe_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing stellar parameter file: {path}")

    bp_rp = np.load(bp_rp_path).reshape(-1, 1)
    g_mag = np.load(g_mag_path).reshape(-1, 1)
    ruwe = np.load(ruwe_path).reshape(-1, 1)

    def standardize(column: np.ndarray) -> np.ndarray:
        return (column - np.mean(column)) / (np.std(column) + 1e-8)

    return np.hstack([
        standardize(bp_rp),
        standardize(g_mag),
        standardize(ruwe),
    ]).astype(np.float32)


def load_pca_model(
    weights_path: str,
    model_name: str,
    input_dim: int,
    num_stellar_params: Optional[int] = None,
) -> torch.nn.Module:
    """Load a trained PCA model from disk and instantiate the corresponding architecture."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    try:
        loaded = joblib.load(weights_path)
    except Exception:
        loaded = torch.load(weights_path, map_location="cpu")

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
        pca_basis = loaded.get("pca_basis")
        pca_mean = loaded.get("pca_mean")
    else:
        raise ValueError(
            "Unsupported weight file format. Expected a dictionary containing "
            "'model_state_dict', 'pca_basis', and 'pca_mean'."
        )

    if pca_basis is None or pca_mean is None:
        raise ValueError("The saved checkpoint must include 'pca_basis' and 'pca_mean'.")

    if model_name == "pca_nn":
        model = PCASpectralPINN(
            input_dim=input_dim,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
        )
    elif model_name == "pca_params_nn":
        if num_stellar_params is None:
            raise ValueError("num_stellar_params is required for pca_params_nn models.")
        model = PCAParamsSpectralPINN(
            input_dim=input_dim,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            num_stellar_params=num_stellar_params,
        )
    else:
        raise ValueError("Unsupported model_name. Choose 'pca_nn' or 'pca_params_nn'.")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_identity_response_metrics(
    y_true: np.ndarray,
    r_pred: np.ndarray,
    x_err: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute metrics that describe how close the model response is to identity."""
    x_zero = y_true * r_pred
    n_wavelengths = y_true.shape[1]
    n_samples = y_true.shape[0]

    r_dev = np.abs(r_pred - 1.0)
    r_mean = float(np.mean(r_pred))
    r_std = float(np.std(r_pred))
    r_mae = float(np.mean(r_dev))
    r_mse = float(np.mean(r_dev**2))
    identity_rate_2pct = float(np.mean(r_dev < 0.02) * 100.0)
    identity_rate_5pct = float(np.mean(r_dev < 0.05) * 100.0)

    pn_mae = calculate_peak_normalized_mae(y_true, x_zero)
    bias_pct = calculate_normalized_bias_percentage(y_true, x_zero)
    cos_sim = calculate_cosine_similarity(y_true, x_zero)

    band_errors = calculate_band_errors(
        np.linspace(330.0, 1050.0, n_wavelengths, dtype=np.float32),
        y_true[0],
        x_zero[0],
    )

    metrics = {
        "R_mean": r_mean,
        "R_std": r_std,
        "R_mae": r_mae,
        "R_mse": r_mse,
        "R_within_2pct_%": identity_rate_2pct,
        "R_within_5pct_%": identity_rate_5pct,
        "flux_peak_normalized_mae_%": pn_mae,
        "flux_normalized_bias_%": bias_pct,
        "flux_cosine_similarity": cos_sim,
        "blue_rel_error_%": band_errors["blue_rel_error_percent"],
        "green_rel_error_%": band_errors["green_rel_error_percent"],
        "red_rel_error_%": band_errors["red_rel_error_percent"],
    }

    if x_err is not None:
        metrics["has_uncertainties"] = 1.0
        metrics["samples"] = float(n_samples)
    else:
        metrics["has_uncertainties"] = 0.0
        metrics["samples"] = float(n_samples)

    return metrics


def run_zero_degradation_test(
    model_name: str,
    weights_path: str,
    y_path: str,
    params_dir: Optional[str] = None,
    x_err_path: Optional[str] = None,
    output_dir: str = "results/evaluation/zero_degradation",
) -> str:
    """Run the zero-degradation response test and export results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.load(y_path).astype(np.float32)
    num_samples, num_channels = y_true.shape

    params_np = None
    if model_name == "pca_params_nn":
        if params_dir is None:
            raise ValueError("params_dir must be provided for pca_params_nn models.")
        params_np = load_stellar_parameters(params_dir)
        if params_np.shape[0] != num_samples:
            raise ValueError(
                f"Stellar parameters sample count {params_np.shape[0]} does not match Y sample count {num_samples}."
            )

    x_err_np = None
    if x_err_path is not None and os.path.exists(x_err_path):
        x_err_np = np.load(x_err_path).astype(np.float32)
        if x_err_np.shape != y_true.shape:
            x_err_np = None

    model = load_pca_model(
        weights_path=weights_path,
        model_name=model_name,
        input_dim=num_channels,
        num_stellar_params=params_np.shape[1] if params_np is not None else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_tensor = torch.tensor(y_true, dtype=torch.float32).to(device)
    params_tensor = torch.tensor(params_np, dtype=torch.float32).to(device) if params_np is not None else None

    with torch.no_grad():
        if params_tensor is not None:
            out = model(y_tensor, params_tensor)
        else:
            out = model(y_tensor)

        if hasattr(model, "reconstruct") and callable(getattr(model, "reconstruct")):
            r_pred = model.reconstruct(out)
        else:
            r_pred = out

        r_pred = r_pred.detach().cpu().numpy().astype(np.float32)

    metrics = compute_identity_response_metrics(
        y_true=y_true,
        r_pred=r_pred,
        x_err=x_err_np,
    )

    report_path = os.path.join(output_dir, f"{model_name}_zero_degradation_metrics.npz")
    np.savez_compressed(
        report_path,
        Y_ref=y_true,
        R_pred=r_pred,
        X_zero=y_true * r_pred,
        **{k: np.array(v, dtype=np.float32) for k, v in metrics.items()},
    )

    summary_lines = [
        f"Zero-Degradation Response Test: {model_name}",
        f"Weights path: {weights_path}",
        f"Reference input shape: {y_true.shape}",
        "",
        "R(lambda) response metrics:",
        f"  mean(R) = {metrics['R_mean']:.6f}",
        f"  std(R) = {metrics['R_std']:.6f}",
        f"  MAE(|R - 1|) = {metrics['R_mae']:.6f}",
        f"  MSE(|R - 1|) = {metrics['R_mse']:.6e}",
        f"  fraction within 2% = {metrics['R_within_2pct_%']:.2f}%",
        f"  fraction within 5% = {metrics['R_within_5pct_%']:.2f}%",
        "",
        "Flux reconstruction metrics:",
        f"  peak-normalized MAE = {metrics['flux_peak_normalized_mae_%']:.4f}%",
        f"  normalized bias = {metrics['flux_normalized_bias_%']:.4f}%",
        f"  cosine similarity = {metrics['flux_cosine_similarity']:.6f}",
        "",
        "Band-relative errors:",
        f"  blue  = {metrics['blue_rel_error_%']:.4f}%",
        f"  green = {metrics['green_rel_error_%']:.4f}%",
        f"  red   = {metrics['red_rel_error_%']:.4f}%",
        "",
        f"Saved compressed results to: {report_path}",
    ]

    summary_path = os.path.join(output_dir, f"{model_name}_zero_degradation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print("\n" + "=" * 70)
    print("Zero-Degradation Response Summary")
    print("=" * 70)
    print("\n".join(summary_lines))
    print("=" * 70 + "\n")

    return report_path


if __name__ == "__main__":
    MODE = "all"
    MODEL_NAME = "pca_nn"  # Choose 'pca_nn' or 'pca_params_nn'

    weights_path = f"results/models/{MODEL_NAME}/{MODE}/pinn_pca_recalibration.joblib"
    y_path = f"data/processed/nn/{MODE}/Y.npy"
    params_dir = f"data/processed/nn/{MODE}" if MODEL_NAME == "pca_params_nn" else None
    x_err_path = f"data/processed/nn/{MODE}/X_error.npy"
    output_dir = f"results/evaluation/zero_degradation/{MODEL_NAME}/{MODE}"

    run_zero_degradation_test(
        model_name=MODEL_NAME,
        weights_path=weights_path,
        y_path=y_path,
        params_dir=params_dir,
        x_err_path=x_err_path,
        output_dir=output_dir,
    )
