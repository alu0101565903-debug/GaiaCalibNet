"""
Module: plot_r_diagnostics.py
Report Section

Diagnostic Visualization Pipeline for Predicted Multiplicative Correction Factors R(lambda).

This module processes exported test result files (.npz) from model inference routines 
and generates diagnostic visualizations for the multiplicative spectral recalibration 
factor R(lambda).

Key Diagnostics:
1. Per-Star Empirical vs. Predicted Correction Curves:
   Compares the continuous, regularized correction factor R(lambda) predicted by the 
   Physics-Informed Neural Network (PINN) against the raw, point-by-point empirical ratio 
   R_ideal = Y_ref / X_orig (which contains photometric noise).
2. Multi-Spectrum Overlay Plot:
   Overlays all predicted R(lambda) curves across the test sample to evaluate global 
   trends, systematic features, and dispersion as a function of wavelength.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_r_curves(npz_path: str, save_dir: str = "plots_r_diagnostics") -> None:
    """
    Load test set evaluation predictions and generate R(lambda) diagnostic plots.

    Extracts spectral arrays, calculates empirical flux ratios, isolates predicted 
    correction profiles, and renders both individual star comparison plots and a consolidated 
    overlay diagnostic plot.

    Parameters
    ----------
    npz_path : str
        Filepath to the saved test prediction file (.npz).
    save_dir : str, default="plots_r_diagnostics"
        Directory where generated diagnostic plots will be written.

    Raises
    - FileNotFoundError: If the specified ".npz" file does not exist.
    - KeyError: If required keys ("wave", "X_orig", "Y_ref") are missing from the array dictionary.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Specified result file not found: '{npz_path}'")

    print(f"[>] Loading test prediction artifacts from: {npz_path}")
    data = np.load(npz_path)

    # Validate required data keys
    required_keys = {"wave", "X_orig", "Y_ref"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise KeyError(f"Missing required keys in .npz file: {missing_keys}")
    
    wave = data["wave"]
    X_orig = data["X_orig"]
    Y_ref = data["Y_ref"]
    
    # Extract predicted correction factor or compute dynamically from model predictions
    if "R_pred" in data:
        R_pred = data["R_pred"]
    elif "X_pinn" in data:
        X_pinn = data["X_pinn"]
        R_pred = X_pinn / (X_orig + 1e-15)
    else:
        raise KeyError("Neither 'R_pred' nor 'X_pinn' found in the prediction archive.")

    num_samples = X_orig.shape[0]
    os.makedirs(save_dir, exist_ok=True)
    print(f"[>] Generating individual R(lambda) diagnostic plots for {num_samples} stars...")

    # -------------------------------------------------------------
    # DIAGNOSTIC PLOT 1: Individual R(lambda) Curves per Star
    # -------------------------------------------------------------
    for i in range(num_samples):
        # Compute noisy empirical point-by-point ratio profile
        r_ideal = Y_ref[i] / (X_orig[i] + 1e-15)
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        # Plot point-by-point empirical flux ratio (includes Gaia photometric noise)
        ax.plot(wave, r_ideal, color="gray", alpha=0.4, linewidth=1, label="Empirical Ratio (Y_ref / X_orig)")
        
        # Plot continuous PINN-predicted correction factor curve R(lambda)
        ax.plot(wave, R_pred[i], color="crimson", linewidth=2, label="PINN Predicted R(λ)")
        
        ax.set_title(f"Blind Test Evaluation | Recalibration Factor R(λ) | Star Index: {i:03d}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavelength [nm]", fontsize=11)
        ax.set_ylabel("Correction Factor R(λ)", fontsize=11)

        # Determine dynamic y-axis limits with small padding around data range
        valid_r = r_ideal[np.isfinite(r_ideal)]
        if len(valid_r) > 0:
            y_min, y_max = np.percentile(valid_r, [1, 99])
            ax.set_ylim(max(0.2, y_min - 0.1), min(2.5, y_max + 0.1))
        else:
            ax.set_ylim(0.8, 1.2)

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"R_curve_star_{i:03d}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"[✓] Individual R(λ) plots saved in: {save_dir}/")

    # -------------------------------------------------------------
    # DIAGNOSTIC PLOT 2: Overlay of All Predicted R(lambda) Curves
    # -------------------------------------------------------------
    print("[>] Generating aggregated overlay plot of all correction curves...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Render all star predictions on a single canvas
    for i in range(num_samples):
        # Truncate legend entries if sample size is large to prevent plot occlusion
        label = f"Star {i:03d}" if num_samples <= 12 else None
        ax.plot(wave, R_pred[i], linewidth=1.5, alpha=0.7, label=label)

    ax.axhline(1.0, color="black", linestyle=":", alpha=0.7, label="Identity (R=1)")
    ax.set_title("Overlay of Predicted R(λ) Across All Blind Test Stars", fontsize=12, fontweight="bold")
    ax.set_xlabel("Wavelength [nm]", fontsize=11)
    ax.set_ylabel("Correction Factor R(λ)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    if num_samples <= 12:
        ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    
    plt.tight_layout()
    overlay_path = os.path.join(save_dir, "R_curves_overlay.png")
    plt.savefig(overlay_path)
    plt.close()
    
    print(f"[✓] Overlay plot saved as: {overlay_path}")

if __name__ == "__main__":
    # Define file paths for prediction source and plot export output
    PATH_TO_NPZ = "results/models/pca_nn/blue/test_predictions_final.npz"
    save_directory = "results/models/pca_nn/blue/plots_r_diagnostics"
    
    if os.path.exists(PATH_TO_NPZ):
        plot_r_curves(npz_path=PATH_TO_NPZ, save_dir=save_directory)
    else:
        print(f"[!] Warning: Prediction file not found at '{PATH_TO_NPZ}'. Please update the path.")