"""
Module: plot_stellar_dependencies.py
Report Section:

Stellar Physical Parameter Residual and Error Dependence Evaluation.

This module evaluates whether the calibration errors or residuals of the 
Physics-Informed Neural Network (PINN) exhibit systematic biases against 
true physical parameters of stars (G Magnitude, BP-RP Color index, and RUWE).

Key Workflow:
1. Data Reconstruction:
   Re-runs K-Fold splits to pair out-of-fold predictions from each CV fold 
   with the exact corresponding stellar physical parameters.
2. Error Metric Calculation per Star:
   Calculates Mean Absolute Error (MAE) and Peak-Normalized Percentage Residuals 
   individually for each star across all test sets.
3. Diagnostic Visualization:
   Generates a 2x3 grid of scatter plots illustrating MAE and percentage residual 
   distributions against G Magnitude, BP-RP Color, and RUWE astrometric quality.
"""

import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def load_raw_parameters(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load unnormalized physical stellar parameters for plotting axes.

    Reads the raw ".npy" files for G magnitude, BP-RP color index, and RUWE 
    from the target directory to ensure plot axes display physical units 
    rather than Z-score standardized values.

    Parameters
    - data_dir (str): Directory path containing raw parameter NumPy files.

    Returns
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing flattened 1D NumPy arrays:
        (g_mag, bp_rp, ruwe).

    Raises
    - FileNotFoundError: If any of the required parameter files are missing.
    """
    bp_rp_path = os.path.join(data_dir, "bp_rp.npy")
    
    # Handle filename variations across dataset versions
    g_mag_path = os.path.join(data_dir, "G_mags.npy")
    if not os.path.exists(g_mag_path):
        g_mag_path = os.path.join(data_dir, "G_mag.npy")
        
    ruwe_path = os.path.join(data_dir, "ruwe.npy")

    # Validate file existence
    for path in [bp_rp_path, g_mag_path, ruwe_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing stellar parameter file: '{path}'")
        
    bp_rp = np.load(bp_rp_path).flatten()
    g_mag = np.load(g_mag_path).flatten()
    ruwe = np.load(ruwe_path).flatten()
    
    return g_mag, bp_rp, ruwe

def evaluate_stellar_dependencies(
    dataset_mode: str = "all", 
    n_splits: int = 6, 
    random_state: int = 42
) -> None:
    """
    Evaluate PINN residual dependencies on stellar parameters and generate plots.

    Reconstructs the K-Fold validation splits to map each star's prediction error 
    to its exact physical properties (G magnitude, BP-RP color, and RUWE). Generates 
    and saves a composite 2x3 scatter plot diagnostic figure.

    Parameters
    - dataset_mode (str, default="all"): Dataset subset mode to evaluate ("solar", "blue", or "all").
    - n_splits (int, default=6): Number of cross-validation splits used during model training.
    - random_state (int, default=42): Random seed used for KFold splitting to ensure exact fold alignment.
    """
    # Define directory paths
    data_dir = f"data/processed/nn/{dataset_mode}"
    models_dir = f"results/models/pca_nn_cv/{dataset_mode}"
    
    # 1. Load full raw dataset and unnormalized stellar parameters
    print(f"[>] Loading original catalog from: {data_dir}")
    X_np = np.load(os.path.join(data_dir, "X.npy"))
    g_mag_full, bp_rp_full, ruwe_full = load_raw_parameters(data_dir)
    
    # 2. Configure KFold matching crossvalidation.py specifications
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage lists for accumulated out-of-fold metrics and stellar parameters
    all_g_mags, all_bp_rp, all_ruwe, all_mae_pinn, all_res_pct = [], [], [], [], []

    print("[>] Mapping test fold predictions to stellar physical parameters...")
    for fold, (train_index, test_index) in enumerate(kf.split(X_np)):
        fold_idx = fold + 1
        filepath = os.path.join(models_dir, f"fold_{fold_idx}", f"test_predictions_fold_{fold_idx}.npz")
        
        if not os.path.exists(filepath):
            print(f"  [!] Warning: Prediction file missing for Fold {fold_idx}: {filepath}")
            continue
            
        data = np.load(filepath)
        Y_ref = data["Y_ref"]
        X_pinn = data["X_pinn"]
        
        # Extract true unnormalized physical parameters for current test split
        all_g_mags.extend(g_mag_full[test_index])
        all_bp_rp.extend(bp_rp_full[test_index])
        all_ruwe.extend(ruwe_full[test_index])
        
        # Calculate per-star Mean Absolute Error (MAE)
        mae_per_star = np.mean(np.abs(Y_ref - X_pinn), axis=1)
        all_mae_pinn.extend(mae_per_star)
        
        # Calculate per-star Peak-Normalized Relative Residual Percentage (%)
        peak_flux = np.max(Y_ref, axis=1)
        peak_flux = np.where(peak_flux < 1e-15, 1e-15, peak_flux)
        res_pct_per_star = np.mean(((X_pinn - Y_ref) / peak_flux[:, None]) * 100.0, axis=1)
        all_res_pct.extend(res_pct_per_star)

    # Convert accumulated lists to NumPy arrays
    all_g_mags = np.array(all_g_mags)
    all_bp_rp = np.array(all_bp_rp)
    all_ruwe = np.array(all_ruwe)
    all_mae_pinn = np.array(all_mae_pinn)
    all_res_pct = np.array(all_res_pct)
    
    # 3. Generate Diagnostic Scatter Plots
    print("[>] Generating stellar parameter dependency scatter plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"PINN Calibration Error Dependence on Stellar Parameters (Mode: {dataset_mode.upper()})",
        fontsize=16
    )

    # Row 1: Absolute MAE vs. Stellar Physical Parameters
    axes[0, 0].scatter(all_g_mags, all_mae_pinn, alpha=0.7, color="#4f88f3", edgecolor="k")
    axes[0, 0].set_xlabel("G Magnitude")
    axes[0, 0].set_ylabel("Absolute MAE")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    axes[0, 1].scatter(all_bp_rp, all_mae_pinn, alpha=0.7, color="#e63946", edgecolor="k")
    axes[0, 1].set_xlabel("Color (BP - RP)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    axes[0, 2].scatter(all_ruwe, all_mae_pinn, alpha=0.7, color="#00cf49", edgecolor="k")
    axes[0, 2].set_xlabel("RUWE")
    axes[0, 2].grid(True, linestyle="--", alpha=0.5)

    # Row 2: Peak-Normalized Percentage Residuals (%) vs. Stellar Physical Parameters
    axes[1, 0].scatter(all_g_mags, all_res_pct, alpha=0.7, color="#4f88f3", edgecolor="k")
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1, 0].set_xlabel("G Magnitude")
    axes[1, 0].set_ylabel("Mean Residual (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    axes[1, 1].scatter(all_bp_rp, all_res_pct, alpha=0.7, color="#e63946", edgecolor="k")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1, 1].set_xlabel("Color (BP - RP)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    axes[1, 2].scatter(all_ruwe, all_res_pct, alpha=0.7, color="#00cf49", edgecolor="k")
    axes[1, 2].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1, 2].set_xlabel("RUWE")
    axes[1, 2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Export plot
    output_path = os.path.join(models_dir, f"stellar_dependencies_{dataset_mode}.png")
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Diagnostic plot successfully saved to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    # Specify target dataset mode ("solar", "blue", or "all")
    DATASET_MODE = "solar"
    N_SPLITS = 6
    RANDOM_STATE = 42

    evaluate_stellar_dependencies(
        dataset_mode=DATASET_MODE, 
        n_splits=N_SPLITS, 
        random_state=RANDOM_STATE
    )
