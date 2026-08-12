"""
Module: plot_stellar_dependencies.py
Report Section:

Stellar Physical Parameter Residual and Error Dependence Evaluation.
This module evaluates whether calibration errors or residuals from a 
Physics-Informed Neural Network (PINN) exhibit systematic biases against 
the true physical parameters of stars (Gaia G Magnitude, BP-RP Color index, 
and RUWE astrometric quality factor).

Key Workflow:
1. Data Reconstruction:
   Reconstructs K-Fold validation splits to map out-of-fold predictions 
   from each cross-validation fold to its corresponding stellar physical parameters.
2. Per-Star Error Metric Calculation:
   Computes Mean Absolute Error (MAE) and Peak-Normalized Percentage Residuals 
   for each star across test folds.
3. Diagnostic Visualization:
   Generates a 2x3 composite grid of scatter plots depicting MAE and percentage 
   residual distributions against G Magnitude, BP-RP Color, and RUWE.
"""

import os
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def load_raw_parameters(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load unnormalized physical stellar parameters for plotting axes.

    Reads the raw binary (.npy) files for G magnitude, BP-RP color index, and RUWE 
    from the target directory. This ensures plot axes display physical observational 
    units rather than Z-score standardized values.

    Parameters
    ----------
    data_dir : str
        Directory path containing the raw parameter NumPy files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of flattened 1D NumPy arrays containing:
        - g_mag: Gaia G-band magnitude.
        - bp_rp: Integrated BP - RP color index.
        - ruwe: Renormalised Unit Weight Error (astrometric quality).

    Raises
    ------
    FileNotFoundError
        If any of the required parameter files are missing from data_dir.
    """
    bp_rp_path = os.path.join(data_dir, "bp_rp.npy")
    
    # Handle filename variations across dataset versions
    g_mag_path = os.path.join(data_dir, "G_mags.npy")
    if not os.path.exists(g_mag_path):
        g_mag_path = os.path.join(data_dir, "G_mag.npy")
        
    ruwe_path = os.path.join(data_dir, "ruwe.npy")

    # Validate file existence existence prior to loading
    for path in [bp_rp_path, g_mag_path, ruwe_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing stellar parameter file: '{path}'")
        
    bp_rp = np.load(bp_rp_path).flatten()
    g_mag = np.load(g_mag_path).flatten()
    ruwe = np.load(ruwe_path).flatten()
    
    return g_mag, bp_rp, ruwe


def evaluate_stellar_dependencies(
    dataset_mode: str = "all", 
    nn_mode: str = "pca_nn",
    n_splits: int = 5, 
    random_state: int = 42,
    models_dir: Optional[str] = None,
    data_dir: Optional[str] = None
) -> None:
    """
    Evaluate PINN residual dependencies on stellar parameters and export diagnostic plots.

    Reconstructs K-Fold validation splits to pair out-of-fold model predictions 
    with individual stellar physical properties (G magnitude, BP-RP color, and RUWE). 
    Generates and saves a publication-ready 2x3 composite diagnostic figure.

    Parameters
    ----------
    dataset_mode : str, default="all"
        Dataset partition key ("solar", "blue", or "all").
    nn_mode : str, default="pca_nn"
        PINN architecture identifier key.
    n_splits : int, default=5
        Number of cross-validation splits used during model training.
    random_state : int, default=42
        Random seed used for KFold splitting to guarantee fold alignment.
    models_dir : str, optional
        Custom root directory containing model fold predictions. If None, resolves automatically.
    data_dir : str, optional
        Custom directory containing dataset features and stellar parameters. If None, resolves automatically.
    """
    # 1. Resolve output models directory
    if models_dir is None:
        models_dir = f"results/cross_validation/{nn_mode}_cv/{dataset_mode}"

    # 2. Resolve input data directory with adaptive fallback options
    if data_dir is None:
        possible_data_dirs = [
            f"data/processed/nn_undegraded/{dataset_mode}",
            f"data/processed/nn/{dataset_mode}",
            "data/processed"
        ]
        data_dir = next((p for p in possible_data_dirs if os.path.exists(p)), possible_data_dirs[0])

    print(f"[>] Loading original catalog from: {data_dir}")
    
    x_file = os.path.join(data_dir, "X.npy")
    if not os.path.exists(x_file):
        print(f"[!] Warning: Input dataset not found at {x_file}. Skipping dependency evaluation.")
        return

    X_np = np.load(x_file)
    try:
        g_mag_full, bp_rp_full, ruwe_full = load_raw_parameters(data_dir)
    except FileNotFoundError as e:
        print(f"[!] Warning: Could not load raw stellar parameters: {e}. Skipping dependency evaluation.")
        return

    # 3. Configure KFold matching cross-validation specifications
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_g_mags, all_bp_rp, all_ruwe, all_mae_pinn, all_res_pct = [], [], [], [], []

    print(f"[>] Mapping test fold predictions from '{models_dir}' to stellar parameters...")
    for fold, (train_index, test_index) in enumerate(kf.split(X_np)):
        fold_idx = fold + 1
        
        # Support multiple file naming conventions across pipelines
        possible_filepaths = [
            os.path.join(models_dir, f"fold_{fold_idx}", "test_predictions.npz"),
            os.path.join(models_dir, f"fold_{fold_idx}", f"test_predictions_fold_{fold_idx}.npz")
        ]
        filepath = next((p for p in possible_filepaths if os.path.exists(p)), None)
        
        if filepath is None:
            print(f"  [!] Warning: Prediction file missing for Fold {fold_idx} in {os.path.join(models_dir, f'fold_{fold_idx}')}")
            continue
            
        data = np.load(filepath)
        
        # Flexibly extract reference spectra and model predictions from npz archives
        Y_ref = data["Y_ref"] if "Y_ref" in data else data.get("Y_true", data.get("Y_test", None))
        X_pinn = data["X_pinn"] if "X_pinn" in data else data.get("Y_pred", data.get("predictions", None))
        
        if Y_ref is None or X_pinn is None:
            print(f"  [!] Warning: Could not find ground truth or prediction keys in {filepath}")
            continue

        # Map unnormalized physical parameters to current test partition
        all_g_mags.extend(g_mag_full[test_index])
        all_bp_rp.extend(bp_rp_full[test_index])
        all_ruwe.extend(ruwe_full[test_index])
        
        # Calculate per-star Mean Absolute Error (MAE) across wavelengths
        mae_per_star = np.mean(np.abs(Y_ref - X_pinn), axis=1)
        all_mae_pinn.extend(mae_per_star)
        
        # Calculate per-star Peak-Normalized Relative Residual Percentage (%)
        peak_flux = np.max(Y_ref, axis=1)
        peak_flux = np.where(peak_flux < 1e-15, 1e-15, peak_flux)
        res_pct_per_star = np.mean(((X_pinn - Y_ref) / peak_flux[:, None]) * 100.0, axis=1)
        all_res_pct.extend(res_pct_per_star)

    # Check if any predictions were successfully collected
    if not all_g_mags:
        print("[!] No valid fold prediction files found. Skipping plot generation.")
        return

    # Convert accumulated lists to NumPy arrays
    all_g_mags = np.array(all_g_mags)
    all_bp_rp = np.array(all_bp_rp)
    all_ruwe = np.array(all_ruwe)
    all_mae_pinn = np.array(all_mae_pinn)
    all_res_pct = np.array(all_res_pct)
    
    # 4. Generate Diagnostic Scatter Plots
    print("[>] Generating stellar parameter dependency scatter plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"PINN Calibration Error Dependence on Stellar Parameters (Model: {nn_mode} | Mode: {dataset_mode.upper()})",
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
    
    # 5. Export plot with directory safety check
    os.makedirs(models_dir, exist_ok=True)
    output_path = os.path.join(models_dir, f"stellar_dependencies_{dataset_mode}.png")
    
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Diagnostic plot successfully saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Specify target dataset mode ("solar", "blue", or "all")
    DATASET_MODE = "blue"
    NN_MODE = "cheb_undegraded_nn"
    N_SPLITS = 5
    RANDOM_STATE = 42

    evaluate_stellar_dependencies(
        dataset_mode=DATASET_MODE, 
        nn_mode=NN_MODE,
        n_splits=N_SPLITS, 
        random_state=RANDOM_STATE
    )