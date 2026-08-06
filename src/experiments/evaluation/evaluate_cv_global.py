"""
Module: evaluate_cv_global.py
Report Section

Global Cross-Validation Aggregation and Evaluation Pipeline.
This module consolidates validation predictions across all K-Fold cross-validation 
splits by reading out-of-fold prediction files ("test_predictions_fold_X.npz"). 
It computes global quantitative metrics (Mean Absolute Error, Root Mean Squared Error) 
and peak-normalized percentage residual distributions (mean bias ± standard deviation) 
to evaluate total recalibration performance across the entire dataset without data leakage.

Key Features:
1. Data Aggregation:
   Reads and concatenates out-of-fold reference spectra (Y_ref), calibrated PINN 
   spectra (X_pinn), and uncorrected baseline spectra (X_orig) across all folds.
2. Global Metric Computation:
   Calculates global absolute MAE and RMSE metrics across all wavelength bins and samples.
3. Normalized Residual Analysis:
   Computes scale-invariant percentage residuals scaled by each spectrum's peak flux 
   value to evaluate calibration bias and scatter across diverse brightness levels.
"""

import os
import numpy as np

def aggregate_kfold_results(base_model_dir: str, n_splits: int):
    """
    Consolidate out-of-fold prediction files (.npz) across all cross-validation folds.

    Traverses the directory structure generated during cross-validation, loads each 
    fold's evaluation file ("test_predictions_fold_{fold}.npz"), extracts spectral arrays, 
    and concatenates them along the sample axis (axis 0).

    Parameters
    - base_model_dir (str): Root directory containing cross-validation fold subdirectories 
        (e.g., 'results/models/pca_nn_cv/all').
    - n_splits (int): Total number of cross-validation splits/folds executed during training.

    Returns
        dict
            A dictionary containing consolidated numpy arrays:
            - "Y_ref" : Ground-truth reference target spectra array of shape (N_total, wavelength_dim).
            - "X_pinn": Calibrated spectra predicted by the PINN of shape (N_total, wavelength_dim).
            - "X_orig": Uncalibrated baseline Gaia spectra of shape (N_total, wavelength_dim).
            - "wave"  : 1D array of wavelength values corresponding to spectral bins.

    Raises
    - ValueError: If no fold prediction files could be found or loaded from the specified directory.
    """
    aggregated_data = {
        "Y_ref": [],
        "X_pinn": [],
        "X_orig": [],
        "wave": None
    }
    
    print(f"[>] Searching for fold predictions in: {base_model_dir}")
    
    for fold in range(1, n_splits + 1):
        # Construct exact directory path matching crossvalidation.py outputs
        fold_dir = os.path.join(base_model_dir, f"fold_{fold}")
        filepath = os.path.join(fold_dir, f"test_predictions_fold_{fold}.npz")
        
        if not os.path.exists(filepath):
            print(f"  [!] Warning: Missing prediction file for Fold {fold}: {filepath}")
            continue
            
        print(f"  [✓] Loading Fold {fold}...")
        data = np.load(filepath)
        
        aggregated_data["Y_ref"].append(data["Y_ref"])
        aggregated_data["X_pinn"].append(data["X_pinn"])
        aggregated_data["X_orig"].append(data["X_orig"])
        
        # Store common wavelength grid (assumed uniform across all folds)
        if aggregated_data["wave"] is None:
            aggregated_data["wave"] = data["wave"]
            
    # Concatenate accumulated sample arrays along the dataset sample dimension (axis 0)
    if not aggregated_data["Y_ref"]:
        raise ValueError(
            f"No fold data found to aggregate in '{base_model_dir}'. "
            "Verify directory path and fold count."
        )
        
    final_data = {
        "Y_ref": np.concatenate(aggregated_data["Y_ref"], axis=0),
        "X_pinn": np.concatenate(aggregated_data["X_pinn"], axis=0),
        "X_orig": np.concatenate(aggregated_data["X_orig"], axis=0),
        "wave": aggregated_data["wave"]
    }

    print(f"\n[✓] All folds successfully consolidated. Total validation samples: {final_data['Y_ref'].shape[0]}")
    return final_data

def calculate_global_metrics(consolidated_data: dict):
    """
    Calculate and display comprehensive global statistical error metrics.

    Computes global performance indicators comparing uncalibrated input spectra 
    (X_orig) and calibrated output spectra (X_pinn) against reference standards (Y_ref):
    
    1. Absolute Error Metrics:
       - Mean Absolute Error (MAE): Average absolute flux magnitude error.
       - Root Mean Squared Error (RMSE): Square root of mean squared flux error.
    
    2. Peak-Normalized Relative Residuals:
       - Residual percentage calculated relative to local spectrum peak flux:
         $\text{Residual}_{\%} = \frac{X - Y_{\text{ref}}}{\max(Y_{\text{ref}})} \times 100\%$
       - Evaluates mean offset bias (mean) and residual variance dispersion (std).

    Parameters
    - consolidated_data (dict): Dictionary containing merged arrays ("Y_ref", "X_pinn", "X_orig") generated 
        by "aggregate_kfold_results".
    """
    Y_ref = consolidated_data["Y_ref"]
    X_pinn = consolidated_data["X_pinn"]
    X_orig = consolidated_data["X_orig"]
    
    # 1. Global Absolute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    mae_pinn = np.mean(np.abs(Y_ref - X_pinn))
    rmse_pinn = np.sqrt(np.mean((Y_ref - X_pinn)**2))
    
    mae_orig = np.mean(np.abs(Y_ref - X_orig))
    rmse_orig = np.sqrt(np.mean((Y_ref - X_orig)**2))
    
    # 2. Peak-Normalized Relative Percentage Residuals
    # (Matches visualization normalization logic used in plot_model_comparison.py)
    peak_flux = np.max(Y_ref, axis=1, keepdims=True)
    
    # Avoid zero division or extreme numerical instability on faint noise spectra
    peak_flux = np.where(peak_flux < 1e-15, 1e-15, peak_flux)
    
    res_pinn_pct = ((X_pinn - Y_ref) / peak_flux) * 100.0
    res_orig_pct = ((X_orig - Y_ref) / peak_flux) * 100.0
    
    # Mean (calibration bias) and standard deviation (residual dispersion)
    mean_res_pinn = np.mean(res_pinn_pct)
    std_res_pinn = np.std(res_pinn_pct)
    
    mean_res_orig = np.mean(res_orig_pct)
    std_res_orig = np.std(res_orig_pct)

    # Print formatted summary table to console
    print("\n" + "=" * 60)
    print("        GLOBAL CROSS-VALIDATION PERFORMANCE SUMMARY")
    print("=" * 60)
    print("ABSOLUTE FLUX METRICS:")
    print(f"  MAE Original (Uncalibrated Gaia):  {mae_orig:.4e}")
    print(f"  MAE PINN (Calibrated Model):      {mae_pinn:.4e}")
    print(f"  RMSE Original:                    {rmse_orig:.4e}")
    print(f"  RMSE PINN:                        {rmse_pinn:.4e}")
    print("-" * 60)
    print("PEAK-NORMALIZED PERCENTAGE RESIDUALS (%):")
    print(f"  Original Bias (Mean ± Std):       {mean_res_orig:+.2f}% ± {std_res_orig:.2f}%")
    print(f"  PINN Bias (Mean ± Std):           {mean_res_pinn:+.2f}% ± {std_res_pinn:.2f}%")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Define root directory containing cross-validation fold subfolders and fold split count
    BASE_DIR = "results/models/pca_nn_cv/all"
    N_SPLITS = 6
    
    try:
        consolidated_data = aggregate_kfold_results(BASE_DIR, N_SPLITS)
        calculate_global_metrics(consolidated_data)
    except Exception as error:
        print(f"\n[!] Error encountered during global evaluation: {error}")