"""
Module: train_baselines.py
Report Section: Chapter 7 Training Strategies & Baseline Evaluation

This script trains and evaluates classical machine learning baselines for Gaia-
CALSPEC spectral recalibration. Baseline regressors include Ridge, Random Forest,
and SVR, and each model is trained with the same scaling/descaling protocol used
by the physics-informed neural network pipeline.

The goal is to establish a robust performance benchmark in the same numerical
space as the main PINN model, enabling fair comparison of baseline and neural
recalibration behavior.
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from src.models.baselines import get_baseline_models
from src.utils.alternative_metrics import calculate_mape, calculate_pseudo_chi_square, calculate_band_errors


def evaluate_baselines(mode: str = "spectrum"):
    """
    Loads processed dataset matrices, normalizes inputs/targets matching the PINN,
    evaluates classical baseline models, and saves both models and scalers.

    Parameters:
    - mode (str): Training mode, either 'spectrum' (autoencoder/recalibrator) 
                  or 'features' (stellar parameters mapping).
    """
    print("=" * 60)
    print(f"[7.0] EXECUTING CLASSICAL BASELINE MODELS | MODE: {mode.upper()}")
    print("=" * 60)
    
    # 1. Path and Dimensionality Configuration
    if mode == "spectrum":
        features_path = "data/processed/X.npy"
    elif mode == "features":
        features_path = "data/processed/X_features.npy"
    else:
        raise ValueError("Mode must be either 'spectrum' or 'features'.")
        
    targets_path = "data/processed/Y.npy"
    
    # 2. Path Validation
    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        print(f"[!] Error: Dataset binaries not found for mode '{mode}'.")
        print("Please ensure your preprocessing pipeline has been executed prior to training.")
        return

    # 3. Data Loading
    print("-> Loading pre-processed data matrices into memory...")
    X = np.load(features_path)
    Y = np.load(targets_path)
    
    # 4. Train/Validation split using an 80/20 ratio to mirror the PINN setup.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"-> Training Set size : {X_train.shape[0]} samples")
    print(f"-> Testing Set size  : {X_test.shape[0]} samples")
    print("-" * 60)
    
    # 5. Pipeline Alignment: Coordinated Scaling Protocol
    scalers_dict = {}
    Y_max = float(np.max(np.abs(Y_train)))
    scalers_dict['Y_max'] = Y_max
    
    if mode == "spectrum":
        # Scale the spectrum inputs by the maximum absolute training flux.
        # This keeps the spectrum values in a consistent relative range.
        X_train_scaled = X_train / Y_max
        X_test_scaled = X_test / Y_max
        Y_train_scaled = Y_train / Y_max
    else:
        # Feature mode: standardize stellar parameter inputs with Z-score scaling.
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        scalers_dict['scaler_X'] = scaler_X
        
        # Target spectrum flux values still require linear scaling by Y_max.
        Y_train_scaled = Y_train / Y_max
        
    # 6. Model Initialization
    models = get_baseline_models(random_seed=42)
    
    output_dir = "results/models/baselines"
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. Training, Evaluation and Saving Loop
    for model_name, model in models.items():
        print(f"[*] Fitting classical regressor: {model_name}...")
        
        # Fit the baseline regressor on the scaled training data.
        model.fit(X_train_scaled, Y_train_scaled)
        
        # Predict on the scaled test set.
        Y_pred_scaled = model.predict(X_test_scaled)
        
        # Rescale predictions back to physical flux units for evaluation.
        Y_pred = Y_pred_scaled * Y_max
        
        # Compute alternative domain-specific metrics
        Y_test_flat = Y_test.flatten()
        Y_pred_flat = Y_pred.flatten()
        
        wave_grid_np = np.linspace(330.0, 1050.0, 300)
        wave_flat = np.tile(wave_grid_np, len(Y_test))
        
        mape = calculate_mape(Y_test_flat, Y_pred_flat)
        chi2 = calculate_pseudo_chi_square(Y_test_flat, Y_pred_flat)
        band_errs = calculate_band_errors(wave_flat, Y_test_flat, Y_pred_flat)
        
        print(f"    -> Final MAPE (%)             : {mape:.4f}%")
        print(f"    -> Final Pseudo Chi-Square    : {chi2:.4e}")
        print(f"    -> MAE Blue Band (<500nm)     : {band_errs['blue']:.4e}")
        print(f"    -> MAE Green Band (500-700nm) : {band_errs['green']:.4e}")
        print(f"    -> MAE Red Band (>700nm)      : {band_errs['red']:.4e}")
        
        # Package the fitted model together with the scaling parameters so
        # that the baseline can be safely loaded and used for inference later.
        sanitized_name = model_name.lower().replace(" ", "_")
        model_save_path = os.path.join(output_dir, f"{sanitized_name}_{mode}.joblib")
        
        checkpoint = {
            "model": model,
            "scalers": scalers_dict,
            "mode": mode
        }
        
        joblib.dump(checkpoint, model_save_path)
        print(f"    [+] Saved baseline package to: {model_save_path}\n")
        
    print("=" * 60)
    print(">>> Classical Baseline Evaluation Pipeline Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    # Execute the baseline pipeline sequentially for both supported modes.
    evaluate_baselines(mode="spectrum")
    print("\n")

    if os.path.exists("data/processed/X_features.npy"):
        evaluate_baselines(mode="features")
    else:
        print("[i] Skipping 'features' mode: 'data/processed/X_features.npy' not found.")