"""
Module: train_baselines.py
Report Section: Chapter 7 Training Strategies & Baseline Evaluation

This script executes the training and evaluation loop for the classical Machine 
Learning baseline regressors (Ridge, Random Forest, SVR). It establishes a 
quantitative performance threshold to justify the implementation of the complex 
Physics-Informed Neural Network (PINN).
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.models.baselines import get_baseline_models

def evaluate_baselines(mode: str = "spectrum"):
    """
    Loads processed dataset matrices, evaluates classical baseline models, 
    saves them to disk, and prints standard statistical metrics.

    Args:
        mode (str): Training mode, either "spectrum" for full 285D data (280D + 5D features), 
                    or "features" for baseline 5D feature training.
    """
    print("\n" + "=" * 60)
    print(f"[7.0] EXECUTING CLASSICAL BASELINES | MODE: {mode.upper()}")
    print("=" * 60)
    
    is_spectrum = (mode == "spectrum")
    targets_path = "data/processed/Y.npy"
    
    # 1. Path Validation & Data Loading (Mirroring PINN logic)
    if not os.path.exists(targets_path):
        print("[!] Error: Target dataset binary not found.")
        return

    print("-> Loading pre-processed data matrices into memory...")
    if is_spectrum:
        X_spec = np.load("data/processed/X.npy")
        X_feat = np.load("data/processed/X_features.npy")
        X_data = np.concatenate([X_spec, X_feat], axis=1)  # 285D array
    else:
        X_data = np.load("data/processed/X_features.npy")  # 5D array
        
    Y_data = np.load(targets_path)
    
    # 2. Train/Validation Splitting (80/20 ratio as per standard procedures)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    
    print(f"-> Training Set size : {X_train.shape[0]} samples | Features: {X_train.shape[1]}D")
    print(f"-> Testing Set size  : {X_test.shape[0]} samples")
    print("-" * 60)
    
    # 3. Model Initialization
    models = get_baseline_models(random_seed=42)
    
    # Ensure the target directory for saving models exists
    output_dir = f"results/models/baselines/{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Training, Evaluation and Saving Loop
    for model_name, model in models.items():
        print(f"[*] Compiling and fitting: {model_name}...")
        
        # Fit the model to the training manifold
        model.fit(X_train, Y_train)
        
        # Predict continuous channels over the unseen test set
        Y_pred = model.predict(X_test)
        
        # Compute performance metrics
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        
        print(f"    -> Mean Squared Error (MSE) : {mse:.4e}")
        print(f"    -> R-squared (R2) Variance  : {r2:.4f}")
        
        # Format the filename (e.g., "Random_Forest" -> "random_forest.joblib")
        sanitized_name = model_name.lower().replace(" ", "_")
        model_save_path = os.path.join(output_dir, f"{sanitized_name}.joblib")
        
        joblib.dump(model, model_save_path)
        print(f"    [+] Model successfully saved to: {model_save_path}\n")
        
    print("=" * 60)
    print(f">>> Baseline Evaluation Completed for {mode.upper()} mode.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_baselines(mode="spectrum")
    
    if os.path.exists("data/processed/X_features.npy"):
        evaluate_baselines(mode="features")
    else:
        print("[i] Skipping 'features' mode: Dataset not found.")