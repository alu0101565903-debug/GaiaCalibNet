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

def evaluate_baselines(features_path: str, targets_path: str):
    """
    Loads processed dataset matrices, evaluates classical baseline models, 
    saves them to disk, and prints standard statistical metrics.

    Parameters:
    - features_path (str): Path to the normalized features array (X.npy).
    - targets_path (str): Path to the standardized target flux array (Y.npy).
    """
    print("=" * 60)
    print("[7.0] EXECUTING CLASSICAL BASELINE MODELS EVALUATION")
    print("=" * 60)
    
    # 1. Path Validation
    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        print(f"[!] Error: Dataset binaries not found in {os.path.dirname(features_path)}")
        print("Please ensure 'prepare_dataset.py' has been executed prior to training.")
        return

    # 2. Data Loading
    print("-> Loading pre-processed data matrices into memory...")
    X = np.load(features_path)
    Y = np.load(targets_path)
    
    # 3. Train/Validation Splitting (80/20 ratio as per standard procedures)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"-> Training Set size : {X_train.shape[0]} samples")
    print(f"-> Testing Set size  : {X_test.shape[0]} samples")
    print("-" * 60)
    
    # 4. Model Initialization
    models = get_baseline_models(random_seed=42)
    
    # Ensure the target directory for saving models exists
    output_dir = "results/models/baselines"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Training, Evaluation and Saving Loop
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
        # -------------------------------------------
        
    print("=" * 60)
    print(">>> Baseline Evaluation and Saving Completed.")
    print("=" * 60)


if __name__ == "__main__":
    # Standardized paths routing according to the project layout
    X_DATA_PATH = "data/processed/X.npy"
    Y_DATA_PATH = "data/processed/Y.npy"
    
    evaluate_baselines(X_DATA_PATH, Y_DATA_PATH)