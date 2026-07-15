"""
Module: train_nn.py
Report Section: Chapter 7 Training Strategies & PINN Evaluation

Orchestrates the training of the Physics-Informed Neural Network (PINN).
Original version: Full 300-pixel spectrum without edge trimming.
"""

import random
import os
import numpy as np
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from src.models.neural_net import SpectralRecalibrationMLP
from src.train.loss.losses import PhysicsInformedLoss, scaled_physics_wrapper
from src.train.training_functions.adp_training_strategies import train_2stage_adaptive

def set_seed(seed: int = 42):
    """
    Sets the computational seed across all libraries to ensure 
    reproducibility of the results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_network(mode: str = "spectrum"):
    """
    Executes the PINN training and evaluation pipeline.
    
    Args:
        mode (str): Training mode, either "spectrum" for full 305D data (300D + 5D features), 
                    or "features" for baseline 5D feature training.
    """
    print("\n" + "=" * 60)
    print(f"[7.2] INITIATING PINN TRAINING PIPELINE | MODE: {mode.upper()}")
    print("=" * 60)
    
    is_spectrum = (mode == "spectrum")
    targets_path = "data/processed/Y.npy"
    
    # 1. DATA LOADING AND DIMENSION MERGING (Original Full Size)
    if is_spectrum:
        print("-> Loading both Spectrum (300D) and Features (5D) arrays...")
        X_spec = np.load("data/processed/X.npy")
        X_feat = np.load("data/processed/X_features.npy")
        X_data = np.concatenate([X_spec, X_feat], axis=1)  # Master tensor: 305D
        input_dim = 305
    else:
        X_data = np.load("data/processed/X_features.npy")
        input_dim = 5
        
    Y_data = np.load(targets_path)

    # 2. DATA SPLIT (Perfectly aligned arrays)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data, test_size=0.2, random_state=42
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Computation Device allocated: {device}")
    
    save_path = os.path.join("results/models/pinn", f"pinn_{mode}.joblib")
    hidden_architecture = [256, 512, 256]
    
    model = SpectralRecalibrationMLP(
        input_dim=input_dim, 
        output_dim=300,  # Original full output size
        hidden_dims=hidden_architecture, 
        dropout_rate=0.0 if is_spectrum else 0.2
    ).to(device)

    scalers_dict = {}

    # 3. CONDITIONAL TRAINING BYPASS
    if os.path.exists(save_path):
        print(f"[+] Pre-trained model found at: {save_path}")
        checkpoint = joblib.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        scalers_dict = checkpoint["scalers"]
        
        if is_spectrum:
            X_test_spec_scaled = X_test[:, :300] / scalers_dict["Y_max"]
            X_test_feat_scaled = scalers_dict["scaler_feat"].transform(X_test[:, 300:305])
            X_test_scaled = np.concatenate([X_test_spec_scaled, X_test_feat_scaled], axis=1)
        else:
            X_test_scaled = scalers_dict["scaler_X"].transform(X_test)                  
    else:
        print("[-] No pre-trained model found. Proceeding with full training pipeline...")
        
        Y_max = float(np.max(np.abs(Y_train)))
        Y_train_scaled = Y_train / Y_max
        scalers_dict["Y_max"] = Y_max
        
        # --- HYBRID SCALING LOGIC ---
        if is_spectrum:
            X_train_spec, X_train_feat = X_train[:, :300], X_train[:, 300:305]
            X_test_spec, X_test_feat = X_test[:, :300], X_test[:, 300:305]
            
            X_train_spec_scaled = X_train_spec / Y_max
            X_test_spec_scaled = X_test_spec / Y_max
            
            scaler_feat = StandardScaler()
            X_train_feat_scaled = scaler_feat.fit_transform(X_train_feat)
            X_test_feat_scaled = scaler_feat.transform(X_test_feat)
            scalers_dict["scaler_feat"] = scaler_feat
            
            X_train_scaled = np.concatenate([X_train_spec_scaled, X_train_feat_scaled], axis=1)
            X_test_scaled = np.concatenate([X_test_spec_scaled, X_test_feat_scaled], axis=1)
        else:
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            scalers_dict["scaler_X"] = scaler_X
        
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32).to(device), 
                torch.tensor(Y_train_scaled, dtype=torch.float32).to(device)
            ), 
            batch_size=4, shuffle=True
        )
        
        alpha_val = 0.1 if is_spectrum else 0.1
        beta_val = 0.0001 if is_spectrum else 0.1
        
        # Original full wavelength grid
        wave_grid_np = np.linspace(330.0, 1050.0, 300)
        wave_grid_tensor = torch.tensor(wave_grid_np, dtype=torch.float32, device=device)
        
        pinn_loss_engine = PhysicsInformedLoss(
            wavelength_grid=wave_grid_tensor, 
            alpha=alpha_val, beta=beta_val, scale_logcosh=1.0
        )
            
        physics_criterion = lambda m, x, y: scaled_physics_wrapper(m, x, y, pinn_loss_engine)

        X_full_train = train_loader.dataset.tensors[0]
        Y_full_train = train_loader.dataset.tensors[1]
        
        model = train_2stage_adaptive(
            model=model, dataloader=train_loader,
            X_full=X_full_train, Y_full=Y_full_train,
            physics_criterion=physics_criterion, epochs_adam=500, steps_lbfgs=15
        )
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": input_dim,
            "hidden_dims": hidden_architecture,
            "output_dim": 300,
            "scalers": scalers_dict
        }, save_path)
        print(f"    [+] Model & Scalers successfully saved to: {save_path}")

    # 4. EVALUATION LOOP
    print("\n" + "-" * 60)
    print(">>> Evaluating PINN on Validation Set")
    print("-" * 60)
    
    model.eval()
    with torch.no_grad():
        Y_pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        Y_pred = Y_pred_scaled * scalers_dict["Y_max"]
        
    if is_spectrum:
        X_test_descaled = X_test_scaled[:, :300] * scalers_dict["Y_max"]
        print(f"    -> Baseline Gaia MSE (No NN) : {mean_squared_error(Y_test, X_test_descaled):.4e}")
        print(f"    -> Baseline Gaia R2  (No NN) : {r2_score(Y_test.T, X_test_descaled.T, multioutput='uniform_average'):.4f}")
        print("    " + "." * 56)
        
    print(f"    -> Final PINN MSE : {mean_squared_error(Y_test, Y_pred):.4e}")
    print(f"    -> Final PINN R2  : {r2_score(Y_test.T, Y_pred.T, multioutput='uniform_average'):.4f}\n")

if __name__ == "__main__":
    set_seed(42)
    train_network(mode="spectrum")
    if os.path.exists("data/processed/X_features.npy"):
        train_network(mode="features")