"""
Module: train_nn.py
Report Section: Chapter 7 Training Strategies & PINN Evaluation

This script orchestrates the training of the Physics-Informed Neural Network (PINN).
It applies coordinated scaling, executes the robust 2-stage adaptive training pipeline,
and handles seamless descaling during evaluation to guarantee mathematical accuracy.
"""

import os
import numpy as np
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# CORREGIDO: Importamos la clase exacta que modificamos con la física multiplicativa
from src.models.neural_net import SpectralRecalibrationMLP
from src.train.loss.losses import PhysicsInformedLoss, adaptive_physics_wrapper
from src.train.training_functions.adp_training_strategies import train_2stage_adaptive

def train_network(mode: str = "spectrum"):
    """
    Executes or skips the training pipeline for the PINN based on whether a 
    pre-trained joblib checkpoint exists. It then evaluates the model on the test set.
    """
    print("=" * 60)
    print(f"[7.2] INITIATING PINN TRAINING PIPELINE | MODE: {mode.upper()}")
    print("=" * 60)
    
    # 1. Path and Dimensionality Configuration
    if mode == "spectrum":
        features_path = "data/processed/X.npy"
        input_dim = 300
    elif mode == "features":
        features_path = "data/processed/X_features.npy"
        input_dim = 5
    else:
        raise ValueError("Mode must be either 'spectrum' or 'features'.")
        
    targets_path = "data/processed/Y.npy"
    
    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        print(f"[!] Error: Dataset binaries not found for mode '{mode}'.")
        return

    # 2. Data Loading and Splitting
    X = np.load(features_path)
    Y = np.load(targets_path)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Computation Device allocated: {device}")
    
    output_dir = "results/models/pinn"
    save_path = os.path.join(output_dir, f"pinn_{mode}.joblib")
    
    # Configuración dinámica de capas ocultas
    hidden_architecture = [256, 512, 256]
    
    # Inicializamos la arquitectura corregida
    model = SpectralRecalibrationMLP(
        input_dim=input_dim, 
        output_dim=300, 
        hidden_dims=hidden_architecture, 
        dropout_rate=0.2
    ).to(device)

    # Diccionario contenedor para el mapeo de los escaladores
    scalers_dict = {}

    # CONDITIONAL TRAINING BYPASS
    if os.path.exists(save_path):
        print(f"[+] Pre-trained model found at: {save_path}")
        print("-> Loading trained weights and scalers via joblib. Skipping training stage entirely...")
        
        checkpoint = joblib.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # CORRECCIÓN CRÍTICA: Recuperamos los escaladores del archivo para que la evaluación funcione
        scalers_dict = checkpoint['scalers']
        
        # Aplicamos la transformación de test usando los mismos parámetros históricos
        if mode == "spectrum":
            X_test_scaled = X_test / scalers_dict['Y_max']
        else:
            X_test_scaled = scalers_dict['scaler_X'].transform(X_test)
                 
    else:
        print("[-] No pre-trained model found. Proceeding with full training pipeline...")
        
        # CORRECCIÓN DE ESCALA: Alineación para la física residual-multiplicativa
        if mode == "spectrum":
            # Escalamos AMBOS catálogos por el Y_max del set de entrenamiento.
            # Así, si la red predice un factor de 1.0, el flujo predicho coincide exactamente con Gaia.
            Y_max = float(np.max(np.abs(Y_train)))
            X_train_scaled = X_train / Y_max
            X_test_scaled = X_test / Y_max
            Y_train_scaled = Y_train / Y_max
            scalers_dict['Y_max'] = Y_max
        else:
            # Para el modo features (parámetros estelares escalamos la X estándar)
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            scalers_dict['scaler_X'] = scaler_X
            
            # El target Y sigue requiriendo escalado global de flujo absoluto
            Y_max = float(np.max(np.abs(Y_train)))
            Y_train_scaled = Y_train / Y_max
            scalers_dict['Y_max'] = Y_max
        
        # Prepare Tensors and DataLoader
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        Y_train_t = torch.tensor(Y_train_scaled, dtype=torch.float32).to(device)
        
        train_dataset = TensorDataset(X_train_t, Y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Setup physics loss evaluation engine
        wave_grid = torch.linspace(330.0, 1050.0, 300, device=device)
        pinn_loss_engine = PhysicsInformedLoss(wavelength_grid=wave_grid)
        physics_criterion = lambda m, x, y: adaptive_physics_wrapper(m, x, y, pinn_loss_engine)
        
        # Execute 2-Stage Adaptive Training (Adam + LBFGS)
        model = train_2stage_adaptive(
            model=model,
            dataloader=train_loader,
            X_full=X_train_t,
            Y_full=Y_train_t,
            physics_criterion=physics_criterion,
            epochs_adam=500,
            steps_lbfgs=10
        )
            
        # Serialize checkpoint mapping
        os.makedirs(output_dir, exist_ok=True)
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        
        checkpoint = {
            'model_state_dict': state_dict_cpu,
            'input_dim': input_dim,
            'hidden_dims': hidden_architecture, # CORREGIDO: Guardamos las dimensiones reales entrenadas
            'output_dim': 300,
            'scalers': scalers_dict
        }
        
        joblib.dump(checkpoint, save_path)
        print(f"    [+] Model & Scalers successfully saved via joblib to: {save_path}")

    # EVALUATION LOOP
    print("\n" + "=" * 60)
    print(">>> Evaluating PINN on Validation Set")
    print("=" * 60)
    
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    model.eval()
    
    with torch.no_grad():
        Y_pred_scaled_t = model(X_test_t)
        Y_pred_scaled = Y_pred_scaled_t.cpu().numpy()
        
        # CORRECCIÓN MAESTRA: Desescalamos multiplicando la predicción de la red por su factor original
        Y_pred = Y_pred_scaled * scalers_dict['Y_max']
        
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    
    print(f"    -> Final Mean Squared Error (MSE) : {mse:.4e}")
    print(f"    -> Final R-squared (R2) Variance  : {r2:.4f}\n")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Nota técnica: El modo "spectrum" aprovecha la potencia de la arquitectura multiplicativa residual
    train_network(mode="spectrum")
    print("\n")
    
    if os.path.exists("data/processed/X_features.npy"):
        # Nota técnica: Si ejecutas el modo features, recuerda que al no tener espectro base en X,
        # la red actuará directamente estimando flujos sobre la base normalizada.
        train_network(mode="features")
    else:
        print("[i] Skipping 'features' mode: 'data/processed/X_features.npy' not found.")