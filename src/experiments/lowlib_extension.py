"""
LOWLIB Database Extension and Retraining Script.

Expands the foundational CALSPEC dataset by integrating the LOWLIB white dwarf 
database. Retrains the Physics-Informed Neural Network (PINN) on the augmented 
dataset to evaluate improvements in generalization and spectral feature fidelity.
"""

import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.models.neural_net import CalspecFluxMLP
from src.models.losses import PhysicsInformedLoss

def load_and_parse_dataset(filepath: str) -> pd.DataFrame:
    """Helper function to load and safely parse string arrays from CSV."""
    df = pd.read_csv(filepath)
    df['BP_Coefficients'] = df['BP_Coefficients'].apply(ast.literal_eval)
    df['RP_Coefficients'] = df['RP_Coefficients'].apply(ast.literal_eval)
    df['Target_Flux'] = df['Target_Flux'].apply(ast.literal_eval)
    df['Target_Wavelength_Grid'] = df['Target_Wavelength_Grid'].apply(ast.literal_eval)
    return df

def run_extended_training(calspec_path: str, lowlib_path: str, model_save_path: str, epochs: int = 500):
    """
    Combines CALSPEC and LOWLIB datasets and trains the PINN model.
    """
    print("-" * 60)
    print("[CHAPTER 8] Extended PINN Training with LOWLIB Database...")
    print("-" * 60)

    # 1. Check for datasets
    if not os.path.exists(calspec_path):
        print(f"[!] Error: Missing CALSPEC dataset at {calspec_path}")
        return
        
    if not os.path.exists(lowlib_path):
        print(f"[!] Warning: LOWLIB dataset not found at {lowlib_path}.")
        print("[i] Please ensure you have processed the LOWLIB FITS files into the identical format.")
        return

    # 2. Load and concatenate datasets
    print("[i] Loading and combining CALSPEC and LOWLIB datasets...")
    df_calspec = load_and_parse_dataset(calspec_path)
    df_lowlib = load_and_parse_dataset(lowlib_path)
    
    df_combined = pd.concat([df_calspec, df_lowlib], ignore_index=True)
    print(f"[i] Total augmented sample size: {len(df_combined)} white dwarfs.")

    # 3. Prepare Tensors
    X = np.array([bp + rp for bp, rp in zip(df_combined['BP_Coefficients'], df_combined['RP_Coefficients'])], dtype=np.float32)
    y = np.array(df_combined['Target_Flux'].tolist(), dtype=np.float32)
    wavelength_grid = np.array(df_combined['Target_Wavelength_Grid'].iloc[0], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    lambda_tensor = torch.tensor(wavelength_grid)

    # 4. Initialize Architecture and PINN Loss
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = CalspecFluxMLP(input_dim=input_dim, output_dim=output_dim)
    criterion = PhysicsInformedLoss(wavelength_grid=lambda_tensor, alpha_flux=0.1, beta_grad=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print(f"[i] Starting extended training for {epochs} epochs...")

    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] | Augmented PINN Loss: {loss.item():.6f}")

    # 6. Save final model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\n[✓] LOWLIB extended training complete! Final weights saved to: {model_save_path}")

if __name__ == "__main__":
    base_dataset = "data/processed/final_dataset.csv"
    lowlib_dataset = "data/processed/lowlib_dataset.csv"
    extended_model_out = "results/models/mlp_physics_lowlib.pt"
    
    run_extended_training(base_dataset, lowlib_dataset, extended_model_out)