"""
Evaluation and Visualization Script for Trained MLPs.

Loads the test dataset and the trained model weights (Standard vs. PINN),
computes performance metrics (RMSE, MAE), and generates comparative plots.
"""

import argparse
import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from src.models.neural_net import CalspecFluxMLP
from src.processing.prepare_dataset import WAVELENGTH_GRID
from src.training.losses import PhysicsInformedLoss

DEFAULT_WAVELENGTH_GRID = WAVELENGTH_GRID.astype(np.float32)


def load_evaluation_dataset(dataset_path: str):
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        df['BP_Coefficients'] = df['BP_Coefficients'].apply(ast.literal_eval)
        df['RP_Coefficients'] = df['RP_Coefficients'].apply(ast.literal_eval)
        df['Target_Flux'] = df['Target_Flux'].apply(ast.literal_eval)
        df['Target_Wavelength_Grid'] = df['Target_Wavelength_Grid'].apply(ast.literal_eval)

        X = np.array([bp + rp for bp, rp in zip(df['BP_Coefficients'], df['RP_Coefficients'])], dtype=np.float32)
        y = np.array(df['Target_Flux'].tolist(), dtype=np.float32)
        wavelength_grid = np.array(df['Target_Wavelength_Grid'].iloc[0], dtype=np.float32)
        names = df['Name'].values
        return X, y, wavelength_grid, names

    npy_dir = os.path.join('data', 'processed')
    x_path = os.path.join(npy_dir, 'X.npy')
    y_path = os.path.join(npy_dir, 'Y.npy')
    names_path = os.path.join(npy_dir, 'star_names.npy')

    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32)
        names = np.load(names_path) if os.path.exists(names_path) else np.array([f"star_{i}" for i in range(len(X))])
        return X, y, DEFAULT_WAVELENGTH_GRID, names

    raise FileNotFoundError(
        f"Dataset not found. Tried CSV at '{dataset_path}' and numpy arrays at '{x_path}'/'{y_path}'."
    )


def train_standard_model(X_train: np.ndarray, y_train: np.ndarray, model_save_path: str, epochs: int = 100):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = CalspecFluxMLP(input_dim=input_dim, output_dim=output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    X_tensor = torch.tensor(X_train)
    y_tensor = torch.tensor(y_train)

    print(f"[i] Training standard model for {epochs} epochs as '{model_save_path}' does not exist.")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[✓] Saved standard model weights to: {model_save_path}")
    return model


def train_physics_model(X_train: np.ndarray, y_train: np.ndarray, wavelength_grid: np.ndarray, model_save_path: str, epochs: int = 100):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = CalspecFluxMLP(input_dim=input_dim, output_dim=output_dim)
    wavelength_tensor = torch.tensor(wavelength_grid)
    criterion = PhysicsInformedLoss(wavelength_grid=wavelength_tensor, alpha=0.1, beta=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    X_tensor = torch.tensor(X_train)
    y_tensor = torch.tensor(y_train)

    print(f"[i] Training physics-informed model for {epochs} epochs as '{model_save_path}' does not exist.")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[✓] Saved physics-informed model weights to: {model_save_path}")
    return model


def evaluate_models(dataset_path: str, std_model_path: str, phys_model_path: str, plots_dir: str):
    """
    Evaluates both neural networks and plots a spectral comparison.
    """
    print("-" * 60)
    print("[PHASE 4] Model Evaluation and Plot Generation...")
    print("-" * 60)

    # 1. Load and parse the dataset (supports CSV or precomputed NumPy arrays)
    X, y, wavelength_grid, names = load_evaluation_dataset(dataset_path)

    # Split to get the exact same test set as training scripts
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, random_state=42
    )

    X_test_tensor = torch.tensor(X_test)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    # 2. Load Models
    if os.path.exists(std_model_path):
        model_std = CalspecFluxMLP(input_dim, output_dim)
        model_std.load_state_dict(torch.load(std_model_path))
        model_std.eval()
    else:
        model_std = train_standard_model(X_train, y_train, std_model_path)
        model_std.eval()

    if os.path.exists(phys_model_path):
        model_phys = CalspecFluxMLP(input_dim, output_dim)
        model_phys.load_state_dict(torch.load(phys_model_path))
        model_phys.eval()
    else:
        model_phys = train_physics_model(X_train, y_train, wavelength_grid, phys_model_path)
        model_phys.eval()

    # 3. Generate Predictions
    with torch.no_grad():
        preds_std = model_std(X_test_tensor).numpy()
        preds_phys = model_phys(X_test_tensor).numpy()

    # 4. Calculate Metrics
    rmse_std = np.sqrt(mean_squared_error(y_test, preds_std))
    mae_std = mean_absolute_error(y_test, preds_std)
    
    rmse_phys = np.sqrt(mean_squared_error(y_test, preds_phys))
    mae_phys = mean_absolute_error(y_test, preds_phys)

    print("\n[i] TEST SET METRICS (Copy these to your Report Table):")
    print(f"    Standard Model -> RMSE: {rmse_std:.6f} | MAE: {mae_std:.6f}")
    print(f"    PINN Model     -> RMSE: {rmse_phys:.6f} | MAE: {mae_phys:.6f}")

    # 5. Generate Spectral Plot for the first star in the test set
    os.makedirs(plots_dir, exist_ok=True)
    
    star_idx = 0  # Feel free to change this to see other stars
    star_name = names_test[star_idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength_grid, y_test[star_idx], label='Ground Truth (CALSPEC)', color='black', linewidth=2)
    plt.plot(wavelength_grid, preds_std[star_idx], label='Standard MLP', color='red', linestyle='--', alpha=0.8)
    plt.plot(wavelength_grid, preds_phys[star_idx], label='PINN (Physics-Informed)', color='blue', linestyle='-.', alpha=0.8)
    
    plt.title(f"Spectral Reconstruction Comparison - Star: {star_name}", fontsize=14)
    plt.xlabel(r"Wavelength $\lambda$ (nm)", fontsize=12)
    plt.ylabel("Flux", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "spectral_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[✓] Spectral comparison plot saved at: {plot_path}")

if __name__ == "__main__":
    dataset_in = "data/processed/final_dataset.csv"
    std_model = "results/models/mlp_standard.pt"
    phys_model = "results/models/mlp_physics.pt"
    plots_folder = "results/plots/"
    
    evaluate_models(dataset_in, std_model, phys_model, plots_folder)