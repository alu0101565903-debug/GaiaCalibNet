"""
Module: plot_results.py
Report Section: Chapter 8 Results & Discussion

This script generates publication-ready visualizations comparing the true CALSPEC 
spectra against the predictions of the Random Forest and the Neural Network.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import your models
from src.models.baselines import get_baseline_models
from src.models.neural_net import SpectralRecalibrationMLP

def generate_comparative_plot(star_index: int = 0):
    print("=" * 60)
    print("[8.0] GENERATING PUBLICATION-READY SPECTRAL PLOTS")
    print("=" * 60)

    # 1. Load and prepare the datasets
    X = np.load("data/processed/X.npy")
    Y = np.load("data/processed/Y.npy")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Global Scaling
    X_max = np.max(np.abs(X_train))
    Y_max = np.max(np.abs(Y_train))
    
    X_train_scaled = X_train / X_max
    X_test_scaled = X_test / X_max
    Y_train_scaled = Y_train / Y_max

    # 2. Train the Random Forest Baseline
    print("-> Training Random Forest (Baseline)...")
    rf_model = get_baseline_models(random_seed=42)["Random_Forest"]
    rf_model.fit(X_train_scaled, Y_train_scaled)
    
    Y_pred_rf = rf_model.predict(X_test_scaled) * Y_max  

    # 3. Train the Neural Network directly on the Spectral Data
    print("-> Training Neural Network (Bottleneck + Cosine Annealing)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train_scaled, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # --- ARCHITECTURAL FIX: THE BOTTLENECK ---
    # We compress 300 -> 64 to force the network to discard high-frequency noise
    # and only learn the physical macro-structure of the spectrum.
    pinn = SpectralRecalibrationMLP(
        input_dim=300, 
        output_dim=300, 
        hidden_dims=[256, 64, 256], 
        dropout_rate=0.0  # Zero dropout to prevent micro-jitters
    ).to(device)
    
    # --- ENHANCED TRAINING LOOP ---
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.002)
    epochs = 1000  # Increased time to learn the fine details
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
    
    pinn.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            predictions = pinn(batch_x)
            
            loss_data = F.mse_loss(predictions, batch_y)
            # TV Loss is weaker now because the bottleneck does the heavy smoothing
            loss_tv = torch.mean((predictions[:, 1:] - predictions[:, :-1]) ** 2)
            
            total_loss = loss_data + (2.0 * loss_tv)
            
            total_loss.backward()
            optimizer.step()
            
        scheduler.step() # Cools down the learning rate smoothly

    # Predict
    pinn.eval()
    with torch.no_grad():
        Y_pred_pinn = pinn(X_test_t).cpu().numpy() * Y_max  

    # 4. Configure and generate the Plot
    print(f"-> Generating plot for test star at index {star_index}...")
    
    wavelengths = np.linspace(3000, 10000, 300)
    
    y_true = Y_test[star_index]
    y_rf = Y_pred_rf[star_index]
    y_pinn = Y_pred_pinn[star_index]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    # --- TOP PANEL ---
    ax1.set_title("Spectral Recalibration Output vs Ground Truth (CALSPEC)", fontsize=14, pad=15)
    ax1.plot(wavelengths, y_true, color='black', linewidth=2, label="CALSPEC (True Flux)")
    ax1.plot(wavelengths, y_rf, color='#2ca02c', linewidth=1.5, linestyle='--', label="Random Forest (Baseline)")
    ax1.plot(wavelengths, y_pinn, color='#d62728', linewidth=1.5, alpha=0.9, label="Neural Net (Bottleneck + Smooth)")
    
    ax1.set_ylabel(r"Absolute Flux $[erg \cdot s^{-1} \cdot cm^{-2} \cdot \AA^{-1}]$", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)

    # --- BOTTOM PANEL ---
    res_rf = (y_rf - y_true) / (y_true + 1e-8) * 100  
    res_pinn = (y_pinn - y_true) / (y_true + 1e-8) * 100

    ax2.plot(wavelengths, res_rf, color='#2ca02c', linewidth=1.2, linestyle='--')
    ax2.plot(wavelengths, res_pinn, color='#d62728', linewidth=1.2, alpha=0.9)
    ax2.axhline(0, color='black', linewidth=1, linestyle=':')
    
    ax2.set_xlabel(r"Wavelength $\lambda$ $[\AA]$", fontsize=12)
    ax2.set_ylabel("Residuals [%]", fontsize=12)
    
    # We tighten the Y-axis constraints because the fit will be much better now
    ax2.set_ylim(-20, 20) 
    ax2.grid(True, alpha=0.3)

    os.makedirs("results/plots", exist_ok=True)
    save_path = f"results/plots/spectrum_comparison_idx{star_index}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[*] Plot successfully saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparative_plot(star_index=2)