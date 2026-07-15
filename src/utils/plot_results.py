"""
Module: plot_results.py
Report Section: Chapter 7/8 Visual Validation

Generates comparative plots in Normalized Flux space.
Original version: Full 300-pixel spectrum.
"""

import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.models.neural_net import SpectralRecalibrationMLP

def generate_comparison_plots(mode="spectrum", num_stars=2):
    print("\n" + "=" * 60)
    print(f"GENERATING PLOTS FOR MODE: {mode.upper()} (NORMALIZED DOMAIN)")
    print("=" * 60)
    
    is_spectrum = (mode == "spectrum")
    targets_path = "data/processed/Y.npy"
    model_path = f"results/models/pinn/pinn_{mode}.joblib"
    output_dir = "results/plots/pinn_comparisons"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"[!] Error: Trained model not found at {model_path}. Please train it first!")
        return

    print("-> Loading dataset for validation...")
    if is_spectrum:
        X_spec = np.load("data/processed/X.npy")
        X_feat = np.load("data/processed/X_features.npy")
        X = np.concatenate([X_spec, X_feat], axis=1)  # 305D master array
    else:
        X = np.load("data/processed/X_features.npy")  # 5D array
        
    Y = np.load(targets_path)
    
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"-> Loading pre-trained weights from {model_path}...")
    checkpoint = joblib.load(model_path)
    scalers = checkpoint['scalers']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SpectralRecalibrationMLP(
        input_dim=checkpoint['input_dim'], 
        output_dim=checkpoint['output_dim'], 
        hidden_dims=checkpoint['hidden_dims'], 
        dropout_rate=0.0 if is_spectrum else 0.2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Original full grid
    wavelengths = np.linspace(330.0, 1050.0, 300) 
    
    np.random.seed(42) 
    random_indices = np.random.choice(len(X_test), num_stars, replace=False)
    
    for idx in random_indices:
        x_sample = X_test[idx:idx+1]
        y_true_sample = Y_test[idx]
        
        y_max_val = float(scalers['Y_max'])
        
        if is_spectrum:
            x_spec_raw = x_sample[:, :300]
            x_feat_raw = x_sample[:, 300:305]
            
            x_spec_scaled = x_spec_raw / y_max_val
            
            feat_scaler_key = 'scaler_feat' if 'scaler_feat' in scalers else 'scaler_X'
            x_feat_scaled = scalers[feat_scaler_key].transform(x_feat_raw)
            
            x_scaled = np.concatenate([x_spec_scaled, x_feat_scaled], axis=1)
        else:
            feat_scaler_key = 'scaler_feat' if 'scaler_feat' in scalers else 'scaler_X'
            x_scaled = scalers[feat_scaler_key].transform(x_sample)
            
        with torch.no_grad():
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
            y_pred_scaled = model(x_tensor).cpu().numpy()[0]
            
        y_true_norm = y_true_sample / y_max_val
        y_pred_norm = y_pred_scaled 
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.05)
        
        ax1.plot(wavelengths, y_true_norm, label='CALSPEC Target (Y)', color='red', linewidth=2, zorder=3)
        # Fix de LaTeX incluido
        ax1.plot(wavelengths, y_pred_norm, label=r'PINN Prediction ($\hat{Y}$)', color='forestgreen', linewidth=2, linestyle='--', zorder=4)
        
        if is_spectrum:
            x_gaia_norm = x_sample[0, :300] / y_max_val
            ax1.plot(wavelengths, x_gaia_norm, label='Gaia Input (X)', color='royalblue', alpha=0.6, linewidth=1.5, zorder=2)
            
        ax1.set_title(f'Physical Calibration (300 px) - Star {idx}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Flux', fontsize=12)
        ax1.legend(loc='upper right', frameon=True)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        res_pinn = (y_pred_norm - y_true_norm) / (y_true_norm + 1e-8) * 100
        ax2.plot(wavelengths, res_pinn, color='forestgreen', linewidth=1.2)
        
        if is_spectrum:
            res_gaia = (x_gaia_norm - y_true_norm) / (y_true_norm + 1e-8) * 100
            ax2.plot(wavelengths, res_gaia, color='royalblue', alpha=0.5, linewidth=1.2)
            
        ax2.axhline(0, color='black', linewidth=1, linestyle=':')
        # Fix de LaTeX incluido
        ax2.set_xlabel(r'Wavelength $\lambda$ (nm)', fontsize=12)
        ax2.set_ylabel('Residuals [%]', fontsize=12)
        ax2.set_ylim(-20, 20)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        save_file = os.path.join(output_dir, f"plot_{mode}_star_{idx}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"    [+] Saved: {save_file}")
        plt.close()

if __name__ == "__main__":
    generate_comparison_plots(mode="spectrum", num_stars=4)
    if os.path.exists("data/processed/X_features.npy"):
        generate_comparison_plots(mode="features", num_stars=4)