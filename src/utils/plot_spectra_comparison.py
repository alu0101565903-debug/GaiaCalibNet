"""
Module: plot_spectra_comparison.py
Report Section: Chapter 8 Results & Discussion

This script produces comparison plots of stellar spectra.
It loads previously processed Gaia/CALSPEC data, restores the scaling used
during training, generates predictions from both a classical Random Forest
baseline and the trained Physics-Informed Neural Network (PINN), and saves
comparison figures showing both absolute flux and relative residuals.

The plotting pipeline is designed for pure inference only; no model training
is performed in this module.
"""

import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.models.neural_net import SpectralRecalibrationMLP


def generate_comparative_plots(indices="all", max_plots=None):
    """
    Generates comparative spectral plots for the requested test stars.
    """
    print("=" * 60)
    print("[8.0] INITIATING SECURE INFERENCE & PLOTTING PIPELINE")
    print("=" * 60)

    # 1. DATA LOADING (DETERMINISTIC SPLIT)
    print("-> Loading data matrices and isolating the test set...")
    X = np.load("data/processed/X.npy")
    Y = np.load("data/processed/Y.npy")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 2. MODEL LOADING & CHECKPOINT METADATA (PINN)
    print("-> Loading pre-trained PINN checkpoint & metadata...")
    pinn_path = "results/models/pinn/pinn_spectrum.joblib"
    if not os.path.exists(pinn_path):
        raise FileNotFoundError(f"PINN model not found at: {pinn_path}")
    
    pinn_checkpoint = joblib.load(pinn_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract physical scaling factor used during training
    if isinstance(pinn_checkpoint, dict) and "scalers" in pinn_checkpoint:
        Y_max = pinn_checkpoint["scalers"]["Y_max"]
    else:
        Y_max = float(np.max(np.abs(Y_train)))

    # Apply physical scaling consistent with training pipeline
    X_test_scaled = X_test / Y_max

    # 3. MODEL LOADING & BATCH INFERENCE (RANDOM FOREST)
    print("-> Loading pre-trained Random Forest model...")
    rf_path = "results/models/baselines/random_forest_spectrum.joblib"
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Random Forest model not found at: {rf_path}")
    
    rf_checkpoint = joblib.load(rf_path)
    rf_model = rf_checkpoint["model"] if isinstance(rf_checkpoint, dict) and "model" in rf_checkpoint else rf_checkpoint

    print("   [*] Running RF batch inference...")
    Y_pred_rf_all = rf_model.predict(X_test_scaled) * Y_max  

    # 4. RECONSTRUCT PINN & BATCH INFERENCE
    print("-> Reconstructing Neural Network (PINN)...")
    hidden_dims = pinn_checkpoint.get("hidden_dims", [64, 32, 64]) if isinstance(pinn_checkpoint, dict) else [64, 32, 64]
    input_dim = pinn_checkpoint.get("input_dim", 300) if isinstance(pinn_checkpoint, dict) else 300
    output_dim = pinn_checkpoint.get("output_dim", 300) if isinstance(pinn_checkpoint, dict) else 300

    pinn_model = SpectralRecalibrationMLP(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dims=hidden_dims, 
        dropout_rate=0.2  
    ).to(device)

    state_dict = None
    if isinstance(pinn_checkpoint, dict):
        for key in ["model_state_dict", "state_dict"]:
            if key in pinn_checkpoint:
                state_dict = pinn_checkpoint[key]
                break
        if not state_dict and "model" in pinn_checkpoint:
            state_dict = pinn_checkpoint["model"] if isinstance(pinn_checkpoint["model"], dict) else None
    else:
        state_dict = pinn_checkpoint

    if state_dict is not None:
        pinn_model.load_state_dict(state_dict)
        print("   [+] PINN weights successfully loaded.")

    print("   [*] Running PINN batch inference (Strict Evaluation Mode)...")
    pinn_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        Y_pred_pinn_all = pinn_model(X_test_t).detach().cpu().numpy() * Y_max  

    # 5. PLOTTING ENGINE
    total_test_stars = len(X_test)
    if isinstance(indices, str) and indices.lower() == "all":
        plot_list = list(range(total_test_stars))
    elif isinstance(indices, list):
        plot_list = indices
    elif isinstance(indices, int):
        plot_list = [indices]
    else:
        raise ValueError("Indices must be an int, a list of ints, or 'all'.")

    if max_plots is not None:
        plot_list = plot_list[:max_plots]

    print(f"\n-> Generating {len(plot_list)} comparative plot(s)...")
    os.makedirs("results/plots/models_performance_comparison", exist_ok=True)

    wavelengths = np.linspace(330, 1050, 300)

    for star_index in plot_list:
        if star_index >= total_test_stars:
            print(f"   [!] Skipping index {star_index}: Out of bounds (Max: {total_test_stars-1})")
            continue
            
        x_gaia = X_test[star_index]    
        y_true = Y_test[star_index]      
        y_rf = Y_pred_rf_all[star_index]     
        y_pinn = Y_pred_pinn_all[star_index] 

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.05)

        # --- TOP PANEL: Absolute Spectra ---
        ax1.set_title(f"Spectral Recalibration Output vs Degraded Ground Truth | Test Star Index: {star_index}", fontsize=14, pad=15, fontweight="bold")
        ax1.plot(wavelengths, x_gaia, color="#264ab7", linewidth=1.8, alpha=0.5, label="Gaia Input (Uncalibrated)")
        ax1.plot(wavelengths, y_true, color="black", linewidth=2, label="CALSPEC (Degraded Ground Truth)")
        ax1.plot(wavelengths, y_rf, color="#2ca02c", linewidth=1.5, linestyle="--", label="Random Forest (Baseline)")
        ax1.plot(wavelengths, y_pinn, color="#d62728", linewidth=1.5, alpha=0.9, label="Neural Net (PINN)")
        
        ax1.set_ylabel(r"Absolute Flux $[erg \cdot s^{-1} \cdot cm^{-2} \cdot \AA^{-1}]$", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", frameon=True, fontsize=10)

        # --- BOTTOM PANEL: Percentage Residuals ---
        y_true_safe = np.clip(y_true, a_min=1e-15, a_max=None)
        res_gaia = ((x_gaia - y_true) / y_true_safe) * 100
        res_rf = ((y_rf - y_true) / y_true_safe) * 100  
        res_pinn = ((y_pinn - y_true) / y_true_safe) * 100

        ax2.plot(wavelengths, res_gaia, color="#4542e2", linewidth=1.2, alpha=0.4, label="Gaia Original Error")
        ax2.plot(wavelengths, res_rf, color="#2ca02c", linewidth=1.2, linestyle="--")
        ax2.plot(wavelengths, res_pinn, color="#d62728", linewidth=1.2, alpha=0.9)
        ax2.axhline(0, color="black", linewidth=1.5, linestyle="-")
        
        ax2.set_xlabel(r"Wavelength $\lambda$ $[nm]$", fontsize=12)
        ax2.set_ylabel("Residuals [%]", fontsize=12)
        ax2.set_ylim(-40, 40) 
        ax2.grid(True, alpha=0.3)

        save_path = f"results/plots/models_performance_comparison/spectrum_comparison_idx{star_index}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig) 
        
    print(f"[*] All requested plots ({len(plot_list)}) successfully saved in results/plots/models_performance_comparison/.")

if __name__ == "__main__":
    generate_comparative_plots(indices="all")