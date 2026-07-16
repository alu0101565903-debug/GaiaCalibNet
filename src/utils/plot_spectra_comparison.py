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

    The function follows this pipeline:
    1. Load the processed X/Y dataset and reproduce the deterministic train/test split.
    2. Recover the scaling factors from the training set and apply them to the test set.
    3. Load the saved Random Forest baseline and compute predictions for the test set.
    4. Reconstruct the saved PINN architecture, load its weights, and compute predictions.
    5. For each requested star index, plot absolute spectra and percentage residuals.

    Parameters:
        indices (int, list, or str): 
            - int: Generates a plot for a single specific index (e.g., 4)
            - list: Generates plots for a specific list of indices (e.g., [0, 4, 15])
            - "all": Generates plots for ALL stars in the test set.
        max_plots (int, optional): Safety limit if indices="all" to avoid generating 
                                   thousands of images. Default is None.
    """
    print("=" * 60)
    print("[8.0] INITIATING SECURE INFERENCE & PLOTTING PIPELINE")
    print("=" * 60)

    # 1. DATA LOADING (DETERMINISTIC SPLIT)
    # Load the processed feature and target arrays produced by the preprocessing
    # pipeline. Then reproduce the same train/test split used during training so
    # the selected test examples remain consistent across runs.
    print("-> Loading data matrices and isolating the test set...")
    X = np.load("data/processed/X.npy")
    Y = np.load("data/processed/Y.npy")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Reconstruct the global scaling values from the training set only. These
    # scaling factors are used to invert the normalization applied during model
    # training so the plotted outputs are in physical flux units.
    X_max = np.max(np.abs(X_train))
    Y_max = np.max(np.abs(Y_train))
    
    X_test_scaled = X_test / X_max

    # 2. MODEL LOADING & BATCH INFERENCE (RANDOM FOREST)
    # Load the saved Random Forest baseline checkpoint and extract the trained
    # model handle. This allows the baseline predictions to be compared directly
    # against both the Gaia input and the PINN output.
    print("-> Loading pre-trained Random Forest model...")
    rf_path = "results/models/baselines/random_forest_spectrum.joblib"
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Random Forest model not found at: {rf_path}")
    
    rf_checkpoint = joblib.load(rf_path)
    rf_model = rf_checkpoint['model'] if isinstance(rf_checkpoint, dict) and 'model' in rf_checkpoint else rf_checkpoint

    print("   [*] Running RF batch inference...")
    # Batch inference on the full test split, with scaling reversed to physical flux.
    Y_pred_rf_all = rf_model.predict(X_test_scaled) * Y_max  

    # 3. MODEL LOADING & BATCH INFERENCE (PINN)
    # Load the saved physics-informed neural network (PINN) checkpoint and
    # reconstruct the exact model architecture used during training.
    print("-> Loading and reconstructing the Neural Network (PINN)...")
    pinn_path = "results/models/pinn/pinn_spectrum.joblib"
    if not os.path.exists(pinn_path):
        raise FileNotFoundError(f"PINN model not found at: {pinn_path}")
    
    pinn_checkpoint = joblib.load(pinn_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract architecture metadata from the checkpoint if available.
    hidden_dims = pinn_checkpoint.get('hidden_dims', [256, 512, 256]) if isinstance(pinn_checkpoint, dict) else [256, 512, 256]
    input_dim = pinn_checkpoint.get('input_dim', 300) if isinstance(pinn_checkpoint, dict) else 300
    output_dim = pinn_checkpoint.get('output_dim', 300) if isinstance(pinn_checkpoint, dict) else 300

    # Recreate the exact network structure before loading weights.
    pinn_model = SpectralRecalibrationMLP(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dims=hidden_dims, 
        dropout_rate=0.2  
    ).to(device)

    # Extract state dict robustly from common checkpoint formats.
    state_dict = None
    if isinstance(pinn_checkpoint, dict):
        for key in ['model_state_dict', 'state_dict']:
            if key in pinn_checkpoint:
                state_dict = pinn_checkpoint[key]
                break
        if not state_dict and 'model' in pinn_checkpoint:
            state_dict = pinn_checkpoint['model'] if isinstance(pinn_checkpoint['model'], dict) else None
    else:
        state_dict = pinn_checkpoint

    if state_dict is not None:
        pinn_model.load_state_dict(state_dict)
        print("   [+] PINN weights successfully loaded.")

    # Strict evaluation mode: disable dropout and gradient tracking so the
    # model behaves deterministically during plotting.
    print("   [*] Running PINN batch inference (Strict Evaluation Mode)...")
    pinn_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        Y_pred_pinn_all = pinn_model(X_test_t).cpu().numpy() * Y_max  

    # 4. PLOTTING ENGINE
    # Determine which test examples should be plotted based on the provided
    # indices argument. This supports single indices, custom lists, or the full
    # test set.
    total_test_stars = len(X_test)
    if isinstance(indices, str) and indices.lower() == "all":
        plot_list = list(range(total_test_stars))
    elif isinstance(indices, list):
        plot_list = indices
    elif isinstance(indices, int):
        plot_list = [indices]
    else:
        raise ValueError("Indices must be an int, a list of ints, or 'all'.")

    # Apply a safety cap when plotting all examples to avoid excessive file output.
    if max_plots is not None:
        plot_list = plot_list[:max_plots]

    print(f"\n-> Generating {len(plot_list)} comparative plot(s)...")
    os.makedirs("results/plots/models_performance_comparison", exist_ok=True)

    # Build a physical wavelength axis for plotting. The values here are in nm.
    wavelengths = np.linspace(330, 1050, 300)

    for star_index in plot_list:
        # Guard against invalid indices when the user asks to plot beyond the
        # number of stored test examples.
        if star_index >= total_test_stars:
            print(f"   [!] Skipping index {star_index}: Out of bounds (Max: {total_test_stars-1})")
            continue
            
        # Extract the original Gaia input spectrum, the CALSPEC reference
        # spectrum, and the two model prediction spectra for this star.
        x_gaia = X_test[star_index]    
        y_true = Y_test[star_index]      
        y_rf = Y_pred_rf_all[star_index]     
        y_pinn = Y_pred_pinn_all[star_index] 

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.05)

        # --- TOP PANEL: Absolute Spectra ---
        # This panel shows the original uncalibrated Gaia spectrum, the true
        # CALSPEC reference, and the two model-corrected outputs.
        ax1.set_title(f"Spectral Recalibration Output vs Ground Truth | Star Index: {star_index}", fontsize=14, pad=15, fontweight='bold')
        ax1.plot(wavelengths, x_gaia, color='#264ab7', linewidth=1.8, alpha=0.5, label="Gaia Input (Uncalibrated)")
        ax1.plot(wavelengths, y_true, color='black', linewidth=2, label="CALSPEC (True Flux)")
        ax1.plot(wavelengths, y_rf, color='#2ca02c', linewidth=1.5, linestyle='--', label="Random Forest (Baseline)")
        ax1.plot(wavelengths, y_pinn, color='#d62728', linewidth=1.5, alpha=0.9, label="Neural Net (PINN)")
        
        ax1.set_ylabel(r"Absolute Flux $[erg \cdot s^{-1} \cdot cm^{-2} \cdot \AA^{-1}]$", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", frameon=True, fontsize=10)

        # --- BOTTOM PANEL: Percentage Residuals ---
        # The residual panel highlights how far each prediction deviates from
        # the true CALSPEC flux, normalized by the true values.
        y_true_safe = np.clip(y_true, a_min=1e-15, a_max=None)
        res_gaia = ((x_gaia - y_true) / y_true_safe) * 100
        res_rf = ((y_rf - y_true) / y_true_safe) * 100  
        res_pinn = ((y_pinn - y_true) / y_true_safe) * 100

        ax2.plot(wavelengths, res_gaia, color="#4542e2", linewidth=1.2, alpha=0.4, label="Gaia Original Error")
        ax2.plot(wavelengths, res_rf, color='#2ca02c', linewidth=1.2, linestyle='--')
        ax2.plot(wavelengths, res_pinn, color='#d62728', linewidth=1.2, alpha=0.9)
        ax2.axhline(0, color='black', linewidth=1.5, linestyle=':')
        
        ax2.set_xlabel(r"Wavelength $\lambda$ $[nm]$", fontsize=12)
        ax2.set_ylabel("Residuals [%]", fontsize=12)
        ax2.set_ylim(-40, 40) 
        ax2.grid(True, alpha=0.3)

        save_path = f"results/plots/models_performance_comparison/spectrum_comparison_idx{star_index}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Free memory by closing the figure
        
    # Summary message after all requested plots have been written to disk.
    print(f"[*] All requested plots ({len(plot_list)}) successfully saved in results/plots/models_performance_comparison/.")

if __name__ == "__main__":
    # USAGE EXAMPLES:
    
    # 1. Plot a single star:
    # generate_comparative_plots(indices=4)
    
    # 2. Plot a specific list of stars:
    generate_comparative_plots(indices=[0, 4, 15, 42])
    
    # 3. Plot ALL stars (warning: this may use a large amount of disk space):
    # generate_comparative_plots(indices="all")
    
    # 4. Plot the first 20 stars from the entire dataset:
    # generate_comparative_plots(indices="all", max_plots=20)