"""
Module: predict.py

This script performs inference using the trained PyTorch PINN model (SpectralRecalibrationMLP). 
It dynamically scales the raw unseen BP/RP spectrum by its own local maximum (Instance Scaling)
to bypass unit mismatches, allows the network to correct the spectral shape in its mathematical 
sweet spot [0, 1], and restores the physical absolute flux scale post-inference.
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import torch

from src.processing.download_gaia import load_real_gaia_spectrum
from src.models.neural_net import SpectralRecalibrationMLP

def main():
    """
    Main pipeline for the Virtual Telescope inference mode with PyTorch.
    """
    print("============================================================")
    print(" [8.0] INITIATING VIRTUAL TELESCOPE (PYTORCH INFERENCE)")
    print("============================================================")

    # 1. Configuration Parameters
    target_source_id = "4881345510044485120" # HD 031128
    model_path = "results/models/pinn/pinn_spectrum.joblib"
    wave_length_grid = np.linspace(330, 1050, 300)
    
    # Hardware acceleration check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Computation Device allocated: {device}")

    # 2. Load the PyTorch checkpoint
    if not os.path.exists(model_path):
        print(f"[!] ERROR: Checkpoint not found at {model_path}")
        return

    print(f" -> Loading PyTorch state dictionary from: {model_path}...")
    checkpoint = joblib.load(model_path)
    
    # 3. Extract architecture components
    input_dim = checkpoint.get('input_dim')
    hidden_dims = checkpoint.get('hidden_dims')
    output_dim = checkpoint.get('output_dim', 300)
    state_dict = checkpoint.get('model_state_dict')

    if state_dict is None:
        print("[!] ERROR: 'model_state_dict' not found in the checkpoint.")
        return

    # 4. Instantiate the PyTorch Model
    print(" -> Reconstructing neural network architecture...")
    model = SpectralRecalibrationMLP(
        input_dim=input_dim, 
        output_dim=output_dim,
        hidden_dims=hidden_dims, 
        dropout_rate=0.2 
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Download the unseen star from the ESA archive
    print(f" -> Connecting to Gaia DR3 to fetch source ID: {target_source_id}...")
    gaia_raw_spectrum = load_real_gaia_spectrum(target_source_id, wave_length_grid)
    
    if gaia_raw_spectrum is None:
        print("[!] Process aborted: Could not retrieve real Gaia spectrum.")
        return

    # 6. Neural Network Inference (INSTANCE SCALING)
    print(" -> Applying instrumental corrections (Neural Processing)...")
    
    gaia_input_2d = gaia_raw_spectrum.reshape(1, -1)
    
    # --- AUTO-ESCALADO (INSTANCE SCALING) ---
    # Encontramos el valor más alto de ESTA estrella específica
    local_max = np.max(np.abs(gaia_input_2d))
    
    # Llevamos los datos al rango [0, 1] para que la red trabaje perfecta
    gaia_input_scaled = gaia_input_2d / local_max
    
    print("\n [DEBUG] DATOS SANEADOS:")
    print(f"  -> Flujo máximo crudo: {local_max:.2e}")
    print(f"  -> Flujo máximo inyectado a la red: {np.max(gaia_input_scaled):.2f} (¡Zona de confort!)\n")
        
    input_tensor = torch.tensor(gaia_input_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction_tensor = model(input_tensor)
        
    predicted_calspec_scaled = prediction_tensor.cpu().numpy()
    
    # --- DESESCALADO ---
    # Devolvemos la predicción a sus unidades físicas originales multiplicando por su máximo
    predicted_calspec_2d = predicted_calspec_scaled * local_max
    clean_spectrum = predicted_calspec_2d[0]

    # 7. Visualization of the Results
    print(" -> Generating comparative plot...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(wave_length_grid, gaia_raw_spectrum, label="Raw Observation (Gaia BP/RP)", 
             color='gray', linestyle='--', alpha=0.8)
    plt.plot(wave_length_grid, clean_spectrum, label="AI Reconstruction (CALSPEC Quality)", 
             color='red', linewidth=2)
    
    plt.title(f"Virtual Telescope Reconstruction - Gaia ID: {target_source_id}", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Absolute Flux", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"prediction_{target_source_id}.png")
    
    plt.savefig(save_path, dpi=300)
    print(f" [+] Success! Figure saved at: {save_path}")

if __name__ == "__main__":
    main()