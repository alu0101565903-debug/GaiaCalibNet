"""
Module: generate_from_features.py
Uses the trained generative PINN to synthesize a 300-channel spectrum 
from 5 input Gaia astrometric/photometric features.
"""

import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from src.models.neural_net import SpectralRecalibrationMLP

def main():
    print("============================================================")
    print(" [8.3] INITIATING STELLAR SPECTRUM GENERATOR (PINN)")
    print("============================================================")

    model_path = "results/models/pinn/pinn_features.joblib"
    wave_length_grid = np.linspace(330, 1050, 300)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Computation Device allocated: {device}")

    if not os.path.exists(model_path):
        print(f"[!] ERROR: Checkpoint not found at {model_path}")
        return

    checkpoint = joblib.load(model_path)
    
    model = SpectralRecalibrationMLP(
        input_dim=checkpoint.get('input_dim', 5), 
        output_dim=checkpoint.get('output_dim', 300),
        hidden_dims=checkpoint.get('hidden_dims'), 
        dropout_rate=0.2 
    )
    model.load_state_dict(checkpoint.get('model_state_dict'))
    model.to(device)
    model.eval()

    # --- EXTRACCIÓN DE ESCALADORES ---
    scalers_dict = checkpoint.get('scalers', {})
    scaler_x = scalers_dict.get('scaler_X')
    y_max = scalers_dict.get('Y_max', 1.0)
    
    if scaler_x is not None:
        print(" -> [+] Successfully loaded input scaler ('scaler_X').")
    if y_max != 1.0:
        print(f" -> [+] Successfully loaded output scalar ('Y_max'): {y_max}")

    # --- PARÁMETROS DE ENTRADA (GAIA DR3) ---
    # [phot_g_mean_mag, pmra, pmdec, parallax, E(B-V)]
    # Usamos la "Estrella Promedio" del dataset para máxima estabilidad
    gaia_features = np.array([[14.99398139, 18.50681638, -8.85694027, 9.43189684, 0.04984697]])
    print(f"\n -> Target Gaia Features (Dataset Means): {gaia_features[0]}")

    # 1. Normalizar entrada
    if scaler_x is not None:
        features_scaled = scaler_x.transform(gaia_features)
    else:
        features_scaled = gaia_features

    input_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    # 2. Generar el espectro (Saldrá normalizado)
    print(" -> Generating synthetic spectrum...")
    with torch.no_grad():
        generated_spectrum_norm = model(input_tensor).cpu().numpy()[0]

    # 3. Deshacer la normalización de salida
    generated_spectrum = generated_spectrum_norm * y_max

    # --- VISUALIZACIÓN ---
    plt.figure(figsize=(10, 6))
    plt.plot(wave_length_grid, generated_spectrum, color='purple', linewidth=2, 
             label='AI Synthesized Spectrum')
    
    # Título corregido con los nombres astronómicos reales
    title_text = f"Synthetic Star Mean Profile\n(G-mag: {gaia_features[0][0]:.2f}, Plx: {gaia_features[0][3]:.2f} mas)"
    plt.title(title_text, fontsize=13, fontweight='bold')
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel(r"Absolute Flux ($W \cdot m^{-2} \cdot nm^{-1}$)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = "results/figures/generative"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "synthetic_star_final.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"\n [+] Success! Clean synthetic spectrum saved at: {save_path}")
    print("============================================================")

if __name__ == "__main__":
    main()