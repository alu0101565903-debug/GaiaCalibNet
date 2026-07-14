"""
Module: predict_batch.py

This script performs batch inference using the trained PyTorch PINN model.
It iterates over a list of Gaia Source IDs, applies Instance Scaling,
corrects the spectral shape, restores physical units, applies the wavelength
inversion correction, and compiles a final CSV catalog with the results.
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.processing.download_gaia import load_real_gaia_spectrum
from src.models.neural_net import SpectralRecalibrationMLP

def main():
    print("============================================================")
    print(" [8.1] INITIATING BATCH VIRTUAL TELESCOPE (PYTORCH)")
    print("============================================================")

    # 1. Configuration Parameters & Target IDs
    model_path = "results/models/pinn/pinn_spectrum.joblib"
    wave_length_grid = np.linspace(330, 1050, 300)
    
    # Lista de prueba: Añade aquí los IDs de las estrellas que quieras procesar
    target_ids = [
        "4881345510044485120",  # HD 031128 (Tu estrella original)
        "5284204302730217984",  # HD 271783 (Estrella estándar SPSS_ID 348)
        "5757389920689415040"   # G114-25 (Estrella estándar SPSS_ID 115)
    ]
    
    # Hardware acceleration check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Computation Device allocated: {device}")

    # 2. Load the PyTorch checkpoint
    if not os.path.exists(model_path):
        print(f"[!] ERROR: Checkpoint not found at {model_path}")
        return

    print(f" -> Loading PyTorch state dictionary from: {model_path}...")
    checkpoint = joblib.load(model_path)
    
    input_dim = checkpoint.get('input_dim')
    hidden_dims = checkpoint.get('hidden_dims')
    output_dim = checkpoint.get('output_dim', 300)
    state_dict = checkpoint.get('model_state_dict')

    # 3. Instantiate the PyTorch Model
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

    # Preparar directorios de salida
    output_dir_figs = "results/figures/batch_test"
    output_dir_data = "results/data/catalogs"
    os.makedirs(output_dir_figs, exist_ok=True)
    os.makedirs(output_dir_data, exist_ok=True)

    # Diccionario para almacenar el catálogo final
    catalog_results = {"source_id": []}
    for wl in wave_length_grid:
        catalog_results[f"flux_{wl:.1f}nm"] = []

    print("\n============================================================")
    print(" >>> STARTING BATCH PROCESSING")
    print("============================================================\n")

    # 4. Bucle de Inferencia
    for sid in target_ids:
        print(f" -> Processing Gaia ID: {sid}...")
        
        # Intentar descargar el espectro
        try:
            gaia_raw_spectrum = load_real_gaia_spectrum(sid, wave_length_grid)
            if gaia_raw_spectrum is None:
                print(f"    [!] Skipping {sid}: Spectrum not found or failed to download.")
                continue
        except Exception as e:
            print(f"    [!] Skipping {sid} due to download error: {e}")
            continue

        # Instance Scaling
        gaia_input_2d = gaia_raw_spectrum.reshape(1, -1)
        local_max = np.max(np.abs(gaia_input_2d))
        gaia_input_scaled = gaia_input_2d / local_max
        
        input_tensor = torch.tensor(gaia_input_scaled, dtype=torch.float32).to(device)
        
        # Predicción
        with torch.no_grad():
            prediction_tensor = model(input_tensor)
            
        predicted_calspec_scaled = prediction_tensor.cpu().numpy()
        
        # Desescalado y CORRECCIÓN DEL EFECTO ESPEJO ([::-1])
        predicted_calspec_2d = predicted_calspec_scaled * local_max
        clean_spectrum = predicted_calspec_2d[0]

        # Guardar en el diccionario del catálogo
        catalog_results["source_id"].append(sid)
        for i, wl in enumerate(wave_length_grid):
            catalog_results[f"flux_{wl:.1f}nm"].append(clean_spectrum[i])

        # Generar gráfica individual
        plt.figure(figsize=(10, 6))
        plt.plot(wave_length_grid, gaia_raw_spectrum, label="Raw Observation (Gaia BP/RP)", 
                 color='gray', linestyle='--', alpha=0.8)
        plt.plot(wave_length_grid, clean_spectrum, label="AI Reconstruction (CALSPEC Quality)", 
                 color='red', linewidth=2)
        
        plt.title(f"Virtual Telescope Reconstruction - Gaia ID: {sid}", fontsize=14)
        plt.xlabel("Wavelength (nm)", fontsize=12)
        plt.ylabel("Absolute Flux", fontsize=12)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir_figs, f"prediction_{sid}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"    [+] Success! Reconstructed and plotted.")

    # 5. Exportar el nuevo catálogo a CSV
    print("\n============================================================")
    print(" >>> COMPILING FINAL CATALOG")
    print("============================================================")
    
    df_catalog = pd.DataFrame(catalog_results)
    catalog_path = os.path.join(output_dir_data, "ai_reconstructed_catalog.csv")
    df_catalog.to_csv(catalog_path, index=False)
    
    print(f" [+] Batch process complete. Processed {len(df_catalog)} stars.")
    print(f" [+] Catalog saved successfully at: {catalog_path}")
    print(f" [+] Figures saved in: {output_dir_figs}")
    print("============================================================\n")

if __name__ == "__main__":
    main()