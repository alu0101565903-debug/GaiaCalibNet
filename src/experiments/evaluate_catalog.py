"""
Module: evaluate_catalog.py
Reads the batch-generated spectrum catalog and calculates global statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("============================================================")
    print(" [9.0] EVALUATING AI CATALOG STATISTICS")
    print("============================================================")

    catalog_path = "results/data/catalogs/ai_reconstructed_catalog.csv"
    
    if not os.path.exists(catalog_path):
        print(f"[!] ERROR: Catalog not found at {catalog_path}")
        return

    print(" -> Loading catalog into memory...")
    df = pd.read_csv(catalog_path)
    
    # Separamos los IDs de los datos numéricos (los flujos)
    flux_cols = [c for c in df.columns if c.startswith('flux_')]
    flux_data = df[flux_cols].values
    
    # Extraemos las longitudes de onda de los nombres de las columnas para el eje X
    wavelengths = [float(c.replace('flux_', '').replace('nm', '')) for c in flux_cols]

    print(f" -> Total stars in catalog: {len(df)}")
    print(f" -> Spectral resolution: {len(wavelengths)} bins")

    # Cálculos estadísticos
    mean_spectrum = np.mean(flux_data, axis=0)
    std_spectrum = np.std(flux_data, axis=0)

    print("\n -> Generating statistical plot...")
    plt.figure(figsize=(10, 6))
    
    # Ploteamos la media y una banda de sombra para la desviación estándar
    plt.plot(wavelengths, mean_spectrum, color='blue', linewidth=2, label='Mean AI Spectrum')
    plt.fill_between(wavelengths, 
                     mean_spectrum - std_spectrum, 
                     mean_spectrum + std_spectrum, 
                     color='blue', alpha=0.2, label='Standard Deviation')
    
    plt.title("Statistical Overview of AI-Reconstructed Catalog", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Absolute Flux (Mean)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "results/figures/catalog_statistics.png"
    plt.savefig(save_path, dpi=300)
    
    print(f" [+] Success! Statistical plot saved at: {save_path}")
    print("============================================================")

if __name__ == "__main__":
    main()