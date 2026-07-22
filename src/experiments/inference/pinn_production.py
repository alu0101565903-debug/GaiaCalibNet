"""
Module: run_pinn_production.py
Description: Applies the trained SpectralRecalibrationMLP to arbitrary, new 
Gaia DR3 sources in the wild. Handles missing data gracefully, fetches specific 
edge-cases from the Gaia Archive via ADQL, and exports individual plots.
"""

import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from astroquery.gaia import Gaia
from gaiaxpy import calibrate
from src.models.neural_net import SpectralRecalibrationMLP

def fetch_wild_gaia_spectra(source_ids: list, wl_grid: np.ndarray):
    """
    Fetches and calibrates continuous BP/RP spectra from the ESA Gaia servers.

    Args:
        source_ids (list): List of Gaia DR3 source IDs (as strings).
        wl_grid (np.ndarray): Wavelength grid for sampling the spectra.

    Returns:
        tuple: (flux_matrix_cgs, valid_source_ids) or (None, []) if no data found.
    """
    print(f"-> Connecting to ESA servers...")
    print(f"-> Attempting to download spectra for {len(source_ids)} Gaia DR3 stars...")
    
    try:
        # gaiaxpy handles the download and Hermite polynomial reconstruction
        calibrated_spectra, _ = calibrate(source_ids, sampling=wl_grid)
    except ValueError as e:
        if "No continuous BP/RP data found" in str(e):
            print("\n[!] DATA ERROR: None of these stars have published spectra in Gaia DR3.")
            print("[!] Try with different 'source_id's.")
            return None, []
        else:
            raise e
            
    # Extract IDs that were successfully downloaded (filtering out any failures)
    valid_source_ids = calibrated_spectra["source_id"].astype(str).tolist()
    flux_matrix = np.vstack(calibrated_spectra["flux"].values)
    
    # Mandatory conversion from SI (W/m^2/nm) to CGS (erg/s/cm^2/A)
    flux_matrix_cgs = flux_matrix * 100.0
    
    print(f"[✓] Spectra successfully downloaded: {len(valid_source_ids)} out of {len(source_ids)}")
    return flux_matrix_cgs, valid_source_ids


def run_production_inference(source_ids: list, weights_path: str):
    """
    Loads the trained PINN, processes the raw Gaia spectra, and applies the physical recalibration.

    Args:
        source_ids (list): List of Gaia DR3 source IDs to process.
        weights_path (str): Path to the trained model weights (.joblib or .pt).

    Returns:
        tuple: (wl_grid, raw_fluxes, recalibrated_fluxes, valid_ids)
    """
    wl_grid = np.linspace(350, 1050, 300)
    
    # 1. Download raw data
    result = fetch_wild_gaia_spectra(source_ids, wl_grid)
    if result[0] is None:
        return None, None, None, None
        
    gaia_fluxes, valid_ids = result
    
    # 2. Setup environment and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[!] Model weights not found at {weights_path}")
        
    try:
        loaded_data = torch.load(weights_path, map_location=device, weights_only=False)
    except Exception:
        loaded_data = joblib.load(weights_path)
        
    y_max = None
    scaler_X = None

    # Handle dictionary-based model states (which include scalers)
    if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
        state_dict = loaded_data["model_state_dict"]
        model_kwargs = {k: loaded_data[k] for k in ["input_dim", "hidden_dims", "output_dim"] if k in loaded_data}
        model = SpectralRecalibrationMLP(**model_kwargs).to(device)
        
        if "scalers" in loaded_data:
            scalers = loaded_data["scalers"]
            y_max = scalers.get("Y_max", None)
            scaler_X = scalers.get("scaler_X", None)
    else:
        state_dict = loaded_data
        model = SpectralRecalibrationMLP().to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 3. Normalize input to match training distribution
    if y_max is not None:
        gaia_fluxes_scaled = gaia_fluxes / y_max
    elif scaler_X is not None:
        gaia_fluxes_scaled = scaler_X.transform(gaia_fluxes)
    else:
        gaia_fluxes_scaled = gaia_fluxes

    # 4. Neural Network Forward Pass (Inference)
    print("-> Cleaning and recalibrating spectra...")
    inputs_tensor = torch.tensor(gaia_fluxes_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        recalibrated_scaled = model(inputs_tensor).cpu().numpy()

    # 5. Denormalize back to physical CGS units
    if y_max is not None:
        recalibrated_fluxes = recalibrated_scaled * y_max
    else:
        recalibrated_fluxes = recalibrated_scaled

    print("[✓] Process completed successfully.")
    return wl_grid, gaia_fluxes, recalibrated_fluxes, valid_ids


def plot_production_results(wl, raw_fluxes, pinn_fluxes, valid_ids, output_dir="results/plots/production_wild"):
    """
    Generates and saves individual comparative plots for each star.

    Args:
        wl (np.ndarray): Wavelength grid.
        raw_fluxes (np.ndarray): Original noisy fluxes from Gaia.
        pinn_fluxes (np.ndarray): Recalibrated fluxes from the PINN.
        valid_ids (list): List of successful Gaia Source IDs.
        output_dir (str): Directory to save the output plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_samples = len(valid_ids)
    
    print(f"\n-> Generating {num_samples} independent plots...")
    
    for i in range(num_samples):
        plt.figure(figsize=(10, 6))
        
        # Plot Input (Blue) and Prediction (Red). No Ground Truth available in the wild.
        plt.plot(wl, raw_fluxes[i], label="Gaia DR3 (Raw/Noisy)", color="blue", alpha=0.5, linewidth=1.5)
        plt.plot(wl, pinn_fluxes[i], label="PINN (Recalibrated)", color="red", linestyle="-", linewidth=2)
        
        plt.title(f"Production Inference | Gaia Source ID: {valid_ids[i]}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux (erg/s/cm²/Å)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save each plot individually
        save_path = os.path.join(output_dir, f"wild_star_{valid_ids[i]}.png")
        plt.savefig(save_path, dpi=150)
        plt.close() # Free memory
        
        print(f"  [+] Saved: {save_path}")


if __name__ == "__main__":
    WEIGHTS_PATH = "results/models/pinn/pinn_spectrum.joblib" 
    
    # =====================================================================
    # --- CONFIGURATION PANEL ---
    # Choose which test mode to run: "noisy", "red_dwarfs", "blue_stars", or "manual"
    # =====================================================================
    TEST_MODE = "manual"
    
    # Provide IDs here if TEST_MODE is set to "manual"
    MANUAL_IDS = [
        "5813181197970338560", 
        "5287373841830460416"
    ]
    
    target_ids = []

    print(f"\n=== Starting Production Run | Mode: {TEST_MODE.upper()} ===")

    if TEST_MODE == "manual":
        print(f"-> Using {len(MANUAL_IDS)} manually provided IDs.")
        target_ids = MANUAL_IDS

    elif TEST_MODE == "noisy":
        # Stars with G Magnitude > 15 (High instrumental noise)
        query = """
            SELECT TOP 3 source_id 
            FROM gaiadr3.gaia_source 
            WHERE has_xp_continuous = 'true' AND phot_g_mean_mag > 15.0 
            ORDER BY random_index
        """
        print("-> Querying Gaia Archive for faint/noisy stars...")
        job = Gaia.launch_job(query)
        target_ids = job.get_results()["source_id"].astype(str).tolist()

    elif TEST_MODE == "red_dwarfs":
        # Very cold stars with prominent molecular absorption lines (TiO)
        query = """
            SELECT TOP 3 source_id 
            FROM gaiadr3.gaia_source 
            WHERE has_xp_continuous = 'true' AND bp_rp > 2.5
            ORDER BY random_index
        """
        print("-> Querying Gaia Archive for Red Dwarfs...")
        job = Gaia.launch_job(query)
        target_ids = job.get_results()["source_id"].astype(str).tolist()

    elif TEST_MODE == "blue_stars":
        # Very hot stars stressing the blue/UV edge of the instrument
        query = """
            SELECT TOP 3 source_id 
            FROM gaiadr3.gaia_source 
            WHERE has_xp_continuous = 'true' AND bp_rp < -0.1
            ORDER BY random_index
        """
        print("-> Querying Gaia Archive for Hot Blue Stars...")
        job = Gaia.launch_job(query)
        target_ids = job.get_results()["source_id"].astype(str).tolist()

    else:
        print("[!] Invalid TEST_MODE selected.")

    # --- INFERENCE AND PLOTTING PIPELINE ---
    if target_ids:
        print(f"-> Target IDs acquired: {target_ids}")
        
        # 1. Run the data through the neural network
        wl, raw, pinn, valid_ids = run_production_inference(
            source_ids=target_ids,
            weights_path=WEIGHTS_PATH
        )
        
        # 2. Plot the results using the returned data
        if valid_ids:
            plot_production_results(wl, raw, pinn, valid_ids)
        else:
            print("[!] Pipeline finished but no valid data was plotted.")
    else:
        print("[!] No target IDs to process.")