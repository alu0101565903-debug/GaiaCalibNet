import os
from typing import Tuple
import numpy as np
import torch
from torch import nn
from scipy.interpolate import interp1d
from astroquery.gaia import Gaia
from astroquery.mast import Observations
from astropy.io import fits

# Añade esto arriba junto a tus otros imports
try:
    from src.data_processing.gaia_xp_lsf_convolution import degrade_to_gaia_xp as c_module_correct_lsf
except ImportError:
    # Si falla, importa la versión fallback desde prepare_degraded_dataset
    from src.data_processing.prepare_degraded_dataset import c_module_correct_lsf

# Asegúrate de importar tus nuevas clases correctamente según la estructura de tu proyecto
from src.experiments.loocv_residual_pinn import DirectConvPINN, BlueWiggleNet, EnsembleExportModel
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file

def calibrate_gaia_spectrum(
    wave_gaia_raw: np.ndarray, 
    flux_gaia_raw: np.ndarray, 
    bp_rp: float, 
    g_mag: float, 
    ruwe: float, 
    model_path: str, 
    pca_basis: np.ndarray, 
    pca_mean: np.ndarray, 
    mean_p: np.ndarray, 
    std_p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the trained Ensemble Direct PINN (Base + BlueWiggle) to calibrate an uncalibrated Gaia spectrum.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Standardize the wavelength grid
    wave_gaia_standard = np.arange(336.0, 1022.0, 2.0, dtype=np.float32)
    num_channels = len(wave_gaia_standard)
    
    interp_func = interp1d(
        wave_gaia_raw, 
        flux_gaia_raw, 
        kind="cubic", 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    flux_interp = interp_func(wave_gaia_standard).astype(np.float32)
    
    # 2. Define the Blue Mask configuration used during training
    blue_mask = wave_gaia_standard < 450.0
    num_blue_channels = np.sum(blue_mask)
    
    # 3. Standardize stellar parameters (Z-score scaling)
    params_raw = np.array([bp_rp, g_mag, ruwe], dtype=np.float32)
    params_std = (params_raw - mean_p) / (std_p + 1e-8)
    
    # 4. Convert arrays to PyTorch Tensors (Batch dimension = 1)
    x_t = torch.tensor(flux_interp, dtype=torch.float32).unsqueeze(0).to(device)
    params_t = torch.tensor(params_std, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 5. Instantiate the specific architectures
    base_model = DirectConvPINN(
        input_dim=num_channels, 
        pca_basis=pca_basis, 
        pca_mean=pca_mean, 
        num_params=3
    ).to(device)
    
    blue_model = BlueWiggleNet(
        input_dim=num_blue_channels, 
        num_params=3
    ).to(device)
    
    ensemble = EnsembleExportModel(
        base_model=base_model, 
        blue_model=blue_model, 
        blue_mask=blue_mask, 
        wave_grid=wave_gaia_standard
    ).to(device)
    
    # 6. Load pre-trained weights for the ensemble
    ensemble.load_state_dict(torch.load(model_path, map_location=device))
    ensemble.eval()
    
    # 7. Execute forward pass (Inference)
    with torch.no_grad():
        r_final_t = ensemble(x_t, params_t)
        
    r_final_np = r_final_t.squeeze(0).cpu().numpy()
    
    # 8. Reconstruct the physical flux
    calibrated_flux = flux_interp * r_final_np
    
    return wave_gaia_standard, flux_interp, calibrated_flux


if __name__ == "__main__":
    # 1. Configuración de rutas locales
    dir_modelo = "results/models/loocv_adp_direct_pinn"
    dir_datos_nn = "data/processed/nn_data"

    ruta_modelo = os.path.join(dir_modelo, "modelo_direct_pinn.pth")
    ruta_pca_basis = os.path.join(dir_modelo, "pca_basis.npy")
    ruta_pca_mean = os.path.join(dir_modelo, "pca_mean.npy")
    ruta_mean_p = os.path.join(dir_datos_nn, "stellar_mean_p.npy")
    ruta_std_p = os.path.join(dir_datos_nn, "stellar_std_p.npy")

    # 2. Configuración de la estrella objetivo
    target_name = "hd011964"  
    # ID real de HD 11964 en Gaia DR3
    gaia_source_id = "2462426800883134336"
    
    bp_rp_val = 1.2           
    g_mag_val = 15.4          
    ruwe_val = 1.05           

    try:
        # =====================================================================
        # PASO A: DESCARGAR ESPECTRO ORIGINAL DE GAIA (XP_SAMPLED)
        # =====================================================================
        print(f"[>] Descargando espectro de Gaia para el ID {gaia_source_id}...")
        
        datalink = Gaia.load_data(
            ids=[gaia_source_id], 
            data_release='Gaia DR3', 
            retrieval_type='XP_SAMPLED', 
            data_structure='INDIVIDUAL',
            valid_data=True
        )
        
        dl_keys = list(datalink.keys())
        if not dl_keys:
            raise ValueError(f"No se devolvieron datos de datalink para el ID {gaia_source_id}.")
            
        # Extraer la tabla VOTable con el espectro
        tabla_gaia = datalink[dl_keys[0]][0].to_table()
        
        # En Gaia DR3 XP_SAMPLED, las columnas correctas son 'wavelength' y 'flux'
        wave_gaia_raw = tabla_gaia['wavelength'].value.astype(np.float32)
        flux_gaia_raw = tabla_gaia['flux'].value.astype(np.float32)
        print(f"    Espectro de Gaia descargado con éxito ({len(wave_gaia_raw)} puntos).")

        # =====================================================================
        # PASO B: DESCARGAR ESPECTRO DE LOWLIB (GROUND-TRUTH DE REFERENCIA)
        # =====================================================================
        print(f"[>] Buscando espectro de referencia {target_name} en MAST (LOWLIB)...")
        all_obs = Observations.query_criteria(provenance_name="lowlib", target_name=target_name)
        if len(all_obs) == 0:
            raise ValueError(f"No se encontró el objetivo '{target_name}' en la librería LOWLIB.")

        data_products = Observations.get_product_list(all_obs)
        spect_products = Observations.filter_products(data_products, extension="fits")
        
        manifest = Observations.download_products(spect_products)
        if manifest is None or len(manifest) == 0:
            raise FileNotFoundError("Fallo al descargar el archivo FITS desde MAST.")
            
        ruta_fits_descargado = manifest['Local Path'][0]
        
        with fits.open(ruta_fits_descargado) as hdul:
            datos_espectro = hdul[1].data
            
            # ¡AQUÍ ESTÁ LA CORRECCIÓN! 
            # Convertir Angstroms a nm y erg/s/cm²/Å a W/m²/nm
            wave_lowlib = (datos_espectro['Wavelength'] / 10.0).astype(np.float32)
            flux_lowlib = (datos_espectro['Uncorrected'] * 1e-2).astype(np.float32)

        # =====================================================================
        # PASO C: CALIBRAR EL ESPECTRO DE GAIA CON LA PINN
        # =====================================================================
        print("[>] Cargando archivos de entrenamiento y parámetros...")
        pca_basis_real = np.load(ruta_pca_basis)
        pca_mean_real = np.load(ruta_pca_mean)
        mean_p_real = np.load(ruta_mean_p)
        std_p_real = np.load(ruta_std_p)

        print("[>] Iniciando calibración del espectro de Gaia...")
        wave_grid, flux_orig_gaia, flux_cal_pinn = calibrate_gaia_spectrum(
            wave_gaia_raw=wave_gaia_raw,
            flux_gaia_raw=flux_gaia_raw,
            bp_rp=bp_rp_val,
            g_mag=g_mag_val,
            ruwe=ruwe_val,
            model_path=ruta_modelo,
            pca_basis=pca_basis_real,
            pca_mean=pca_mean_real,
            mean_p=mean_p_real,
            std_p=std_p_real
        )
        
        # =====================================================================
        # PROCESAMIENTO DE REFERENCIA IDÉNTICO AL ENTRENAMIENTO
        # =====================================================================
        
        # 0. Ordenar de menor a mayor longitud de onda (Crítico para la integral LSF)
        sort_idx = np.argsort(wave_lowlib)
        wave_lowlib = wave_lowlib[sort_idx]
        flux_lowlib = flux_lowlib[sort_idx]

        # 1. Extrapolación lineal densa hasta 1200 nm
        if wave_lowlib[-1] < 1200.0:
            step = np.median(np.diff(wave_lowlib))
            ext_wave = np.arange(wave_lowlib[-1] + step, 1200.0 + step, step)
            
            n_points = min(10, len(wave_lowlib))
            dy = flux_lowlib[-1] - flux_lowlib[-n_points]
            dx = wave_lowlib[-1] - wave_lowlib[-n_points]
            slope = dy / (dx + 1e-8)
            
            ext_flux = flux_lowlib[-1] + slope * (ext_wave - wave_lowlib[-1])
            ext_flux = np.clip(ext_flux, 0.0, None)
            
            wave_lowlib = np.concatenate([wave_lowlib, ext_wave])
            flux_lowlib = np.concatenate([flux_lowlib, ext_flux])

        # 2. Convolución LSF DIRECTA sobre la rejilla final de Gaia (336 - 1022 nm)
        # Pasamos 'wave_grid' como x_eval para que el módulo evalúe el suavizado
        # exactamente en los 343 puntos estándar. Esto elimina la necesidad de 
        # usar un Spline Cúbico después y soluciona el exceso de resolución.
        flux_ref_lowlib = c_module_correct_lsf(wave_lowlib, flux_lowlib, wave_grid).astype(np.float32)
        
        print("[✓] Calibración completada con éxito.")

        # =====================================================================
        # PASO D: EXPORTACIÓN Y EVALUACIÓN AUTOMÁTICA
        # =====================================================================
        print("[>] Preparando datos para evaluación y graficado...")
        os.makedirs("results", exist_ok=True)
        ruta_npz = "results/lowlib_inference_temp.npz"
        
        # Ahora asignamos correctamente:
        # X_orig = Gaia sin calibrar
        # Y_ref = LOWLIB (Referencia ideal)
        # X_pinn = Gaia calibrado por tu red
        np.savez(
            ruta_npz,
            wave=wave_grid,
            X_orig=np.expand_dims(flux_orig_gaia, axis=0),
            Y_ref=np.expand_dims(flux_ref_lowlib, axis=0), 
            X_pinn=np.expand_dims(flux_cal_pinn, axis=0)
        )

        dir_graficas = "plots/lowlib_validation_figures"
        generate_plots_from_file(
            input_filepath=ruta_npz,
            output_dir=dir_graficas,
            num_samples=1
        )
        
        dir_metricas = "results/lowlib_metrics"
        evaluate_test_set_from_file(
            input_filepath=ruta_npz,
            output_dir=dir_metricas,
            num_knots=20,
            model_name="PINN-LOWLIB"
        )
        
        print(f"[✓] Proceso completo. Revisa las carpetas '{dir_graficas}' y '{dir_metricas}'.")

    except FileNotFoundError as e:
        print(f"[!] Error de archivo: {e}")
    except Exception as e:
        print(f"[!] Ocurrió un error inesperado durante el proceso: {e}")