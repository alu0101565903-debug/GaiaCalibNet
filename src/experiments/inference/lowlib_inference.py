"""Validate Gaia XP calibration against a LOWLIB reference spectrum.

The script downloads Gaia DR3 and LOWLIB spectra, applies the trained PINN
calibration, and exports validation plots and metrics.

Notes
-----
Requires trained model artifacts and network access to Gaia and MAST.
"""

import os
import numpy as np
from astroquery.gaia import Gaia
from astroquery.mast import Observations
from astropy.io import fits

try:
    from src.data_processing.gaia_xp_lsf_convolution import degrade_to_gaia_xp as c_module_correct_lsf
except ImportError:
    # Fall back to the implementation in prepare_degraded_dataset.
    from src.data_processing.prepare_degraded_dataset import c_module_correct_lsf

from src.utils.processing.pinn_inference import calibrate_spectrum
from src.utils.visualize_models_performance.export_utils import save_prediction_archive
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file


if __name__ == "__main__":
    # Configure local paths.
    model_dir = "results/models/full_pinn_training"

    model_path = os.path.join(model_dir, "model_bundle.pt")
    mean_path = os.path.join(model_dir, "stellar_mean_p.npy")
    std_path = os.path.join(model_dir, "stellar_std_p.npy")

    # Configure the target star.
    target_name = "hd011964"  
    # Gaia DR3 source ID for HD 11964.
    gaia_source_id = "2462426800883134336"
    
    bp_rp_val = 1.2           
    g_mag_val = 15.4          
    ruwe_val = 1.05           

    try:
        # Step A: download the original Gaia XP_SAMPLED spectrum.
        print(f"[>] Downloading Gaia spectrum for source {gaia_source_id}...")
        
        datalink = Gaia.load_data(
            ids=[gaia_source_id], 
            data_release='Gaia DR3', 
            retrieval_type='XP_SAMPLED', 
            data_structure='INDIVIDUAL',
            valid_data=True
        )
        
        dl_keys = list(datalink.keys())
        if not dl_keys:
            raise ValueError(f"No datalink data were returned for source {gaia_source_id}.")
            
        # Extract the VOTable containing the spectrum.
        tabla_gaia = datalink[dl_keys[0]][0].to_table()
        
        # Gaia DR3 XP_SAMPLED uses the 'wavelength' and 'flux' columns.
        wave_gaia_raw = tabla_gaia['wavelength'].value.astype(np.float32)
        flux_gaia_raw = tabla_gaia['flux'].value.astype(np.float32)
        print(f"    Gaia spectrum downloaded ({len(wave_gaia_raw)} samples).")

        # Step B: download the LOWLIB reference spectrum.
        print(f"[>] Searching MAST (LOWLIB) for reference spectrum {target_name}...")
        all_obs = Observations.query_criteria(provenance_name="lowlib", target_name=target_name)
        if len(all_obs) == 0:
            raise ValueError(f"Target '{target_name}' was not found in the LOWLIB library.")

        data_products = Observations.get_product_list(all_obs)
        spect_products = Observations.filter_products(data_products, extension="fits")
        
        manifest = Observations.download_products(spect_products)
        if manifest is None or len(manifest) == 0:
            raise FileNotFoundError("Failed to download the FITS file from MAST.")
            
        ruta_fits_descargado = manifest['Local Path'][0]
        
        with fits.open(ruta_fits_descargado) as hdul:
            datos_espectro = hdul[1].data
            
            # Convert Angstroms to nm and erg/s/cm^2/Angstrom to W/m^2/nm.
            wave_lowlib = (datos_espectro['Wavelength'] / 10.0).astype(np.float32)
            flux_lowlib = (datos_espectro['Uncorrected'] * 1e-2).astype(np.float32)

        # Step C: calibrate the Gaia spectrum with the PINN.
        print("[>] Loading training artifacts and feature statistics...")
        mean_p_real = np.load(mean_path)
        std_p_real = np.load(std_path)

        print("[>] Calibrating the Gaia spectrum...")
        wave_grid, flux_orig_gaia, flux_cal_pinn = calibrate_spectrum(
            wave_raw=wave_gaia_raw,
            flux_raw=flux_gaia_raw,
            bp_rp=bp_rp_val,
            g_mag=g_mag_val,
            ruwe=ruwe_val,
            model_bundle_path=model_path,
            mean_p=mean_p_real,
            std_p=std_p_real
        )
        
        # Process the reference using the same steps as in training.
        # Sort wavelengths in ascending order for LSF integration.
        sort_idx = np.argsort(wave_lowlib)
        wave_lowlib = wave_lowlib[sort_idx]
        flux_lowlib = flux_lowlib[sort_idx]

        # Extrapolate linearly to 1200 nm using a dense wavelength grid.
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

        # Evaluate LSF convolution directly on Gaia's final wavelength grid.
        # This avoids cubic-spline interpolation and excess spectral resolution.
        flux_ref_lowlib = c_module_correct_lsf(wave_lowlib, flux_lowlib, wave_grid).astype(np.float32)
        
        print("[✓] Calibration completed successfully.")

        # Step D: export predictions and run validation.
        print("[>] Preparing data for evaluation and plotting...")
        os.makedirs("results", exist_ok=True)
        archive_path = "results/lowlib_inference_temp.npz"
        
        # Store the uncalibrated Gaia spectrum, LOWLIB reference, and PINN output.
        save_prediction_archive(
            output_filepath=archive_path,
            wave=wave_grid,
            X_orig=np.expand_dims(flux_orig_gaia, axis=0),
            Y_ref=np.expand_dims(flux_ref_lowlib, axis=0),
            X_pinn=np.expand_dims(flux_cal_pinn, axis=0),
        )

        dir_graficas = "plots/lowlib_validation_figures"
        generate_plots_from_file(
            input_filepath=archive_path,
            output_dir=dir_graficas,
            num_samples=1
        )
        
        dir_metricas = "results/lowlib_metrics"
        evaluate_test_set_from_file(
            input_filepath=archive_path,
            output_dir=dir_metricas,
            num_knots=20,
            model_name="PINN-LOWLIB"
        )
        
        print(f"[✓] Validation complete. See '{dir_graficas}' and '{dir_metricas}'.")

    except FileNotFoundError as e:
        print(f"[!] File error: {e}")
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")