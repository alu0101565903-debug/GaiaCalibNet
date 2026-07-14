"""
Module: download_gaia.py

This module handles the connection to the European Space Agency (ESA) Gaia DR3 archive.
It downloads the raw BP/RP continuous spectra for specific stars and interpolates them 
onto the standardized 300-channel grid used by the neural network.
"""

import numpy as np
from scipy.interpolate import interp1d
import warnings

# Importamos la herramienta oficial de la ESA para calibrar espectros BP/RP
from gaiaxpy import calibrate

def load_real_gaia_spectrum(source_id: str, wave_length_grid: np.ndarray) -> np.ndarray:
    """
    Downloads the real BP/RP continuous spectrum for a given Gaia DR3 source_id,
    calibrates it to absolute flux, and interpolates it onto the project's common grid.
    
    Parameters:
    - source_id (str): The unique Gaia DR3 identifier.
    - wave_length_grid (np.ndarray): The 300-channel grid (Lambda) from the pipeline.
    
    Returns:
    - np.ndarray: The real observed flux mapped to the common grid, or None if it fails.
    """
    try:
        # 1. Fetch from ESA. 
        # calibrate() returns a DataFrame with the flux, and a NumPy array with the wavelengths
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated_data, sampling_wave = calibrate([int(source_id)])
        
        # Comprobamos si la estrella realmente tiene espectro publicado
        if calibrated_data is None or calibrated_data.empty:
            print(f"  [!] No BP/RP spectrum available in Gaia DR3 for source {source_id}")
            return None

        # 2. Extraemos los datos físicos
        real_wave_nm = sampling_wave  # El array de longitudes de onda en nanómetros
        real_flux = calibrated_data['flux'].iloc[0]  # El array de flujo de la estrella
        
        # 3. Limpiamos valores NaN (huecos vacíos en los bordes de los sensores de Gaia)
        valid_mask = ~np.isnan(real_flux)
        real_wave_nm = real_wave_nm[valid_mask]
        real_flux = real_flux[valid_mask]
        
        # 4. Interpolamos al "cuello de botella" de tu proyecto (300 canales)
        # Usamos interpolación cúbica para suavizar la curva de forma física
        interp_func = interp1d(real_wave_nm, real_flux, kind='cubic', fill_value="extrapolate")
        aligned_real_flux = interp_func(wave_length_grid)
        
        return aligned_real_flux

    except Exception as e:
        # Si hay un problema de conexión con el servidor o la ID es inválida, lo capturamos aquí
        print(f"  [!] Gaia API Error for {source_id}: {str(e)}")
        return None