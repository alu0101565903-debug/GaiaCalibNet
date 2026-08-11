"""
Module: prepare_undegraded_dataset.py
Report Section: Chapter 5 Data Processing & Feature Engineering

This script constructs the full regression dataset (X and Y matrices) for training 
and evaluating the PINN model. It maps CALSPEC target FITS spectra to Gaia DR3 
source IDs, converts wavelengths from Ångströms to nanometers, and interpolates the 
data onto a unified wavelength grid WITHOUT applying LSF degradation to the target Y.
The resulting matrices are saved as single-precision float32 NumPy arrays.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from gaiaxpy import calibrate
except ImportError:
    calibrate = None


def fetch_real_gaia_spectrum(source_id: int, wave_grid: np.ndarray) -> np.ndarray:
    """
    Obtiene el espectro calibrado real de Gaia BP/RP mediante GaiaXPy.
    Lanza una excepción si falla la consulta (sin código de fallback derivado de Y).
    """
    if calibrate is None:
        raise ImportError(
            "La librería 'gaiaxpy' es necesaria para consultar la API de Gaia. "
            "Instálala con: pip install gaiaxpy"
        )

    try:
        # Consulta y calibración oficial sobre la rejilla de longitudes de onda
        calibrated_spectra, _ = calibrate(
            [int(source_id)], 
            sampling=wave_grid, 
            truncation=False, 
            verbose=False
        )
        flux = np.array(calibrated_spectra["flux"].iloc[0], dtype=np.float32)

        if np.any(np.isnan(flux)) or len(flux) == 0:
            raise ValueError(f"El espectro devuelto para Gaia Source ID {source_id} contiene valores NaN.")

        return flux

    except Exception as e:
        # ESTRICTO: Lanza el error directamente para evitar contaminar X
        raise RuntimeError(
            f"No se pudo recuperar el espectro real de Gaia para Source ID {source_id}: {str(e)}"
        ) from e


def load_calspec_pristine(file_path: str, wave_grid: np.ndarray) -> np.ndarray:
    """
    Carga el espectro de CALSPEC e interpola a la rejilla definida
    SIN aplicar ninguna convolución de degradación LSF.
    """
    if file_path.endswith(".npy"):
        data = np.load(file_path, allow_pickle=True).item()
        raw_wave = data["wave"]
        raw_flux = data["flux"]
    elif file_path.endswith(".fits"):
        from astropy.io import fits
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            # CORRECCIÓN: CALSPEC usa Angstroms, convertimos a nanómetros dividiendo por 10.0
            raw_wave = data['WAVELENGTH'] / 10.0
            raw_flux = data['FLUX']
    else:
        # Asume CSV
        df_spec = pd.read_csv(file_path)
        raw_wave = df_spec["wavelength"].values
        raw_flux = df_spec["flux"].values

    # Interpolación directa a la nueva rejilla sin filtro gaussiano ni LSF
    y_pristine = np.interp(wave_grid, raw_wave, raw_flux)
    return y_pristine.astype(np.float32)


def create_dataset(csv_path: str, calspec_dir: str, output_dir: str, n_channels: int, wave_min_nm: float, wave_max_nm: float):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Crear la rejilla dinámicamente
    wave_grid = np.linspace(wave_min_nm, wave_max_nm, n_channels)
    
    # 2. Comprobar catálogo
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el catálogo de entrada en {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Intentar reutilizar X real antiguo si existe para ahorrar peticiones
    # Asumimos que la ruta equivalente degradada sigue la misma estructura de nombre
    mode = Path(output_dir).name
    degraded_x_cache = Path(f"data/processed/nn/{mode}/X.npy")
    
    cached_X = None
    if degraded_x_cache.exists():
        print(f"[INFO] Reutilizando matriz X (datos reales de Gaia) desde: {degraded_x_cache}")
        cached_X = np.load(degraded_x_cache)

    X_list = []
    Y_list = []
    
    print(f"[INFO] Procesando {len(df)} estrellas para dataset NO DEGRADADO (Modo: {mode})...")

    for idx, row in df.iterrows():
        source_id = row["gaia_source_id"]
        
        # Construir ruta al FITS original de CALSPEC leyendo la columna 'fits_file'
        filename = row["fits_file"]
        calspec_path = os.path.join(calspec_dir, filename)
        
        try:
            # --- Y: CALSPEC Prístino ---
            y_pristine = load_calspec_pristine(calspec_path, wave_grid)

            # --- X: Gaia Real ---
            if cached_X is not None and idx < len(cached_X):
                x_gaia = cached_X[idx].astype(np.float32)
            else:
                x_gaia = fetch_real_gaia_spectrum(source_id, wave_grid)

            X_list.append(x_gaia)
            Y_list.append(y_pristine)

        except Exception as err:
            print(f"[WARN] Omite estrella {idx} (Gaia: {source_id}) -> {err}")
            continue

    if len(X_list) == 0:
        raise RuntimeError("[ERROR CRÍTICO] Ninguna estrella pudo ser procesada correctamente.")

    # 3. Guardar matrices finales
    X_final = np.array(X_list, dtype=np.float32)
    Y_final = np.array(Y_list, dtype=np.float32)

    np.save(out_path / "X.npy", X_final)
    np.save(out_path / "Y.npy", Y_final)
    np.save(out_path / "wave_grid.npy", wave_grid)

    print("\n[ÉXITO] Dataset generado correctamente:")
    print(f" - Matriz X (Gaia real, sin artefactos): {X_final.shape}")
    print(f" - Matriz Y (CALSPEC prístino, alta res): {Y_final.shape}")
    print(f" - Directorio: {out_path}")


if __name__ == "__main__":
    mode = "all"
    create_dataset(
        csv_path = f"data/processed/catalogs_by_color/{mode}_catalog.csv",
        calspec_dir = "data/raw/calspec_fits",
        output_dir = f"data/processed/nn_undegraded/{mode}",
        n_channels = 300,
        wave_min_nm = 340.0,
        wave_max_nm = 1000.0
    )