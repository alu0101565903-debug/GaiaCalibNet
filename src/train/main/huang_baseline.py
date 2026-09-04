"""
Module: huang_baseline.py
Report Section: 5.0 Empirical Baselines & Comparative Evaluation

Standalone evaluation script for the empirical baseline from Huang et al.

This module isolates the test set split, applies the official `GaiaDR3XPspectracorrectionV1`
correction algorithm, and evaluates both raw uncalibrated Gaia spectra and Huang-calibrated
spectra using the standard evaluation module (`metrics_eval.py`). Baseline predictions are
aliased under the `"X_pinn"` array key within temporary `.npz` files to maintain uniform
metrics generation.
"""

from pathlib import Path
from typing import Union
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from GaiaXPcorrection.GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
from src.utils.visualize_models_performance.metrics_eval import (
    evaluate_test_set_from_file,
)

# Suppress non-critical third-party runtime warnings during evaluation
warnings.filterwarnings("ignore")


def run_huang_evaluation(
    base_data_dir: Union[str, Path] = "data/processed/nn/all",
    test_size: float = 0.2,
    random_state: int = 42,
    output_base_dir: Union[str, Path] = "results/models/baselines",
) -> None:
    """
    Execute evaluation pipeline for Huang et al. baseline and raw Gaia XP spectra.

    Parameters
    ----------
    base_data_dir : str or Path, default="data/processed/nn/all"
        Path to directory containing input NumPy dataset files.
    test_size : float, default=0.2
        Proportion of dataset allocated to the test split.
    random_state : int, default=42
        Seed used to replicate identical dataset split across evaluation scripts.
    output_base_dir : str or Path, default="results/models/baselines"
        Output directory for saving generated evaluation NPZ archives and plots.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If required dataset files are missing from `base_data_dir`.
    """
    print("=" * 60)
    print("  BASELINE EVALUATION: HUANG ET AL. & UNCALIBRATED GAIA ")
    print("=" * 60)

    base_dir = Path(base_data_dir).resolve()
    out_base = Path(output_base_dir).resolve()

    # 1. Load dataset matrices
    print("[>] Loading regression dataset...")
    try:
        X_np = np.load(base_dir / "X.npy")
        Y_np = np.load(base_dir / "Y.npy")
        X_err_np = np.load(base_dir / "X_error.npy")
        G_mags = np.load(base_dir / "G_mags.npy")
        gaia_ids = np.load(base_dir / "gaia_source_ids.npy")
    except FileNotFoundError as e:
        print(f"[!] Error: Could not find dataset files in '{base_dir}': {e}")
        return

    # 2. Replicate train-test split for unbiased evaluation
    (
        _, X_test, 
        _, Y_test, 
        _, X_err_test,
        _, G_mags_test,
        _, ids_test
    ) = train_test_split(
        X_np, Y_np, X_err_np, G_mags, gaia_ids, 
        test_size=test_size, random_state=random_state
    )

    n_test_samples = X_test.shape[0]
    print(f"[>] Samples in Blind Test Set: {n_test_samples}")

    # 3. Format input for Huang et al. correction module
    print("[>] Applying Huang et al. correction...")
    
    df_huang_input = pd.DataFrame({
        'source_id': np.asarray(ids_test).flatten(),
        'flux': list(X_test),
        'flux_error': list(X_err_test)
    })
    
    g_mags_series = pd.Series(np.asarray(G_mags_test).flatten())

    try:
        flux_out = Gaia_Correction_V1.correction_df(
            df_huang_input, 
            g_mags_series, 
            have_error=True, 
            Truncation=True
        )
        X_huang_test = np.vstack(flux_out['flux_cor'].values)
        print("[✓] Huang correction applied successfully.")
        
    except Exception as e:
        print(f"[!] Failed to execute Huang model correction: {e}")
        return

    # =====================================================================
    # 4. PREPARE DATA ARCHIVES FOR METRICS_EVAL MODULE (ALIASING AS "X_pinn")
    # =====================================================================
    print("\n[>] Preparing .npz archives for evaluation module...")
    
    # Standard wavelength sampling grid (336.0 nm to 1020.0 nm across 343 channels)
    wave = np.linspace(336.0, 1020.0, 343)
    
    # Create distinct target directories for raw and Huang baseline evaluation
    out_dir_uncalib = out_base / "uncalibrated"
    out_dir_huang = out_base / "huang"
    out_dir_uncalib.mkdir(parents=True, exist_ok=True)
    out_dir_huang.mkdir(parents=True, exist_ok=True)
    
    npz_uncalib_path = out_dir_uncalib / "eval_data.npz"
    npz_huang_path = out_dir_huang / "eval_data.npz"

    # Save uncalibrated spectra (X_test) aliased as "X_pinn"
    np.savez(
        npz_uncalib_path, 
        wave=wave, 
        Y_ref=Y_test, 
        X_pinn=X_test,       # Alias raw spectra to match standard evaluation key
        X_err=X_err_test
    )
    
    # Save Huang-calibrated spectra (X_huang_test) aliased as "X_pinn"
    np.savez(
        npz_huang_path, 
        wave=wave, 
        Y_ref=Y_test, 
        X_pinn=X_huang_test, # Alias Huang corrected spectra to match standard evaluation key
        X_err=X_err_test
    )

    # =====================================================================
    # 5. EXECUTE METRICS EVALUATION PIPELINE
    # =====================================================================
    print("\n[>>] RUNNING EVALUATION: UNCALIBRATED GAIA DATA")
    # Pass num_knots=0 because uncalibrated raw data contains no model knot parameters
    evaluate_test_set_from_file(str(npz_uncalib_path), str(out_dir_uncalib), num_knots=0, model_name="Uncalibrated Gaia")

    print("\n[>>] RUNNING EVALUATION: HUANG ET AL. BASELINE")
    # Pass num_knots=0 because Huang et al. is an empirical lookup correction
    evaluate_test_set_from_file(str(npz_huang_path), str(out_dir_huang), num_knots=0, model_name="Huang et al. Baseline")
    print("\n[✓] All baseline evaluations completed successfully!")
    print(f"[✓] Output directory (Uncalibrated): '{out_dir_uncalib}'")
    print(f"[✓] Output directory (Huang et al.): '{out_dir_huang}'")


if __name__ == "__main__":
    run_huang_evaluation(base_data_dir="data/processed/nn/all")