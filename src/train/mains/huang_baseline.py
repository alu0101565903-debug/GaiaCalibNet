"""Evaluate the Huang et al. empirical spectral correction baseline.

The script isolates the test split, applies the official
``GaiaDR3XPspectracorrectionV1`` algorithm, and evaluates raw and corrected
spectra through the shared metrics module.

Notes
-----
Report context: Section 5.0, Empirical Baselines and Comparative Evaluation.
"""

import warnings
from pathlib import Path
from typing import Union

import numpy as np

from src.utils.processing.pinn_data import apply_huang_correction, load_pinn_dataset
from src.utils.visualize_models_performance.export_utils import save_prediction_archive
from src.utils.visualize_models_performance.metrics_eval import (
    evaluate_test_set_from_file,
)

# Suppress non-critical third-party runtime warnings during evaluation
warnings.filterwarnings("ignore")


def run_huang_evaluation(
    base_data_dir: Union[str, Path] = "data/processed/nn_data",
    output_base_dir: Union[str, Path] = "results/models/baselines",
) -> None:
    """
    Execute evaluation pipeline for Huang et al. baseline and raw Gaia XP spectra
    across the FULL dataset (76 stars), matching PINN LOOCV evaluation.
    """
    print("=" * 60)
    print(" BASELINE EVALUATION: HUANG ET AL. & UNCALIBRATED GAIA (FULL SAMPLE)")
    print("=" * 60)

    base_dir = Path(base_data_dir).resolve()
    out_base = Path(output_base_dir).resolve()

    # 1. Load the full dataset matrices (all 76 stars).
    print("[>] Loading full dataset...")
    try:
        X_test, Y_test, X_err_test, _, _, G_mags_test, ids_test = load_pinn_dataset(base_dir)
    except FileNotFoundError as e:
        print(f"[!] Error: Could not find dataset files in '{base_dir}': {e}")
        return

    n_test_samples = X_test.shape[0]
    print(f"[>] Total samples evaluated: {n_test_samples} stars")

    # 2. Apply the Huang et al. correction to the full dataset.
    print("[>] Applying Huang et al. correction...")
    try:
        X_huang_test = apply_huang_correction(
            X_test,
            X_err_test,
            ids_test,
            G_mags_test,
        )
        print("[✓] Huang correction applied successfully.")
        
    except Exception as e:
        print(f"[!] Failed to execute Huang model correction: {e}")
        return

    # 3. Prepare data archives for the metrics evaluation module.
    print("\n[>] Preparing .npz archives for evaluation module...")
    
    wave = np.linspace(336.0, 1020.0, X_test.shape[1])
    
    out_dir_uncalib = out_base / "uncalibrated"
    out_dir_huang = out_base / "huang"
    out_dir_uncalib.mkdir(parents=True, exist_ok=True)
    out_dir_huang.mkdir(parents=True, exist_ok=True)
    
    npz_uncalib_path = out_dir_uncalib / "eval_data.npz"
    npz_huang_path = out_dir_huang / "eval_data.npz"

    # Save uncalibrated spectra (full dataset)
    save_prediction_archive(
        output_filepath=str(npz_uncalib_path),
        wave=wave,
        X_orig=X_test,
        Y_ref=Y_test,
        X_pinn=X_test,
        X_err=X_err_test,
    )
    
    # Save Huang-calibrated spectra (full dataset)
    save_prediction_archive(
        output_filepath=str(npz_huang_path),
        wave=wave,
        X_orig=X_test,
        Y_ref=Y_test,
        X_pinn=X_huang_test,
        X_err=X_err_test,
    )

    # 4. Run the metrics evaluation pipeline.
    print("\n[>>] RUNNING EVALUATION: UNCALIBRATED GAIA DATA")
    evaluate_test_set_from_file(
        str(npz_uncalib_path), 
        str(out_dir_uncalib), 
        num_knots=0, 
        model_name="Uncalibrated Gaia"
    )

    print("\n[>>] RUNNING EVALUATION: HUANG ET AL. BASELINE")
    evaluate_test_set_from_file(
        str(npz_huang_path), 
        str(out_dir_huang), 
        num_knots=0, 
        model_name="Huang et al. Baseline"
    )
    print("\n[✓] All baseline evaluations completed successfully over all stars")


if __name__ == "__main__":
    run_huang_evaluation(base_data_dir="data/processed/nn_data")