"""
Correct a selected Gaia DR3 spectrum with the saved neural ensemble.

Edit the configuration in the main block, then run from the repository root:
    python examples/correct_gaia_spectrum.py

Notes
-----
Train the final model once with ``python -m src.train.mains.main_pinn`` before
running this example. Internet access is required to retrieve the Gaia spectrum
and stellar properties. The output CSV contains wavelength in nm and corrected
flux density in W/m^2/nm.
"""

from pathlib import Path
import sys

import numpy as np


# Support direct execution from either the repository root or examples/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.inference.gaia_inference import (
    DEFAULT_MODEL_DIR,
    calibrate_gaia_spectrum,
)


if __name__ == "__main__":
    # Configuration: choose the Gaia DR3 source to correct.
    gaia_source_id = "2462426800883134336"
    model_dir = DEFAULT_MODEL_DIR
    device = None  # Automatic selection; use "cpu" or "cuda" to override.
    output_dir = PROJECT_ROOT / "results" / "gaia_inference"

    print(f"[>] Correcting Gaia DR3 source {gaia_source_id}...")
    wavelength, corrected_flux = calibrate_gaia_spectrum(
        gaia_source_id,
        model_dir=model_dir,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gaia_{gaia_source_id}_corrected.csv"
    np.savetxt(
        output_file,
        np.column_stack((wavelength, corrected_flux)),
        delimiter=",",
        header="wavelength_nm,corrected_flux_W_m2_nm",
        comments="",
    )
    print(f"[✓] Saved {len(wavelength)} corrected spectral samples to {output_file}")
