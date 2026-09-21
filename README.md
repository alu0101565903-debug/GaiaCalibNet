# GaiaCalibNet

A spectral recalibration pipeline for Gaia DR3 and CALSPEC data. The project combines catalog construction, Gaia/CALSPEC crossmatching, dataset preparation, baseline regression training, and physics-informed neural network (PINN) experiments for spectral correction.

## Repository Structure

- `data/`
  - `raw/` - raw inputs such as CALSPEC FITS files and source catalogs
  - `processed/` - processed data artifacts such as spectra matrices, labels, and crossmatch outputs
- `GaiaXPcorrection/` - reference correction utilities and supporting files
- `results/`
  - `models/` - saved checkpoints and trained models
  - `lowlib_metrics/` - evaluation metrics and plots for the low-resolution library
- `src/`
  - `data_processing/` - catalog generation, crossmatching, downloads, and dataset preparation
  - `train/` - PINN losses, training functions, and entry points
  - `models/` - PINN architectures and export wrappers
  - `experiments/` - inference, LOOCV, and metric-analysis scripts
  - `utils/` - plotting and spectral helper utilities
- `requirements.txt` and `pyproject.toml` - Python dependencies and project metadata

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
cd /home/yaiza/GaiaCalibNet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Typical Workflow

### 1. Build the base CALSPEC catalog

Generate the initial CALSPEC catalog from the remote STScI index:

```bash
python3 -m src.data_processing.create_catalog
```

This writes the raw catalog to `data/raw/calspec_catalog.csv`.

### 2. Cross-match CALSPEC targets with Gaia DR3

Create the Gaia/CALSPEC crossmatch catalog:

```bash
python3 -m src.data_processing.crossmatch
```

This produces `data/processed/crossmatch_catalog.csv`.

### 3. Download CALSPEC FITS files

Download the required CALSPEC FITS spectra referenced by the catalog:

```bash
python3 -m src.data_processing.download_calspec
```

Files are stored under `data/raw/calspec_fits/`.

### 4. Prepare the regression datasets

Prepare the degraded or mock datasets used for training and evaluation:

```bash
python3 -m src.data_processing.prepare_degraded_dataset
```

These scripts generate the processed arrays used by the training pipeline.

## Training

### Full-dataset PINN training

To train the direct PINN ensemble on all prepared spectra and save an
inference-ready model bundle, run:

```bash
python3 -m src.train.mains.main_pinn \
  --data-dir data/processed/nn_data \
  --output-dir results/models/full_pinn_training
```

The main trains the base calibration model followed by the specialized blue
region correction. The most useful options are:

```text
--num-components   Number of PCA components (default: 40)
--epochs-adam      Base-model epochs (default: 600)
--epochs-blue      Blue-correction epochs (default: 150)
--lambda-smooth    Spectral smoothness weight
--lambda-reg       Regularization weight
--seed             Random seed (default: 42)
```

The output directory contains:

- `model_ensemble_state.pth` - ensemble weights for inference
- `model_bundle.pt` - weights plus model metadata in one PyTorch bundle
- `pca_basis.npy` and `pca_mean.npy` - PCA calibration-ratio parameters
- `stellar_mean_p.npy` and `stellar_std_p.npy` - stellar-feature normalization
- `wave_grid.npy` - wavelength grid used by the model
- `training_config.json` - training configuration and dataset dimensions

The dataset directory must contain `X.npy`, `Y.npy`, `X_error.npy`,
`wave_grid.npy`, `bp_rp.npy`, `G_mags.npy`, `ruwe.npy`,
`stellar_mean_p.npy`, and `stellar_std_p.npy`.

### Leave-one-out residual PINN

Run the residual PINN experiment on the corrected dataset:

```bash
python3 -m src.experiments.loocv_residual_pinn
```

The experiment uses `data/processed/nn_data_c/` and writes its results to
`results/models/loocv_residual_c_pinn/`. The output includes the aggregated
predictions, training histories, PCA parameters, evaluation metrics, and
comparison plots. Since LOOCV trains one model per source, this experiment is
substantially more expensive than full-dataset training.

### Applying a trained correction

The shared inference function is available in
`src/experiments/inference/lowlib_inference.py`. The full training module also
provides `calibrate_spectrum`. Both receive a raw Gaia wavelength/flux pair and
stellar parameters, then return the standard wavelength grid, the interpolated
Gaia flux, and the corrected flux.

Example:

```python
import numpy as np
from src.experiments.inference.lowlib_inference import calibrate_gaia_spectrum

output_dir = "results/models/full_pinn_training"
wave, raw_flux, corrected_flux = calibrate_gaia_spectrum(
    wave_gaia_raw=np.load("wave_raw.npy"),
    flux_gaia_raw=np.load("flux_raw.npy"),
    bp_rp=1.2,
    g_mag=15.4,
    ruwe=1.05,
    model_path=f"{output_dir}/model_ensemble_state.pth",
    pca_basis=np.load(f"{output_dir}/pca_basis.npy"),
    pca_mean=np.load(f"{output_dir}/pca_mean.npy"),
    mean_p=np.load(f"{output_dir}/stellar_mean_p.npy"),
    std_p=np.load(f"{output_dir}/stellar_std_p.npy"),
)
```

## Evaluation and Visualization

The repository includes utilities for plotting and inspecting spectra and model outputs. Examples are available through the scripts in `src/experiments/` and `src/utils/visualize_spectra/`.

## Key Scripts

- `src/data_processing/create_catalog.py` - build the base CALSPEC catalog
- `src/data_processing/crossmatch.py` - crossmatch CALSPEC objects with Gaia DR3
- `src/data_processing/download_calspec.py` - download CALSPEC FITS spectra
- `src/data_processing/prepare_degraded_dataset.py` - prepare degraded spectral datasets
- `src/train/mains/main_pinn.py` - train the full-dataset PINN and save inference artifacts
- `src/train/mains/huang_baseline.py` - run the Huang et al. Gaia XP correction baseline
- `src/experiments/loocv_residual_pinn.py` - run residual PINN leave-one-out validation
- `src/experiments/inference/lowlib_inference.py` - apply a trained model to a spectrum
- `src/experiments/inference/pinn_production.py` - production inference utilities

## Recent Updates

- The requested source modules now have English comments and docstrings for better readability.
- The processing pipeline includes catalog generation, Gaia crossmatching, CALSPEC downloads, and degraded-dataset preparation entry points.
- The repository keeps generated artifacts and checkpoints under `data/processed/` and `results/`.

## Notes

- The workflow downloads or otherwise assumes that the required CALSPEC FITS files are available under `data/raw/calspec_fits/`.
- The pipeline uses a common wavelength grid and normalized spectral representations for training.
- Model implementations are defined in `src/models/`, while losses and training logic are organized under `src/train/`.

## Testing

A test suite can be run with:

```bash
python3 -m pytest
```
