# IAC_GAIA

A spectral recalibration pipeline for Gaia DR3 and CALSPEC data. This repository supports data crossmatching, CALSPEC FITS download and processing, dataset generation, classical baseline training, physics-informed neural network (PINN) training, and model evaluation.

## Repository Structure

- `data/`
  - `raw/` - raw input files, including CALSPEC FITS and initial catalogs
  - `processed/` - processed dataset artifacts (`X.npy`, `Y.npy`, `star_names.npy`, `crossmatch_catalog.csv`, `X_features.npy`)
- `results/`
  - `models/` - saved model checkpoints for baselines and PINN
  - `plots/` - evaluation plots and visualizations
- `src/`
  - `processing/` - data preparation and feature extraction
  - `train/` - training code for PINN and baseline models
  - `models/` - neural network and baseline model definitions
  - `experiments/` - evaluation and experiment scripts
  - `utils/` - plotting utilities and spectra helpers
- `requirements.txt` - Python dependencies
- `report/` - project report files and LaTeX sources

## Setup

Recommended environment setup:

```bash
cd /home/yaiza/IAC_GAIA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Basic Workflow

### 1. Cross-match CALSPEC with Gaia DR3

Generate the processed crossmatch catalog from the CALSPEC source list:

```bash
python3 -m src.processing.crossmatch_wd
```

This creates `data/processed/crossmatch_catalog.csv` by querying Gaia DR3 and filtering for high-quality XP spectra.

### 2. Download missing CALSPEC FITS files

Fetch the required CALSPEC FITS files referenced by the catalog:

```bash
python3 -m src.processing.download_calspec
```

Downloaded FITS files are saved under `data/raw/calspec_fits/`.

### 3. Prepare the regression dataset

Build the spectral dataset matrices and save them as NumPy binaries:

```bash
python3 -m src.processing.prepare_dataset
```

This generates:

- `data/processed/X.npy` - input spectra or feature matrices
- `data/processed/Y.npy` - target normalized flux spectra
- `data/processed/star_names.npy` - sample names

### 4. Extract Gaia scalar features (optional)

If you want the feature-based training branch, extract the five Gaia scalar features:

```bash
python3 -m src.processing.extract_features
```

This creates `data/processed/X_features.npy`.

## Training

### Classical baseline models

Train and evaluate Ridge, Random Forest, and SVR baselines:

```bash
python3 -m src.train.mains.train_baselines
```

Results are stored in `results/models/baselines/`.

### Physics-Informed Neural Network (PINN)

Train the PINN model using the preprocessed data:

```bash
python3 -m src.train.mains.train_neural_net
```

This script supports two modes:

- `spectrum` - uses `data/processed/X.npy` and `data/processed/Y.npy`
- `features` - uses `data/processed/X_features.npy` and `data/processed/Y.npy`

Saved checkpoints are written to `results/models/pinn/`.

## Evaluation

Evaluate saved models and generate comparison plots:

```bash
python3 -m src.experiments.evaluate
```

If the required checkpoint files are missing, `evaluate.py` can train fallback models temporarily and still produce metrics and plot output.

## Key Scripts

- `src/processing/crossmatch_wd.py` - crossmatch CALSPEC targets to Gaia DR3
- `src/processing/download_calspec.py` - download CALSPEC FITS spectra
- `src/processing/prepare_dataset.py` - produce `X.npy`, `Y.npy`, and normalized targets
- `src/processing/extract_features.py` - extract Gaia scalar features for alternative training
- `src/train/mains/train_baselines.py` - train classical baseline regressors
- `src/train/mains/train_neural_net.py` - train the PINN model and evaluate on validation set
- `src/experiments/evaluate.py` - compute metrics and save spectral comparison plots

## Recent Repository Updates

- Added missing dependencies to `requirements.txt`: `joblib`, `requests`
- Updated `.gitignore` to exclude generated outputs and checkpoints: `results/`, `*.joblib`, `*.pt`, `.pytest_cache/`, `*.log`
- Enhanced `src/experiments/evaluate.py` with fallback dataset loading and temporary model training when checkpoint files are absent
- Added compatibility alias in `src/models/neural_net.py` so legacy imports of `CalspecFluxMLP` still work

## Notes

- The repository assumes `data/raw/calspec_fits/` contains the target `.fits` files referenced by the processed crossmatch catalog.
- The processing pipeline normalizes spectra using trapezoidal integration and aligns all spectra on a common wavelength grid.
- The neural model architecture is implemented in `src/models/neural_net.py`, and baseline regressors are defined in `src/models/baselines.py`.

## Requirements

Install dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Troubleshooting

- If a required dataset file is missing, verify the outputs of the processing scripts in `data/processed/`.
- If model weights are not available, run the training scripts before evaluation.
- Use the active Python virtual environment to ensure imports resolve correctly.

## Testing

A lightweight import test is included to verify that core modules can be loaded successfully.

```bash
python3 -m pytest
```

If you want to run only the smoke test for import sanity, use:

```bash
python3 -m pytest tests/test_imports.py
```
