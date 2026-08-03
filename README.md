# IAC_GAIA

A spectral recalibration pipeline for Gaia DR3 and CALSPEC data. The project combines catalog construction, Gaia/CALSPEC crossmatching, dataset preparation, baseline regression training, and physics-informed neural network (PINN) experiments for spectral correction.

## Repository Structure

- `data/`
  - `raw/` - raw inputs such as CALSPEC FITS files and source catalogs
  - `processed/` - processed data artifacts such as spectra matrices, labels, and crossmatch outputs
- `GaiaXPcorrection/` - reference correction utilities and supporting files
- `results/`
  - `models/` - saved checkpoints and trained models
  - `plots/` - figures and diagnostic outputs
- `src/`
  - `processing/` - catalog generation, crossmatching, downloads, and dataset preparation
  - `train/` - training entry points and training utilities
  - `models/` - baseline and PINN model implementations
  - `experiments/` - evaluation and experiment scripts
  - `utils/` - plotting and spectral helper utilities
- `requirements.txt` - Python dependencies
- `report/` - LaTeX report and supporting files

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
cd /home/yaiza/IAC_GAIA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Typical Workflow

### 1. Build the base CALSPEC catalog

Generate the initial CALSPEC catalog from the remote STScI index:

```bash
python3 -m src.processing.create_catalog
```

This writes the raw catalog to `data/raw/calspec_catalog.csv`.

### 2. Cross-match CALSPEC targets with Gaia DR3

Create the Gaia/CALSPEC crossmatch catalog:

```bash
python3 -m src.processing.crossmatch
```

This produces `data/processed/crossmatch_catalog.csv`.

### 3. Download CALSPEC FITS files

Download the required CALSPEC FITS spectra referenced by the catalog:

```bash
python3 -m src.processing.download_calspec
```

Files are stored under `data/raw/calspec_fits/`.

### 4. Prepare the regression datasets

Prepare the degraded or mock datasets used for training and evaluation:

```bash
python3 -m src.processing.prepare_degraded_dataset
```

or

```bash
python3 -m src.processing.prepare_mock_dataset
```

These scripts generate the processed arrays used by the training pipeline.

### 5. Extract Gaia scalar features (optional)

If you want the feature-based branch, run:

```bash
python3 -m src.processing.extract_features
```

This creates `data/processed/X_features.npy` when the feature extraction pipeline is available.

## Training

### Classical baselines

Train and evaluate baseline models:

```bash
python3 -m src.train.mains.train_baselines
```

### PINN and related neural models

Several training entry points are available under `src/train/mains/`:

```bash
python3 -m src.train.mains.train_neural_net
python3 -m src.train.mains.train_pca_nn
python3 -m src.train.mains.train_pol_nn
python3 -m src.train.mains.train_simplified_nn
```

Outputs are written under `results/models/`.

## Evaluation and Visualization

The repository includes utilities for plotting and inspecting spectra and model outputs. Examples are available through the scripts in `src/experiments/` and `src/utils/visualize_spectra/`.

## Key Scripts

- `src/processing/create_catalog.py` - build the base CALSPEC catalog
- `src/processing/crossmatch.py` - crossmatch CALSPEC objects with Gaia DR3
- `src/processing/download_calspec.py` - download CALSPEC FITS spectra
- `src/processing/prepare_degraded_dataset.py` - prepare degraded spectral datasets
- `src/processing/prepare_mock_dataset.py` - prepare mock datasets for experimentation
- `src/processing/extract_features.py` - extract Gaia scalar features for alternative training
- `src/train/mains/train_baselines.py` - train classical regressors
- `src/train/mains/train_neural_net.py` - train neural network variants
- `src/train/mains/train_pca_nn.py` - PCA-based neural training entry point
- `src/train/mains/train_pol_nn.py` - polynomial-neural training entry point
- `src/train/mains/train_simplified_nn.py` - simplified neural training entry point

## Recent Updates

- The requested source modules now have English comments and docstrings for better readability.
- The processing pipeline includes catalog generation, Gaia crossmatching, and dataset preparation entry points.
- The repository keeps generated artifacts and checkpoints under `data/processed/` and `results/`.

## Notes

- The workflow assumes that the required CALSPEC FITS files are available under `data/raw/calspec_fits/`.
- The pipeline uses a common wavelength grid and normalized spectral representations for training.
- Model implementations are defined in `src/models/` and training logic is organized under `src/train/`.

## Testing

A lightweight import smoke test is available:

```bash
python3 -m pytest
```

You can target the import test directly with:

```bash
python3 -m pytest tests/test_imports.py
```
