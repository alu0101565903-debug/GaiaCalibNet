# GaiaCalibNet

GaiaCalibNet is a Python pipeline for spectral recalibration with Gaia DR3 and CALSPEC data. The main workflow is:

1. build a CALSPEC reference catalog,
2. cross-match it to Gaia DR3,
3. download the required spectral files,
4. prepare the training arrays,
5. train a PINN-based spectral correction model,
6. run validation or comparison experiments.

## Repository structure

- `data/`
  - `raw/` - source catalogs and downloaded CALSPEC FITS files
  - `processed/` - processed crossmatch outputs and ML-ready dataset arrays
- `GaiaXPcorrection/` - comparison/reference correction utilities shipped with the project
- `results/`
  - `models/` - trained model checkpoints and output bundles
  - `diagrams/` - architecture figures and generated diagrams
- `src/`
  - `data_processing/` - catalog generation, crossmatching, downloads, and dataset preparation
  - `train/` - training entry points, losses, and baseline scripts
  - `models/` - PINN model architectures
  - `experiments/` - validation and inference experiment scripts
  - `utils/` - plotting, metrics, and helper utilities
- `requirements.txt` and `pyproject.toml` - package dependencies and project metadata

## Setup

Create a virtual environment and install the project dependencies:

```bash
cd /home/yaiza/GaiaCalibNet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Core workflow

The current project is centered on the main data-processing and training flow below. These are the scripts that are part of the active repository pipeline.

### 1. Build the CALSPEC catalog

```bash
python3 -m src.data_processing.create_catalog
```

This creates the base catalog used for Gaia cross-matching.

### 2. Cross-match CALSPEC objects with Gaia DR3

```bash
python3 -m src.data_processing.crossmatch
```

This builds the Gaia/CALSPEC association table used later in the data-preparation step.

### 3. Download CALSPEC FITS spectra

```bash
python3 -m src.data_processing.download_calspec
```

The downloaded files are expected under `data/raw/calspec_fits/`.

### 4. Prepare the degraded/training dataset

```bash
python3 -m src.data_processing.prepare_degraded_dataset
```

This step generates the processed arrays used by the training pipeline, such as the spectral matrices and derived stellar features.

## Training

### Full-dataset PINN training

Run the main training pipeline:

```bash
python3 -m src.train.mains.main_pinn
```

This script trains the PINN pipeline on the prepared dataset and saves the model artifacts in `results/models/full_pinn_training/` by default. The training configuration is defined directly in the `if __name__ == "__main__":` block of `src/train/mains/main_pinn.py`.

The script expects processed arrays such as:

- `X.npy`
- `Y.npy`
- `X_error.npy`
- `wave_grid.npy`
- `bp_rp.npy`
- `G_mags.npy`
- `ruwe.npy`
- `gaia_source_ids.npy`

The generated output includes training checkpoints, normalization statistics, and export metadata used for inference.

### Baseline comparison

A reference baseline is available in:

```bash
python3 -m src.train.mains.huang_baseline
```

This script evaluates the Huang-style empirical correction baseline for comparison with the PINN results.

### Leave-one-out validation

To run the residual PINN leave-one-out experiment:

```bash
python3 -m src.experiments.loocv_residual_pinn
```

This is a validation-oriented experiment and is substantially more expensive than the full-dataset training run. It writes results under `results/models/loocv_residual_pinn/`.

## Additional utilities

The repository also contains auxiliary scripts for diagnostics and analysis, including:

- `src/experiments/inference/lowlib_inference.py` - model application on external reference spectra
- `src/utils/visualize_spectra/` - plotting utilities for spectra inspection
- `src/utils/visualize_models_performance/` - visualization of training and validation diagnostics
- `src/utils/diagram_pinn_architecture/` - architecture diagram generation

These are useful for analysis and figure generation, but they are not the core data-production steps required to build the training dataset or fit the main model.

## Notes

- The repository assumes the required CALSPEC FITS files are available in `data/raw/calspec_fits/` before training.
- The main active pipeline is the sequence: catalog build -> crossmatch -> download -> dataset preparation -> PINN training.
- Model implementations live under `src/models/`, while training logic and experiment entry points are organized under `src/train/` and `src/experiments/`.

## Testing

A test suite can be run with:

```bash
python3 -m pytest
```

