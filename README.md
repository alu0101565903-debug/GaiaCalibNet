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
python3 -m src.train.mains.main_pinn
```

The main trains the base calibration model followed by the specialized blue
region correction. Configure the run by editing the `train_full_pinn(...)`
call in the `if __name__ == "__main__":` block of
`src/train/mains/main_pinn.py`. There is no command-line argument parser.
The available settings are:

```text
data_dir         Dataset directory (default: data/processed/nn_data)
output_dir       Output directory (default: results/models/full_pinn_training)
num_components   Number of PCA components (default: 20)
epochs_adam      Base-model epochs (default: 600)
epochs_blue      Blue-correction epochs (default: 150)
lambda_smooth    Spectral smoothness weight (default: 1e-4)
lambda_reg       Regularization weight (default: 1e-4)
lr_base          Base Adam learning rate (default: 1e-3)
lr_blue          Blue Adam learning rate (default: 1e-3)
weight_decay_base  Base Adam weight decay (default: 1e-4)
weight_decay_blue  Blue Adam weight decay (default: 0)
lambda_reg_blue    Blue identity penalty (default: 1e-3)
lambda_smooth_blue Blue smoothness weight (default: 1e-4)
random_state     Random seed (default: 42)
n_test_samples   Stars reserved for testing (default: 0)
device           Torch device (default: None, automatically selected)
```

The training defaults match the selected search configuration (candidate 12,
seed 42), including hidden widths of 128 for both networks, blue dropout of 0.2,
and BP-RP, G and RUWE inputs. `training_config.json` records the learning rates,
weight decay and loss weights used. The current main block reserves three test
stars and writes to `results/models/full_pinn_training/3_test_samples`; set
`n_test_samples=0` to train a final inference model on all available stars.

The output directory contains:

- `model_ensemble_state.pth` - ensemble weights for inference
- `model_bundle.pt` - weights plus model metadata in one PyTorch bundle
- `pca_basis.npy` and `pca_mean.npy` - PCA calibration-ratio parameters
- `stellar_mean_p.npy` and `stellar_std_p.npy` - stellar-feature normalization
- `ruwe_imputation_median.npy` - observed training RUWE median, also stored in the bundle
- `ruwe_missing.npy` - missing-value mask aligned with the dataset rows
- `wave_grid.npy` - wavelength grid used by the model
- `training_config.json` - training configuration and dataset dimensions

The dataset directory must contain `X.npy`, `Y.npy`, `X_error.npy`,
`wave_grid.npy`, `bp_rp.npy`, `G_mags.npy`, `ruwe.npy`, and `gaia_source_ids.npy`.
Missing RUWE is stored as NaN with a boolean `ruwe_missing.npy` mask. Imputation
uses the median of observed training values before fitting the scaler. The mask
is metadata, not an extra network input. In inference, pass `ruwe=None` or NaN
to reuse the bundle's median; older bundles without it require observed RUWE.

To reserve a reproducible test set, set `n_test_samples=3` and
`output_dir=Path("results/models/full_pinn_training/3_test_stars")` in that call,
then run the same module command above.

The reserved stars are excluded from PCA fitting, parameter-normalization fitting,
and model training. Stellar parameters, including RUWE, use training-only
normalization without clipping in both evaluation and inference. The output
includes `train_indices.npy`, `test_indices.npy`, `test_predictions.npz`,
`test_metrics/`, and `test_plots/`. Training loss histories are saved in
`training_loss_history.npz` and `training_loss.pdf`.

### Leave-one-out residual PINN

Run the residual PINN experiment on the corrected dataset:

```bash
python3 -m src.experiments.loocv_residual_pinn
```

The experiment uses `data/processed/nn_data/` and writes its results to
`results/models/loocv_residual_pinn/`. The output includes the aggregated
predictions, training histories, PCA parameters, evaluation metrics, and
comparison plots. Since LOOCV trains one model per source, this experiment is
substantially more expensive than full-dataset training.

The script and `run_loocv` defaults use the selected configuration (candidate 12):
600 base epochs, 150 blue epochs, 20 PCA components, and learning rate `1e-3`
for both networks. Base/blue Adam weight decay is `1e-4`/`0`; base regularization
and smoothness are both `1e-4`, and blue regularization/smoothness are
`1e-3`/`1e-4`. RUWE is included and seed 42 is reset at the start of every fold.

LOOCV evaluation figures use the original layout, colours and Matplotlib
fonts, with English labels. Loss and aggregate comparison figures are saved as
PDFs, individual spectra as PNGs, and the four-column metrics table as LaTeX
and PNG. Loss curves show fold medians and interquartile ranges without
smoothing; epoch limits follow the saved histories. The hyperparameter-search
report retains its separate publication style.

LOOCV fits stellar-parameter normalization on the training stars separately in
each fold, including RUWE median imputation before normalization.
`fold_normalization.npz` stores each fold's mean, standard deviation, RUWE median,
missing-value mask, validation index, and feature names; the exported `params` row uses the scaler
of the fold that predicted that star. The first model's scaler is saved as
`stellar_mean_first_fold.npy` and `stellar_std_first_fold.npy`.
RUWE is standardized without clipping. To exclude RUWE
from both networks, call `run_loocv(..., use_ruwe=False)`.

### Hyperparameter comparison

Run the 24-candidate search on a completed dataset:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.experiments.tune_pinn_loocv --workers 4
```

Results are written to `results/experiments/pinn_hyperparameter_search/`. The
search freezes a copy of the dataset, fits all preprocessing inside each training
split, and repeats the full three-fold selection inside five outer folds. It
then evaluates the selected configuration in LOOCV with seeds 42 and 123 and
compares it with the reference configuration and Huang. Final fixed-configuration
LOOCV scores are post-selection diagnostics; the nested assessment evaluates the
selection procedure separately. Use a new `--output` directory if code or data
changes. Interrupted runs reuse completed fits when the manifest matches.

Export the report, per-star comparison and loss figures after completion:

```bash
python -m src.experiments.report_pinn_search results/experiments/pinn_hyperparameter_search
```

The report includes a `run_loocv` example loading
`recommended_loocv_kwargs.json`. The search fixes both hidden widths at 128 and
resets the training seed in each fold; use `reset_seed_each_fold=True` to reproduce
its results in the production LOOCV function.

To compare both variants with shared initial weights and train-only normalization:

```bash
python3 -m src.experiments.ruwe_ablation \
 --output-dir results/experiments/ruwe_comparison
```

This trains both phases on five reproducibly selected folds and reports the
diagnostic fold 70 separately. The experiment keeps `weight_decay=1e-4` fixed.

### Applying a trained correction

The shared inference function is available in `src/utils/pinn_inference.py`
as `calibrate_spectrum`. It receives a raw Gaia wavelength/flux pair and stellar
parameters, then returns the model wavelength grid, the interpolated Gaia flux,
and the corrected flux.

Example:

```python
import numpy as np
from src.utils.pinn_inference import calibrate_spectrum

output_dir = "results/models/full_pinn_training"
wave, raw_flux, corrected_flux = calibrate_spectrum(
    wave_raw=np.load("wave_raw.npy"),
    flux_raw=np.load("flux_raw.npy"),
    bp_rp=1.2,
    g_mag=15.4,
    ruwe=1.05,
    model_bundle_path=f"{output_dir}/model_bundle.pt",
    mean_p=np.load(f"{output_dir}/stellar_mean_p.npy"),
    std_p=np.load(f"{output_dir}/stellar_std_p.npy"),
)
```

## Evaluation and Visualization

### Compact residual calibration with nested validation

A new experimental alternative, `ResidualKernelCalibrator`, learns an additive
spectral residual with RBF kernel ridge regression. It uses each input spectrum's
own peak for normalization and selects hyperparameters inside every outer
leave-one-star-out fold. Run the complete comparison and export a fitted model:

```bash
python3 -m src.experiments.residual_kernel_comparison --output-dir results/experiments/residual_kernel_new_run --historical-archive results/models/loocv_residual_pinn/loocv_aggregated_predictions.npz
```

The historical archive is optional; Huang is recomputed from the local correction
tables. Use a fresh output directory. On the available 76-star sample, nested
PN-MAE is **0.403284%**, versus **0.736866%** for Huang (45.27% lower error).
This is internal validation after model-family exploration, not an external test.
The fitted model is a separate experimental alternative to the PINN.
See [the historical comparison, full results and inference example](docs/comparacion_metodos.md).

Evaluation tables report the mean across stars for each metric.

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
