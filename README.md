# GaiaCalibNet

GaiaCalibNet learns a spectral correction from Gaia DR3 XP spectra to CALSPEC
reference spectra degraded to Gaia's resolution. The neural ensemble combines a
global calibration model with a specialized correction below 450 nm, using
BP-RP, G magnitude, and RUWE as auxiliary inputs.

The workflow is:

1. Build and cross-match the reference catalog, then select the training stars.
2. Download CALSPEC spectra and prepare aligned Gaia/CALSPEC arrays.
3. Validate the approach with the available cross-validation and baseline scripts.
4. Refit the final model and preprocessing on all available samples.
5. Apply the saved correction to a chosen Gaia source ID.

If you already have a trained `model_bundle.pt`, you can go directly to
[correcting a Gaia spectrum](#correct-a-gaia-spectrum).

## Repository structure

| Location | Purpose |
| --- | --- |
| `src/data_processing/` | Catalog creation, crossmatching, downloads, and dataset preparation |
| `src/models/` | Neural architectures and inference wrappers |
| `src/train/` | Training entry points, losses, and baseline evaluation |
| `src/experiments/` | Cross-validation and inference workflows |
| `src/utils/` | Preprocessing, metrics, plotting, and model-loading helpers |
| `examples/` | Editable examples, including correction of a selected Gaia spectrum |
| `tests/` | Inference and training/export regression tests |
| `data/raw/` | Source catalogs and CALSPEC FITS spectra |
| `data/processed/` | Selected catalogs and training arrays |
| `results/` | Model bundles, configurations, metrics, plots, and corrected spectra |
| `GaiaXPcorrection/` | Reference empirical correction used for baseline comparisons |
| `report/` | Report and presentation sources |

## Setup

Use Python 3.11 or newer. From your local repository directory:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

For baseline and cross-validation comparisons, initialize the reference-correction
submodule if it is not already available:

```bash
git submodule update --init --recursive GaiaXPcorrection
```

Run the commands below from the repository root with the virtual environment
activated. Configuration is set in the scripts' `if __name__ == "__main__"`
blocks or through their Python function arguments.

Catalog downloads, crossmatching, dataset acquisition, and Gaia-ID inference
require internet access. Training uses the prepared local arrays. Downloaded data
and generated model files are excluded from Git, so a fresh checkout does not
supply the datasets or a trained checkpoint.

## Prepare the dataset

Skip this section if the required arrays already exist in
`data/processed/nn_data/`.

### 1. Build the CALSPEC catalog

```bash
python -m src.data_processing.create_catalog
```

This writes `data/raw/calspec_catalog.csv`.

### 2. Cross-match with Gaia DR3

```bash
python -m src.data_processing.crossmatch
```

This writes `data/processed/crossmatch_catalog.csv` with Gaia source IDs,
CALSPEC FITS filenames, and stellar properties.

### 3. Select the training catalog

Review the matched sources and save the selected rows as
`data/processed/training_catalog.csv`. Preserve the catalog columns, including
`calspec_name`, `fits_file`, `gaia_source_id`, `phot_g_mean_mag`, `bp_rp`, and
`ruwe`. Keep Gaia IDs as full decimal strings to avoid losing digits.

This is a separate selection step: the crossmatch script writes
`crossmatch_catalog.csv`, while both subsequent scripts default to
`training_catalog.csv`. The active pipeline does not automatically create the
selected catalog. If you intend to use all matches, save all its rows under the
training-catalog filename.

### 4. Download the selected CALSPEC spectra

```bash
python -m src.data_processing.download_calspec
```

This reads `data/processed/training_catalog.csv` and downloads the referenced
FITS files into `data/raw/calspec_fits/`.

### 5. Prepare aligned spectral arrays

```bash
python -m src.data_processing.prepare_degraded_dataset
```

This reconstructs Gaia XP spectra, degrades the CALSPEC references, and writes
arrays to `data/processed/nn_data/`. The default wavelength grid is 336–1020 nm
with 2 nm spacing.

The training pipeline loads:

| File | Contents |
| --- | --- |
| `X.npy` | Gaia flux spectra |
| `Y.npy` | CALSPEC reference flux spectra on the same grid |
| `X_error.npy` | Gaia flux uncertainties |
| `wave_grid.npy` | Wavelengths in nm |
| `bp_rp.npy`, `G_mags.npy`, `ruwe.npy` | Stellar features |
| `gaia_source_ids.npy` | Source IDs aligned with the spectral rows |

Preparation also saves derived arrays and metadata such as `ruwe_missing.npy`
and `star_names.npy`. Missing RUWE remains missing until its replacement value
is fitted on the training subset. CALSPEC FITS files are needed for dataset
preparation; training itself reads the prepared arrays.

## Validate before the final fit

### Leave-one-out cross-validation

```bash
python -m src.experiments.loocv_residual_pinn
```

This trains a separate model for each held-out star. PCA, stellar normalization,
and RUWE imputation are fitted on each fold's training stars. The current entry
point writes results under `results/models/loocv_eps_residual_pinn/`, including
`loocv_aggregated_predictions.npz`, metrics, and comparison plots. This requires
many more training runs than the final all-samples fit.

### Empirical baseline comparison

```bash
python -m src.train.mains.huang_baseline
```

This compares the uncorrected Gaia spectra and the Huang empirical correction
against the prepared reference data. Outputs go to `results/models/baselines/`.

The repository includes these validation scripts, but no automated
hyperparameter-search entry point. The final training script uses the fixed
configuration documented below; changing it requires editing its main block or
calling `train_full_pinn(...)` with different arguments.

For a single held-out split, call `train_full_pinn(n_test_samples=...)` with a
separate `output_dir`. Such a run trains only on the non-reserved stars and
exports test predictions and metrics. It is distinct from the final all-samples
checkpoint described below.

## Train the final model

After validation and hyperparameter selection, run:

```bash
python -m src.train.mains.main_pinn
```

The current entry point uses `n_test_samples=0`. It refits both neural networks,
the PCA basis, stellar normalization, and RUWE imputation on all samples loaded
from `data/processed/nn_data/`. The final artifacts are saved under:

```text
results/models/full_pinn_training/all_samples/
```

The entry point currently uses 20 PCA components, 600 base-model epochs, 200
blue-correction epochs, and seed 42. Both learning rates are `1e-3`; base and
blue weight decay are `1e-4` and `0`, respectively. The loss weights are
`1e-4` for base smoothness and regularization, and `1e-4` / `1e-3` for blue
smoothness / regularization.

| Artifact | Purpose |
| --- | --- |
| `model_bundle.pt` | Self-contained model used by Gaia-ID inference |
| `training_config.json` | Hyperparameters, sample counts, training scope, and export paths |
| `train_indices.npy`, `test_indices.npy` | Dataset partition; the final fit has no test indices |
| `training_loss_history.npz`, `training_loss.pdf` | Training losses and their plot |
| `model_ensemble_state.pth` | Ensemble state dictionary |
| PCA, wavelength, and stellar-statistics `.npy` files | Separate copies of preprocessing information |

The bundle contains both networks' weights, their required dimensions, PCA basis
and mean, wavelength grid, blue-region mask, stellar mean and standard deviation,
and the training RUWE median. Inference does not require separate `.npy` files.

The final fit does not produce independent test metrics. Use the earlier
validation results to assess generalization; training error on all samples is
not an independent accuracy estimate.

Training can also be called from Python:

```python
from src.train.mains.main_pinn import train_full_pinn

result = train_full_pinn()  # Run once after choosing the configuration.
```

The return value is a configuration dictionary, not the network object:

- `result["model_dir"]`: absolute directory accepted by the inference function.
- `result["model_bundle_path"]`: absolute path to the saved bundle.
- `result["trained_on_all_samples"]`: whether no stars were reserved for testing.

## Correct a Gaia spectrum

### Run the example

Edit `gaia_source_id` in
[examples/correct_gaia_spectrum.py](examples/correct_gaia_spectrum.py), then run:

```bash
python examples/correct_gaia_spectrum.py
```

You can also edit `model_dir`, `device`, and `output_dir` in its main block. By
default it loads `results/models/full_pinn_training/all_samples/model_bundle.pt`
and saves:

```text
results/gaia_inference/gaia_<source_id>_corrected.csv
```

The CSV columns are `wavelength_nm` and `corrected_flux_W_m2_nm`. This example
performs inference only: train the final model first if its bundle is absent.

### Call the function

```python
from src.experiments.inference.gaia_inference import calibrate_gaia_spectrum

wavelength, corrected_flux = calibrate_gaia_spectrum("2462426800883134336")
```

The function retrieves the source's continuous Gaia DR3 XP spectrum and its
BP-RP, G magnitude, and RUWE. GaiaXPy samples the spectrum directly on the saved
model grid, then the neural ensemble applies its learned correction. The returned
NumPy arrays contain wavelengths in **nm** and corrected flux density in
**W/m²/nm**. The function itself does not write files.

No LOWLIB or CALSPEC reference is needed for inference. The source must have a
published continuous XP spectrum and usable BP-RP and G measurements. Missing
RUWE uses the saved training median; unavailable spectra or required features
raise an error. Archive/network errors propagate to the caller.

To use the result of a training call or a different saved model:

```python
wavelength, corrected_flux = calibrate_gaia_spectrum(
    "2462426800883134336",
    model_dir=result["model_dir"],
    device="cpu",
)
```

You can also supply a model directory directly. The default all-samples path is
resolved relative to the repository; an explicitly supplied relative path is
resolved from the current working directory. CUDA is selected automatically
when available unless `device` is specified. Inference never trains a model or
silently substitutes a different checkpoint when its bundle is missing.

## Additional utilities

- [lowlib_inference.py](src/experiments/inference/lowlib_inference.py) compares a
  corrected Gaia spectrum with a downloaded LOWLIB reference. Configure the
  matching `target_name`, `gaia_source_id`, stellar properties, and `model_dir`
  first. Its current model-directory setting points to the parent training
  folder; set it to an actual run such as
  `results/models/full_pinn_training/all_samples`. Unlike Gaia-ID inference,
  this validation script needs a LOWLIB counterpart and separate
  `stellar_mean_p.npy` / `stellar_std_p.npy` files from the same run.
- `src/utils/visualize_spectra/` contains spectrum and catalog inspection tools.
- `src/utils/visualize_models_performance/` contains prediction plots and metrics.
- `src/utils/diagram_pinn_architecture/` contains architecture-diagram utilities.

## Tests

```bash
python -m pytest
```

The tests cover Gaia-ID validation, mocked archive responses, feature handling,
and the training → bundle export → inference handoff. The export tests train
tiny synthetic datasets; they do not rerun the full scientific training or make
live Gaia requests.
