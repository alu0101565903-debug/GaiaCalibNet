"""Verify the real training-to-inference bundle round trip on tiny datasets."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.train.mains import main_pinn as training
from src.experiments.inference import gaia_inference as inference
from src.models.pinn_export_models import EnsembleExportModel
from src.utils.processing.pinn_inference import load_ensemble_from_bundle
from src.utils.processing.stellar_loader import transform_stellar_parameters


@pytest.fixture
def synthetic_dataset(tmp_path):
    rng = np.random.default_rng(42)
    n_samples, n_wave = 8, 32
    wave = np.linspace(336., 1020., n_wave, dtype=np.float32)
    flux = (rng.uniform(.7, 1.3, (n_samples, n_wave)) * 1e-15).astype(np.float32)
    ratio = np.exp(rng.normal(0., .05, flux.shape)).astype(np.float32)
    features = np.column_stack((np.linspace(-.2, 1.5, n_samples),
                                np.linspace(8., 14., n_samples),
                                np.linspace(.8, 1.5, n_samples))).astype(np.float32)
    features[0, 2] = np.nan
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    arrays = {
        'X': flux, 'Y': flux*ratio, 'X_error': flux*.01, 'wave_grid': wave,
        'bp_rp': features[:, 0], 'G_mags': features[:, 1], 'ruwe': features[:, 2],
        'gaia_source_ids': np.arange(1, n_samples+1, dtype=np.int64),
    }
    for name, values in arrays.items():
        np.save(data_dir / f'{name}.npy', values)
    return data_dir, flux, features, wave


@pytest.mark.parametrize('n_test_samples', [0, 2])
def test_training_bundle_is_ready_for_gaia_inference(synthetic_dataset, tmp_path, monkeypatch, n_test_samples):
    data_dir, flux, features, wave = synthetic_dataset
    # Use real preprocessing, optimization, serialization, and model loading.
    # Plotting and the external empirical baseline are outside this contract.
    for name in ('plot_training_history', 'generate_plots_from_file',
                 'evaluate_test_set_from_file', 'export_test_predictions'):
        monkeypatch.setattr(training, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(training, 'apply_huang_correction', lambda X, *args: X)
    original_train_fold = training.train_fold
    trained = {}

    def train_and_capture(**kwargs):
        result = original_train_fold(**kwargs)
        trained['base'], trained['blue'] = result[:2]
        trained['rows_seen'] = kwargs['X_train'].shape[0]
        return result

    monkeypatch.setattr(training, 'train_fold', train_and_capture)
    monkeypatch.chdir(tmp_path)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = training.train_full_pinn(
            data_dir=data_dir, output_dir='export_model', num_components=2,
            epochs_adam=1, epochs_blue=1, n_test_samples=n_test_samples, device='cpu',
        )
        assert result['trained_on_all_samples'] is (n_test_samples == 0)
        assert result['n_train_samples'] == trained['rows_seen'] == len(flux)-n_test_samples
        bundle_path = Path(result['model_bundle_path'])
        model_dir = Path(result['model_dir'])
        assert bundle_path.is_absolute() and bundle_path.is_file()
        assert bundle_path == model_dir / 'model_bundle.pt'
        saved_config = json.loads((model_dir / 'training_config.json').read_text())
        assert saved_config == result
        train_indices = np.load(model_dir / 'train_indices.npy')
        assert len(np.load(model_dir / 'test_indices.npy')) == n_test_samples
        if n_test_samples == 0:
            np.testing.assert_array_equal(train_indices, np.arange(len(flux)))

        loaded_model, metadata = load_ensemble_from_bundle(bundle_path, torch.device('cpu'))
        assert not loaded_model.training and not loaded_model.blue_model.training
        assert metadata['trained_on_all_samples'] is (n_test_samples == 0)
        assert metadata['n_test_samples'] == n_test_samples
        assert metadata['n_train_samples'] == len(train_indices)
        median = np.nanmedian(features[train_indices, 2])
        assert metadata['ruwe_imputation_median'] == pytest.approx(median)
        imputed = features[train_indices].copy()
        imputed[np.isnan(imputed[:, 2]), 2] = median
        np.testing.assert_allclose(metadata['stellar_mean'], imputed.mean(axis=0))
        np.testing.assert_allclose(metadata['stellar_std'], imputed.std(axis=0))

        expected_model = EnsembleExportModel(trained['base'], trained['blue'], wave < 450., wave).eval()
        params = transform_stellar_parameters(features[0], metadata['stellar_mean'],
                                              metadata['stellar_std'], median)
        with torch.inference_mode():
            expected_ratio = expected_model(torch.tensor(flux[:1]), torch.tensor(params[None])).numpy()[0]

        # Only the Gaia download is mocked: inference loads the newly trained bundle.
        monkeypatch.setattr(inference, '_download_gaia_data', lambda source_id, grid: (flux[0], features[0]))
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        returned_wave, corrected = inference.calibrate_gaia_spectrum(
            '2462426800883134336', model_dir=result['model_dir'], device='cpu'
        )
        np.testing.assert_array_equal(returned_wave, wave)
        np.testing.assert_allclose(corrected, flux[0]*expected_ratio, rtol=1e-6, atol=0.)
        assert np.isfinite(corrected).all()
    finally:
        torch.set_num_threads(original_threads)
