"""Single-source inference tests with mocked Gaia services."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from astropy.table import Table, MaskedColumn

from src.experiments.inference import gaia_inference as inference

SOURCE_ID = "2462426800883134336"


class FixedRatio(torch.nn.Module):
    def forward(self, flux, params):
        self.params_seen = params.detach().cpu().numpy()
        self.flux_seen = flux.detach().cpu().numpy()
        return torch.full_like(flux, 2.)


@pytest.fixture
def setup_inference(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_bundle.pt").touch()
    wave = np.arange(336., 346., 2., dtype=np.float32)
    flux = np.arange(1., 6., dtype=np.float32) * 1e-15
    metadata = {
        "wave_grid": wave, "num_params": 3,
        "stellar_mean": np.array([1., 10., 2.]),
        "stellar_std": np.array([2., 2., .5]),
        "ruwe_imputation_median": 2.,
    }
    model = FixedRatio()
    loader = Mock(return_value=(model, metadata))
    monkeypatch.setattr(inference, "load_ensemble_from_bundle", loader)
    monkeypatch.setattr(inference, "DEFAULT_MODEL_DIR", model_dir)
    catalog = Table({
        "source_id": [int(SOURCE_ID)], "bp_rp": [.5], "phot_g_mean_mag": [11.],
        "ruwe": [1.5], "has_xp_continuous": [True],
    })
    job = Mock()
    job.get_results.return_value = catalog
    gaia = ModuleType("astroquery.gaia")
    gaia.Gaia = SimpleNamespace(launch_job=Mock(return_value=job))
    gaiaxpy = ModuleType("gaiaxpy")
    gaiaxpy.calibrate = Mock(return_value=(pd.DataFrame({"source_id": [int(SOURCE_ID)], "flux": [flux]}), wave))
    monkeypatch.setitem(sys.modules, "astroquery.gaia", gaia)
    monkeypatch.setitem(sys.modules, "gaiaxpy", gaiaxpy)
    return SimpleNamespace(model_dir=model_dir, wave=wave, flux=flux, metadata=metadata,
                           model=model, loader=loader, job=job, catalog=catalog,
                           gaia=gaia.Gaia, calibrate=gaiaxpy.calibrate)


def test_download_features_and_apply_neural_correction(setup_inference):
    env = setup_inference
    wave, corrected = inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")
    np.testing.assert_array_equal(wave, env.wave)
    np.testing.assert_allclose(corrected, 2*env.flux, rtol=1e-6, atol=0)
    np.testing.assert_allclose(env.model.params_seen, [[-.25, .5, -1.]])
    np.testing.assert_array_equal(env.model.flux_seen, [env.flux])
    env.calibrate.assert_called_once()
    assert env.calibrate.call_args.args == ([SOURCE_ID],)
    assert env.calibrate.call_args.kwargs["save_file"] is False
    np.testing.assert_array_equal(env.calibrate.call_args.kwargs["sampling"], env.wave)
    assert f"WHERE source_id = {SOURCE_ID}" in env.gaia.launch_job.call_args.args[0]
    assert env.gaia.launch_job.call_args.kwargs["dump_to_file"] is False


@pytest.mark.parametrize("source_id", [int(SOURCE_ID), np.int64(SOURCE_ID), f"  0{SOURCE_ID}  "])
def test_ids_preserve_integer_precision(source_id, setup_inference):
    env = setup_inference
    inference.calibrate_gaia_spectrum(source_id, env.model_dir, device="cpu")
    assert env.calibrate.call_args.args[0] == [SOURCE_ID]


@pytest.mark.parametrize("source_id", [True, 0, -1, 1.5, float(SOURCE_ID), "", "1 OR 1=1", "1.0", "１２３", 2**63])
def test_invalid_id_rejected_before_loading_or_network(source_id, setup_inference):
    env = setup_inference
    with pytest.raises(ValueError, match="gaia_source_id"):
        inference.calibrate_gaia_spectrum(source_id, env.model_dir)
    env.loader.assert_not_called()
    env.gaia.launch_job.assert_not_called()


def test_missing_bundle_fails_before_network(setup_inference, tmp_path):
    env = setup_inference
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        inference.calibrate_gaia_spectrum(SOURCE_ID, tmp_path / "absent")
    env.gaia.launch_job.assert_not_called()


def test_missing_ruwe_uses_saved_training_median(setup_inference):
    env = setup_inference
    env.catalog.replace_column("ruwe", MaskedColumn([1.], mask=[True]))
    inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")
    assert env.model.params_seen[0, 2] == 0.


def test_missing_ruwe_without_saved_median_is_rejected(setup_inference):
    env = setup_inference
    env.catalog["ruwe"][0] = np.nan
    env.metadata.pop("ruwe_imputation_median")
    with pytest.raises(ValueError, match="Missing RUWE"):
        inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")


def test_missing_stellar_feature_is_rejected(setup_inference):
    env = setup_inference
    env.catalog.replace_column("bp_rp", MaskedColumn([1.], mask=[True]))
    with pytest.raises(ValueError, match="missing BP-RP or G"):
        inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")
    env.calibrate.assert_not_called()


def test_missing_source_is_rejected(setup_inference):
    env = setup_inference
    env.job.get_results.return_value = env.catalog[:0]
    with pytest.raises(ValueError, match="was not found"):
        inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")
    env.calibrate.assert_not_called()


def test_source_without_xp_is_rejected(setup_inference):
    env = setup_inference
    env.catalog["has_xp_continuous"][0] = False
    with pytest.raises(ValueError, match="no published continuous XP"):
        inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")
    env.calibrate.assert_not_called()


@pytest.mark.parametrize("failure", ["empty", "wrong_id", "nan_flux", "wrong_grid"])
def test_invalid_download_is_rejected(failure, setup_inference):
    env = setup_inference
    spectra, wave = env.calibrate.return_value
    if failure == "empty": spectra = pd.DataFrame()
    elif failure == "wrong_id": spectra.loc[0, "source_id"] = 123
    elif failure == "nan_flux": env.flux[0] = np.nan
    else: wave = wave + 1
    env.calibrate.return_value = spectra, wave
    with pytest.raises(ValueError):
        inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")


def test_archive_error_propagates(setup_inference):
    env = setup_inference
    env.gaia.launch_job.side_effect = ConnectionError("Archive unavailable")
    with pytest.raises(ConnectionError, match="Archive unavailable"):
        inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")


def test_two_feature_bundle(setup_inference):
    env = setup_inference
    env.metadata["num_params"] = 2
    env.metadata["stellar_mean"] = env.metadata["stellar_mean"][:2]
    env.metadata["stellar_std"] = env.metadata["stellar_std"][:2]
    inference.calibrate_gaia_spectrum(SOURCE_ID, env.model_dir, device="cpu")
    np.testing.assert_allclose(env.model.params_seen, [[-.25, .5]])
