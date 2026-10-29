"""
Train the full Gaia-to-CALSPEC PINN model, optionally reserving a test subset.

This script fits the same direct calibration pipeline used in the LOOCV setup,
trains it on all non-reserved spectra, and saves the artifacts required to apply
the learned correction to new Gaia spectra afterwards.

Saved artifacts:
- model weights for the full ensemble
- PCA basis/mean for the calibration ratio reconstruction
- stellar mean/std for parameter standardization
- wavelength grid and metadata needed for inference
- a JSON config capturing the training setup
- optional test predictions, metrics, and plots

The output directory is meant to be reusable in inference with the project helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.models.pinn_architecture import BlueWiggleNet, DirectConvPINN
from src.models.pinn_export_models import EnsembleExportModel
from src.train.loss.physics_loss import BluePhysicsLoss, DirectPhysicsLoss
from src.train.training_functions.pinn_trainer import train_fold
from src.utils.processing.pinn_data import apply_huang_correction, fit_log_ratio_pca, load_pinn_dataset
from src.utils.reproducibility_utils import set_seed
from src.utils.processing.stellar_loader import standardize_stellar_parameters
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import (
    generate_plots_from_file,
    plot_training_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/processed/nn_data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results/models/full_pinn_training/all_samples"


def train_full_pinn(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    num_components: int = 20,
    lambda_smooth: float = 1e-4,
    lambda_reg: float = 1e-4,
    epochs_adam: int = 600,
    epochs_blue: int = 200,
    random_state: int = 42,
    n_test_samples: int = 0,
    device: str | torch.device | None = None,
    *,
    lr_base: float = 1e-3,
    lr_blue: float = 1e-3,
    weight_decay_base: float = 1e-4,
    weight_decay_blue: float = 0.0,
    lambda_reg_blue: float = 1e-3,
    lambda_smooth_blue: float = 1e-4,
) -> Dict[str, Any]:
    """Train with the selected seed-42 hyperparameters, optionally reserving stars.

    Defaults match candidate 12 from the hyperparameter search, with both
    network widths fixed at 128 and BP-RP, G and RUWE as auxiliary inputs.
    This performs one fit, with the seed set once at the start. The default
    ``n_test_samples=0`` refits the ensemble, PCA, and feature normalization
    on all available samples after hyperparameter selection and validation.

    Returns
    -------
    dict of str to Any
        Training configuration and export locations. ``model_dir`` is the
        absolute directory accepted by ``calibrate_gaia_spectrum``;
        ``model_bundle_path`` identifies its self-contained checkpoint.
        ``trained_on_all_samples`` records whether any stars were reserved.
        Returning the network object is unnecessary for file-based inference.

    Examples
    --------
    >>> result = train_full_pinn()
    >>> from src.experiments.inference.gaia_inference import calibrate_gaia_spectrum
    >>> wave, flux = calibrate_gaia_spectrum(
    ...     "2462426800883134336", model_dir=result["model_dir"]
    ... )
    """
    set_seed(random_state)
    data_dir = Path(data_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    X, Y, E, params_raw, wave, g_mags, source_ids = load_pinn_dataset(data_dir, standardize_params=False)

    n_samples = X.shape[0]
    if n_test_samples < 0 or n_test_samples >= n_samples:
        raise ValueError(
            f"n_test_samples must be between 0 and {n_samples - 1}; "
            f"got {n_test_samples}."
        )
    if n_samples - n_test_samples < num_components:
        raise ValueError(
            "The number of non-reserved stars must be at least "
            f"num_components ({num_components}); got {n_samples - n_test_samples}."
        )

    sample_order = np.random.default_rng(random_state).permutation(n_samples)
    test_indices = np.sort(sample_order[:n_test_samples])
    train_indices = np.sort(sample_order[n_test_samples:])
    np.save(out_dir / "train_indices.npy", train_indices)
    np.save(out_dir / "test_indices.npy", test_indices)

    params, mean_p, std_p, ruwe_median = standardize_stellar_parameters(params_raw, train_indices)
    X_train, Y_train = X[train_indices], Y[train_indices]
    E_train, params_train = E[train_indices], params[train_indices]

    # PCA basis for the absolute calibration ratio R(lambda)
    pca_basis, pca_mean = fit_log_ratio_pca(
        X_train, Y_train, num_components=num_components, random_state=random_state
    )

    np.save(out_dir / "pca_basis.npy", pca_basis)
    np.save(out_dir / "pca_mean.npy", pca_mean)

    np.save(out_dir / "stellar_mean_p.npy", mean_p)
    np.save(out_dir / "stellar_std_p.npy", std_p)
    np.save(out_dir / "ruwe_imputation_median.npy", ruwe_median)
    np.save(out_dir / "ruwe_missing.npy", np.isnan(params_raw[:, 2]))
    np.save(out_dir / "wave_grid.npy", wave)

    blue_mask = wave < 450.0
    num_blue_channels = int(np.sum(blue_mask))

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    E_t = torch.tensor(E_train, dtype=torch.float32, device=device)
    P_t = torch.tensor(params_train, dtype=torch.float32, device=device)

    hidden_dim = 128  # Match the architecture used in the winning search.
    base_model = DirectConvPINN(
        input_dim=X.shape[1],
        pca_basis=pca_basis,
        pca_mean=pca_mean,
        num_params=params.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)

    blue_model = BlueWiggleNet(
        input_dim=num_blue_channels,
        num_params=params.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer_base = torch.optim.Adam(
        base_model.parameters(), lr=lr_base, weight_decay=weight_decay_base
    )
    optimizer_blue = torch.optim.Adam(
        blue_model.parameters(), lr=lr_blue, weight_decay=weight_decay_blue
    )

    criterion_base = DirectPhysicsLoss(lambda_smooth=lambda_smooth, lambda_reg=lambda_reg).to(device)
    criterion_blue = BluePhysicsLoss(
        lambda_reg=lambda_reg_blue, lambda_smooth=lambda_smooth_blue
    ).to(device)

    trained_model, trained_blue_model, training_history = train_fold(
        model=base_model,
        blue_model=blue_model,
        criterion_base=criterion_base,
        criterion_blue=criterion_blue,
        optimizer_base=optimizer_base,
        optimizer_blue=optimizer_blue,
        X_train=X_t,
        Y_train=Y_t,
        P_train=P_t,
        E_train=E_t,
        blue_mask=blue_mask,
        epochs_adam=epochs_adam,
        epochs_blue=epochs_blue,
        return_history=True,
    )

    history_filepath = out_dir / "training_loss_history.npz"
    np.savez_compressed(history_filepath, **training_history)
    plot_training_history(history_filepath, out_dir / "training_loss.pdf")

    ensemble = EnsembleExportModel(
        base_model=trained_model,
        blue_model=trained_blue_model,
        blue_mask=blue_mask,
        wave_grid=wave,
    ).to(device)

    ensemble.eval()
    bundle_path = out_dir / "model_bundle.pt"
    trained_on_all_samples = len(test_indices) == 0

    ensemble_state = ensemble.state_dict()
    torch.save(ensemble_state, out_dir / "model_ensemble_state.pth")
    torch.save({
        "base_model": trained_model.state_dict(),
        "blue_model": trained_blue_model.state_dict(),
        "blue_mask": blue_mask,
        "wave_grid": wave,
        "stellar_mean": mean_p,
        "stellar_std": std_p,
        "ruwe_imputation_median": ruwe_median,
        "pca_basis": pca_basis,
        "pca_mean": pca_mean,
        "num_components": num_components,
        "num_params": params.shape[1],
        "input_dim": X.shape[1],
        "hidden_dim": hidden_dim,
        "trained_on_all_samples": trained_on_all_samples,
        "n_samples": n_samples,
        "n_train_samples": len(train_indices),
        "n_test_samples": len(test_indices),
    }, bundle_path)

    config = {
        "random_state": random_state,
        "normalization": "fit on training stars only",
        "ruwe_imputation": "median of observed training RUWE before normalization",
        "ruwe_imputation_median": ruwe_median,
        "n_missing_ruwe_train": int(np.isnan(params_raw[train_indices, 2]).sum()),
        "n_missing_ruwe_test": int(np.isnan(params_raw[test_indices, 2]).sum()),
        "device": str(device),
        "num_components": num_components,
        "lambda_smooth": lambda_smooth,
        "lambda_reg": lambda_reg,
        "epochs_adam": epochs_adam,
        "epochs_blue": epochs_blue,
        "lr_base": lr_base,
        "lr_blue": lr_blue,
        "weight_decay_base": weight_decay_base,
        "weight_decay_blue": weight_decay_blue,
        "lambda_reg_blue": lambda_reg_blue,
        "lambda_smooth_blue": lambda_smooth_blue,
        "use_ruwe": True,
        "blue_hidden_dim": hidden_dim,
        "input_dim": X.shape[1],
        "hidden_dim": hidden_dim,
        "n_samples": n_samples,
        "n_train_samples": len(train_indices),
        "n_test_samples": len(test_indices),
        "num_params": params.shape[1],
        "blue_threshold_nm": 450.0,
        "blue_mask_n_channels": num_blue_channels,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "model_dir": str(out_dir),
        "model_bundle_path": str(bundle_path),
        "trained_on_all_samples": trained_on_all_samples,
    }

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if n_test_samples:
        test_results_filepath = out_dir / "test_predictions.npz"
        X_test_t = torch.tensor(X[test_indices], dtype=torch.float32, device=device)
        Y_test_t = torch.tensor(Y[test_indices], dtype=torch.float32, device=device)
        E_test_t = torch.tensor(E[test_indices], dtype=torch.float32, device=device)
        P_test_t = torch.tensor(params[test_indices], dtype=torch.float32, device=device)
        G_mags_test_t = torch.tensor(g_mags[test_indices], dtype=torch.float32, device=device)
        X_huang_test = apply_huang_correction(
            X[test_indices],
            E[test_indices],
            source_ids[test_indices],
            g_mags[test_indices],
        )
        X_huang_test_t = torch.tensor(X_huang_test, dtype=torch.float32, device=device)

        export_test_predictions(
            model=ensemble,
            X_tensor=X_test_t,
            Y_tensor=Y_test_t,
            X_err_tensor=E_test_t,
            output_filepath=str(test_results_filepath),
            params_tensor=P_test_t,
            wave=wave,
            G_mags_tensor=G_mags_test_t,
            X_huang_tensor=X_huang_test_t
        )
        evaluate_test_set_from_file(
            input_filepath=str(test_results_filepath),
            output_dir=str(out_dir / "test_metrics"),
            num_knots=num_components,
            model_name="PINN Test Set"
        )
        generate_plots_from_file(
            input_filepath=str(test_results_filepath),
            output_dir=str(out_dir / "test_plots"),
            num_samples=n_test_samples,
            include_gaussian=False
        )

    print(f"[✓] Full PINN training complete.")
    print(f"[✓] Artifacts saved to: {out_dir}")
    print(f"[✓] Inference bundle: {bundle_path}")
    return config


if __name__ == "__main__":
    train_full_pinn(
        data_dir=DEFAULT_DATA_DIR,
        output_dir=DEFAULT_MODEL_DIR,
        num_components=20,
        lambda_smooth=1e-4,
        lambda_reg=1e-4,
        epochs_adam=600,
        epochs_blue=200,
        random_state=42,
        lr_base=1e-3,
        lr_blue=1e-3,
        weight_decay_base=1e-4,
        weight_decay_blue=0.0,
        lambda_reg_blue=1e-3,
        lambda_smooth_blue=1e-4,
        n_test_samples=0,
        device=None,
    )
