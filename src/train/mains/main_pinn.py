"""
Train the full Gaia-to-CALSPEC PINN model on the complete dataset.

This script fits the same direct calibration pipeline used in the LOOCV setup,
trains it on all available spectra, and saves the artifacts required to apply the
learned correction to new Gaia spectra afterwards.

Saved artifacts:
- model weights for the full ensemble
- PCA basis/mean for the calibration ratio reconstruction
- stellar mean/std for parameter standardization
- wavelength grid and metadata needed for inference
- a JSON config capturing the training setup

The output directory is meant to be reusable in inference with the project helpers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from src.models.pinn_architecture import BlueWiggleNet, DirectConvPINN
from src.models.pinn_export_models import EnsembleExportModel
from src.train.loss.physics_loss import BluePhysicsLoss, DirectPhysicsLoss
from src.train.training_functions.pinn_trainer import train_fold
from src.utils.reproducibility_utils import set_seed
from src.utils.stellar_loader import load_stellar_parameters


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load spectra, flux errors, stellar params and wavelength grid."""
    X = np.load(data_dir / "X.npy").astype(np.float32)
    Y = np.load(data_dir / "Y.npy").astype(np.float32)
    E = np.load(data_dir / "X_error.npy").astype(np.float32)
    wave = np.load(data_dir / "wave_grid.npy").astype(np.float32)
    params = load_stellar_parameters(data_dir)
    g_mags = np.load(data_dir / "G_mags.npy").astype(np.float32)
    return X, Y, E, params, wave, g_mags


def fit_pca_ratio(X: np.ndarray, Y: np.ndarray, num_components: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on the log calibration-ratio field using the whole dataset."""
    ratio = Y / np.clip(X, 1e-16, None)
    ratio = np.clip(ratio, 0.1, 15.0)
    ratio_log = np.log(ratio)
    ratio_log_smooth = gaussian_filter1d(ratio_log, sigma=0.5, axis=1)

    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(ratio_log_smooth)
    return pca.components_, pca.mean_


def train_full_pinn(
    data_dir: str | Path,
    output_dir: str | Path,
    num_components: int = 40,
    lambda_smooth: float = 1e-4,
    lambda_reg: float = 1e-4,
    epochs_adam: int = 600,
    epochs_blue: int = 150,
    random_state: int = 42,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    """Train the full ensemble on all spectra and save all artifacts for inference."""
    set_seed(random_state)
    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    X, Y, E, params, wave, _ = load_dataset(data_dir)

    # PCA basis for the absolute calibration ratio R(lambda)
    pca_basis, pca_mean = fit_pca_ratio(X, Y, num_components=num_components, random_state=random_state)

    np.save(out_dir / "pca_basis.npy", pca_basis)
    np.save(out_dir / "pca_mean.npy", pca_mean)

    mean_p = np.load(data_dir / "stellar_mean_p.npy")
    std_p = np.load(data_dir / "stellar_std_p.npy")
    np.save(out_dir / "stellar_mean_p.npy", mean_p)
    np.save(out_dir / "stellar_std_p.npy", std_p)
    np.save(out_dir / "wave_grid.npy", wave)

    blue_mask = wave < 450.0
    num_blue_channels = int(np.sum(blue_mask))

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    E_t = torch.tensor(E, dtype=torch.float32, device=device)
    P_t = torch.tensor(params, dtype=torch.float32, device=device)

    base_model = DirectConvPINN(
        input_dim=X.shape[1],
        pca_basis=pca_basis,
        pca_mean=pca_mean,
        num_params=params.shape[1],
    ).to(device)

    blue_model = BlueWiggleNet(
        input_dim=num_blue_channels,
        num_params=params.shape[1],
    ).to(device)

    optimizer_base = torch.optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer_blue = torch.optim.Adam(blue_model.parameters(), lr=1e-3)

    criterion_base = DirectPhysicsLoss(lambda_smooth=lambda_smooth, lambda_reg=lambda_reg).to(device)
    criterion_blue = BluePhysicsLoss(lambda_reg=1e-3, lambda_smooth=1e-4).to(device)

    trained_model, trained_blue_model = train_fold(
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
    )

    ensemble = EnsembleExportModel(
        base_model=trained_model,
        blue_model=trained_blue_model,
        blue_mask=blue_mask,
        wave_grid=wave,
    ).to(device)

    ensemble_state = ensemble.state_dict()
    torch.save(ensemble_state, out_dir / "model_ensemble_state.pth")
    torch.save({
        "base_model": trained_model.state_dict(),
        "blue_model": trained_blue_model.state_dict(),
        "blue_mask": blue_mask,
        "wave_grid": wave,
        "stellar_mean": mean_p,
        "stellar_std": std_p,
        "pca_basis": pca_basis,
        "pca_mean": pca_mean,
        "num_components": num_components,
        "num_params": params.shape[1],
        "input_dim": X.shape[1],
    }, out_dir / "model_bundle.pt")

    config = {
        "random_state": random_state,
        "device": str(device),
        "num_components": num_components,
        "lambda_smooth": lambda_smooth,
        "lambda_reg": lambda_reg,
        "epochs_adam": epochs_adam,
        "epochs_blue": epochs_blue,
        "input_dim": X.shape[1],
        "n_samples": X.shape[0],
        "num_params": params.shape[1],
        "blue_threshold_nm": 450.0,
        "blue_mask_n_channels": num_blue_channels,
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
    }

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[✓] Full PINN training complete.")
    print(f"[✓] Artifacts saved to: {out_dir}")
    return config


def calibrate_spectrum(
    wave_raw: np.ndarray,
    flux_raw: np.ndarray,
    bp_rp: float,
    g_mag: float,
    ruwe: float,
    model_bundle_path: str | Path,
    mean_p: np.ndarray,
    std_p: np.ndarray,
    device: str | torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the trained ensemble correction to a single Gaia spectrum.

    Returns wave_grid, original_flux, corrected_flux.
    """
    from scipy.interpolate import interp1d

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    bundle = torch.load(model_bundle_path, map_location=device)
    wave_grid = bundle["wave_grid"]
    blue_mask = bundle["blue_mask"]
    pca_basis = bundle["pca_basis"]
    pca_mean = bundle["pca_mean"]
    num_params = bundle["num_params"]

    wave_standard = np.arange(336.0, 1022.0, 2.0, dtype=np.float32)
    interp = interp1d(wave_raw, flux_raw, kind="cubic", bounds_error=False, fill_value="extrapolate")
    flux_interp = interp(wave_standard).astype(np.float32)

    params_raw = np.array([bp_rp, g_mag, ruwe], dtype=np.float32)
    params_std = (params_raw - mean_p) / (std_p + 1e-8)

    x_t = torch.tensor(flux_interp, dtype=torch.float32).unsqueeze(0).to(device)
    params_t = torch.tensor(params_std, dtype=torch.float32).unsqueeze(0).to(device)

    base_model = DirectConvPINN(
        input_dim=len(wave_standard),
        pca_basis=pca_basis,
        pca_mean=pca_mean,
        num_params=num_params,
    ).to(device)

    blue_model = BlueWiggleNet(
        input_dim=int(np.sum(blue_mask)),
        num_params=num_params,
    ).to(device)

    ensemble = EnsembleExportModel(
        base_model=base_model,
        blue_model=blue_model,
        blue_mask=blue_mask,
        wave_grid=wave_standard,
    ).to(device)

    ensemble.load_state_dict(torch.load(model_bundle_path, map_location=device))
    ensemble.eval()

    with torch.no_grad():
        r_final = ensemble(x_t, params_t).squeeze(0).cpu().numpy()

    corrected_flux = flux_interp * r_final
    return wave_standard, flux_interp, corrected_flux


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the full Gaia PINN calibration model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/nn_data"), help="Directory with X.npy, Y.npy, X_error.npy, ...")
    parser.add_argument("--output-dir", type=Path, default=Path("results/models/full_pinn_training"), help="Directory for saved weights and metadata")
    parser.add_argument("--num-components", type=int, default=40, help="Number of PCA components for the calibration ratio.")
    parser.add_argument("--lambda-smooth", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--epochs-adam", type=int, default=600)
    parser.add_argument("--epochs-blue", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_full_pinn(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_components=args.num_components,
        lambda_smooth=args.lambda_smooth,
        lambda_reg=args.lambda_reg,
        epochs_adam=args.epochs_adam,
        epochs_blue=args.epochs_blue,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
