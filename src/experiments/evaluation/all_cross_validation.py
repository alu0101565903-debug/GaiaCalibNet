"""
Module: all_cross_validation.py
Report Section:

Unified K-Fold Cross-Validation Pipeline for All Registered PINN Architectures.

This script generalizes the existing cross-validation workflow so it can train,
validate, and export results for every PINN architecture declared in
`src/models/pinn_architectures.py` using the matching loss function from
`src/train/loss/losses.py`.

Supported model keys:
- residual_bicubic
- residual_gaussian
- pca_nn
- pca_params_nn

The pipeline supports the following capabilities:
- fold-isolated PCA basis generation for PCA-based models
- proper dataset construction for parameter-conditioned models
- optional stochastic uncertainty handling for Gaussian PINNs
- per-fold model checkpointing, loss history export, prediction export,
  quantitative evaluation, and validation plots
"""

import os
from typing import Dict, Optional

import joblib
import numpy as np
import scipy.signal as signal
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from src.models.pinn_architectures import get_pinn_model
from src.train.loss.losses import (
    BiCubicPhysicsInformedLoss,
    GaussianPhysicsInformedLoss,
    PCAPhysicsInformedLoss,
    PCAParamsPhysicsInformedLoss,
    UndegradedPCAPhysicsInformedLoss
)
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed
from src.utils.extra.seed import set_seed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_loss import plot_loss_evolution
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file
from src.experiments.evaluation.plot_stellar_dependencies import evaluate_stellar_dependencies

SUPPORTED_MODELS = {
    "residual_bicubic",
    "residual_gaussian",
    "pca_nn",
    "pca_params_nn",
    "pca_undegraded_nn"
}

REQUIRES_STELLAR_PARAMS = {"pca_params_nn"}
REQUIRES_PCA_BASIS = {"pca_nn", "pca_params_nn", "pca_undegraded_nn"}
SUPPORTS_SIGMA = {"residual_gaussian"}


# =====================================================================
# DATA HELPERS
# =====================================================================

def load_stellar_parameters(params_dir: str) -> np.ndarray:
    """
    Load and standardize stellar parameters from a directory.

    This helper reads `bp_rp.npy`, `G_mags.npy` (or `G_mag.npy`), and
    `ruwe.npy`, applies per-feature Z-score normalization, and concatenates
    the results into a single array of shape (N, 3).
    """
    bp_rp_path = os.path.join(params_dir, "bp_rp.npy")
    g_mag_path = os.path.join(params_dir, "G_mags.npy")
    if not os.path.exists(g_mag_path):
        g_mag_path = os.path.join(params_dir, "G_mag.npy")
    ruwe_path = os.path.join(params_dir, "ruwe.npy")

    for p in [bp_rp_path, g_mag_path, ruwe_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required stellar parameter file not found: {p}")

    bp_rp = np.load(bp_rp_path).reshape(-1, 1)
    g_mag = np.load(g_mag_path).reshape(-1, 1)
    ruwe = np.load(ruwe_path).reshape(-1, 1)

    def standardize(column: np.ndarray) -> np.ndarray:
        return (column - np.mean(column)) / (np.std(column) + 1e-8)

    params = np.hstack([
        standardize(bp_rp),
        standardize(g_mag),
        standardize(ruwe),
    ])
    return params.astype(np.float32)


def compute_pca_basis(X: np.ndarray, Y: np.ndarray, num_components: int, random_state: int, window_length: int = 5) -> Dict[str, np.ndarray]:
    """
    Compute a fold-specific PCA basis from the empirical ratio R = Y / X.
    """
    R_raw = Y / (X + 1e-15)
    R_clipped = np.clip(R_raw, 0.5, 2.0)
    R_smooth = signal.savgol_filter(R_clipped, window_length=window_length, polyorder=2, axis=1, mode="nearest")

    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(R_smooth)

    return {
        "basis": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
    }


def build_model_and_loss(
    model_name: str,
    input_dim: int,
    num_stellar_params: int = 0,
    pca_basis: Optional[np.ndarray] = None,
    pca_mean: Optional[np.ndarray] = None,
    lambda_flux: float = 1.0,
    lambda_smooth: float = 1e-3,
    lambda_reg: float = 1e-4,
) -> Dict[str, object]:
    """
    Instantiate the requested model architecture and its matching loss module.
    """
    if model_name == "residual_bicubic":
        model = get_pinn_model("residual_bicubic", input_dim=input_dim)
        loss = BiCubicPhysicsInformedLoss(lambda_flux=lambda_flux, lambda_smooth=lambda_smooth)
        physics_criterion = lambda m, x, y, extra=None: loss(m, x, y)

    elif model_name == "residual_gaussian":
        model = get_pinn_model("residual_gaussian", input_dim=input_dim)
        loss = GaussianPhysicsInformedLoss()
        physics_criterion = lambda m, x, y, sigma=None: loss(m, x, y, sigma)

    elif model_name == "pca_nn":
        if pca_basis is None or pca_mean is None:
            raise ValueError("PCA basis and mean are required for pca_nn models.")
        model = get_pinn_model(
            "pca_nn",
            input_dim=input_dim,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
        )
        loss = PCAPhysicsInformedLoss(pca_basis=pca_basis, pca_mean=pca_mean)
        physics_criterion = lambda m, x, y, extra=None: loss(m, x, y)

    elif model_name == "pca_params_nn":
        if pca_basis is None or pca_mean is None:
            raise ValueError("PCA basis and mean are required for pca_params_nn models.")
        model = get_pinn_model(
            "pca_params_nn",
            input_dim=input_dim,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            num_stellar_params=num_stellar_params,
        )
        loss = PCAParamsPhysicsInformedLoss(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            lambda_smooth=lambda_smooth,
            lambda_reg=lambda_reg,
        )
        physics_criterion = lambda m, x, y, params=None: loss(m, x, y, params)

    elif model_name == "pca_undegraded_nn":
        if pca_basis is None or pca_mean is None:
            raise ValueError("PCA basis and mean are required for pca_undegraded_nn models.")
        
        # Asegúrate de haber registrado 'pca_undegraded_nn' en tu get_pinn_model
        model = get_pinn_model(
            "pca_undegraded_nn",
            input_dim=input_dim,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
        )
        loss = UndegradedPCAPhysicsInformedLoss(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            grad_weight=0.5,
            huber_beta=0.05  # <-- Añade esto para controlarlo centralizadamente
        )
        
        # Como la pérdida mejorada devuelve (total_loss, loss_flux, loss_grad), 
        # nos quedamos solo con total_loss [0] para la retropropagación del trainer.
        physics_criterion = lambda m, x, y, extra=None: loss(m, x, y)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return {
        "model": model,
        "loss": loss,
        "physics_criterion": physics_criterion,
    }


def prepare_fold_dataset(
    model_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_err_train: np.ndarray,
    params_train: Optional[np.ndarray],
    has_sigma: bool,
) -> (TensorDataset, Optional[torch.Tensor]):
    """
    Build the fold training dataset and the optional full-batch extra tensor.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)

    if model_name == "pca_params_nn":
        if params_train is None:
            raise ValueError("Stellar parameters are required for model 'pca_params_nn'.")
        params_train_t = torch.tensor(params_train, dtype=torch.float32)
        dataset = TensorDataset(X_train_t, Y_train_t, params_train_t)
        extra_full = params_train_t

    elif model_name == "residual_gaussian" and has_sigma:
        Sigma_train_t = torch.tensor(X_err_train, dtype=torch.float32)
        dataset = TensorDataset(X_train_t, Y_train_t, Sigma_train_t)
        extra_full = Sigma_train_t

    else:
        dataset = TensorDataset(X_train_t, Y_train_t)
        extra_full = None

    return dataset, extra_full


# =====================================================================
# CROSS-VALIDATION PIPELINE
# =====================================================================

def run_cross_validation(
    model_name: str,
    dataset_mode: str = "all",
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy",
    params_dir: str = "data/processed",
    output_model_dir: str = "results/models/cv_all",
    n_splits: int = 5,
    random_state: int = 42,
    num_components: int = 5,
    lambda_flux: float = 1.0,
    lambda_smooth: float = 1e-3,
    lambda_reg: float = 1e-4,
    epochs_adam: int = 1000,
    steps_lbfgs: int = 100,
    max_batch_size: int = 32,
    effective_params: int = 20,
) -> None:
    """
    Run a generic K-Fold cross-validation workflow for the selected PINN model.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Choose from {sorted(SUPPORTED_MODELS)}")

    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running cross-validation for '{model_name}' on device: {device}")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing dataset at '{x_path}' or '{y_path}'.")

    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path) if os.path.exists(x_err_path) else np.zeros_like(X_np, dtype=np.float32)
    has_sigma = os.path.exists(x_err_path)

    params_np = None
    if model_name in REQUIRES_STELLAR_PARAMS:
        params_np = load_stellar_parameters(params_dir)

    num_samples = X_np.shape[0]
    print(f"[>] Total samples loaded: {num_samples}")
    if params_np is not None:
        print(f"[>] Stellar parameters shape: {params_np.shape}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_np), 1):
        print(f"\n{'='*60}\n[>] Fold {fold}/{n_splits}\n{'='*60}")

        X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
        Y_train_np, Y_test_np = Y_np[train_idx], Y_np[test_idx]
        X_err_train_np, X_err_test_np = X_err_np[train_idx], X_err_np[test_idx]
        params_train_np = params_np[train_idx] if params_np is not None else None
        params_test_np = params_np[test_idx] if params_np is not None else None

        print(f"    Training samples: {len(X_train_np)} | Validation samples: {len(X_test_np)}")

        pca_basis = None
        pca_mean = None
        if model_name in REQUIRES_PCA_BASIS:
            pca_stats = compute_pca_basis(X_train_np, Y_train_np, num_components, random_state + fold)
            pca_basis = pca_stats["basis"]
            pca_mean = pca_stats["mean"]

        model_and_loss = build_model_and_loss(
            model_name=model_name,
            input_dim=X_train_np.shape[1],
            num_stellar_params=params_train_np.shape[1] if params_train_np is not None else 0,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            lambda_flux=lambda_flux,
            lambda_smooth=lambda_smooth,
            lambda_reg=lambda_reg,
        )

        model = model_and_loss["model"].to(device)
        physics_criterion = model_and_loss["physics_criterion"]

        dataset_train, extra_full_train = prepare_fold_dataset(
            model_name=model_name,
            X_train=X_train_np,
            Y_train=Y_train_np,
            X_err_train=X_err_train_np,
            params_train=params_train_np,
            has_sigma=has_sigma,
        )

        batch_size = min(max_batch_size, len(dataset_train))
        generator = torch.Generator()
        generator.manual_seed(random_state + fold)
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=generator)

        X_full_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        Y_full_train = torch.tensor(Y_train_np, dtype=torch.float32).to(device)
        params_full_train = (
            torch.tensor(params_train_np, dtype=torch.float32).to(device)
            if params_train_np is not None
            else None
        )
        extra_full_train = (
            torch.tensor(X_err_train_np, dtype=torch.float32).to(device)
            if model_name == "residual_gaussian" and has_sigma
            else params_full_train
        )

        model, loss_history = train_2stage_fixed(
            model=model,
            dataloader=dataloader,
            X_full=X_full_train,
            Y_full=Y_full_train,
            params_full=extra_full_train,
            physics_criterion=physics_criterion,
            epochs_adam=epochs_adam,
            steps_lbfgs=steps_lbfgs,
        )

        fold_dir = os.path.join(output_model_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.cpu().state_dict(),
            "model_name": model_name,
            "input_dim": X_train_np.shape[1],
            "output_dim": X_train_np.shape[1],
            "num_components": num_components,
            "lambda_flux": lambda_flux,
            "lambda_smooth": lambda_smooth,
            "lambda_reg": lambda_reg,
            "pca_basis": pca_basis,
            "pca_mean": pca_mean,
            "num_stellar_params": params_train_np.shape[1] if params_train_np is not None else 0,
        }
        joblib.dump(checkpoint, os.path.join(fold_dir, "model_checkpoint.joblib"))

        loss_plot_path = os.path.join(fold_dir, "loss_evolution.png")
        plot_loss_evolution(loss_history, loss_plot_path)

        params_test_tensor = (
            torch.tensor(params_test_np, dtype=torch.float32).to(device)
            if params_test_np is not None
            else None
        )

        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
        X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)

        export_test_predictions(
            model=model,
            X_tensor=X_test_tensor,
            Y_tensor=Y_test_tensor,
            X_err_tensor=X_err_test_tensor,
            params_tensor=params_test_tensor,
            output_filepath=os.path.join(fold_dir, "test_predictions.npz"),
        )

        evaluate_test_set_from_file(
            input_filepath=os.path.join(fold_dir, "test_predictions.npz"),
            output_dir=fold_dir,
            num_knots=num_components if model_name.startswith("pca") else effective_params,
        )

        generate_plots_from_file(
            input_filepath=os.path.join(fold_dir, "test_predictions.npz"),
            output_dir=os.path.join(fold_dir, "comparison"),
            num_samples=min(18, len(X_test_np)),
        )

    evaluate_stellar_dependencies(
        dataset_mode=dataset_mode,
        nn_mode = model_name,
        n_splits = n_splits, 
        random_state = random_state
    )   

    print(f"\n[✓] Cross-validation finished. Results saved to: {output_model_dir}")


if __name__ == "__main__": 
    # Options: "all", "blue", "solar"
    MODE = "all"

    # Options: "residual_bicubic", "residual_gaussian", "pca_nn", "pca_params_nn", "pca_degraded_nn"
    MODEL_NAME = "pca_params_nn"

    run_cross_validation(
        model_name=MODEL_NAME,
        dataset_mode=MODE,
        x_path=f"data/processed/nn_undegraded/{MODE}/X.npy",
        y_path=f"data/processed/nn_undegraded/{MODE}/Y.npy",
        x_err_path=f"data/processed/nn_undegraded/{MODE}/X_error.npy",
        params_dir=f"data/processed/nn_undegraded/{MODE}",
        output_model_dir=f"results/models/{MODEL_NAME}_cv/{MODE}",
        n_splits=5,
        random_state=42,
        num_components=40,
        lambda_flux=1.0,
        lambda_smooth=1e-3,
        lambda_reg=1e-4,
        epochs_adam=1000,
        steps_lbfgs=100,
        max_batch_size=32,
        effective_params=20,
    )
