"""
Module: train_undegraded_pca_nn.py
Report Section: Chapter 5 Data Processing & Feature Engineering / Model Training

High-Resolution PCA Physics-Informed Neural Network (PINN) Training Pipeline.

This script executes the complete training and evaluation pipeline for a 
Physics-Informed Neural Network (PINN) designed to recalibrate astrophysical 
spectra (e.g., Gaia BP/RP optical spectra) against high-precision reference 
standards (e.g., CALSPEC) using undegraded target baselines.

Architecture Overview:
    - High-Resolution PCA Basis: Computes empirical PCA components directly from 
      high-resolution CALSPEC target spectra normalized via percentile scaling.
    - Neural Network: Employs a convolutional/fully connected architecture to predict 
      PCA expansion coefficients, conditioned on uncalibrated spectra and auxiliary 
      stellar physical parameters (BP-RP color index, G magnitude, and RUWE).

Workflow:
    1. Loads processed spectral data and input uncertainties.
    2. Splits data into training and test sets.
    3. Computes the High-Resolution PCA basis over the normalized training targets.
    4. Executes two-stage optimization (Adam warm-up followed by L-BFGS fine-tuning).
    5. Exports model artifacts, parameters, and test set predictions.
    6. Computes quantitative performance metrics and generates diagnostic plots.
"""

import os
from typing import Optional, List

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.train.loss.losses import UndegradedPCAPhysicsInformedLoss
from src.models.pinn_architectures import UndegradedPCASpectralPINN
from src.utils.extra.seed import set_seed
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file


# =================================================================
# EXPORT WRAPPER
# =================================================================

class PCAExportWrapper(nn.Module):
    """
    Export Wrapper for High-Resolution PCA-PINN Models.

    Adapts the base model's forward interface for evaluation and inference pipelines 
    when auxiliary stellar physical parameters are required. Instead of returning 
    only compressed latent coefficients, this wrapper automatically invokes the internal 
    reconstruction mechanism to output the full reconstructed and calibrated spectra.

    Parameters
    ----------
    base_model : nn.Module
        The trained UndegradedPCASpectralPINN instance.
    """
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass to predict latent coefficients and reconstructs the calibrated spectra.

        Parameters
        ----------
        x : torch.Tensor
            Input spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Auxiliary stellar parameters tensor of shape `(batch_size, num_params)`.

        Returns
        -------
        torch.Tensor
            The fully reconstructed and calibrated spectra of shape `(batch_size, input_dim)`.
        """
        out = self.base_model(x, params)
        return self.base_model.reconstruct(out)


# =====================================================================
# TRAINING ORCHESTRATION PIPELINE
# =====================================================================

def run_training(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy", 
    params_paths: Optional[List[str]] = None, 
    output_model_dir: str = "results/models/pca_nn",
    num_components: int = 40, 
    test_size: float = 0.2,
    random_state: int = 42,
    lambda_reg: float = 1e-4,
    epochs_adam: int = 1000,
    steps_lbfgs: int = 50,
    max_batch_size: int = 16
) -> str:
    """
    Executes the full training pipeline for the High-Resolution PCA-PINN architecture.
    
    Handles data ingestion, auxiliary stellar parameter integration, train/test splitting, 
    high-resolution target normalization and PCA basis fitting, PyTorch dataset and DataLoader 
    construction, hybrid network initialization, two-stage optimization (Adam warm-up followed 
    by L-BFGS fine-tuning), serialization of model checkpoints and basis elements, and blind test export.

    Parameters
    ----------
    x_path : str, default="data/processed/X.npy"
        File path to input uncalibrated Gaia spectra NumPy array.
    y_path : str, default="data/processed/Y.npy"
        File path to target CALSPEC reference spectra NumPy array.
    x_err_path : str, default="data/processed/X_error.npy"
        File path to input spectral uncertainty NumPy array.
    params_paths : List[str], optional
        List of file paths for auxiliary stellar physical parameter arrays.
    output_model_dir : str, default="results/models/pca_nn"
        Directory where model checkpoints, loss histories, and outputs are saved.
    num_components : int, default=40
        Number of principal components for the high-resolution PCA basis.
    test_size : float, default=0.2
        Proportion of dataset allocated to the blind test split.
    random_state : int, default=42
        Random seed for reproducibility across splits and initialization.
    lambda_reg : float, default=1e-4
        Weighting factor for coefficient regularization in the loss function.
    epochs_adam : int, default=1000
        Number of epochs for the global Adam optimization phase.
    steps_lbfgs : int, default=50
        Number of steps for the secondary L-BFGS fine-tuning phase.
    max_batch_size : int, default=16
        Maximum batch size for the training DataLoader.

    Returns
    -------
    str
        File path to the saved `.npz` file containing the blind test predictions.
    """
    # 1. Setup and device configuration
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_params = len(params_paths) if params_paths is not None else 0
    print(f"[>] Running Undegraded PCA PINN on device: {device} | Components: {num_components}")

    # 2. Data loading & Parameter Stack
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path) if os.path.exists(x_err_path) else np.zeros_like(X_np)

    Params_np = None
    if num_params > 0:
        param_arrays = [np.load(p).reshape(-1, 1) for p in params_paths]
        Params_np = np.concatenate(param_arrays, axis=1)

    # 3. Train-Test Splitting
    if Params_np is not None:
        (
            X_train_np, X_test_np, 
            Y_train_np, Y_test_np, 
            X_err_train_np, X_err_test_np,
            Params_train_np, Params_test_np
        ) = train_test_split(
            X_np, Y_np, X_err_np, Params_np, 
            test_size=test_size, random_state=random_state
        )
    else:
        (
            X_train_np, X_test_np, 
            Y_train_np, Y_test_np, 
            X_err_train_np, X_err_test_np
        ) = train_test_split(
            X_np, Y_np, X_err_np, 
            test_size=test_size, random_state=random_state
        )
        Params_train_np = Params_test_np = None

    print(f"[>] Training samples: {X_train_np.shape[0]}")
    print(f"[>] Blind test samples: {X_test_np.shape[0]}")

    # 4. High-Resolution PCA Computation
    print("[>] Computing PCA basis on high-resolution training targets...")
    # Normalize spectra via 99th percentile scaling to capture intrinsic spectral lines and relative shapes
    #Y_train_scale = np.max(Y_train_np, axis=1, keepdims=True) + 1e-15
    Y_train_scale = np.percentile(Y_train_np, 99, axis=1, keepdims=True) + 1e-15
    Y_train_norm = Y_train_np / Y_train_scale

    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(Y_train_norm)
    
    pca_basis = pca.components_.astype(np.float32)
    pca_mean = pca.mean_.astype(np.float32)

    # 5. PyTorch Dataset & DataLoader Preparation
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    
    if Params_train_np is not None:
        Params_train_tensor = torch.tensor(Params_train_np, dtype=torch.float32)
        dataset_train = TensorDataset(X_train_tensor, Y_train_tensor, Params_train_tensor)
        Params_full_train = Params_train_tensor.to(device)
    else:
        dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
        Params_full_train = None

    batch_size = min(max_batch_size, len(dataset_train))
    g = torch.Generator()
    g.manual_seed(random_state) 
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

    X_full_train = X_train_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)
    num_channels = X_train_tensor.shape[1]

    # 6. Model and Loss initialization
    model = UndegradedPCASpectralPINN(
        input_dim=num_channels,
        pca_basis=pca_basis,
        pca_mean=pca_mean,
        num_params=num_params
    ).to(device)
    
    physics_criterion = UndegradedPCAPhysicsInformedLoss(
        lambda_reg=lambda_reg
    ).to(device)

    # 7. Two-stage Training Optimization (Adam -> L-BFGS)
    trained_model, loss_history = train_2stage_fixed(
        model=model,
        dataloader=dataloader,
        X_full=X_full_train,
        Y_full=Y_full_train,
        params_full=Params_full_train,
        physics_criterion=physics_criterion,
        epochs_adam=epochs_adam,
        steps_lbfgs=steps_lbfgs
    )

    # 8. Model Serialization
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_pca_recalibration.joblib")

    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "history": loss_history,
        "input_dim": num_channels,
        "num_components": num_components,
        "num_params": num_params,
        "pca_basis": pca_basis,
        "pca_mean": pca_mean
    }
    joblib.dump(export_model, save_path)
    print(f"\n[✓] Model successfully serialized to: {save_path}")

    # 9. Test Set Prediction and Export
    export_ready_model = PCAExportWrapper(trained_model)
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)

    Params_test_tensor = None
    if Params_test_np is not None:
        Params_test_tensor = torch.tensor(Params_test_np, dtype=torch.float32).to(device)

    export_test_predictions(
        model=export_ready_model, 
        X_tensor=X_test_tensor,
        Y_tensor=Y_test_tensor,
        X_err_tensor=X_err_test_tensor,
        params_tensor=Params_test_tensor,
        output_filepath=results_filepath,
    )

    return results_filepath


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

if __name__ == "__main__":
    set_seed(42)

    # Select dataset partition mode: "solar", "blue", or "all"
    mode = "all"
    OUTPUT_MODEL_DIR = f"results/models/undegraded_pca_params_nn/{mode}"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")

    BASE_DATA_DIR = f"data/processed/nn_undegraded/{mode}"
    stellar_params = [
        os.path.join(BASE_DATA_DIR, "G_mags.npy"),
        os.path.join(BASE_DATA_DIR, "bp_rp.npy"),
        os.path.join(BASE_DATA_DIR, "ruwe.npy")
    ]

    # 1. Execute Phase 1: High-Resolution PCA Training, Checkpointing, and Export
    npz_results_path = run_training(
        x_path = os.path.join(BASE_DATA_DIR, "X.npy"),
        y_path = os.path.join(BASE_DATA_DIR, "Y.npy"),
        x_err_path = os.path.join(BASE_DATA_DIR, "X_error.npy"),
        params_paths = stellar_params,
        output_model_dir = OUTPUT_MODEL_DIR,
        num_components = 15,
        lambda_reg = 1e-5,
        epochs_adam = 1000,
        steps_lbfgs = 50,
        max_batch_size = 16,
        test_size = 0.2,
        random_state = 42
    )

    # 2. Execute Phase 2: Quantitative Performance Evaluation
    evaluate_test_set_from_file(
        input_filepath = npz_results_path,
        output_dir = OUTPUT_MODEL_DIR,
        num_knots = 40
    )

    # 3. Execute Phase 3: Qualitative Model Comparison & Diagnostic Plots
    generate_plots_from_file(
        input_filepath = npz_results_path,
        output_dir = VIS_DIR,
        num_samples = 18
    )