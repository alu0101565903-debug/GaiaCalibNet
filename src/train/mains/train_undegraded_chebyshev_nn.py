"""
Module: train_undegraded_chebyshev_nn.py
Report Section: Chapter 5 Data Processing & Feature Engineering / Model Training

PINN Spectral Recalibration Training Module (Hybrid Chebyshev Architecture).

This script executes the complete training and evaluation pipeline for a Hybrid 
Physics-Informed Neural Network (PINN) designed to recalibrate astrophysical 
spectra (e.g., Gaia BP/RP optical spectra) against high-precision reference 
standards (e.g., CALSPEC) using undegraded spectral baselines.

Architecture Overview:
    - Global Branch: Fits the macroscopic spectral continuum using a parameterized 
      Chebyshev polynomial basis.
    - Local Residual Branch: Applies high-frequency corrections to capture localized 
      spectral lines, enforced with L1 sparsity regularization to prevent overfitting 
      to instrumental noise.

Workflow:
    1. Loads processed spectral data and input uncertainties.
    2. Splits data into training and test sets.
    3. Executes two-stage optimization (Adam warm-up followed by L-BFGS fine-tuning).
    4. Exports model artifacts, parameters, and test set predictions.
    5. Computes quantitative performance metrics and generates diagnostic plots.
"""

import os
from typing import Optional, List

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.train.loss.losses import UndegradedChebyshevPhysicsInformedLoss
from src.models.pinn_architectures import UndegradedChebyshevSpectralPINN
from src.utils.extra.seed import set_seed
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file


# =================================================================
# EXPORT WRAPPER
# =================================================================

class ChebyshevExportWrapper(nn.Module):
    """
    Export Wrapper for Hybrid Chebyshev-PINN Models.

    Adapts the base model's forward interface for evaluation and inference pipelines 
    when conditional stellar parameters are provided. Instead of returning intermediate 
    latent representations or coefficients, this wrapper invokes the internal model 
    reconstruction routine to output the full wavelength-dependent correction curve R(lambda).

    Parameters
    ----------
    base_model : torch.nn.Module
        The trained UndegradedChebyshevSpectralPINN instance.
    """
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes hybrid Chebyshev expansion coefficients and reconstructs the full correction curve R(lambda).

        Parameters
        ----------
        x : torch.Tensor
            Input uncalibrated spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Normalized auxiliary stellar parameters tensor of shape `(batch_size, num_params)`.

        Returns
        -------
        torch.Tensor
            Reconstructed full recalibration curve R(lambda) of shape `(batch_size, input_dim)`.
        """
        coeffs = self.base_model(x, params)
        return self.base_model.reconstruct(coeffs)


# =====================================================================
# TRAINING ORCHESTRATION PIPELINE
# =====================================================================

def run_training(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy", 
    params_paths: Optional[List[str]] = None, 
    output_model_dir: str = "results/models/nn",
    degree: int = 10, 
    test_size: float = 0.2,
    random_state: int = 42,
    curvature_weight: float = 0.1,
    local_reg_weight: float = 0.01,
    epochs_adam: int = 1000,
    steps_lbfgs: int = 50,
    max_batch_size: int = 16
) -> str:
    """
    Executes the full training pipeline for the Hybrid Physics-Informed Neural Network (PINN).
    
    Handles data ingestion, auxiliary stellar parameter concatenation, train/test splitting, 
    PyTorch dataset and reproducible DataLoader construction, hybrid network initialization, 
    two-stage optimization (Adam warm-up followed by L-BFGS fine-tuning), serialization of 
    model checkpoints and basis vectors, and blind test prediction export.
    
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
    output_model_dir : str, default="results/models/nn"
        Directory where trained model checkpoints, loss histories, and outputs are saved.
    degree : int, default=10
        Polynomial degree for the global Chebyshev expansion branch.
    test_size : float, default=0.2
        Proportion of dataset allocated to the blind test split.
    random_state : int, default=42
        Random seed for reproducibility across splits and initialization.
    curvature_weight : float, default=0.1
        Weighting factor for second-derivative smoothness penalties in the loss function.
    local_reg_weight : float, default=0.01
        Weighting factor for L1 sparsity regularization on the local residual branch.
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
    print(f"[>] Running Hybrid PINN on device: {device} | Degree: {degree}")

    # 2. Data loading & Parameter Stack
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path)

    Params_np = None
    if num_params > 0:
        # Load each individual parameter .npy file, reshape as column vector (N, 1), and stack horizontally
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

    # 4. PyTorch Dataset & DataLoader Preparation
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

    # Initialize DataLoader with a fixed generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(random_state) 
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

    # Load full training tensors to device for evaluation during/after training
    X_full_train = X_train_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)

    num_channels = X_train_tensor.shape[1]

    # 5. Model and Loss initialization
    model = UndegradedChebyshevSpectralPINN(
        input_dim=num_channels, 
        degree=degree,
        num_params=num_params
    ).to(device)
    
    physics_criterion = UndegradedChebyshevPhysicsInformedLoss(
        curvature_weight=curvature_weight, 
        local_reg_weight=local_reg_weight
    ).to(device)

    # 6. Two-stage Training Optimization (Adam -> L-BFGS)
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

    # 7. Model Serialization
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_undegraded_chebyshev_recalibration.joblib")

    # Package the model state alongside its architectural parameters
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "history": loss_history,
        "input_dim": num_channels,
        "degree": degree,
        "num_params": num_params,
        "cheb_basis": trained_model.cheb_basis.cpu().numpy()
    }
    joblib.dump(export_model, save_path)

    # 8. Test Set Prediction and Export
    export_ready_model = ChebyshevExportWrapper(trained_model)
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")

    # Move test tensors to the specified device
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)

    Params_test_tensor = None
    if Params_test_np is not None:
        Params_test_tensor = torch.tensor(Params_test_np, dtype=torch.float32).to(device)

    # Generate and save predictions
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
    OUTPUT_MODEL_DIR = f"results/models/undegraded_cheb_params_nn/{mode}"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")

    BASE_DATA_DIR = f"data/processed/nn_undegraded/{mode}"
    stellar_params = [
        os.path.join(BASE_DATA_DIR, "G_mags.npy"),
        os.path.join(BASE_DATA_DIR, "bp_rp.npy"),
        os.path.join(BASE_DATA_DIR, "ruwe.npy")
    ]

    # 1. Execute Phase 1: Hybrid Training, Checkpointing, and Export
    npz_results_path = run_training(
        x_path = os.path.join(BASE_DATA_DIR, "X.npy"),
        y_path = os.path.join(BASE_DATA_DIR, "Y.npy"),
        x_err_path = os.path.join(BASE_DATA_DIR, "X_error.npy"),
        params_paths = stellar_params,  # <--- Parameter injection
        output_model_dir = OUTPUT_MODEL_DIR,
        degree = 10,
        curvature_weight = 0.1,
        local_reg_weight = 0.015,
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
            num_knots = 10
        )

    # 3. Execute Phase 3: Qualitative Model Comparison & Diagnostic Plots
    generate_plots_from_file(
            input_filepath = npz_results_path,
            output_dir = VIS_DIR,
            num_samples = 18
        )