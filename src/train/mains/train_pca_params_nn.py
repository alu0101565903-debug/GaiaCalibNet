"""
Module: train_pca_params_nn.py
Report Section: Chapter 5 Data Processing & Feature Engineering / Model Training

Parameter-Conditioned PCA Physics-Informed Neural Network (PINN) Training Pipeline.

This module orchestrates the end-to-end training and evaluation pipeline for a 
stellar-parameter-conditioned PCA PINN architecture ("PCAParamsSpectralPINN"). 
The model predicts expansion coefficients c_i for a dynamically generated PCA 
basis, conditioned on auxiliary stellar physical parameters (BP-RP color index, 
G magnitude, and RUWE astrometric quality factor). It outputs a smooth multiplicative 
correction profile R(lambda) to recalibrate Gaia BP/RP optical spectra against 
CALSPEC ground-truth references.
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import joblib
import scipy.signal as signal
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.utils.extra.seed import set_seed
from src.models.pinn_architectures import PCAParamsSpectralPINN
from src.train.loss.losses import PCAParamsPhysicsInformedLoss 
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed

from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file


# =================================================================
# EXPORT WRAPPER
# =================================================================
class PCAExportWrapper(torch.nn.Module):
    """
    Export Wrapper for Parameter-Conditioned PCA-PINN Models.

    Adapts the base model's forward interface for evaluation and inference pipelines 
    when auxiliary stellar physical parameters are required. Instead of returning 
    only the compressed coefficients, this wrapper automatically invokes the internal 
    reconstruction mechanism to output the full wavelength-dependent recalibration curve R(lambda).

    Parameters
    ----------
    base_model : torch.nn.Module
        The trained PCAParamsSpectralPINN instance.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x, params: torch.Tensor = None):
        """
        Extracts predicted PCA coefficients conditioned on stellar parameters 
        and reconstructs the full correction curve R(lambda).

        Parameters
        ----------
        x : torch.Tensor
            Input uncalibrated spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Normalized stellar parameters tensor of shape `(batch_size, num_params)`.

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
    output_model_dir: str = "results/models/pca_nn",
    num_components: int = 10,
    lambda_smooth: float = 10,
    lambda_reg: float = 1e-4,
    epochs_adam: int = 5000,
    steps_lbfgs: int = 100,
    max_batch_size: int = 16,
    test_size: float = 0.2,
    random_state: int = 42
) -> str:
    """
    Executes the end-to-end parameter-conditioned PCA-PINN training and export pipeline.

    Handles stellar parameter ingestion and standardization, leakage-free dynamic 
    PCA basis extraction, multi-objective physics-informed loss setup, hybrid 
    two-stage optimization (Adam + L-BFGS), serialization of scaling statistics and checkpoints, 
    and blind test inference export.

    Parameters
    ----------
    x_path : str, default="data/processed/X.npy"
        Filepath to input degraded Gaia spectra NumPy array.
    y_path : str, default="data/processed/Y.npy"
        Filepath to target CALSPEC reference spectra NumPy array.
    x_err_path : str, default="data/processed/X_error.npy"
        Filepath to input spectral uncertainty NumPy array.
    params_paths : List[str], optional
        List of file paths for individual auxiliary stellar parameter arrays.
    output_model_dir : str, default="results/models/pca_nn"
        Directory where model checkpoints, loss histories, and outputs are saved.
    num_components : int, default=10
        Number of principal components for the dynamic PCA basis.
    lambda_smooth : float, default=10.0
        Weighting factor for the second-derivative curvature penalty.
    lambda_reg : float, default=1e-4
        Weighting factor for L2 regularization on higher-order PCA coefficients.
    epochs_adam : int, default=5000
        Number of epochs for the global Adam optimization phase.
    steps_lbfgs : int, default=100
        Number of steps for the secondary L-BFGS fine-tuning phase.
    max_batch_size : int, default=16
        Maximum batch size for the training DataLoader.
    test_size : float, default=0.2
        Proportion of dataset allocated to the blind test split.
    random_state : int, default=42
        Random seed for reproducibility across splits and initialization.

    Returns
    -------
    str
        File path to the saved `.npz` file containing blind test predictions.
    """
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running PINN training on device: {device}")

    # --- 1. Data Ingestion & Stellar Feature Stacking ---
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path)

    if params_paths is not None:
        params_list = [np.load(p) for p in params_paths]
        stellar_params_np = np.column_stack(params_list)
    else:
        raise ValueError("Parameter paths must be provided for PCAParamsSpectralPINN training.")

    # --- 2. Train / Test Split ---
    (
        X_train_np, X_test_np, 
        Y_train_np, Y_test_np, 
        X_err_train_np, X_err_test_np,
        params_train_np, params_test_np  
    ) = train_test_split(
        X_np, Y_np, X_err_np, stellar_params_np, 
        test_size=test_size, random_state=random_state
    )
    
    print(f"[>] Training samples: {X_train_np.shape[0]}")
    print(f"[>] Blind test samples: {X_test_np.shape[0]}")

    # --- 3. Stellar Parameter Standardization ---
    mean_params = np.mean(params_train_np, axis=0)
    std_params = np.std(params_train_np, axis=0) + 1e-8

    params_train_norm = (params_train_np - mean_params) / std_params
    params_test_norm = (params_test_np - mean_params) / std_params

    # Tensor conversion
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    params_train_tensor = torch.tensor(params_train_norm, dtype=torch.float32)
    
    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor, params_train_tensor)
    batch_size = min(max_batch_size, len(dataset_train))

    # Reproducible DataLoader configuration
    g = torch.Generator()
    g.manual_seed(random_state) 
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

    X_full_train = X_train_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)
    params_full_train = params_train_tensor.to(device) 

    # --- 4. Dynamic PCA Basis Generation (Leakage-Free) ---
    print(f"[>] Computing PCA basis ({num_components} components) from training targets...")

    # Step A: Compute raw ratio R = Y / X
    R_train_raw = Y_train_np / (X_train_np + 1e-15)
    
    # Step B: Clip extreme ratio outliers to prevent edge divergence (e.g., red-edge blowup)
    R_train_clipped = np.clip(R_train_raw, 0.5, 2.0)
    
    # Step C: Smooth with a controlled Savitzky-Golay filter window
    print("[>] Applying Savitzky-Golay filter to R before PCA fit...")
    R_train_smooth = signal.savgol_filter(R_train_clipped, window_length=11, polyorder=2, axis=1)
    
    # Step D: Fit PCA decomposition on training ratios
    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(R_train_smooth)
    
    pca_basis_np = pca.components_  
    pca_mean_np = pca.mean_
    
    # --- 5. Model & Physics Loss Initialization ---
    num_channels = X_train_tensor.shape[1]
    num_params = params_train_tensor.shape[1] # e.g.: 3 (bp_rp, g_mags, ruwe)
    
    # Initialize the network by passing the calculated basis AND the mean
    model = PCAParamsSpectralPINN(
        input_dim=num_channels, 
        pca_basis=pca_basis_np,
        pca_mean=pca_mean_np,
        num_params=num_params
    ).to(device)
    
    # Initialize the physics loss function with the same basis and mean
    physics_criterion = PCAParamsPhysicsInformedLoss(
        pca_basis=pca_basis_np,
        pca_mean=pca_mean_np,
        lambda_smooth=lambda_smooth,
        lambda_reg=lambda_reg
    ).to(device)

    # --- 6. Two-Stage Optimization (Adam + L-BFGS) ---
    trained_model, loss_history = train_2stage_fixed(
        model=model,
        dataloader=dataloader,
        X_full=X_full_train,
        Y_full=Y_full_train,
        params_full=params_full_train,
        physics_criterion=physics_criterion,
        epochs_adam=epochs_adam,
        steps_lbfgs=steps_lbfgs
    )

    # --- 7. Model Serialization ---
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_pca_recalibration.joblib")
    
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "history": loss_history,
        "input_dim": num_channels,
        "num_components": num_components,
        "num_params": num_params,
        "pca_basis": pca_basis_np,
        "pca_mean": pca_mean_np,
        "mean_params": mean_params,
        "std_params": std_params
    }

    joblib.dump(export_model, save_path)
    print(f"\n[✓] Model successfully saved to: {save_path}")

    # --- 8. Export Blind Test Predictions (.npz) ---
    export_ready_model = PCAExportWrapper(trained_model)
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)
    params_test_tensor = torch.tensor(params_test_norm, dtype=torch.float32).to(device)

    export_test_predictions(
        model=export_ready_model,  # <-- Pass the Wrapper, NOT the raw model
        X_tensor=X_test_tensor,
        Y_tensor=Y_test_tensor,
        X_err_tensor=X_err_test_tensor,
        params_tensor=params_test_tensor,
        output_filepath=results_filepath,
    )

    return results_filepath


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

if __name__ == "__main__":
    set_seed(42)

    # Define experiment mode and output directories
    mode = "all"  # change to "blue" for the blue set, or "solar" for the solar set
    OUTPUT_MODEL_DIR = f"results/models/pca_params_nn/{mode}"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")

    BASE_DATA_DIR = f"data/processed/nn/{mode}"
    stellar_params = [
        os.path.join(BASE_DATA_DIR, "G_mags.npy"),
        os.path.join(BASE_DATA_DIR, "bp_rp.npy"),
        os.path.join(BASE_DATA_DIR, "ruwe.npy")
    ]

    # Execute the training pipeline with specified hyperparameters
    npz_results_path = run_training(
        x_path = os.path.join(BASE_DATA_DIR, "X.npy"),
        y_path = os.path.join(BASE_DATA_DIR, "Y.npy"),
        x_err_path = os.path.join(BASE_DATA_DIR, "X_error.npy"),
        params_paths = stellar_params,  # <--- Parameter injection
        output_model_dir = OUTPUT_MODEL_DIR,
        num_components = 15,
        lambda_smooth = 10,
        lambda_reg = 1e-4,
        epochs_adam = 1000,
        steps_lbfgs = 100,
        max_batch_size = 15,
        test_size = 0.2,
        random_state = 42
    )
    
    # Run evaluation metrics and generate visual comparison plots
    evaluate_test_set_from_file(
        input_filepath = npz_results_path,
        output_dir = OUTPUT_MODEL_DIR,
        num_knots = 10
    )

    generate_plots_from_file(
        input_filepath = npz_results_path,
        output_dir = VIS_DIR,
        num_samples = 18
    )