"""
PINN Spectral Recalibration Training Module.

This script executes the complete training and evaluation pipeline for a Hybrid 
Physics-Informed Neural Network (PINN) designed to recalibrate astrophysical 
spectra (e.g., Gaia BP/RP) against high-precision reference standards (e.g., CALSPEC).

Architecture Overview:
    - Global Branch: Fits the macroscopic spectral continuum using a parameterized 
      PCA/Chebyshev polynomial basis.
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
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.train.loss.losses import UndegradedPCAPhysicsInformedLoss
from src.models.pinn_architectures import UndegradedPCASpectralPINN

from src.utils.extra.seed import set_seed
from src.train.training_functions.fixed_training_strategies import train_2stage_fixed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file


# =================================================================
# EXPORT WRAPPER
# =================================================================

class PCAExportWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        """
        Initializes the wrapper with the trained base model.
        
        Args:
        - base_model (nn.Module): The trained PINN model.
        """
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass and immediately reconstructs the final output.
        
        Args:
        - x (torch.Tensor): Input spectra tensor.
            
        Returns:
        - torch.Tensor: The fully reconstructed and calibrated spectra.
        """
        out = self.base_model(x)
        return self.base_model.reconstruct(out)


# =================================================================
# Training
# =================================================================

def run_training(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy",  
    output_model_dir: str = "results/models/pca_nn",
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
    
    This function handles data loading, train/test splitting, PyTorch dataset preparation, 
    model initialization, training (using a two-stage Adam + L-BFGS strategy), 
    and model serialization. Finally, it exports the test set predictions.
    
    Args:
    - x_path (str): File path to the input data (e.g., uncalibrated Gaia spectra).
    - y_path (str): File path to the target data (e.g., CALSPEC reference spectra).
    - x_err_path (str): File path to the input error/uncertainty data.
    - output_model_dir (str): Directory where the trained model and results will be saved.
    - degree (int): Number of global components (e.g., PCA/Chebyshev basis degree).
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Seed used by the random number generator for reproducibility.
    - curvature_weight (float): Weight for the curvature penalty in the loss function.
    - local_reg_weight (float): Weight for the sparsity (L1) penalty on the local residual branch.
    - epochs_adam (int): Number of training epochs using the Adam optimizer.
    - steps_lbfgs (int): Number of training steps using the L-BFGS optimizer.
    - max_batch_size (int): Maximum batch size for the DataLoader.
        
    Returns:
    - str: The file path to the saved .npz file containing the test predictions.
    """

    # 1. Setup and device configuration
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running Hybrid PINN on device: {device} | Degree: {degree}")

    # 2. Data loading
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path)

    # 3. Train-Test Splitting
    (
        X_train_np, X_test_np, 
        Y_train_np, Y_test_np, 
        X_err_train_np, X_err_test_np
    ) = train_test_split(
        X_np, Y_np, X_err_np, 
        test_size=test_size, random_state=random_state
    )

    # 4. PyTorch Dataset preparation
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    
    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
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
    model = UndegradedPCASpectralPINN(
        input_dim=num_channels, 
        degree=degree
    ).to(device)
    
    physics_criterion = UndegradedPCAPhysicsInformedLoss(
        curvature_weight=curvature_weight, 
        local_reg_weight=local_reg_weight
    ).to(device)

    # 6. Two-stage Training (Adam -> L-BFGS)
    trained_model = train_2stage_fixed(
        model=model,
        dataloader=dataloader,
        X_full=X_full_train,
        Y_full=Y_full_train,
        physics_criterion=physics_criterion,
        epochs_adam=epochs_adam,
        steps_lbfgs=steps_lbfgs
    )

    # 7. Model Serialization
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_pca_recalibration.joblib")

    # Package the model state alongside its architectural parameters
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "input_dim": num_channels,
        "num_components": degree,
        "pca_basis": trained_model.pca_basis.cpu().numpy(),
        "pca_mean": np.ones(num_channels, dtype=np.float32)
    }
    joblib.dump(export_model, save_path)

    # 8. Test Set Prediction and Export
    export_ready_model = PCAExportWrapper(trained_model)
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")

    # Move test tensors to the specified device
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)

    # Generate and save predictions
    export_test_predictions(
        model=export_ready_model, 
        X_tensor=X_test_tensor,
        Y_tensor=Y_test_tensor,
        X_err_tensor=X_err_test_tensor,
        output_filepath=results_filepath,
    )

    return results_filepath


if __name__ == "__main__":
    # Define experiment mode and output directories
    mode = "all" 
    OUTPUT_MODEL_DIR = f"results/models/undegraded_pca_nn/{mode}"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")

    # Execute the training pipeline with specified hyperparameters
    npz_results_path = run_training(
        x_path=f"data/processed/nn_undegraded/{mode}/X.npy",
        y_path=f"data/processed/nn_undegraded/{mode}/Y.npy",
        x_err_path=f"data/processed/nn_undegraded/{mode}/X_error.npy",   
        output_model_dir=OUTPUT_MODEL_DIR,
        degree=10,
        curvature_weight=0.1,
        local_reg_weight=0.015,
        epochs_adam=1000,
        steps_lbfgs=50,
        max_batch_size=16,
        test_size=0.2,
        random_state=42
    )

    # Run evaluation metrics and generate visual comparison plots
    evaluate_test_set_from_file(npz_results_path, OUTPUT_MODEL_DIR, 10)
    generate_plots_from_file(npz_results_path, VIS_DIR, 18)