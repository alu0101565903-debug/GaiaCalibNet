"""
Module: train_pca_nn.py

PCA Physics-Informed Neural Network (PINN) Spectral Recalibration Module.

This module orchestrates the end-to-end training pipeline for a PCA-based PINN 
designed to predict a multiplicative correction factor R(lambda) for recalibrating 
Gaia BP/RP optical spectra using CALSPEC references. It handles data loading, 
dynamic PCA basis generation, model initialization, a two-stage training process 
(Adam + L-BFGS), model saving, and triggering the export of inference results.
"""

import os
import numpy as np
import torch
import joblib
import scipy.signal as signal
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.utils.extra.seed import set_seed
from src.models.pinn_architectures import PCASpectralPINN
from src.train.loss.losses import PCAPhysicsInformedLoss 
from src.train.training_functions.fixed_training_strategies import train_2stage_fixed

from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file

# =================================================================
# EXPORT WRAPPER
# =================================================================
class PCAExportWrapper(torch.nn.Module):
    """
    A wrapper class to adapt the PCA model's output for the export function.
    Instead of just returning the PCA coefficients, it automatically applies 
    the inverse transform to output the reconstructed correction factor R.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        # Extract the PCA coefficients and reconstruct R automatically
        coeffs = self.base_model(x)
        return self.base_model.reconstruct(coeffs)

def run_training(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy",  
    output_model_dir: str = "results/models/pca_nn",
    test_size: float = 0.2,
    random_state: int = 42,
    num_components: int = 10,
    epochs_adam: int = 5000,
    steps_lbfgs: int = 100,
    max_batch_size: int = 32
) -> str:
    """
    Handles data preparation, dynamic PCA basis calculation, PINN model training, 
    and exporting inference results to disk.

    Args:
    - x_path (str): Filepath to the input degraded spectra (X) numpy array.
    - y_path (str): Filepath to the target reference spectra (Y) numpy array.
    - x_err_path (str): Filepath to the input spectra errors numpy array.
    - output_model_dir (str): Directory where the trained model and results will be saved.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Seed used by the random number generator for reproducibility.
    - num_components (int): Number of principal components for the PCA basis.
    - epochs_adam (int): Number of epochs for the Adam optimizer phase.
    - steps_lbfgs (int): Number of steps for the L-BFGS optimizer phase.
    - max_batch_size (int): Maximum batch size for the DataLoader.

    Returns:
    - str: The filepath to the saved .npz file containing the test predictions.
    """
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running PINN training on device: {device}")

    # --- Data Loading ---
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Verify that '{x_path}' and '{y_path}' are located in the correct path.")

    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path)

    # --- Train / Test Split ---
    (
        X_train_np, X_test_np, 
        Y_train_np, Y_test_np, 
        X_err_train_np, X_err_test_np
    ) = train_test_split(
        X_np, Y_np, X_err_np, 
        test_size=test_size, random_state=random_state
    )
    
    print(f"[>] Training samples: {X_train_np.shape[0]}")
    print(f"[>] Blind test samples: {X_test_np.shape[0]}")

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    
    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
    batch_size = min(max_batch_size, len(dataset_train))

    # Enforce reproducibility in the Dataloader
    g = torch.Generator()
    g.manual_seed(random_state) 
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

    X_full_train = X_train_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)

    # --- Dynamic PCA Basis Generation ---
    print(f"[>] Computing PCA basis ({num_components} components) from training targets...")

    # 1. Calculate the raw ratio R = Y / X
    R_train_raw = Y_train_np / (X_train_np + 1e-15)
    
    # 2. CRITICAL: Clip anomalous values. 
    # Force max R to 2.0 and min to 0.5. 
    # This removes infinities at the red edge that distort the PCA fit.
    R_train_clipped = np.clip(R_train_raw, 0.5, 2.0)
    
    # 3. Smooth with a SMALLER window (11 instead of 31).
    # This prevents the Savitzky-Golay "drooping tail" effect at the spectrum edges.
    print("[>] Applying Savitzky-Golay filter to R before PCA fit...")
    R_train_smooth = signal.savgol_filter(R_train_clipped, window_length=11, polyorder=2, axis=1)
    
    # 4. Fit the PCA on the cleaned, smoothed data
    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(R_train_smooth)
    
    pca_basis_np = pca.components_  
    pca_mean_np = pca.mean_
    
    # --- Model Initialization ---
    num_channels = X_train_tensor.shape[1]
    
    # Initialize the network by passing the calculated basis AND the mean
    model = PCASpectralPINN(
        input_dim=num_channels, 
        pca_basis=pca_basis_np,
        pca_mean=pca_mean_np
    ).to(device)
    
    # Initialize the physics loss function with the same basis and mean
    physics_criterion = PCAPhysicsInformedLoss(
        pca_basis=pca_basis_np,
        pca_mean=pca_mean_np
    ).to(device)

    # --- Training ---
    trained_model = train_2stage_fixed(
        model=model,
        dataloader=dataloader,
        X_full=X_full_train,
        Y_full=Y_full_train,
        physics_criterion=physics_criterion,
        epochs_adam=epochs_adam,
        steps_lbfgs=steps_lbfgs
    )

    # --- Save Model ---
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_pca_recalibration.joblib")
    
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "input_dim": num_channels,
        "num_components": num_components,
        "pca_basis": pca_basis_np,
        "pca_mean": pca_mean_np 
    }
    joblib.dump(export_model, save_path)
    print(f"\n[✓] Model successfully saved to: {save_path}")

    # Wrap the trained model for inference/export
    export_ready_model = PCAExportWrapper(trained_model)
    # =================================================================

    # --- Export Predictions to Disk (.npz) ---
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)

    export_test_predictions(
        model=export_ready_model,  # <-- Pasamos el Wrapper, NO el modelo crudo
        X_tensor=X_test_tensor,
        Y_tensor=Y_test_tensor,
        X_err_tensor=X_err_test_tensor,
        output_filepath=results_filepath,
    )

    return results_filepath

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    set_seed(42)

    mode = "all"  # change to "blue" for the blue set, or "solar" for the solar set

    # 1. Configure main directories
    OUTPUT_MODEL_DIR = f"results/models/pca_nn/{mode}"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")

    # 2. Execute Phase 1: Training and Exporting
    npz_results_path = run_training(
        x_path=f"data/processed/nn/{mode}/X.npy",
        y_path=f"data/processed/nn/{mode}/Y.npy",
        x_err_path=f"data/processed/nn/{mode}/X_error.npy",   
        output_model_dir=OUTPUT_MODEL_DIR,
        test_size=0.2,
        random_state=42,
        num_components=10,
        epochs_adam=1000,
        steps_lbfgs=100,
        max_batch_size=15
    )
    
    # 3. Execute Phase 2: Quantitative Evaluation 
    evaluate_test_set_from_file(
        input_filepath=npz_results_path,
        output_dir=OUTPUT_MODEL_DIR,
        num_knots=10
    )

    # 4. Execute Phase 3: Visualization 
    generate_plots_from_file(
        input_filepath=npz_results_path,
        output_dir=VIS_DIR,
        num_samples=18
    )