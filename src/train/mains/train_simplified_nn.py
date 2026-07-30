"""
Module: train_simplified_nn.py

Residual Physics-Informed Neural Network (PINN) Spectral Recalibration Module.

This module orchestrates the end-to-end training pipeline for a Residual PINN 
(utilizing a Gaussian smoothing layer) designed to predict a multiplicative 
correction factor R(lambda) for recalibrating Gaia BP/RP optical spectra. 
It handles data ingestion, model initialization, optimization via a two-stage 
strategy (Adam + L-BFGS), state persistence, and the generation of inference 
exports for downstream evaluation and visualization.
"""

import os
import numpy as np
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils.extra.seed import set_seed
from src.models.pinn_architectures import GaussianSpectralPINN
from src.train.loss.losses import GaussianPhysicsInformedLoss
from src.train.training_functions.fixed_training_strategies import train_2stage_fixed

from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file

def run_training(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy",  
    output_model_dir: str = "results/models/simplified_nn",
    test_size: float = 0.2,
    random_state: int = 42,
    poly_deg: int = 5,
    epochs_adam: int = 5000,
    steps_lbfgs: int = 100,
    max_batch_size: int = 32
) -> str:
    """
    Handles data preparation, PINN model training, and exporting inference results to disk.

    This function reads preprocessed uncalibrated spectra and their CALSPEC references, 
    splits them into training and blind test sets, and initializes a PyTorch DataLoader. 
    It instantiates the `GaussianSpectralPINN` and its paired physics-informed loss 
    function. The model is trained using a fixed two-stage strategy (Adam followed by 
    L-BFGS). Upon completion, the model state dictionary is saved to a `.joblib` file, 
    and the test set predictions are exported to a compressed `.npz` archive.

    Args:
    - x_path (str, optional): Filepath to the uncalibrated input spectra NumPy array. Defaults to "data/processed/X.npy".
    - y_path (str, optional): Filepath to the reference CALSPEC spectra NumPy array. Defaults to "data/processed/Y.npy".
    - x_err_path (str, optional): Filepath to the input spectra flux errors. Defaults to "data/processed/X_error.npy".
    - output_model_dir (str, optional): Directory to save the trained model and inference results. Defaults to "results/models/simplified_nn".
    - test_size (float, optional): Proportion of the dataset to include in the blind test split. Defaults to 0.2.
    - random_state (int, optional): Random seed for reproducibility across splits and initialization. Defaults to 42.
    - poly_deg (int, optional): Degree of the Chebyshev polynomial used for baseline calculations during export. Defaults to 5.
    - epochs_adam (int, optional): Number of epochs for the initial Adam optimization stage. Defaults to 5000.
    - steps_lbfgs (int, optional): Number of optimization steps for the fine-tuning L-BFGS stage. Defaults to 100.
    - max_batch_size (int, optional): Maximum batch size for the DataLoader. Defaults to 32.

    Returns:
    - str: The absolute or relative filepath to the generated `.npz` archive containing the test predictions.

    Raises:
    - FileNotFoundError: If the specified data files (`x_path` or `y_path`) do not exist.
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
    X_train_np, X_test_np, Y_train_np, Y_test_np, X_err_train_np, X_err_test_np = train_test_split(
        X_np, Y_np, X_err_np, test_size=test_size, random_state=random_state
    )
    
    print(f"[>] Training samples: {X_train_np.shape[0]}")
    print(f"[>] Blind test samples: {X_test_np.shape[0]}")

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    
    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
    batch_size = min(max_batch_size, len(dataset_train))

    # Reproducibility in Dataloader
    g = torch.Generator()
    g.manual_seed(random_state) 
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

    X_full_train = X_train_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)

    # --- Model Initialization ---
    num_channels = X_train_tensor.shape[1]

    # Instantiate the Gaussian-smoothed SpectralPINN architecture
    model = GaussianSpectralPINN(input_dim=num_channels).to(device)

    # Instantiate the specific physics-informed loss function for this architecture
    physics_criterion = GaussianPhysicsInformedLoss()

    # --- Training ---
    # Using the fixed 2-stage strategy (Adam + L-BFGS) suitable for this network
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
    save_path = os.path.join(output_model_dir, "pinn_spectral_recalibration.joblib")
    
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "input_dim": num_channels,
    }
    joblib.dump(export_model, save_path)
    print(f"\n[✓] Model successfully saved to: {save_path}")

    # --- Export Predictions to Disk (.npz) ---
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)
    
    export_test_predictions(
        model=trained_model,
        X_tensor=X_test_tensor,
        Y_tensor=Y_test_tensor,
        X_err_tensor=X_err_test_tensor, 
        output_filepath=results_filepath,
        poly_deg=poly_deg
    )

    # Return the results file path so it can be read by downstream scripts
    return results_filepath

# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    set_seed(42)

    # 1. Configure main directories
    OUTPUT_MODEL_DIR = "results/models/simplified_nn/all"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")

    # 2. Execute Phase 1: Training and Exporting
    
    npz_results_path = run_training(
        x_path="data/processed/X.npy",
        y_path="data/processed/Y.npy",
        x_err_path="data/processed/X_error.npy",  
        output_model_dir=OUTPUT_MODEL_DIR,
        test_size=0.2,
        random_state=42,
        poly_deg=5,
        epochs_adam=5000,
        steps_lbfgs=100,
        max_batch_size=15
    )

    # Alternative configurations mapping specific bands (e.g., blue or solar)
    """
    npz_results_path = run_training(
        x_path="data/processed/nn/blue/X.npy",
        y_path="data/processed/nn/blue/Y.npy",
        x_err_path="data/processed/nn/blue/X_error.npy",  
        output_model_dir=OUTPUT_MODEL_DIR,
        test_size=0.2,
        random_state=42,
        poly_deg=5,
        epochs_adam=5000,
        steps_lbfgs=100,
        max_batch_size=15
    )
    """
    """
    npz_results_path = run_training(
        x_path="data/processed/nn/solar/X.npy",
        y_path="data/processed/nn/solar/Y.npy",
        x_err_path="data/processed/nn/solar/X_error.npy",  
        output_model_dir=OUTPUT_MODEL_DIR,
        test_size=0.2,
        random_state=42,
        poly_deg=5,
        epochs_adam=1000,
        steps_lbfgs=100,
        max_batch_size=15
    )
    """
    

    # 3. Execute Phase 2: Quantitative Evaluation (Reads the .npz file)
    evaluate_test_set_from_file(
        input_filepath=npz_results_path,
        output_dir=OUTPUT_MODEL_DIR
    )

    # 4. Execute Phase 3: Visualization (Reads the .npz file)
    generate_plots_from_file(
        input_filepath=npz_results_path,
        output_dir=VIS_DIR,
        num_samples=18
    )