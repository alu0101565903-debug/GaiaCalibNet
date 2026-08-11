"""
Module: all_cross_validation.py
Report Section:

Unified K-Fold Cross-Validation Pipeline for All Registered PINN Architectures.

This script generalizes the existing cross-validation workflow so it can train,
validate, and export results for every PINN architecture declared in
"src/models/pinn_architectures.py" using the matching loss function from
"src/train/loss/losses.py".

Supported model keys:
- residual_bicubic
- residual_gaussian
- pca_nn
- pca_params_nn
- cheb_undegraded_nn

The pipeline supports the following capabilities:
- fold-isolated PCA basis generation for PCA-based models
- proper dataset construction for parameter-conditioned models
- optional stochastic uncertainty handling for Gaussian PINNs
- per-fold model checkpointing, loss history export, prediction export,
  quantitative evaluation, and validation plots
"""

import os
from typing import Dict, Tuple, Optional
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
    UndegradedChebyshevPhysicsInformedLoss
)
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed
from src.utils.extra.seed import set_seed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_loss import plot_loss_evolution
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file
from src.experiments.evaluation.plot_stellar_dependencies import evaluate_stellar_dependencies


# Global collections to dictate behavior based on specific model architecture types
SUPPORTED_MODELS = {
    "residual_bicubic",
    "residual_gaussian",
    "pca_nn",
    "pca_params_nn",
    "cheb_undegraded_nn"
}

REQUIRES_STELLAR_PARAMS = {"pca_params_nn", "cheb_undegraded_nn"}
REQUIRES_PCA_BASIS = {"pca_nn", "pca_params_nn"}
SUPPORTS_SIGMA = {"residual_gaussian"}


# =====================================================================
# EXPORT WRAPPER
# =====================================================================
class InferenceExportWrapper(torch.nn.Module):
    """
    Wrapper class to unify the inference output for the export function.
    Automatically applies inverse transform for PCA/Undegraded models.
    """
    def __init__(self, base_model: torch.nn.Module, model_name: str):
        """
        Initializes the InferenceExportWrapper instance.

        Args:
        - base_model (torch.nn.Module): The underlying PINN neural network model.
        - model_name (str): Identifier string representing the specific model architecture.
        """
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        
    def forward(self, x: torch.Tensor, params: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward propagation and applies necessary inverse reconstructions
        depending on the target model architecture.

        Args:
        - x (torch.Tensor): Input feature tensor of shape (N, input_dim).
        - params (torch.Tensor, optional): Optional stellar parameter conditioning tensor. 
            Defaults to None.

        Returns:
            torch.Tensor: The reconstructed output predictions tensor.
        """
        # Check if the current architecture requires structural reconstruction (PCA or Chebyshev)
        if self.model_name in {"pca_params_nn", "cheb_undegraded_nn"}:
            if params is not None:
                out = self.base_model(x, params)
            else:
                out = self.base_model(x)
            # Reconstruct the physical space domain from the latent coefficients or polynomial representation
            return self.base_model.reconstruct(out)
        elif self.model_name == "pca_nn":
            out = self.base_model(x)
            return self.base_model.reconstruct(out)
        else:
            # Standard residual structures directly yield the targeted output format
            return self.base_model(x)
        
# =====================================================================
# DATA HELPERS
# =====================================================================

def load_stellar_parameters(params_dir: str) -> np.ndarray:
    """
    Load and standardize stellar parameters from a directory.

    This helper reads "bp_rp.npy", "G_mags.npy", and "ruwe.npy", 
    applies per-feature Z-score normalization, and concatenates
    the results into a single array of shape (N, 3).
    
    Args:
    - params_dir (str): Directory path containing the stellar parameter .npy files.

    Returns:
    - np.ndarray: A 2D numpy array of standardized stellar parameters with shape (N, 3).

    Raises:
        FileNotFoundError: If any of the required stellar parameter files are missing.
    """
    bp_rp_path = os.path.join(params_dir, "bp_rp.npy")
    g_mag_path = os.path.join(params_dir, "G_mags.npy")
    ruwe_path = os.path.join(params_dir, "ruwe.npy")

    # Validate file presence before loading arrays into memory
    for p in [bp_rp_path, g_mag_path, ruwe_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required stellar parameter file not found: {p}")

    # Load arrays and reshape them uniformly to column vectors
    bp_rp = np.load(bp_rp_path).reshape(-1, 1)
    g_mag = np.load(g_mag_path).reshape(-1, 1)
    ruwe = np.load(ruwe_path).reshape(-1, 1)

    def standardize(column: np.ndarray) -> np.ndarray:
        """
        Applies Z-score standardization to a feature column to stabilize network training.

        Args:
        - column (np.ndarray): Input feature vector to be standardized.

        Returns:
        - np.ndarray: Standardized feature column with zero mean and unit variance.
        """
        # Add a tiny epsilon term (1e-8) in the denominator to avoid division by zero errors
        return (column - np.mean(column)) / (np.std(column) + 1e-8)

    # Horizontally stack individual normalized attributes into a unified multi-parameter dataset matrix
    params = np.hstack([
        standardize(bp_rp),
        standardize(g_mag),
        standardize(ruwe),
    ])
    return params.astype(np.float32)


def compute_pca_basis(X: np.ndarray, Y: np.ndarray, num_components: int, random_state: int, window_length: int = 5) -> Dict[str, np.ndarray]:
    """
    Args:
    - X (np.ndarray): Input training features array.
    - Y (np.ndarray): Target training features array.
    - num_components (int): Number of principal components to retain in the PCA basis.
    - random_state (int): Seed value for reproducible random number generation.
    - window_length (int, optional): Window length parameter for the Savitzky-Golay filter. Defaults to 5.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the PCA "basis" (components) and "mean".
    """
    # Calculate the empirical ratio with an epsilon buffer to prevent division by zero anomalies
    R_raw = Y / (X + 1e-15)
    # Clip outliers to stabilize the smoothing and decomposition stages
    R_clipped = np.clip(R_raw, 0.5, 2.0)
    # Apply Savitzky-Golay filtering along the spectral axis to mitigate high-frequency noise
    R_smooth = signal.savgol_filter(R_clipped, window_length=window_length, polyorder=2, axis=1, mode="nearest")

    # Fit Principal Component Analysis on the processed ratio space
    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(R_smooth)

    return {
        "basis": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
    }


def build_model_and_loss(
    model_name: str,
    input_dim: int,
    num_params: int = 0,
    pca_basis: Optional[np.ndarray] = None,
    pca_mean: Optional[np.ndarray] = None,
    num_components: int = 15,
    lambda_flux: float = 1.0,
    lambda_smooth: float = 1e-3,
    lambda_reg: float = 1e-4,
) -> Dict[str, object]:
    """
    Instantiate the requested model architecture and its matching loss module.
    Args:
    - model_name (str): Identifier string of the target PINN model architecture.
    - input_dim (int): Dimensionality of the input feature vectors.
    - num_params (int, optional): Number of extra conditioning stellar parameters. Defaults to 0.
    - pca_basis (Optional[np.ndarray], optional): Precomputed PCA basis matrix. Defaults to None.
    - pca_mean (Optional[np.ndarray], optional): Precomputed PCA mean vector. Defaults to None.
    - num_components (int, optional): Number of principal components or polynomial degree. Defaults to 15.
    - lambda_flux (float, optional): Weight penalty coefficient for flux loss terms. Defaults to 1.0.
    - lambda_smooth (float, optional): Weight penalty coefficient for smoothness or curvature terms. Defaults to 1e-3.
    - lambda_reg (float, optional): Regularization weight penalty coefficient. Defaults to 1e-4.

    Returns:
    - Dict[str, object]: A dictionary containing the instantiated "model", "loss" object, 
        and the callable "physics_criterion".

    Raises:
        ValueError: If an unsupported model name is supplied, or if PCA components are missing when required.
    """
    # Instantiate the correct model configuration and matching physics criterion dynamically based on name
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
            num_params=num_params,
        )
        loss = PCAParamsPhysicsInformedLoss(
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            lambda_smooth=lambda_smooth,
            lambda_reg=lambda_reg
        )
        physics_criterion = lambda m, x, y, params=None: loss(m, x, y, params)

    elif model_name == "cheb_undegraded_nn":
        model = get_pinn_model(
            "cheb_undegraded_nn",
            input_dim=input_dim,
            degree=num_components,
            num_params=num_params
        )
        loss = UndegradedChebyshevPhysicsInformedLoss(
            curvature_weight=lambda_smooth, # Maping lambda_smooth to curvature_weight
            local_reg_weight=lambda_reg # Maping lambda_reg to local_reg_weight
        )
        
        physics_criterion = lambda m, x, y, params=None: loss(m, x, y, params)
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
) -> Tuple[TensorDataset, Optional[torch.Tensor]]:
    """
    Build the specific fold training dataset and the optional full-batch extra tensor 
    tailored to the requirements of the chosen PINN architecture.

    Args:
    - model_name (str): Identifier string of the target model architecture.
    - X_train (np.ndarray): Training inputs array.
    - Y_train (np.ndarray): Training targets array.
    - X_err_train (np.ndarray): Training error/uncertainty array.
    - params_train (Optional[np.ndarray]): Training stellar parameters array if applicable.
    - has_sigma (bool): Boolean flag indicating whether uncertainty values are available.

    Returns:
    - Tuple[TensorDataset, Optional[torch.Tensor]]: A tuple consisting of the PyTorch 
        TensorDataset for batching and the optional full-batch extra tensor.
    """
    # Convert numpy inputs and targets into core PyTorch tensor structures
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)

    # Wrap additional attributes into the dataset based on model architectural dependencies
    if model_name in REQUIRES_STELLAR_PARAMS and params_train is not None:
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
    Execute a generic K-Fold cross-validation workflow for the selected PINN model architecture,
    managing training iterations, evaluation checkpoints, loss evolution plots, and predictions export.

    Args:
    - model_name (str): Identifier name of the PINN architecture to evaluate.
    - dataset_mode (str, optional): Dataset configuration mode subset (e.g., "all", "blue", "solar"). Defaults to "all".
    - x_path (str, optional): File system path to the input feature array matrix X. Defaults to "data/processed/X.npy".
    - y_path (str, optional): File system path to the target array matrix Y. Defaults to "data/processed/Y.npy".
    - x_err_path (str, optional): File system path to error data matrix. Defaults to "data/processed/X_error.npy".
    - params_dir (str, optional): Directory path containing stellar parameter assets. Defaults to "data/processed".
    - output_model_dir (str, optional): Target directory path to store checkpoints and evaluation logs. Defaults to "results/models/cv_all".
    - n_splits (int, optional): Number of folds for cross-validation splitting. Defaults to 5.
    - random_state (int, optional): Global random seed for reproducibility. Defaults to 42.
    - num_components (int, optional): Number of PCA components or basis functions. Defaults to 5.
    - lambda_flux (float, optional): Loss penalty weight for flux constraints. Defaults to 1.0.
    - lambda_smooth (float, optional): Loss penalty weight for model smoothness. Defaults to 1e-3.
    - lambda_reg (float, optional): Loss penalty weight for weight regularization. Defaults to 1e-4.
    - epochs_adam (int, optional): Total training epochs using the Adam optimizer. Defaults to 1000.
    - steps_lbfgs (int, optional): Total optimization steps using the L-BFGS optimizer. Defaults to 100.
    - max_batch_size (int, optional): Upper bound for mini-batch sizes during training. Defaults to 32.
    - effective_params (int, optional): Effective parameter count for metrics evaluation. Defaults to 20.

    Returns:
    - None: Results, plots, and checkpoints are serialized and written directly to the file system.

    Raises:
        ValueError: If an unsupported model name is supplied.
        FileNotFoundError: If core data file paths do not exist.
    """
    # Validate that the requested model exists inside the supported collection dictionary
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Choose from {sorted(SUPPORTED_MODELS)}")

    # Initialize random seeds and identify the available hardware acceleration device (GPU/CPU)
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running cross-validation for '{model_name}' on device: {device}")

    # Verify foundational feature array file paths exist on the disk
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing dataset at '{x_path}' or '{y_path}'.")

    # Load core numpy arrays for features and targets
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path) if os.path.exists(x_err_path) else np.zeros_like(X_np, dtype=np.float32)
    has_sigma = os.path.exists(x_err_path)

    # Conditionally attempt to load stellar parameters if required by the target model architecture
    params_np = None
    if model_name in REQUIRES_STELLAR_PARAMS:
        try:
            params_np = load_stellar_parameters(params_dir)
        except FileNotFoundError:
            print(f"[!] Warning: Parameters requested but not found in {params_dir}. Model will run without parameters.")
            params_np = None

    num_samples = X_np.shape[0]
    print(f"[>] Total samples loaded: {num_samples}")
    if params_np is not None:
        print(f"[>] Stellar parameters shape: {params_np.shape}")

    # Setup K-Fold split configuration with shuffling enabled
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Iterate over each cross-validation fold index partition
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_np), 1):
        print(f"\n{'='*60}\n[>] Fold {fold}/{n_splits}\n{'='*60}")

        # Slice training and validation subsets for the current fold
        X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
        Y_train_np, Y_test_np = Y_np[train_idx], Y_np[test_idx]
        X_err_train_np, X_err_test_np = X_err_np[train_idx], X_err_np[test_idx]
        params_train_np = params_np[train_idx] if params_np is not None else None
        params_test_np = params_np[test_idx] if params_np is not None else None

        print(f"    Training samples: {len(X_train_np)} | Validation samples: {len(X_test_np)}")

        # Calculate fold-isolated PCA basis components strictly from the training split to avoid data leakage
        pca_basis = None
        pca_mean = None
        if model_name in REQUIRES_PCA_BASIS:
            pca_stats = compute_pca_basis(X_train_np, Y_train_np, num_components, random_state + fold)
            pca_basis = pca_stats["basis"]
            pca_mean = pca_stats["mean"]

        # Build model architecture instances and bind corresponding physics loss equations
        model_and_loss = build_model_and_loss(
            model_name=model_name,
            input_dim=X_train_np.shape[1],
            num_params=params_train_np.shape[1] if params_train_np is not None else 0,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            num_components=num_components,
            lambda_flux=lambda_flux,
            lambda_smooth=lambda_smooth,
            lambda_reg=lambda_reg,
        )

        model = model_and_loss["model"].to(device)
        physics_criterion = model_and_loss["physics_criterion"]

        # Prepare PyTorch training dataset wrappers for mini-batch execution
        dataset_train, extra_full_train = prepare_fold_dataset(
            model_name=model_name,
            X_train=X_train_np,
            Y_train=Y_train_np,
            X_err_train=X_err_train_np,
            params_train=params_train_np,
            has_sigma=has_sigma
        )

        # Configure DataLoader settings with reproducible generator seeding per fold
        batch_size = min(max_batch_size, len(dataset_train))
        generator = torch.Generator()
        generator.manual_seed(random_state + fold)
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=generator)

        # Move full-batch structural tensors to the active compute device
        X_full_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        Y_full_train = torch.tensor(Y_train_np, dtype=torch.float32).to(device)

        if params_train_np is not None:
            params_full_train = torch.tensor(params_train_np, dtype=torch.float32).to(device)
        else:
            params_full_train = None
        
        extra_full_train = (
            torch.tensor(X_err_train_np, dtype=torch.float32).to(device)
            if model_name == "residual_gaussian" and has_sigma
            else params_full_train
        )

        # Execute the two-stage hybrid training strategy (Adam optimization followed by L-BFGS refinement)
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

        # Set up fold directory storage paths and ensure directory creation
        fold_dir = os.path.join(output_model_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Serialize model checkpoints containing network weights and meta-parameters
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

        # Plot and save training loss evolution history curves
        loss_plot_path = os.path.join(fold_dir, "loss_evolution.png")
        plot_loss_evolution(loss_history, loss_plot_path)

        # Move the trained model back to the hardware device for inference evaluation
        model.to(device)

        # Wrap the model to standardize multi-architecture predictions (e.g., R(lambda) reconstruction)
        wrapped_model = InferenceExportWrapper(model, model_name)

        # Cast test subset data into tensor arrays on the target processing device
        params_test_tensor = (
            torch.tensor(params_test_np, dtype=torch.float32).to(device)
            if params_test_np is not None
            else None
        )

        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
        X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)

        # Export test subset predictions to file archives for performance tracking
        export_test_predictions(
            model=wrapped_model,
            X_tensor=X_test_tensor,
            Y_tensor=Y_test_tensor,
            X_err_tensor=X_err_test_tensor,
            params_tensor=params_test_tensor,
            output_filepath=os.path.join(fold_dir, "test_predictions.npz"),
        )

        # Compute quantitative metrics evaluation across the fold test partition
        evaluate_test_set_from_file(
            input_filepath=os.path.join(fold_dir, "test_predictions.npz"),
            output_dir=fold_dir,
            num_knots=num_components if model_name.startswith("pca") else effective_params,
        )

        # Generate comparative model visualization diagrams from test prediction outputs
        generate_plots_from_file(
            input_filepath=os.path.join(fold_dir, "test_predictions.npz"),
            output_dir=os.path.join(fold_dir, "comparison"),
            num_samples=min(18, len(X_test_np)),
        )

    # Evaluate aggregated stellar parameter dependencies across all folds upon completion
    evaluate_stellar_dependencies(
        dataset_mode=dataset_mode,
        nn_mode=model_name,
        n_splits=n_splits, 
        random_state=random_state,
        models_dir=output_model_dir
    ) 

    print(f"\n[✓] Cross-validation finished. Results saved to: {output_model_dir}")


if __name__ == "__main__": 
    # Configuration Options: "all", "blue", "solar"
    MODE = "all"

    # Configuration Options: "residual_bicubic", "residual_gaussian", "pca_nn", "pca_params_nn", "cheb_undegraded_nn"
    MODEL_NAME = "cheb_undegraded_nn"

    # Trigger the full cross-validation pipeline execution sequence
    run_cross_validation(
        model_name=MODEL_NAME,
        dataset_mode=MODE,
        x_path=f"data/processed/nn_undegraded/{MODE}/X.npy",
        y_path=f"data/processed/nn_undegraded/{MODE}/Y.npy",
        x_err_path=f"data/processed/nn_undegraded/{MODE}/X_error.npy",
        params_dir=f"data/processed/nn_undegraded/{MODE}",
        output_model_dir=f"results/cross_validation/{MODEL_NAME}_cv/{MODE}",
        n_splits=5,
        random_state=42,
        num_components=40,
        lambda_flux=0.1,
        lambda_smooth=1e-3,
        lambda_reg=1e-4,
        epochs_adam=1000,
        steps_lbfgs=100,
        max_batch_size=16,
        effective_params=20
    )