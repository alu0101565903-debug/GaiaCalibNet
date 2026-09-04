"""
Module: crossvalidation.py

K-Fold Cross-Validation Pipeline for PCA-based Physics-Informed Neural Networks (PINN).

This module orchestrates a robust K-Fold cross-validation training and evaluation 
pipeline for a PCA-augmented Physics-Informed Neural Network (PINN). The target task 
is to predict a multiplicative, wavelength-dependent spectral correction factor R(lambda) 
used to recalibrate Gaia BP/RP optical spectra against high-precision CALSPEC reference spectra.

Key Architectural & Pipeline Features:
1. Conditional Physical Inputs:
   Integrates stellar physical parameters (BP-RP color index, G magnitude, and RUWE 
   astrometric quality factor) into the neural network to condition calibration corrections 
   on stellar property domains.
2. Leakage-Free Dynamic PCA Basis Generation:
   Computes a reduced-order principal component analysis (PCA) basis independently for 
   each CV fold using solely that fold's training split. This avoids data leakage between 
   training and validation partitions.
3. Signal Preprocessing & Cleaning:
   Applies clipping and Savitzky-Golay smoothing to empirical flux ratios R = Y / X prior 
   to PCA fitting, preventing numerical instabilities or edge artifacts (such as red-edge 
   divergence) from degrading the component basis.
4. Two-Stage Physics-Informed Optimization:
   Trains models using a multi-term physics-informed loss function penalizing unphysical 
   spectral oscillations (smoothness) and large PCA coefficient deviations, leveraging 
   a hybrid Adam + L-BFGS optimization routine.
5. Automated Diagnostics & Artifact Exporting:
   Saves trained model state dictionaries, loss evolution curves, reconstructed test-set 
   predictions, performance metrics, and comparative diagnostic plots for every fold.
"""

import os
import numpy as np
import torch
import joblib
import scipy.signal as signal
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from src.utils.extra.seed import set_seed
from src.models.pinn_architecture import PCAParamsSpectralPINN
from src.train.loss.loss import PCAParamsPhysicsInformedLoss 
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file
from src.utils.visualize_models_performance.plot_loss import plot_loss_evolution

# =================================================================
# EXPORT WRAPPER
# =================================================================
class PCAExportWrapper(torch.nn.Module):
    """
    Inference and Export Wrapper for PCA-based Spectral PINN Models.

    This wrapper adapts the raw forward pass of a "PCAParamsSpectralPINN" instance 
    so that it outputs full high-dimensional spectral correction curves R(lambda) 
    rather than raw low-dimensional PCA latent coefficients.

    When called during evaluation or export routines, it intercepts model predictions 
    and applies the inverse PCA transform using the model's internal basis vectors 
    and empirical mean vector.

    Attributes
    - base_model (torch.nn.Module): The underlying trained "PCAParamsSpectralPINN" model instance.
    """
    def __init__(self, base_model: torch.nn.Module):
        """
        Initialize the wrapper with a base PINN architecture.

        Parameters
        - base_model (torch.nn.Module): The trained model to wrap for inference.
        """
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x: torch.Tensor, params: torch.Tensor = None) -> torch.Tensor:
        """
        Execute forward inference and reconstruct full spectral correction curves.

        Parameters
        - x (torch.Tensor): Input degraded flux spectra tensor of shape (batch_size, num_wavelengths).
        - params (torch.Tensor, optional): Auxiliary stellar physical parameters tensor of shape (batch_size, num_stellar_params).

        Returns
        - torch.Tensor: Reconstructed full spectral response factor R(lambda) of shape (batch_size, num_wavelengths).
        """
        # Execute base forward pass (returns raw latent PCA coefficients or full spectra depending on architecture)
        if params is not None:
            out = self.base_model(x, params)
        else:
            out = self.base_model(x)

        # Reconstruct full-resolution spectrum if the base model exposes a reconstruction method
        if hasattr(self.base_model, "reconstruct"):
            return self.base_model.reconstruct(out)
        return out

# =================================================================
# HELPER DATA LOADER
# =================================================================
def load_stellar_parameters(data_dir: str) -> np.ndarray:
    """
    Load, validate, and standardize auxiliary stellar physical parameters.

    Reads three separate parameter arrays ("bp_rp.npy", "G_mags.npy" / "G_mag.npy", 
    and "ruwe.npy") from the specified directory, validates file existence, applies 
    Z-score standardization (zero mean, unit variance) independently across each 
    feature, and concatenates them into a single 2D NumPy array.

    Parameters
    - data_dir (str): Directory path where the individual ".npy" parameter files reside.

    Returns
    - np.ndarray: Standardized stellar parameter array of shape (N, 3) and data type float32, 
        where columns represent [BP-RP color, G magnitude, RUWE].

    Raises
    - FileNotFoundError: If any of the required parameter files are missing from "data_dir".
    """
    bp_rp_path = os.path.join(data_dir, "bp_rp.npy")
    
    # Account for potential naming convention differences across dataset versions
    g_mag_path = os.path.join(data_dir, "G_mags.npy")
    if not os.path.exists(g_mag_path):
        g_mag_path = os.path.join(data_dir, "G_mag.npy")
        
    ruwe_path = os.path.join(data_dir, "ruwe.npy")

    # Ensure all required parameter files exist on disk
    for path in [bp_rp_path, g_mag_path, ruwe_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stellar parameter file missing: '{path}'")

    # Load and reshape individual column vectors to shape (N, 1)
    bp_rp = np.load(bp_rp_path).reshape(-1, 1)
    g_mag = np.load(g_mag_path).reshape(-1, 1)
    ruwe = np.load(ruwe_path).reshape(-1, 1)

    # Apply per-feature Z-score standardization: z = (x - mu) / (sigma + eps)
    bp_rp_std = (bp_rp - np.mean(bp_rp)) / (np.std(bp_rp) + 1e-8)
    g_mag_std = (g_mag - np.mean(g_mag)) / (np.std(g_mag) + 1e-8)
    ruwe_std = (ruwe - np.mean(ruwe)) / (np.std(ruwe) + 1e-8)

    # Horizontal stack into a unified feature matrix of shape (N, 3)
    params_stacked = np.hstack([bp_rp_std, g_mag_std, ruwe_std])
    return params_stacked.astype(np.float32)

# =================================================================
# CROSS-VALIDATION PIPELINE
# =================================================================
def run_cross_validation(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy",
    params_path: str = "data/processed/params.npy",
    output_model_dir: str = "results/models/pca_nn_cv",
    n_splits: int = 5,
    random_state: int = 42,
    num_components: int = 5,           # Optimal PCA components found to prevent wiggles
    lambda_smooth: float = 1e-3,       # Optimal smoothing loss weight
    lambda_reg: float = 5e-3,          # Optimal PCA coefficient regularization
    epochs_adam: int = 5000,
    steps_lbfgs: int = 100,
    max_batch_size: int = 32
) -> None:
    """
    Handles data preparation, K-Fold splitting, dynamic PCA basis calculation per fold, 
    PINN model training, and exporting inference results to disk.

    Execution Flow per Fold:
    1. Partition dataset into train and validation subsets.
    2. Compute empirical flux ratio curves R_train = Y_train / X_train.
    3. Apply numerical clipping [0.5, 2.0] and Savitzky-Golay filtering to ratio profiles.
    4. Fit a fold-isolated PCA model to construct basis components and empirical mean.
    5. Instantiate "PCAParamsSpectralPINN" network and "PCAParamsPhysicsInformedLoss".
    6. Run two-stage training (Adam stochastic optimization followed by L-BFGS fine-tuning).
    7. Save model artifacts, weights, loss histories, and PCA metadata to disk.
    8. Generate validation predictions, compute performance metrics, and render comparison plots.

    Args:
    - x_path (str): Filepath to the input degraded spectra (X) numpy array.
    - y_path (str): Filepath to the target reference spectra (Y) numpy array.
    - x_err_path (str): Filepath to the input spectra errors numpy array.
    - params_path (str): Filepath to the stellar physical parameters numpy array.
    - output_model_dir (str): Root directory where the trained models and results for each fold will be saved.
    - n_splits (int): Number of folds for the cross-validation.
    - random_state (int): Seed used by the random number generator for reproducibility.
    - num_components (int): Number of principal components for the PCA basis.
    - lambda_smooth (float): Weight for the smoothness regularization term in the loss function.
    - lambda_reg (float): Weight for the PCA coefficient regularization term in the loss function
    - epochs_adam (int): Number of epochs for the Adam optimizer phase.
    - steps_lbfgs (int): Number of steps for the L-BFGS optimizer phase.
    - max_batch_size (int): Maximum batch size for the DataLoader.
    """
    # Seed PyTorch, NumPy, and random number generators for strict reproducibility
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running {n_splits}-Fold Cross-Validation on device: {device}")

    # --- Data Loading & Validation ---
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Verify that '{x_path}' and '{y_path}' exist in the specified paths.")

    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path)
    params_np = load_stellar_parameters(params_path)

    num_total_samples = X_np.shape[0]
    print(f"[>] Total samples loaded: {num_total_samples}")
    print(f"[>] Stellar parameters shape: {params_np.shape}")

    # --- K-Fold Cross-Validation Setup ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # --- Cross-Validation Iteration Loop ---
    for fold, (train_index, test_index) in enumerate(kf.split(X_np)):
        print(f"\n" + "="*50)
        print(f"[>] TRAINING FOLD {fold + 1} / {n_splits}")
        print(f"="*50)

        # 1. Isolate data splits for the current fold
        X_train_np, X_test_np = X_np[train_index], X_np[test_index]
        Y_train_np, Y_test_np = Y_np[train_index], Y_np[test_index]
        X_err_train_np, X_err_test_np = X_err_np[train_index], X_err_np[test_index]
        params_train_np, params_test_np = params_np[train_index], params_np[test_index]

        print(f"    Training samples: {len(X_train_np)} | Validation samples: {len(X_test_np)}")

        # Convert training split to PyTorch float32 tensors
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
        params_train_tensor = torch.tensor(params_train_np, dtype=torch.float32)

        dataset_train = TensorDataset(X_train_tensor, Y_train_tensor, params_train_tensor)
        batch_size = min(max_batch_size, len(dataset_train))

        # Generator seeding for reproducible mini-batch shuffling per fold
        g = torch.Generator()
        g.manual_seed(random_state + fold) 
        dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

        # Move full training matrices to target compute device for full-batch operations
        X_full_train = X_train_tensor.to(device)
        Y_full_train = Y_train_tensor.to(device)
        params_full_train = params_train_tensor.to(device)

        # 2. Dynamic Fold-Isolated PCA Basis Construction
        print(f"[>] Computing PCA basis ({num_components} components) for Fold {fold + 1}...")

        # Calculate unconstrained empirical flux ratio profile: R = Y / X
        R_train_raw = Y_train_np / (X_train_np + 1e-15)
        
        # Clip unphysical ratio extremes (force R within range [0.5, 2.0])
        # This removes zero-division singularities and extreme red-edge distortions
        R_train_clipped = np.clip(R_train_raw, 0.5, 2.0)
        
        # Apply Savitzky-Golay filtering (window length 11, 2nd order polynomial)
        # Prevents high-frequency spectral noise and edge droop from corrupting the PCA basis
        R_train_smooth = signal.savgol_filter(R_train_clipped, window_length=11, polyorder=2, axis=1, mode="nearest")
        
        # Fit PCA strictly on the cleaned, smoothed training split of the current fold
        pca = PCA(n_components=num_components, random_state=random_state)
        pca.fit(R_train_smooth)
        
        pca_basis_np = pca.components_      # Shape: (num_components, num_wavelengths)
        pca_mean_np = pca.mean_             # Shape: (num_wavelengths,)
        
        # 3. Model & Loss Function Initialization
        num_channels = X_train_tensor.shape[1]
        num_stellar_params = params_train_tensor.shape[1]

        # Instantiate network conditioning architecture with fold-specific PCA parameters
        model = PCAParamsSpectralPINN(
            input_dim=num_channels, 
            pca_basis=pca_basis_np,
            pca_mean=pca_mean_np,
            num_stellar_params=num_stellar_params,
        ).to(device)

        # Instantiate physics-informed multi-objective loss function
        physics_criterion = PCAParamsPhysicsInformedLoss(
            pca_basis=pca_basis_np,
            pca_mean=pca_mean_np,
            lambda_smooth=lambda_smooth,
            lambda_reg=lambda_reg
        ).to(device)

        # 4. Model Training Phase (Adam + L-BFGS hybrid optimization)
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

        # 5. Export Model Checkpoint & Loss Evolution Plot
        fold_output_dir = os.path.join(output_model_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        save_path = os.path.join(fold_output_dir, "pinn_pca_recalibration.joblib")
        export_model = {
            "model_state_dict": trained_model.state_dict(),
            "input_dim": num_channels,
            "num_stellar_params": num_stellar_params,
            "num_components": num_components,
            "lambda_smooth": lambda_smooth,
            "lambda_reg": lambda_reg,
            "pca_basis": pca_basis_np,
            "pca_mean": pca_mean_np 
        }
        joblib.dump(export_model, save_path)
        print(f"[✓] Fold {fold + 1} model saved to: {save_path}")

        # Render and save loss optimization curve
        loss_plot_path = os.path.join(fold_output_dir, f"loss_evolution_fold_{fold + 1}.png")
        plot_loss_evolution(loss_history, loss_plot_path)
        
        # Wrap model for test inference (enables inverse PCA transform reconstruction)
        export_ready_model = PCAExportWrapper(trained_model)

        # 6. Test Inference, Prediction Export, and Metric Evaluation
        results_filepath = os.path.join(fold_output_dir, f"test_predictions_fold_{fold + 1}.npz")

        # Prepare validation tensors on compute device
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
        X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)
        params_test_tensor = torch.tensor(params_test_np, dtype=torch.float32).to(device)

        # Compute and export test predictions to disk (.npz format)
        export_test_predictions(
            model=export_ready_model,  
            X_tensor=X_test_tensor,
            Y_tensor=Y_test_tensor,
            X_err_tensor=X_err_test_tensor,
            params_tensor=params_test_tensor,
            output_filepath=results_filepath,
        )

        # Execute quantitative residual & error metric evaluation
        evaluate_test_set_from_file(
            input_filepath=results_filepath,
            output_dir=fold_output_dir,
            num_knots=num_components
        )

        # Generate comparative visualization plots for sample spectra
        vis_dir = os.path.join(fold_output_dir, "comparison")
        generate_plots_from_file(
            input_filepath=results_filepath,
            output_dir=vis_dir,
            num_samples=min(18, len(X_test_np)) # Ensure we don't request more samples than available in the test set
        )
        
    print(f"\n[✓] Cross-Validation successfully completed! All results saved in: {output_model_dir}")


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    # Select dataset partition mode: "solar", "blue", or "all"
    mode = "blue"  
    
    # Define output root directory based on chosen mode
    OUTPUT_MODEL_DIR = f"results/models/pca_nn_cv/{mode}"

    # Execute cross-validation pipeline if mode is valid
    if mode in ["solar", "blue", "all"]:
        print(f"[>] Starting Cross-Validation for mode: '{mode}'")

        run_cross_validation(
            x_path=f"data/processed/nn/{mode}/X.npy",
            y_path=f"data/processed/nn/{mode}/Y.npy",
            x_err_path=f"data/processed/nn/{mode}/X_error.npy",   
            params_path=f"data/processed/nn/{mode}",
            output_model_dir=OUTPUT_MODEL_DIR,
            n_splits=6,
            random_state=42,
            num_components=5,          # 5 principal components
            lambda_smooth=1e-5,        # Smoothness loss weight hyperparameter
            lambda_reg=1e-4,           # L2 coefficient regularization hyperparameter
            epochs_adam=1000,
            steps_lbfgs=100,
            max_batch_size=32
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'solar', 'blue', or 'all'.")