"""
Module: ablation_test.py
Report Section:

Ablation study module for isolating and quantifying the individual and joint
contributions of three core architectural decisions in neural spectral calibration:
1. PCA-based spectral response parameterization (low-dimensional basis projection).
2. Physics-informed loss regularization (PINN smoothness & scale-invariant flux error).
3. Stellar parameter conditioning (incorporating auxiliary physical properties like Teff, logg, [Fe/H]).

This module systematically compares four model variants in a controlled baseline setting:
- Model A: Baseline MSE model without PCA parameterization or physics constraints, with stellar parameters.
- Model B: Direct pixel-wise PINN model without PCA parameterization, with physics loss and stellar parameters.
- Model C: PCA-based spectral response model with physics loss, without stellar parameter conditioning.
- Model D: Full hybrid model combining PCA parameterization, physics loss, and stellar parameter conditioning.

Goal:
Measure the contribution of each design component to the overall normalized error (MAE %)
and evaluate the structural interactions between physics-informed losses and reduced-order priors.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import scipy.signal as signal

from src.models.pinn_architectures import PCAParamsSpectralPINN, PCASpectralPINN
from src.train.loss.losses import PCAParamsPhysicsInformedLoss, PCAPhysicsInformedLoss
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed as train_2stage_params
from src.train.training_functions.fixed_training_strategies import train_2stage_fixed as train_2stage_no_params
from src.utils.extra.seed import set_seed
from src.experiments.evaluation.cross_validation import load_stellar_parameters

# =================================================================
# 1. ABLATION MODULES AND LOSS FUNCTIONS
# =================================================================

class DirectPixelPINN(nn.Module):
    """
    Direct Pixel-Wise Spectral Response Model (Baseline architecture without PCA).

    This neural network directly predicts a multiplicative spectral correction curve 
    R(lambda) at full spectral resolution (pixel-by-pixel). The network receives 
    the uncalibrated spectrum and, optionally, tabular stellar parameters.

    Architecture summary:
    - Input: Combined normalized spectrum (input_dim) + optional stellar parameters.
    - Hidden Layers: Fully connected MLP (128 -> 64) with SiLU activation functions.
    - Output Layer: Linear mapping to full spectral dimension (input_dim), zero-initialized.
    - Residual Connection: R(lambda) = 1.0 + delta_R, bounded to [0.3, 3.0].

    Parameters
    - input_dim (nt): Dimensionality of the input spectrum (number of wavelength bins/pixels).
    - num_stellar_params (int, default=0): Number of auxiliary stellar physical parameters 
        passed as additional inputs.
    """
    def __init__(self, input_dim: int, num_stellar_params: int=0):
        super(DirectPixelPINN, self).__init__()
        self.input_dim = input_dim
        self.use_params = num_stellar_params > 0
        total_input_dim = input_dim + num_stellar_params

        # Shallow feedforward architecture to evaluate baseline pixel-wise performance
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, input_dim)  # Predict the full spectral response vector.
        )

        # Initialize output layer weights and biases to zero so initial output is delta_R = 0
        # (resulting in an initial response multiplier R = 1.0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, stellar_params: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass to compute pixel-wise response curves.

        Parameters
        - x (torch.Tensor): Input flux spectra tensor of shape (batch_size, input_dim).
        - stellar_params (torch.Tensor, optional): Auxiliary stellar parameter tensor of 
            shape (batch_size, num_stellar_params).

        Returns
        - torch.Tensor: Predicted continuous multiplicative response curve R of shape (batch_size, input_dim).
        """
        # Peak normalization for scale stability across different brightness levels
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val

        # Concatenate stellar parameters if available and configured
        if self.use_params and stellar_params is not None:
            x_combined = torch.cat([x_norm, stellar_params], dim=1)
        else:
            x_combined = x_norm

        # Compute residual output delta_R around identity base (R = 1.0 + delta_R)
        delta_R = self.net(x_combined)
        R = 1.0 + delta_R

        # Enforce reasonable lower and upper physical safety boundaries
        return torch.clamp(R, min=0.3, max=3.0)

    def prepare_for_lbfgs(self):
        """Switch network to evaluation mode for L-BFGS second-stage optimization."""
        self.eval()

class DirectMSELoss(nn.Module):
    """
    Pure Mean Squared Error (MSE) Loss without Physics-Informed Regularization.

    Computes unweighted pixel-wise MSE between the calibrated spectrum 
    (input spectrum element-wise multiplied by predicted response) and the ground-truth target.
    Serves as an empirical baseline to evaluate models trained without domain constraints.
    """
    def __init__(self):
        super(DirectMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor, 
        stellar_params: torch.Tensor=None
    ) -> torch.Tensor:
        """
        Compute standard MSE loss.

        Parameters
        - model (nn.Module): Spectral network architecture.
        - x_input (torch.Tensor): Input flux spectra tensor.
        - y_target (torch.Tensor): Target flux spectra tensor.
        - stellar_params (torch.Tensor, optional): Stellar parameter conditioning tensor.

        Returns
        - torch.Tensor: Scalar mean squared error value.
        """
        # Support both architectures that consume stellar params and those that do not.
        if stellar_params is not None:
            R_pred = model(x_input, stellar_params)
        else:
            R_pred = model(x_input)
            
        calibrated_spec = x_input * R_pred
        loss_data = self.mse(calibrated_spec, y_target)

        # Return the same shape as other loss modules, with two placeholder terms.
        return loss_data

class DirectPhysicsLoss(nn.Module):
    """
    Physics-Informed Loss Function for Direct Pixel-Wise Response Estimation.

    Calculates a multi-objective loss composed of:
    1. Normalized Relative Flux Error: Scale-invariant error measurement relative to maximum spectrum flux.
    2. Response Smoothness Regularization: Penalizes discrete second-order derivatives (curvature)
       of predicted response curves to enforce physically smooth adjustments across adjacent wavelengths.

    Parameters
    - lambda_smooth (float, default=1e-3): Regularization coefficient balancing response smoothness against 
        reconstruction accuracy.
    """
    def __init__(self, lambda_smooth: float=1e-3):
        super(DirectPhysicsLoss, self).__init__()
        self.lambda_smooth = lambda_smooth

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor, 
        stellar_params: torch.Tensor=None
    ) -> torch.Tensor:
        """
        Compute combined physics loss, flux error, and smoothness penalties.

        Parameters
        - model (nn.Module): Spectral neural network model.
        - x_input (torch.Tensor): Input flux spectra tensor.
        - y_target (torch.Tensor): Target calibrated spectra tensor.
        - stellar_params (torch.Tensor, optional): Stellar parameter conditioning tensor.

        Returns
        - tuple (torch.Tensor, torch.Tensor, torch.Tensor)
            Total combined loss, relative flux loss component, and smoothness loss component.
        """
        if stellar_params is not None:
            R = model(x_input, stellar_params)
        else:
            R = model(x_input)
            
        calibrated_spec = x_input * R

        # 1. Compute scale-invariant normalized flux error
        y_max = torch.max(y_target, dim=1, keepdim=True)[0]
        epsilon = 0.01 * y_max # Stability constant preventing division by zero
        rel_error = (calibrated_spec - y_target) / (y_max + epsilon)
        loss_flux = torch.mean(rel_error ** 2)

        # 2. Compute second discrete derivative of response R along wavelength dimension
        dR = R[:, 1:] - R[:, :-1]
        ddR = dR[:, 1:] - dR[:, :-1]
        loss_smooth = torch.mean(ddR ** 2)

        total_loss = loss_flux + (self.lambda_smooth * loss_smooth)
        return total_loss, loss_flux, loss_smooth


# =================================================================
# 2. ORCHESTRATED ABLATION LOOP
# =================================================================

def run_ablation_study(
    x_path: str = "data/processed/nn/all/X.npy",
    y_path: str = "data/processed/nn/all/Y.npy",
    params_path: str = "data/processed/nn/all",
    random_state: int = 42,
    epochs_adam: int = 500,
    steps_lbfgs: int = 50
):
    """
    Execute the systematic ablation study across four spectral calibration model configurations.

    Parameters
    - x_path (str): Path to the raw input spectra NumPy file (.npy).
    - y_path (str): Path to the target calibrated spectra NumPy file (.npy).
    - params_path (str): Directory or prefix path containing stellar parameters.
    - random_state (int, default=42): Seed for reproducibility across dataset splitting, 
        PCA construction, and weight initialization.
    - epochs_adam (int, default=500): Number of Adam optimization epochs in stage 1 training.
    - steps_lbfgs (int, default=50): Number of L-BFGS optimization steps in stage 2 fine-tuning.
    """
    # Fix seed across random number generators for exact reproducibility
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[>] Starting ablation study on device: {device}")

    # Load spectral datasets and physical stellar parameters from disk
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    params_np = load_stellar_parameters(params_path)

    # Use a fixed K-Fold cross-validation split (fold 0 used for evaluation)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(kf.split(X_np))
    
    # Convert NumPy arrays to torch tensors once, then reuse them for training and validation.
    X_train, X_test = torch.tensor(X_np[train_idx]).float(), torch.tensor(X_np[test_idx]).float()
    Y_train, Y_test = torch.tensor(Y_np[train_idx]).float(), torch.tensor(Y_np[test_idx]).float()
    params_train, params_test = torch.tensor(params_np[train_idx]).float(), torch.tensor(params_np[test_idx]).float()

    # Create two separate dataloaders so the training routine can be used with
    # both parameter-conditioned and non-parameter-conditioned models.
    # One dataloader includes stellar parameters as input features, while the
    # other only provides the spectral data.
    dataset_params = TensorDataset(X_train, Y_train, params_train)
    dataloader_params = DataLoader(dataset_params, batch_size=32, shuffle=True)
    
    dataset_no_params = TensorDataset(X_train, Y_train)
    dataloader_no_params = DataLoader(dataset_no_params, batch_size=32, shuffle=True)

    input_dim = X_train.shape[1]
    num_params = params_train.shape[1]

    # Estimate the spectral response ratio R = Y/X for the training set.
    # The small epsilon prevents division by zero on any near-zero input values.
    R_train_raw = Y_np[train_idx] / (X_np[train_idx] + 1e-15)
    R_train_clipped = np.clip(R_train_raw, 0.5, 2.0)
    # Apply Savitzky-Golay filtering to smooth ratio vectors before constructing PCA basis
    R_train_smooth = signal.savgol_filter(R_train_clipped, window_length=11, polyorder=2, axis=1, mode="nearest")

    # Fit PCA to the smooth response curves so PCA-based models can predict
    # coefficient vectors instead of directly predicting the full response.
    pca = PCA(n_components=5, random_state=random_state)
    pca.fit(R_train_smooth)
    pca_basis_np, pca_mean_np = pca.components_, pca.mean_

    # Define the four progressive ablation experiments.
    experiments = [
        {"name": "Model A (Pure MSE - Basic Network)", "use_pca": False, "use_physics": False, "use_params": True},
        {"name": "Model B (PINN without PCA - Pixel-wise)", "use_pca": False, "use_physics": True, "use_params": True},
        {"name": "Model C (PCA + PINN - Spectrum only)", "use_pca": True, "use_physics": True, "use_params": False},
        {"name": "Model D (Full PINN - Spectrum + Params)", "use_pca": True, "use_physics": True, "use_params": True}
    ]

    results = []

    print("\n" + "="*70)
    for config in experiments:
        print(f"\n>>> Training {config['name']}")
        
        # 1. Select model architecture and loss function for this experiment.
        #    - PCA models predict coefficients over a shared basis.
        #    - direct models predict the per-pixel response curve.
        #    - physics-informed losses add smoothness and normalized flux constraints.
        if config["use_pca"] and config["use_params"]:
            model = PCAParamsSpectralPINN(input_dim, pca_basis_np, pca_mean_np, num_params).to(device)
            criterion = PCAParamsPhysicsInformedLoss(pca_basis_np, pca_mean_np).to(device) if config["use_physics"] else DirectMSELoss().to(device)
            
        elif config["use_pca"] and not config["use_params"]:
            model = PCASpectralPINN(input_dim, pca_basis_np, pca_mean_np).to(device)
            criterion = PCAPhysicsInformedLoss(pca_basis_np, pca_mean_np).to(device) if config["use_physics"] else DirectMSELoss().to(device)
            
        elif not config["use_pca"]:
            model = DirectPixelPINN(input_dim, num_params if config["use_params"] else 0).to(device)
            criterion = DirectPhysicsLoss().to(device) if config["use_physics"] else DirectMSELoss().to(device)

        # 2. Run the corresponding training strategy.
        #    The parameter-conditioned and non-parameter-conditioned flows use
        #    separate dataloaders but the same two-stage training API.
        if config["use_params"]:
            model = train_2stage_params(
                model=model,
                dataloader=dataloader_params,
                X_full=X_train.to(device),
                Y_full=Y_train.to(device),
                params_full=params_train.to(device),
                physics_criterion=criterion,
                epochs_adam=epochs_adam,
                steps_lbfgs=steps_lbfgs
            )
        else:
            model = train_2stage_no_params(
                model=model,
                dataloader=dataloader_no_params,
                X_full=X_train.to(device),
                Y_full=Y_train.to(device),
                physics_criterion=criterion,
                epochs_adam=epochs_adam,
                steps_lbfgs=steps_lbfgs
            )

        # 3. Model evaluation on held-out test split
        model.eval()
        with torch.no_grad():
            x_test_dev, y_test_dev = X_test.to(device), Y_test.to(device)
            
            if config["use_params"]:
                p_test_dev = params_test.to(device)
                
            if config["use_pca"]:
                pca_coeffs = model(x_test_dev, p_test_dev) if config["use_params"] else model(x_test_dev)
                delta_R = torch.matmul(pca_coeffs, model.pca_basis)
                R_pred = torch.clamp(model.pca_mean + delta_R, min=0.3, max=3.0)
            else:
                R_pred = model(x_test_dev, p_test_dev) if config["use_params"] else model(x_test_dev)
                
            x_cal = x_test_dev * R_pred
            
            # Compute normalized MAE as percentage error relative to the local
            # peak flux of each spectrum, which helps compare spectra of varying brightness.
            y_max = torch.max(y_test_dev, dim=1, keepdim=True)[0]
            mae_normalizado = torch.mean(torch.abs(x_cal - y_test_dev) / y_max).item() * 100
            
        print(f"[*] Test result ({config['name']}): {mae_normalizado:.4f}% normalized MAE")
        results.append((config['name'], mae_normalizado))

    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    for name, res in results:
        print(f"{name:<50} | Normalized MAE: {res:.4f}%")

if __name__ == "__main__":
    mode = "all" # Dataset selection options: "all", "blue", "solar"

    if mode in ["all", "blue", "solar"]:
        run_ablation_study(x_path=f"data/processed/nn/{mode}/X.npy",
            y_path=f"data/processed/nn/{mode}/Y.npy",
            params_path=f"data/processed/nn/{mode}",
            random_state=42,
            epochs_adam=500,
        steps_lbfgs=50
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'all', 'blue', or 'solar'.")