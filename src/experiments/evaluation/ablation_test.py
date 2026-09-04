r"""
Module: ablation_test.py
Report Section:

Ablation Study Module for Neural Spectral Calibration
=====================================================

This module provides a systematic ablation study framework designed to isolate,
quantify, and evaluate the individual and synergistic contributions of three core
architectural innovations in neural astronomical spectral calibration:

1. **PCA-Based Dimensionality Reduction**: Projecting high-dimensional response curves 
   onto a low-dimensional principal component basis versus direct pixel-wise estimation.
2. **Physics-Informed Regularization (PINN)**: Constraining spectral response 
   smoothness and enforcing scale-invariant flux conservation versus pure empirical MSE.
3. **Stellar Parameter Conditioning**: Conditioning neural predictions on auxiliary 
   physical parameters ($T_{\rm eff}$, $\log g$, $[\text{Fe/H}]$) versus pure spectrum-driven inference.

Evaluated Model Variants
------------------------
- **Model A (Pure MSE - Basic Network)**: Direct pixel-wise baseline network trained 
  with pure MSE loss and conditioned on stellar parameters, without physics loss or PCA prior.
- **Model B (PINN without PCA - Pixel-wise)**: Direct pixel-wise network trained with 
  physics-informed loss (smoothness + flux constraints) and stellar parameters, without PCA.
- **Model C1 (PCASpectralPINN - No params arch)**: PCA-based PINN model using a dedicated 
  architecture without stellar parameter inputs (pure spectrum-to-response mapping).
- **Model C2 (PCAParamsSpectralPINN - Zero params)**: Parameter-conditioned PCA-PINN model 
  where stellar parameters are explicitly zeroed out during training and inference to evaluate 
  the structural impact of parameter layers in isolation.
- **Model D (PCAParamsSpectralPINN - With params)**: Full parameter-conditioned PCA-PINN model 
  combining PCA basis projection, physics-informed loss, and real stellar parameter conditioning.
- **Model E (Undegraded PCA PINN)**: Advanced zero-degradation PCA-PINN architecture 
  preventing information loss during parameter propagation.
- **Model F (Undegraded Chebyshev PINN)**: Advanced zero-degradation PINN using 
  analytical Chebyshev polynomial bases instead of data-driven PCA components.

Goal:
Measure the contribution of each design component to the overall normalized error (MAE %)
and evaluate the structural interactions between physics-informed losses and reduced-order priors.
"""

from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import scipy.signal as signal

from src.models.pinn_architecture import (
    PCAParamsSpectralPINN,
    PCASpectralPINN,
    UndegradedPCASpectralPINN,
    UndegradedChebyshevSpectralPINN
)
from src.train.loss.loss import (
    PCAParamsPhysicsInformedLoss, 
    PCAPhysicsInformedLoss,
    UndegradedPCAPhysicsInformedLoss,
    UndegradedChebyshevPhysicsInformedLoss
)
from src.train.training_functions.fixed_params_training_strategies import train_2stage_fixed as train_2stage_params
from src.train.training_functions.fixed_training_strategies import train_2stage_fixed as train_2stage_no_params
from src.utils.extra.seed import set_seed
from src.utils.metrics.alternative_metrics import (
    calculate_peak_normalized_mae,
    calculate_normalized_std_percentage,
    calculate_reduced_chi_square,
    calculate_normalized_bias_percentage,
    calculate_cosine_similarity,
    calculate_band_errors,
    calculate_band_bias
)
from src.experiments.evaluation.cross_validation import load_stellar_parameters


# =================================================================
# 1. ABLATION MODULES AND LOSS FUNCTIONS
# =================================================================

class DirectPixelPINN(nn.Module):
    r"""
    Direct Pixel-Wise Spectral Response Model.

    Predicts a multiplicative spectral correction curve $R(\lambda)$ at full 
    spectral resolution (pixel-by-pixel). The network receives uncalibrated flux 
    spectra and optional tabular stellar parameters.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input spectrum (number of wavelength pixels).
    num_stellar_params : int, default=0
        Number of auxiliary stellar physical parameters passed as additional features.
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
        r"""
        Forward pass computing pixel-wise multiplicative response curves.

        Parameters
        ----------
        x : torch.Tensor
            Input flux spectra tensor of shape `(batch_size, input_dim)`.
        stellar_params : torch.Tensor, optional
            Stellar parameters tensor of shape `(batch_size, num_stellar_params)`.

        Returns
        -------
        torch.Tensor
            Predicted multiplicative response curve $R$ bounded to $[0.3, 3.0]$.
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

    def prepare_for_lbfgs(self) -> None:
        """Switch network to evaluation mode for L-BFGS second-stage optimization."""
        self.eval()

class DirectMSELoss(nn.Module):
    r"""
    Pure Mean Squared Error (MSE) Loss without Physics Regularization.

    Computes standard pixel-wise MSE between calibrated flux spectra and target spectra.

    Methods
    -------
    forward(model, x_input, y_target, stellar_params=None)
        Compute unweighted MSE loss.
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
        r"""
        Compute standard MSE loss.

        Parameters
        ----------
        model : nn.Module
            Neural network architecture under evaluation.
        x_input : torch.Tensor
            Input flux spectra tensor.
        y_target : torch.Tensor
            Target calibrated flux spectra tensor.
        stellar_params : torch.Tensor, optional
            Auxiliary stellar parameter conditioning tensor.

        Returns
        -------
        torch.Tensor
            Scalar Mean Squared Error loss.
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
    r"""
    Physics-Informed Loss Function for Direct Pixel-Wise Models.

    Computes a multi-objective loss combining normalized relative flux error and 
    second-derivative discrete response smoothness regularization.

    Parameters
    ----------
    lambda_smooth : float, default=1e-3
        Regularization hyperparameter balancing response smoothness against reconstruction error.
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute total physics-informed loss along with individual loss components.

        Parameters
        ----------
        model : nn.Module
            Neural network architecture under evaluation.
        x_input : torch.Tensor
            Input flux spectra tensor.
        y_target : torch.Tensor
            Target calibrated flux spectra tensor.
        stellar_params : torch.Tensor, optional
            Auxiliary stellar parameter tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing `(total_loss, loss_flux, loss_smooth)`.
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
# 2. WRAPPER
# =================================================================

class BaselineMSELoss(nn.Module):
    r"""
    Wrapper for MSE Loss Matching PhysicsInformedLoss Signature.

    Returns a 3-element tuple `(total_loss, flux_loss, smooth_loss)` to prevent 
    unpacking exceptions during two-stage training optimization loops.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        y_true: torch.Tensor, 
        params: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute MSE loss and return a physics-compatible 3-tuple output.

        Parameters
        ----------
        model : nn.Module
            Neural network architecture under evaluation.
        x : torch.Tensor
            Input flux spectra tensor.
        y_true : torch.Tensor
            Target flux spectra tensor.
        params : torch.Tensor, optional
            Stellar parameter tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing `(loss, loss, zero_tensor)`.
        """
        # Forward pass (soporta modelos con y sin parámetros)
        if params is not None:
            out = model(x, params)
        else:
            out = model(x)
            
        # Reconstruction depeding on the arquitecture
        if hasattr(model, "reconstruct"):
            r_pred = model.reconstruct(out)
        elif hasattr(model, "pca_basis"):
            delta_r = torch.matmul(out, model.pca_basis)
            r_pred = torch.clamp(model.pca_mean + delta_r, min=0.3, max=3.0)
        else:
            r_pred = out
            
        # Calculate calibrate flux and pure MSE
        x_cal = x * r_pred
        loss = self.mse(x_cal, y_true)
        
        # Return 3 values: total_loss, loss_flux, loss_smooth
        zero_tensor = torch.tensor(0.0, device=loss.device)
        return loss, loss, zero_tensor

    
# =================================================================
# 3. METRICS IMPLEMENTATIONS
# =================================================================

def compute_spectral_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wavelengths: np.ndarray,
    y_err: np.ndarray = None,
    num_params: int = 20
) -> Dict[str, float]:
    r"""
    Compute a full suite of astrophysical evaluation metrics over a test dataset.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target flux array of shape `(N_samples, N_pixels)`.
    y_pred : np.ndarray
        Model-calibrated output flux array of shape `(N_samples, N_pixels)`.
    wavelengths : np.ndarray
        Wavelength grid array of shape `(N_pixels,)`.
    y_err : np.ndarray, optional
        Observational 1-sigma uncertainty array of shape `(N_samples, N_pixels)`.
        If None, a default uncertainty floor of $10^{-4}$ is assumed.
    num_params : int, default=20
        Number of free model parameters used for calculating effective degrees of
        freedom in reduced Chi-Square calculation.

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean evaluation metrics:
        - 'pn_mae': Peak-Normalized Mean Absolute Error (%)
        - 'nstd': Peak-Normalized Standard Deviation (%)
        - 'chi_sq': Reduced Chi-Square ($\chi^2_\nu$)
        - 'bias': Peak-Normalized Mean Bias Error (%)
        - 'cos_sim': Spectral Cosine Similarity
        - 'blue_mae': Relative MAE in Blue passband (%)
        - 'green_mae': Relative MAE in Green passband (%)
        - 'red_mae': Relative MAE in Red passband (%)
    """
    if y_err is None:
        y_err = np.full_like(y_true, 1e-4)

    num_stars = y_true.shape[0]

    pn_mae_list, std_list, chi_sq_list, bias_list, cos_sim_list = [], [], [], [], []
    band_mae_accum = {"blue": 0.0, "green": 0.0, "red": 0.0}

    for i in range(num_stars):
        pn_mae_list.append(calculate_peak_normalized_mae(y_true[i], y_pred[i]))
        std_list.append(calculate_normalized_std_percentage(y_true[i], y_pred[i]))
        chi_sq_list.append(calculate_reduced_chi_square(y_true[i], y_pred[i], y_err[i], n_params=num_params))
        bias_list.append(calculate_normalized_bias_percentage(y_true[i], y_pred[i]))
        cos_sim_list.append(calculate_cosine_similarity(y_true[i], y_pred[i]))

        b_mae = calculate_band_errors(wavelengths, y_true[i], y_pred[i])
        band_mae_accum["blue"] += b_mae["blue_rel_error_percent"]
        band_mae_accum["green"] += b_mae["green_rel_error_percent"]
        band_mae_accum["red"] += b_mae["red_rel_error_percent"]

    return {
        "pn_mae": float(np.mean(pn_mae_list)),
        "nstd": float(np.mean(std_list)),
        "chi_sq": float(np.mean(chi_sq_list)),
        "bias": float(np.mean(bias_list)),
        "cos_sim": float(np.mean(cos_sim_list)),
        "blue_mae": band_mae_accum["blue"] / num_stars,
        "green_mae": band_mae_accum["green"] / num_stars,
        "red_mae": band_mae_accum["red"] / num_stars,
    }


# =================================================================
# 4. ORCHESTRATED ABLATION LOOP
# =================================================================

def run_ablation_study(
    x_path: str = "data/processed/nn/all/X.npy",
    y_path: str = "data/processed/nn/all/Y.npy",
    params_path: str = "data/processed/nn/all",
    wave_path: str = "data/processed/nn/all/wave_grid.npy",
    random_state: int = 42,
    epochs_adam: int = 500,
    steps_lbfgs: int = 50
) -> None:
    r"""
    Execute the multi-metric ablation study across spectral calibration architectures.

    This study systematically compares baseline direct-pixel models, standard PCA PINNs,
    and zero-degradation architectures (Undegraded PCA and Chebyshev) across a complete
    suite of astrophysical evaluation metrics.

    Parameters
    ----------
    x_path : str, default="data/processed/nn/all/X.npy"
        Path to the raw input spectra NumPy file (.npy).
    y_path : str, default="data/processed/nn/all/Y.npy"
        Path to the target calibrated spectra NumPy file (.npy).
    params_path : str, default="data/processed/nn/all"
        Directory or prefix path containing stellar parameters.
    wave_path : str, default="data/processed/nn/all/wave.npy"
        Path to wavelength grid NumPy file (.npy).
    random_state : int, default=42
        Seed for reproducibility across dataset splitting, PCA construction, 
        and weight initialization.
    epochs_adam : int, default=500
        Number of Adam optimization epochs in stage 1 training.
    steps_lbfgs : int, default=50
        Number of L-BFGS optimization steps in stage 2 fine-tuning.

    Returns
    -------
    None
        The function prints the ablation study results to the standard output.
    """
    # Fix seed across random number generators for exact reproducibility
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[>] Starting ablation study on device: {device}")

    # Load spectral datasets and physical stellar parameters from disk
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    params_np = load_stellar_parameters(params_path)
    wavelengths = np.load(wave_path)

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

    # Zero dataset for no params PCAParams model
    params_train_zero = torch.zeros_like(params_train)
    dataset_zero_params = TensorDataset(X_train, Y_train, params_train_zero)
    dataloader_zero_params = DataLoader(dataset_zero_params, batch_size=32, shuffle=True)

    input_dim = X_train.shape[1]
    num_params = params_train.shape[1]

    # Fit PCA to the smooth response curves for PCA-based models
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
        {"name": "Model A (Pure MSE - Basic Network)", "type": "baseline", "use_physics": False, "use_params": True},
        {"name": "Model B (PINN without PCA - Pixel-wise)", "type": "baseline", "use_physics": True, "use_params": True},
        {"name": "Model C1 (PCASpectralPINN - No params arch)", "type": "pca_standard", "use_physics": True, "use_params": False},
        {"name": "Model C2 (PCAParamsSpectralPINN - Empty params)", "type": "pca_params_empty", "use_physics": True, "use_params": False},
        {"name": "Model D (PCAParamsSpectralPINN - With params)", "type": "pca_standard", "use_physics": True, "use_params": True},
        {"name": "Model E (Undegraded PCA PINN)", "type": "pca_undegraded", "use_physics": True, "use_params": True},
        {"name": "Model F (Undegraded Chebyshev PINN)", "type": "cheb_undegraded", "use_physics": True, "use_params": True}
    ]

    results = []

    print("\n" + "="*70)
    for config in experiments:
        print(f"\n>>> Training {config['name']}")
        
        # 1. Select model architecture and loss function for this experiment.
        #    - PCA models predict coefficients over a shared basis.
        #    - direct models predict the per-pixel response curve.
        #    - physics-informed losses add smoothness and normalized flux constraints.
        if config["type"] == "pca_standard" and config["use_params"]:
            # Model D: Standard PCA PINN with full stellar parameter conditioning
            model = PCAParamsSpectralPINN(input_dim, pca_basis_np, pca_mean_np, num_params).to(device)
            criterion = PCAParamsPhysicsInformedLoss(pca_basis_np, pca_mean_np).to(device) if config["use_physics"] else BaselineMSELoss().to(device)
            
        elif config["type"] == "pca_standard" and not config["use_params"]:
            # Model C1: Standard PCA PINN without parameter conditioning architecture
            model = PCASpectralPINN(input_dim, pca_basis_np, pca_mean_np).to(device)
            criterion = PCAPhysicsInformedLoss(pca_basis_np, pca_mean_np).to(device) if config["use_physics"] else BaselineMSELoss().to(device)

        elif config["type"] == "pca_params_empty":
            # Model C2: Parameter-conditioned PCA architecture trained with zeroed/empty stellar parameters
            model = PCAParamsSpectralPINN(input_dim, pca_basis_np, pca_mean_np, num_params).to(device)
            criterion = PCAParamsPhysicsInformedLoss(pca_basis_np, pca_mean_np).to(device) if config["use_physics"] else BaselineMSELoss().to(device)

        elif config["type"] == "baseline":
            # Model A & Model B: Direct pixel-wise architectures (Pure MSE or PINN)
            model = DirectPixelPINN(input_dim, num_params if config["use_params"] else 0).to(device)
            criterion = DirectPhysicsLoss().to(device) if config["use_physics"] else BaselineMSELoss().to(device)
            
        elif config["type"] == "pca_undegraded":
            # Model E: Undegraded PCA PINN architecture
            model = UndegradedPCASpectralPINN(input_dim, pca_basis_np, pca_mean_np, num_params).to(device)
            criterion = UndegradedPCAPhysicsInformedLoss().to(device)
            
        elif config["type"] == "cheb_undegraded":
            # Model F: Undegraded Chebyshev PINN architecture
            model = UndegradedChebyshevSpectralPINN(input_dim, degree=10, num_params=num_params).to(device)
            criterion = UndegradedChebyshevPhysicsInformedLoss().to(device)

        # 2. Run the corresponding training strategy.
        #    The parameter-conditioned and non-parameter-conditioned flows use
        #    separate dataloaders but the same two-stage training API.
        if config["type"] == "pca_params_empty":
            # Train parameter-conditioned architecture using zeroed parameter tensors
            model, _ = train_2stage_params(
                model=model,
                dataloader=dataloader_zero_params,
                X_full=X_train.to(device),
                Y_full=Y_train.to(device),
                params_full=params_train_zero.to(device),
                physics_criterion=criterion,
                epochs_adam=epochs_adam,
                steps_lbfgs=steps_lbfgs
            )
        elif config["use_params"]:
            # Train parameter-conditioned architecture using real stellar parameters
            model, _ = train_2stage_params(
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
            # Train non-parameter architecture
            model = train_2stage_no_params(
                model=model,
                dataloader=dataloader_no_params,
                X_full=X_train.to(device),
                Y_full=Y_train.to(device),
                physics_criterion=criterion,
                epochs_adam=epochs_adam,
                steps_lbfgs=steps_lbfgs
            )

        # 3. Perform inference and evaluation on held-out test split
        model.eval()
        with torch.no_grad():
            x_test_dev, y_test_dev = X_test.to(device), Y_test.to(device)

            if config["type"] == "pca_params_empty":
                p_test_dev = torch.zeros_like(params_test).to(device)
            elif config["use_params"]:
                p_test_dev = params_test.to(device)
            else:
                p_test_dev = None

            # Dynamic reconstruction supporting new Undegraded models natively
            if hasattr(model, "reconstruct"):
                model_out = model(x_test_dev, p_test_dev) if p_test_dev is not None else model(x_test_dev)
                R_pred = model.reconstruct(model_out)
            elif config["type"] in ["pca_standard", "pca_params_empty"]:
                pca_coeffs = model(x_test_dev, p_test_dev) if p_test_dev is not None else model(x_test_dev)
                delta_R = torch.matmul(pca_coeffs, model.pca_basis)
                R_pred = torch.clamp(model.pca_mean + delta_R, min=0.3, max=3.0)
            else:
                R_pred = model(x_test_dev, p_test_dev) if p_test_dev is not None else model(x_test_dev)
            x_cal= x_test_dev * R_pred

            # Convert PyTorch tensors to NumPy arrays prior to metric calculation
            x_cal_np = x_cal.detach().cpu().numpy()
            y_test_np = Y_test.detach().cpu().numpy()
            
            # Compute full metric suite
            metrics = compute_spectral_metrics(
                y_true=y_test_np,
                y_pred=x_cal_np,
                wavelengths=wavelengths,
                num_params=20
            )

        print(f"[*] {config['name']} | PN-MAE: {metrics['pn_mae']:.6f}% | CosSim: {metrics['cos_sim']:.6f} | Chi2: {metrics['chi_sq']:.6f}")
        results.append((config['name'], metrics))

    # Print comparative summary results table
    print("\n" + "="*116)
    print(f"{'MULTIDIMENSIONAL ABLATION STUDY SUMMARY':^116}")
    print("="*116)
    header = f"{'Model':<47} | {'MAE (%)':>9} | {'STD (%)':>9} | {'Chi2_nu':>12} | {'Bias (%)':>10} | {'CosSim':>10}"
    print(header)
    print("-" * 116)
    for name, m in results:
        print(
            f"{name:<47} | "
            f"{m['pn_mae']:>9.4f} | "
            f"{m['nstd']:>9.4f} | "
            f"{m['chi_sq']:>12.4e} | "
            f"{m['bias']:>+10.4f} | "
            f"{m['cos_sim']:>10.6f}"
        )
    print("="*116)

if __name__ == "__main__":
    mode = "all" # Dataset selection options: "all", "blue", "solar"

    if mode in ["all", "blue", "solar"]:
        run_ablation_study(x_path=f"data/processed/nn/{mode}/X.npy",
            y_path=f"data/processed/nn/{mode}/Y.npy",
            params_path=f"data/processed/nn/{mode}",
            wave_path="data/processed/nn/all/wave_grid.npy",
            random_state=42,
            epochs_adam=1000,
            steps_lbfgs=100
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'all', 'blue', or 'solar'.")