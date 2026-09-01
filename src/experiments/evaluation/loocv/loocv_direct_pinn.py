"""
Module: loocv_direct_pinn.py
Report Section: 5.0 Machine Learning Calibration Framework

Leave-One-Out Cross-Validation (LOOCV) Pipeline for Direct PINN.

This module executes leave-one-out cross-validation for a direct Physics-Informed 
Neural Network (PINN) architecture. The model learns to predict the absolute continuous 
spectral response function R(lambda) mapping low-resolution Gaia XP spectra directly to 
degraded CALSPEC physical reference targets without relying on external baseline ratios.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.extra.seed import set_seed
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import generate_plots_from_file


# =================================================================
# 1. DIRECT CALIBRATION ARCHITECTURE
# =================================================================
class DirectConvPINN(nn.Module):
    """
    1D Convolutional Physics-Informed Neural Network for absolute spectral calibration.

    Predicts log-space PCA coefficients representing the continuous calibration ratio 
    R(lambda) such that Y_calspec = X_gaia * R(lambda).

    Parameters
    ----------
    input_dim : int
        Number of spectral flux channels in input spectra.
    pca_basis : numpy.ndarray
        Orthonormal PCA basis components fitted on train-set log calibration ratios.
    pca_mean : numpy.ndarray
        Mean logarithmic calibration ratio vector from PCA fitting.
    num_params : int, default=3
        Number of auxiliary stellar physical parameters (e.g., BP-RP, G mag, RUWE).
    hidden_dim : int, default=128
        Dimension of dense hidden layers in the MLP network.
    """

    def __init__(
        self, 
        input_dim: int, 
        pca_basis: np.ndarray, 
        pca_mean: np.ndarray, 
        num_params: int = 3, 
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Register PCA basis and mean vectors as non-trainable buffers
        self.register_buffer("pca_basis", torch.tensor(pca_basis, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.tensor(pca_mean, dtype=torch.float32))
        num_components = self.pca_basis.shape[0]

        # 1D Convolutional Feature Extractor to extract local spectral morphology
        self.conv_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16)  # Retain 16 spatial bins to capture broad continuum shape
        )

        total_features = (32 * 16) + num_params

        # MLP regression head predicting PCA component coefficients
        self.mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_components)
        )

        # Zero-initialize output layer to default to global average calibration initially
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass predicting PCA log-coefficient weights.

        Parameters
        ----------
        x : torch.Tensor
            Input flux tensor of shape (batch_size, input_dim).
        params : torch.Tensor, optional
            Auxiliary physical stellar parameters tensor of shape (batch_size, num_params).

        Returns
        -------
        torch.Tensor
            Predicted PCA basis coefficients of shape (batch_size, num_components).
        """
        # Scale normalization with floor to preserve small flux levels (~10^-16 W/m^2/nm)
        scale = torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-20
        x_norm = (x / scale).unsqueeze(1) 
        
        conv_feats = self.conv_feature_extractor(x_norm).flatten(1)

        if params is not None:
            combined = torch.cat([conv_feats, params], dim=1)
        else:
            combined = conv_feats

        return self.mlp(combined)

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the continuous absolute calibration ratio R(lambda) in exponential space.

        Parameters
        ----------
        coeffs : torch.Tensor
            Predicted PCA basis coefficients tensor of shape (batch_size, num_components).

        Returns
        -------
        torch.Tensor
            Reconstructed positive calibration ratio R(lambda) of shape (batch_size, input_dim).
        """
        raw_R = self.pca_mean + torch.matmul(coeffs, self.pca_basis)
        return torch.exp(raw_R)


# =================================================================
# 2. DIRECT PHYSICS LOSS
# =================================================================
class DirectPhysicsLoss(nn.Module):
    """
    Physics-informed loss function evaluating relative flux accuracy and curve smoothness.

    Parameters
    ----------
    lambda_smooth : float, default=1e-4
        Regularization weight for second-derivative smoothness penalty on R(lambda).
    lambda_reg : float, default=1e-4
        L2 regularization weight applied to predicted PCA coefficients.
    """

    def __init__(self, lambda_smooth: float = 1e-4, lambda_reg: float = 1e-4) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg

    def forward(
        self, 
        model: DirectConvPINN, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor, 
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total objective loss combining relative flux error and smoothness constraints.

        Parameters
        ----------
        model : DirectConvPINN
            The Direct PINN neural network model.
        x_input : torch.Tensor
            Input Gaia XP spectra tensor.
        y_target : torch.Tensor
            Target ground-truth CALSPEC spectra tensor.
        params : torch.Tensor, optional
            Auxiliary stellar parameter features.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        pca_coeffs = model(x_input, params)
        R_pred = model.reconstruct(pca_coeffs)

        # Predict target spectrum: Y_pred = X_input * R_pred
        y_pred = x_input * R_pred
        
        # Relative Smooth L1 Loss with numerical floor suited for physical flux ranges
        scale = torch.clamp(torch.abs(y_target), min=1e-20)
        rel_err = (y_pred - y_target) / scale
        loss_flux = F.smooth_l1_loss(rel_err, torch.zeros_like(rel_err), beta=0.01)

        # Soft smoothness penalty (discrete 2nd derivative) to avoid high-frequency oscillations
        d2_R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        loss_smooth = torch.mean(d2_R ** 2)

        # L2 penalty on coefficient magnitudes to avoid spurious extrapolation
        loss_reg = torch.mean(pca_coeffs ** 2)

        return loss_flux + (self.lambda_smooth * loss_smooth) + (self.lambda_reg * loss_reg)


# =================================================================
# 3. TRAINING LOOP
# =================================================================
def train_fold(
    model: nn.Module, 
    criterion: nn.Module, 
    X_train: torch.Tensor, 
    Y_train: torch.Tensor, 
    P_train: torch.Tensor, 
    E_train: torch.Tensor,
    epochs_adam: int
) -> nn.Module:
    """
    Train a Direct PINN model instance on a single cross-validation fold.

    Parameters
    ----------
    model : nn.Module
        Initialized PINN model instance.
    criterion : nn.Module
        Physics loss function module.
    X_train : torch.Tensor
        Training set input spectra.
    Y_train : torch.Tensor
        Training set ground truth target spectra.
    P_train : torch.Tensor
        Training set stellar parameters.
    E_train : torch.Tensor
        Observational error/uncertainty matrix for input data augmentation.
    epochs_adam : int
        Number of optimization epochs.

    Returns
    -------
    nn.Module
        Trained network model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam)
    model.train()

    for _ in range(epochs_adam):
        optimizer.zero_grad()
        # Data augmentation: inject scaled Gaussian observational uncertainties
        noise = torch.randn_like(X_train) * E_train * 0.2
        # Floor value at 1e-20 to preserve physical signal levels (~10^-15 W/m^2/nm)
        X_aug = torch.clamp(X_train + noise, min=1e-20)

        loss = criterion(model, X_aug, Y_train, P_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return model


# =================================================================
# 4. STELLAR PARAMETERS LOADER
# =================================================================
def load_stellar_parameters(data_dir: Union[str, Path]) -> np.ndarray:
    """
    Load and standard-scale auxiliary stellar parameters from binary files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing `bp_rp.npy`, `G_mags.npy`, and `ruwe.npy`.

    Returns
    -------
    numpy.ndarray
        Column-stacked float32 array of shape (n_samples, 3) containing standardized features.
    """
    data_path = Path(data_dir).resolve()
    bp_rp = np.load(data_path / "bp_rp.npy").reshape(-1, 1)
    g_mag = np.load(data_path / "G_mags.npy").reshape(-1, 1)
    ruwe = np.load(data_path / "ruwe.npy").reshape(-1, 1)

    bp_rp_std = (bp_rp - np.mean(bp_rp)) / (np.std(bp_rp) + 1e-8)
    g_mag_std = (g_mag - np.mean(g_mag)) / (np.std(g_mag) + 1e-8)
    ruwe_std = (ruwe - np.mean(ruwe)) / (np.std(ruwe) + 1e-8)

    return np.hstack([bp_rp_std, g_mag_std, ruwe_std]).astype(np.float32)


# =================================================================
# 5. LOOCV PIPELINE
# =================================================================
def run_loocv(
    x_path: Union[str, Path], 
    y_path: Union[str, Path], 
    x_err_path: Union[str, Path], 
    params_path: Union[str, Path],
    output_model_dir: Union[str, Path], 
    random_state: int = 42,
    num_components: int = 12, 
    lambda_smooth: float = 1e-4, 
    lambda_reg: float = 1e-4, 
    epochs_adam: int = 400
) -> None:
    """
    Execute Leave-One-Out Cross-Validation for Direct PINN model.

    Parameters
    ----------
    x_path : str or Path
        Path to input spectra NumPy file (X.npy).
    y_path : str or Path
        Path to target spectra NumPy file (Y.npy).
    x_err_path : str or Path
        Path to flux errors NumPy file (X_error.npy).
    params_path : str or Path
        Directory containing stellar parameter NumPy arrays.
    output_model_dir : str or Path
        Target directory to output evaluation metrics, plots, and array binaries.
    random_state : int, default=42
        Random seed for reproducibility.
    num_components : int, default=12
        Number of PCA basis components used for spectral reconstruction.
    lambda_smooth : float, default=1e-4
        Smoothness loss hyperparameter weight.
    lambda_reg : float, default=1e-4
        L2 regularization hyperparameter weight.
    epochs_adam : int, default=400
        Number of training epochs per CV fold.

    Returns
    -------
    None
    """
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_model_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X_np, Y_np = np.load(x_path), np.load(y_path)
    X_err_np = np.load(x_err_path)
    params_np = load_stellar_parameters(params_path)

    n_samples, num_channels = X_np.shape
    num_stellar_params = params_np.shape[1]
    print(f"[>] Initiating Direct LOOCV on {n_samples} samples. Device: {device}")

    Y_pred_all_np = np.zeros_like(X_np)
    loo = LeaveOneOut()

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_np)):
        val_i = test_idx[0]
        print(f"[{val_i+1}/{n_samples}] Training fold (inferring on star index {val_i})...", end="\r")

        X_train, Y_train = X_np[train_idx], Y_np[train_idx]
        P_train, E_train = params_np[train_idx], X_err_np[train_idx]
        X_test, P_test = X_np[test_idx], params_np[test_idx]

        # -------------------------------------------------------------
        # ABSOLUTE CALIBRATION PCA (Log-space mapping)
        # -------------------------------------------------------------
        R_train_raw = Y_train / (X_train + 1e-20)
        R_train_clipped = np.clip(R_train_raw, a_min=1e-5, a_max=np.percentile(R_train_raw, 99))
        
        # Fit PCA decomposition on natural logarithm of the flux calibration ratio
        R_train_log = np.log(R_train_clipped)

        pca = PCA(n_components=num_components, random_state=random_state)
        pca.fit(R_train_log)
        pca_basis_np, pca_mean_np = pca.components_, pca.mean_
        
        # Transfer fold data to compute device
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
        P_train_t = torch.tensor(P_train, dtype=torch.float32).to(device)
        E_train_t = torch.tensor(E_train, dtype=torch.float32).to(device)

        model = DirectConvPINN(
            input_dim=num_channels, 
            pca_basis=pca_basis_np, 
            pca_mean=pca_mean_np,
            num_params=num_stellar_params
        ).to(device)

        criterion = DirectPhysicsLoss(
            lambda_smooth=lambda_smooth, 
            lambda_reg=lambda_reg
        ).to(device)

        trained_model = train_fold(
            model=model, 
            criterion=criterion,
            X_train=X_train_t, 
            Y_train=Y_train_t, 
            P_train=P_train_t, 
            E_train=E_train_t, 
            epochs_adam=epochs_adam
        )

        trained_model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            P_val_t = torch.tensor(P_test, dtype=torch.float32).to(device)

            coeffs = trained_model(X_val_t, P_val_t)
            R_pred = trained_model.reconstruct(coeffs)
            Y_pred = X_val_t * R_pred
            
            Y_pred_all_np[val_i] = Y_pred.cpu().numpy()

    print("\n\n[>] Direct LOOCV Completed. Exporting metrics...")

    # Wrapper class matching standard evaluation pipeline interfaces
    class DirectExportModel(nn.Module):
        """
        Export wrapper returning precomputed aggregated LOOCV predictions.
        """

        def __init__(self, target_device: torch.device, Y_pred_ref: np.ndarray) -> None:
            super().__init__()
            self.target_device = target_device
            self.Y_pred_ref = torch.tensor(Y_pred_ref, dtype=torch.float32, device=target_device)
            self.dummy_param = nn.Parameter(torch.empty(0, device=target_device))

        def forward(
            self, 
            x: torch.Tensor, 
            params: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            return self.Y_pred_ref

        def reconstruct(self, y_pred: torch.Tensor) -> torch.Tensor:
            return y_pred

    X_all_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    Y_all_t = torch.tensor(Y_np, dtype=torch.float32).to(device)
    E_all_t = torch.tensor(X_err_np, dtype=torch.float32).to(device)
    P_all_t = torch.tensor(params_np, dtype=torch.float32).to(device)

    results_filepath = str(out_dir / "loocv_aggregated_predictions.npz")

    export_test_predictions(
        model=DirectExportModel(device, Y_pred_all_np),
        X_tensor=X_all_t,
        Y_tensor=Y_all_t,
        X_err_tensor=E_all_t,
        output_filepath=results_filepath,
        params_tensor=P_all_t,
        X_huang_tensor=None  # Explicitly set to None as Huang baseline is not used
    )

    evaluate_test_set_from_file(
        input_filepath=results_filepath, 
        output_dir=str(out_dir), 
        num_knots=num_components
    )

    vis_dir = str(out_dir / "comparison_plots")
    generate_plots_from_file(
        input_filepath=results_filepath, 
        output_dir=vis_dir, 
        num_samples=min(18, n_samples)
    )

    print(f"\n[✓] Finished. Metrics and plots saved to: {out_dir}")


if __name__ == "__main__":
    mode = "all"
    OUTPUT_MODEL_DIR = f"results/models/loocv_direct_pinn/{mode}"

    run_loocv(
        x_path=f"data/processed/nn/{mode}/X.npy",
        y_path=f"data/processed/nn/{mode}/Y.npy",
        x_err_path=f"data/processed/nn/{mode}/X_error.npy",
        params_path=f"data/processed/nn/{mode}",
        output_model_dir=OUTPUT_MODEL_DIR,
        random_state=42,
        num_components=12,   # Increased capacity to 12 PCA components for model flexibility
        lambda_smooth=1e-4,  # Lower smoothness weight allowing network to fit sharp spectral variations
        lambda_reg=1e-4,
        epochs_adam=400
    )