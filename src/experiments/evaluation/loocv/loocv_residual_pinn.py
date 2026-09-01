"""
Module: loocv_residual_pinn.py
Report Section: 5.0 Machine Learning Calibration Framework

Leave-One-Out Cross-Validation (LOOCV) Pipeline for Residual Linear PINN.

This module implements leave-one-out cross-validation for a residual Physics-Informed 
Neural Network (PINN). The network models multiplicative residual corrections r(lambda) 
relative to normalized Gaia XP spectra, using inverse-variance heteroscedastic loss 
weighting, wavelength-aware blue-band penalties, and physical smoothness priors.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.signal as signal
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
# 1. RESIDUAL ARCHITECTURES
# =================================================================

class DenseResidualPINN(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Residual PINN architecture.

    Predicts a continuous multiplicative residual correction factor r(lambda) 
    from coarsened spectral features and physical stellar parameters.

    Parameters
    ----------
    input_dim : int
        Full output spectral dimension.
    num_params : int, default=2
        Number of auxiliary physical stellar parameters.
    num_coarse_bins : int, default=15
        Number of spatial bins used for downsampled input representation.
    max_correction : float, default=0.8
        Maximum allowable relative correction amplitude (bounded via tanh).
    hidden_dim : int, default=128
        Dimension of dense hidden layers.
    """

    def __init__(
        self, 
        input_dim: int, 
        num_params: int = 2, 
        num_coarse_bins: int = 15, 
        max_correction: float = 0.8, 
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.num_coarse_bins = num_coarse_bins
        self.max_correction = max_correction
        
        total_features = num_coarse_bins + num_params
        
        self.mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)  # Direct output mapping full spectral dimension
        )
        
        # Zero-initialize output layer: model defaults to identity correction (1.0) initially
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass estimating bounded multiplicative residual correction.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input spectrum tensor of shape (batch_size, input_dim).
        params : torch.Tensor
            Standardized stellar parameters tensor of shape (batch_size, num_params).

        Returns
        -------
        torch.Tensor
            Multiplicative residual array r(lambda) of shape (batch_size, input_dim).
        """
        # Downsample spectral vector to coarse representation using adaptive pooling
        x_in = x.unsqueeze(1)
        x_coarse = F.adaptive_avg_pool1d(x_in, self.num_coarse_bins).flatten(1)
        
        combined = torch.cat([x_coarse, params], dim=1)
        raw_delta = self.mlp(combined)
        
        # Enforce physical correction bounds (+/- max_correction)
        bounded_delta = self.max_correction * torch.tanh(raw_delta / self.max_correction)
        return 1.0 + bounded_delta


class Conv1DResidualPINN(nn.Module):
    """
    1D Convolutional Residual PINN architecture.

    Extracts localized spectral features via 1D convolutions combined with 
    stellar parameters to predict continuous residual corrections.

    Parameters
    ----------
    input_dim : int
        Full output spectral dimension.
    num_params : int, default=2
        Number of auxiliary physical stellar parameters.
    max_correction : float, default=0.8
        Maximum allowable relative correction amplitude.
    """

    def __init__(
        self, 
        input_dim: int, 
        num_params: int = 2, 
        max_correction: float = 0.8
    ) -> None:
        super().__init__()
        self.max_correction = max_correction
        
        # Localized 1D convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16)  # Output fixed spatial dimension regardless of input length
        )
        
        # Flattened convolutional features (32 * 16 = 512) concatenated with stellar parameters
        self.fc = nn.Sequential(
            nn.Linear(512 + num_params, 256),
            nn.GELU(),
            nn.Linear(256, input_dim)
        )
        
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass extracting local spectral features and predicting residual correction.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input spectrum tensor of shape (batch_size, input_dim).
        params : torch.Tensor
            Standardized stellar parameters tensor of shape (batch_size, num_params).

        Returns
        -------
        torch.Tensor
            Multiplicative residual array r(lambda) of shape (batch_size, input_dim).
        """
        x_in = x.unsqueeze(1)
        conv_features = self.conv_blocks(x_in).flatten(1)
        
        combined = torch.cat([conv_features, params], dim=1)
        raw_delta = self.fc(combined)
        
        bounded_delta = self.max_correction * torch.tanh(raw_delta / self.max_correction)
        return 1.0 + bounded_delta


# =================================================================
# 2. PHYSICS-INFORMED LOSS (WAVELENGTH-AWARE)
# =================================================================

class HeteroscedasticRatioLoss(nn.Module):
    """
    Inverse-variance heteroscedastic loss with blue-band weighting and physical priors.

    Applies error-based weighting to accommodate observational uncertainties, 
    incorporates an artificial loss multiplier for noisy short wavelengths (< 500 nm), 
    and adds discrete derivative smoothness and anchor regularization.

    Parameters
    ----------
    lambda_smooth : float, default=0.8
        Weight coefficient for second-derivative curve smoothness penalty.
    lambda_anchor : float, default=0.1
        Weight coefficient preventing global baseline drift.
    blue_penalty : float, default=5.0
        Loss weighting multiplier applied to channels below `blue_cutoff_idx`.
    blue_cutoff_idx : int, default=82
        Channel index corresponding to the blue wavelength boundary (~500 nm).
    """

    def __init__(
        self, 
        lambda_smooth: float = 0.8, 
        lambda_anchor: float = 0.1, 
        blue_penalty: float = 5.0, 
        blue_cutoff_idx: int = 82
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_anchor = lambda_anchor
        self.blue_penalty = blue_penalty
        self.blue_cutoff_idx = blue_cutoff_idx

    def forward(
        self, 
        r_pred: torch.Tensor, 
        r_true: torch.Tensor, 
        errors_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute composite weighted heteroscedastic residual loss.

        Parameters
        ----------
        r_pred : torch.Tensor
            Predicted multiplicative residual array.
        r_true : torch.Tensor
            Target ground-truth ratio array.
        errors_norm : torch.Tensor
            Normalized observational flux uncertainties.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # 1. Base weights: Inverse-variance weighting based on Gaia errors
        weights = 1.0 / (errors_norm**2 + 1e-8)
        
        # 2. Wavelength-Aware Mask: Boost loss importance in short-wavelength blue region
        mask = torch.ones_like(weights)
        mask[:, :self.blue_cutoff_idx] = self.blue_penalty
        weights = weights * mask
        
        # Normalize weights per sample to prevent gradient scaling distortion
        weights = weights / torch.mean(weights, dim=1, keepdim=True)
        data_loss = torch.mean(weights * (r_pred - r_true)**2)

        # 3. Smoothness Prior: Penalize sharp high-frequency jumps (2nd discrete derivative)
        d1 = r_pred[:, 1:] - r_pred[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        smooth_loss = torch.mean(d2**2)

        # 4. Anchor Prior: Prevent overall baseline drift away from unity
        anchor_loss = torch.mean((torch.mean(r_pred, dim=1) - 1.0)**2)

        return data_loss + self.lambda_smooth * smooth_loss + self.lambda_anchor * anchor_loss


# =================================================================
# 3. 2-STAGE TRAINING 
# =================================================================

def train_fold(
    model: nn.Module, 
    criterion: nn.Module, 
    X_norm_train: torch.Tensor, 
    Y_ratio_train: torch.Tensor, 
    Errors_norm_train: torch.Tensor, 
    P_train: torch.Tensor, 
    epochs_adam: int, 
    steps_lbfgs: int
) -> nn.Module:
    """
    Train model instance using a two-stage AdamW followed by L-BFGS optimization strategy.

    Parameters
    ----------
    model : nn.Module
        Initialized PyTorch model instance.
    criterion : nn.Module
        Physics-informed loss module.
    X_norm_train : torch.Tensor
        Normalized training spectra.
    Y_ratio_train : torch.Tensor
        Target calibration ratio vectors.
    Errors_norm_train : torch.Tensor
        Normalized uncertainties for heteroscedastic weighting.
    P_train : torch.Tensor
        Stellar parameter features.
    epochs_adam : int
        Number of AdamW gradient descent iterations.
    steps_lbfgs : int
        Maximum iterations for fine-tuning via L-BFGS.

    Returns
    -------
    nn.Module
        Optimized PyTorch model instance.
    """
    optimizer_adam = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    model.train()

    # Stage 1: AdamW global optimization
    for _ in range(epochs_adam):
        optimizer_adam.zero_grad()
        r_pred = model(X_norm_train, P_train)
        loss = criterion(r_pred, Y_ratio_train, Errors_norm_train)
        loss.backward()
        optimizer_adam.step()

    # Stage 2: L-BFGS second-order fine-tuning
    if steps_lbfgs > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=steps_lbfgs, history_size=10, line_search_fn="strong_wolfe"
        )
        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad()
            r_pred = model(X_norm_train, P_train)
            loss = criterion(r_pred, Y_ratio_train, Errors_norm_train)
            loss.backward()
            return loss
        optimizer_lbfgs.step(closure)

    return model


# =================================================================
# 4. STELLAR PARAMETERS LOADER
# =================================================================

def load_stellar_parameters(data_dir: Union[str, Path]) -> np.ndarray:
    """
    Load and standard-scale auxiliary stellar physical parameters from directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing parameter files (`bp_rp.npy`, `G_mags.npy`/`G_mag.npy`, `ruwe.npy`).

    Returns
    -------
    numpy.ndarray
        Standardized feature matrix of shape (n_samples, 3).

    Raises
    ------
    FileNotFoundError
        If any required stellar parameter file is missing.
    """
    data_path = Path(data_dir).resolve()
    bp_rp_path = data_path / "bp_rp.npy"
    g_mag_path = data_path / "G_mags.npy"
    if not g_mag_path.exists():
        g_mag_path = data_path / "G_mag.npy"
    ruwe_path = data_path / "ruwe.npy"

    for path in [bp_rp_path, g_mag_path, ruwe_path]:
        if not path.exists():
            raise FileNotFoundError(f"Stellar parameter file missing: '{path}'")

    bp_rp = np.load(bp_rp_path).reshape(-1, 1)
    g_mag = np.load(g_mag_path).reshape(-1, 1)
    ruwe = np.load(ruwe_path).reshape(-1, 1)

    bp_rp_std = (bp_rp - np.mean(bp_rp)) / (np.std(bp_rp) + 1e-8)
    g_mag_std = (g_mag - np.mean(g_mag)) / (np.std(g_mag) + 1e-8)
    ruwe_std = (ruwe - np.mean(ruwe)) / (np.std(ruwe) + 1e-8)

    return np.hstack([bp_rp_std, g_mag_std, ruwe_std]).astype(np.float32)


# =================================================================
# 5. LOOCV PIPELINE
# =================================================================

def run_loocv(
    data_dir: Union[str, Path], 
    output_model_dir: Union[str, Path], 
    random_state: int = 42,
    architecture: str = "dense", 
    num_coarse_bins: int = 15,
    lambda_smooth: float = 0.8, 
    lambda_anchor: float = 0.1,
    epochs_adam: int = 500, 
    steps_lbfgs: int = 20
) -> None:
    """
    Execute Leave-One-Out Cross-Validation (LOOCV) for Residual PINN model.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing input NumPy array datasets.
    output_model_dir : str or Path
        Target directory to output predictions, metrics, and visualization figures.
    random_state : int, default=42
        Seed value for reproducibility.
    architecture : str, default="dense"
        Network architecture choice ("dense" or "conv1d").
    num_coarse_bins : int, default=15
        Number of spatial bins for coarse spectral downsampling (used in dense architecture).
    lambda_smooth : float, default=0.8
        Weight multiplier for curve smoothness prior loss.
    lambda_anchor : float, default=0.1
        Weight multiplier for mean drift anchor loss.
    epochs_adam : int, default=500
        Number of initial AdamW optimization epochs per fold.
    steps_lbfgs : int, default=20
        Maximum L-BFGS fine-tuning steps per fold.

    Returns
    -------
    None
    """
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = Path(data_dir).resolve()
    out_dir = Path(output_model_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load absolute and normalized arrays
    X_np = np.load(data_path / "X.npy")
    Y_np = np.load(data_path / "Y.npy")
    X_err_np = np.load(data_path / "X_error.npy")
    
    X_norm_np = np.load(data_path / "X_norm.npy")
    Y_ratio_np = np.load(data_path / "Y_ratio.npy")
    Errors_norm_np = np.load(data_path / "Errors_norm.npy")
    
    params_np = load_stellar_parameters(data_path)

    n_samples, num_channels = X_norm_np.shape
    num_stellar_params = params_np.shape[1]
    print(f"[>] Initiating LOOCV on {n_samples} samples. Architecture: {architecture}. Device: {device}")

    R_pred_all_np = np.zeros_like(X_norm_np)
    R_huang_all_np = np.zeros_like(X_norm_np)
    loo = LeaveOneOut()

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_norm_np)):
        val_i = test_idx[0]
        print(f"[{val_i+1}/{n_samples}] Training fold (inferring on star index {val_i})...", end="\r")

        # -------------------------------------------------------------
        # BASELINE HUANG ET AL.
        # -------------------------------------------------------------
        X_train_abs, Y_train_abs = X_np[train_idx], Y_np[train_idx]
        R_train_raw = Y_train_abs / (X_train_abs + 1e-30)
        median_ratio = np.median(R_train_raw, axis=0)
        
        win_len = min(31, num_channels if num_channels % 2 != 0 else num_channels - 1)
        R_huang_all_np[val_i] = signal.savgol_filter(median_ratio, window_length=win_len, polyorder=3)

        # -------------------------------------------------------------
        # PINN TRAINING ON NORMALIZED SPACE
        # -------------------------------------------------------------
        X_norm_train = torch.tensor(X_norm_np[train_idx], dtype=torch.float32).to(device)
        Y_ratio_train = torch.tensor(Y_ratio_np[train_idx], dtype=torch.float32).to(device)
        Errors_norm_train = torch.tensor(Errors_norm_np[train_idx], dtype=torch.float32).to(device)
        P_train = torch.tensor(params_np[train_idx], dtype=torch.float32).to(device)
        
        X_norm_test = torch.tensor(X_norm_np[test_idx], dtype=torch.float32).to(device)
        P_test = torch.tensor(params_np[test_idx], dtype=torch.float32).to(device)

        if architecture.lower() == "dense":
            model = DenseResidualPINN(
                input_dim=num_channels, 
                num_params=num_stellar_params, 
                num_coarse_bins=num_coarse_bins
            ).to(device)
        else:
            model = Conv1DResidualPINN(
                input_dim=num_channels, 
                num_params=num_stellar_params
            ).to(device)

        criterion = HeteroscedasticRatioLoss(
            lambda_smooth=lambda_smooth, 
            lambda_anchor=lambda_anchor,
            blue_penalty=5.0,      # 5x penalty weight for short-wavelength blue region
            blue_cutoff_idx=82     # Cutoff channel corresponding to ~500 nm
        ).to(device)

        trained_model = train_fold(
            model=model, 
            criterion=criterion,
            X_norm_train=X_norm_train, 
            Y_ratio_train=Y_ratio_train, 
            Errors_norm_train=Errors_norm_train, 
            P_train=P_train,
            epochs_adam=epochs_adam, 
            steps_lbfgs=steps_lbfgs
        )

        trained_model.eval()
        with torch.no_grad():
            r_pred = trained_model(X_norm_test, P_test)
            R_pred_all_np[val_i] = r_pred.cpu().numpy()

    print("\n\n[>] LOOCV Completed. Evaluating global performance...")

    class RobustExportModel(nn.Module):
        """
        Export wrapper returning precomputed predictions based on nearest input vector match.
        """

        def __init__(
            self, 
            target_device: torch.device, 
            X_ref_norm: np.ndarray, 
            R_pred_ref: np.ndarray
        ) -> None:
            super().__init__()
            self.X_ref_norm = torch.tensor(X_ref_norm, dtype=torch.float32, device=target_device)
            self.R_pred_ref = torch.tensor(R_pred_ref, dtype=torch.float32, device=target_device)

            # Dummy parameter to avoid StopIteration during export inspection
            self.dummy_param = nn.Parameter(torch.empty(0, device=target_device))

        def forward(
            self, 
            x: torch.Tensor, 
            params: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            x_norm = x / (torch.max(x, dim=1, keepdim=True)[0] + 1e-30)
            dists = torch.cdist(x_norm, self.X_ref_norm, p=2)
            idx = torch.argmin(dists, dim=1)
            return self.R_pred_ref[idx]

    # Convert absolute arrays to tensors for evaluation and plotting
    X_all_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    Y_all_t = torch.tensor(Y_np, dtype=torch.float32).to(device)
    E_all_t = torch.tensor(X_err_np, dtype=torch.float32).to(device)
    P_all_t = torch.tensor(params_np, dtype=torch.float32).to(device)

    # Huang reconstruction
    y_pred_huang_np = X_np * R_huang_all_np
    X_huang_t = torch.tensor(y_pred_huang_np, dtype=torch.float32).to(device)

    run_specific_dir = out_dir / f"{architecture}_bins{num_coarse_bins}_smooth{lambda_smooth}_anchor{lambda_anchor}"
    run_specific_dir.mkdir(parents=True, exist_ok=True)
    
    results_filepath = str(run_specific_dir / "loocv_aggregated_predictions.npz")

    export_test_predictions(
        model=RobustExportModel(device, X_norm_np, R_pred_all_np), 
        X_tensor=X_all_t, 
        Y_tensor=Y_all_t,
        X_err_tensor=E_all_t, 
        output_filepath=results_filepath,
        params_tensor=P_all_t,
        X_huang_tensor=X_huang_t 
    )

    evaluate_test_set_from_file(
        input_filepath=results_filepath, 
        output_dir=str(run_specific_dir), 
        num_knots=0
    )

    vis_dir = str(run_specific_dir / "comparison_plots")
    generate_plots_from_file(
        input_filepath=results_filepath, 
        output_dir=vis_dir, 
        num_samples=min(18, n_samples)
    )

    print(f"\n[✓] Finished. Aggregated metrics and plots saved to: {out_dir}")


if __name__ == "__main__":
    mode = "all"
    DATA_DIR = f"data/processed/nn/{mode}"
    OUTPUT_MODEL_DIR = f"results/models/loocv_residual_pinn/{mode}"

    run_loocv(
        data_dir=DATA_DIR,
        output_model_dir=OUTPUT_MODEL_DIR,
        random_state=42,
        architecture="conv1d",   # Select "conv1d" or "dense" architecture
        num_coarse_bins=60, 
        lambda_smooth=0.05,  
        lambda_anchor=0.0,  
        epochs_adam=500,    
        steps_lbfgs=20      
    )