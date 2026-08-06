"""
Module: pinn_architectures.py

Physics-Informed Neural Network (PINN) Architectures for Spectral Recalibration.

This module defines neural network architectures tailored for predicting continuous, 
smooth multiplicative correction profiles R(lambda) to recalibrate Gaia BP/RP optical 
spectra against high-precision CALSPEC reference standards.

Supported Architectures:
1. BiCubicSpectralPINN:
   Predicts correction values at sparse control knots and upsamples to full 
   spectral resolution via 2D bicubic interpolation to enforce smoothness.
2. PCASpectralPINN:
   Predicts expansion coefficients c_i for a pre-computed Principal Component Analysis 
   (PCA) basis set constructed from residual spectral features.
3. PCAParamsSpectralPINN:
   Extends PCASpectralPINN by incorporating stellar physical properties (e.g., G mag, 
   BP-RP color, RUWE) alongside spectral flux input.
4. GaussianSpectralPINN:
   Embeds an internal 1D Gaussian kernel smoothing layer in the forward path to 
   filter high-frequency noise prior to evaluating dense layer transformations.
"""

from typing import Dict, Optional, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# BASE CLASS DEFINITION
# =====================================================================

class BaseSpectralPINN(nn.Module):
    """
    Base class for all spectral recalibration PINN models.

    Establishes common structural contracts and utility methods across 
    derived spectral PINN architectures.
    """
    def prepare_for_lbfgs(self) -> None:
        """
        Prepare model state for L-BFGS second-order optimization.
        
        Sets model to evaluation mode ("self.eval()") to disable stochastic 
        dropout and freeze batch-normalization statistics during line-search steps.
        """
        self.eval()


# =====================================================================
# 1. BICUBIC CONTROL KNOT PINN ARCHITECTURE
# =====================================================================

class BiCubicSpectralPINN(BaseSpectralPINN):
    """
    Physics-Informed Neural Network based on Control Knots and Cubic Interpolation.

    Predicts a smooth $C^1$-continuous recalibration profile $R(\lambda)$. Instead of 
    directly parameterizing every wavelength bin, the network outputs values across a 
    sparse set of $K$ control knots. These knot values are upsampled to the full spectral 
    dimension using bicubic interpolation, structurally suppressing high-frequency noise.

    Parameters
    - input_dim (int): Total number of wavelength pixels in input spectra.
    - num_knots (int, default=20): Number of control knot parameters used to represent the correction curve.
    """
    def __init__(self, input_dim: int, num_knots: int = 20) -> None:
        super(BiCubicSpectralPINN, self).__init__()
        self.input_dim = input_dim
        self.num_knots = num_knots
        
        # Multilayer Perceptron mapping flux -> knot delta predictions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_knots)
        )

        # Zero-initialize output projection weights/biases to start at identity mapping R(lambda) = 1.0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass predicting the multiplicative factor R(lambda).

        Parameters
        - x (torch.Tensor): Uncalibrated spectra tensor of shape "(batch_size, input_dim)".

        Returns
        - torch.Tensor: Predicted multiplicative factor $R(\lambda)$ clamped within [0.3, 3.0], 
            shape "(batch_size, input_dim)".
        """
        # 1. Local normalization by peak spectrum flux for numerical stability
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        
        # 2. Predict correction values at control knots -> Shape: (batch_size, num_knots)
        delta_knots = self.net(x_norm)
        
        # 3. Reshape to 4D tensor for 2D spatial interpolation -> (batch_size, 1, 1, num_knots)
        delta_4d = delta_knots.unsqueeze(1).unsqueeze(2)
        
        # 4. Perform smooth bicubic interpolation along the wavelength axis
        delta_interp = F.interpolate(
            delta_4d, 
            size=(1, self.input_dim), 
            mode="bicubic", 
            align_corners=True
        )
        
        # 5. Restore 2D shape -> (batch_size, input_dim)
        delta_R = delta_interp.squeeze(2).squeeze(1)
        
        # 6. Form multiplicative correction R(lambda) = 1 + delta_R
        R = 1.0 + delta_R
        return torch.clamp(R, min=0.3, max=3.0)


# =====================================================================
# 2. PCA BASIS COMPONENT PINN ARCHITECTURE
# =====================================================================

class PCASpectralPINN(BaseSpectralPINN):
    """
    PCA Component Expansion Physics-Informed Neural Network.

    Predicts a vector of expansion coefficients $c_i$ for a pre-computed Principal 
    Component Analysis (PCA) basis constructed from residual calibration factors.

    Parameters
    - input_dim (int): Number of spectral wavelength bins.
    - pca_basis : Union[torch.Tensor, np.ndarray]
        Pre-computed PCA basis matrix of shape "(num_components, input_dim)".
    - pca_mean : Optional[Union[torch.Tensor, np.ndarray]], default=None
        Mean residual vector of shape "(input_dim,)". Defaults to ones if None.
    """
    def __init__(
        self, 
        input_dim: int, 
        pca_basis: Union[torch.Tensor, np.ndarray], 
        pca_mean: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> None:
        super(PCASpectralPINN, self).__init__()
        self.input_dim = input_dim
        
        # Register PCA basis matrix as frozen non-trainable buffer
        if not isinstance(pca_basis, torch.Tensor):
            pca_basis = torch.tensor(pca_basis, dtype=torch.float32)
        self.register_buffer("pca_basis", pca_basis.float())
        self.num_components = self.pca_basis.shape[0]

        # Register PCA mean correction buffer
        if pca_mean is None:
            pca_mean = torch.ones(input_dim, dtype=torch.float32)
        elif not isinstance(pca_mean, torch.Tensor):
            pca_mean = torch.tensor(pca_mean, dtype=torch.float32)
        self.register_buffer("pca_mean", pca_mean.float())
        
        # Output head predicting PCA expansion coefficients c_i
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_components)
        )

        # Zero-initialize final layer so initial predictions yield c_i = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting PCA expansion coefficients $c_i$.

        Parameters
        - x (torch.Tensor): Uncalibrated spectra tensor of shape "(batch_size, input_dim)".

        Returns
        - torch.Tensor: Predicted PCA coefficients $c_i$ of shape "(batch_size, num_components)".
        """
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        return self.net(x_norm)

    def reconstruct(self, pca_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full R(lambda) correction profiles from predicted PCA coefficients.

        Parameters
        - pca_coeffs (torch.Tensor): PCA coefficients tensor of shape "(batch_size, num_components)".

        Returns
        - torch.Tensor: Reconstructed $R(lambda)$ profiles clamped to [0.3, 3.0], 
            shape "(batch_size, input_dim)".
        """
        delta_R = torch.matmul(pca_coeffs, self.pca_basis)
        R = self.pca_mean + delta_R
        return torch.clamp(R, min=0.3, max=3.0)


# =====================================================================
# 3. PCA + STELLAR PARAMETER AUGMENTED PINN ARCHITECTURE
# =====================================================================

class PCAParamsSpectralPINN(BaseSpectralPINN):
    """
    PCA PINN Architecture Augmented with Physical Stellar Parameters.

    Extends "PCASpectralPINN" by concatenating physical stellar parameters 
    (e.g., G Magnitude, BP-RP Color index, RUWE) to normalized spectral input features.

    Parameters
    - input_dim (int): Number of spectral wavelength bins.
    - pca_basis : Union[torch.Tensor, np.ndarray]
        PCA basis matrix of shape "(num_components, input_dim)".
    - pca_mean : Optional[Union[torch.Tensor, np.ndarray]], default=None
        Mean residual profile vector of shape "(input_dim,)".
    - num_stellar_params (int, default=0): Number of physical stellar parameter features appended to inputs.
    """
    def __init__(
        self, 
        input_dim: int, 
        pca_basis: Union[torch.Tensor, np.ndarray], 
        pca_mean: Optional[Union[torch.Tensor, np.ndarray]] = None, 
        num_stellar_params: int = 0
    ) -> None:
        super(PCAParamsSpectralPINN, self).__init__()
        self.input_dim = input_dim
        self.num_stellar_params = num_stellar_params
        
        # Register PCA basis buffer
        if not isinstance(pca_basis, torch.Tensor):
            pca_basis = torch.tensor(pca_basis, dtype=torch.float32)
        self.register_buffer("pca_basis", pca_basis.float())
        self.num_components = self.pca_basis.shape[0]

        # Register PCA mean buffer
        if pca_mean is None:
            pca_mean = torch.ones(input_dim, dtype=torch.float32)
        elif not isinstance(pca_mean, torch.Tensor):
            pca_mean = torch.tensor(pca_mean, dtype=torch.float32)
        self.register_buffer("pca_mean", pca_mean.float())

        total_input_dim = input_dim + num_stellar_params

        self.net = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_components)
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, stellar_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass utilizing combined spectral flux and stellar parameter inputs.

        Parameters
        - x (torch.Tensor): Uncalibrated spectra tensor of shape "(batch_size, input_dim)".
        - stellar_params (torch.Tensor): Stellar properties tensor of shape "(batch_size, num_stellar_params)".

        Returns
        - torch.Tensor: Predicted PCA coefficients $c_i$ of shape "(batch_size, num_components)".
        """
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val

        # Concatenate normalized flux vector with physical parameter vector
        x_combined = torch.cat([x_norm, stellar_params], dim=1)
        return self.net(x_combined)

    def reconstruct(self, pca_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct $R(\lambda)$ profiles from predicted PCA coefficients.

        Parameters
        - pca_coeffs (torch.Tensor): PCA expansion coefficients of shape "(batch_size, num_components)".

        Returns
        - torch.Tensor: Reconstructed multiplicative factors $R(\lambda)$ of shape "(batch_size, input_dim)".
        """
        delta_R = torch.matmul(pca_coeffs, self.pca_basis)
        R = self.pca_mean + delta_R
        return torch.clamp(R, min=0.3, max=3.0)


# =====================================================================
# 4. GAUSSIAN SMOOTHED RESIDUAL PINN ARCHITECTURE
# =====================================================================

class GaussianSpectralPINN(BaseSpectralPINN):
    """
    Residual PINN Architecture featuring Embedded 1D Gaussian Filtering.

    Applies an internal fixed 1D Gaussian smoothing convolution operation to inputs 
    prior to feedforward transformation. This structural filter prevents overfitting 
    to localized absorption/emission lines and photon noise.

    Parameters
    - input_dim (int): Number of spectral wavelength bins.
    - kernel_size (int, default=7): Width of 1D Gaussian convolution kernel window (must be an odd integer).
    - sigma (float, default=1.5): Standard deviation governing Gaussian kernel smoothing width.
    """
    def __init__(self, input_dim: int, kernel_size: int = 7, sigma: float = 1.5) -> None:
        super(GaussianSpectralPINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, input_dim)
        )
        
        # Initialize output projection layer to zeros
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Construct normalized 1D Gaussian Kernel
        x_coord = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-(x_coord ** 2) / (2 * (sigma ** 2)))
        g = g / g.sum()
        
        # Register kernel as buffer tensor of shape (out_channels=1, in_channels=1, kernel_size)
        self.register_buffer("gaussian_kernel", g.view(1, 1, kernel_size))
        self.kernel_padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass applying Gaussian smoothing filter beforehand.

        Parameters
        x (torch.Tensor): Uncalibrated input spectra tensor of shape "(batch_size, input_dim)".

        Returns
        - torch.Tensor: Predicted multiplicative factor $R(lambda)$ clamped to [0.3, 3.0], 
            shape "(batch_size, input_dim)".
        """
        # 1. Local normalization
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        
        # 2. Apply 1D Gaussian convolution filter
        x_norm_3d = x_norm.unsqueeze(1) 
        x_smoothed = F.conv1d(x_norm_3d, self.gaussian_kernel, padding=self.kernel_padding)
        x_smoothed = x_smoothed.squeeze(1)

        # 3. Predict residual shift delta_R from smoothed input spectrum
        delta_R = self.net(x_smoothed)
        
        # 4. Form multiplicative factor R(lambda)
        R = 1.0 + delta_R
        return torch.clamp(R, min=0.3, max=3.0)


# =====================================================================
# FACTORY REGISTRY & INSTANTIATION ROUTINE
# =====================================================================

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "residual_bicubic": BiCubicSpectralPINN,
    "residual_gaussian": GaussianSpectralPINN,
    "pca_nn": PCASpectralPINN,
    "pca_params_nn": PCAParamsSpectralPINN
}


def get_pinn_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate PINN architecture instances by string key.

    Parameters
    - model_name (str): Registered architecture identifier (e.g., "pca_nn", "residual_bicubic").
    **kwargs
        Arbitrary keyword arguments forwarded to the selected class constructor.

    Returns
    - nn.Module: Initialized PyTorch PINN model instance.

    Raises
    - ValueError: If "model_name" is not recognized in "MODEL_REGISTRY".
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported architecture model key '{model_name}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)