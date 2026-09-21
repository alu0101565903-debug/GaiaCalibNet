"""
Module: pinn_architecture.py
Report Section: 4.5 Neural Network Architectures

This module defines the 1D Convolutional Physics-Informed Neural Network architectures 
used for absolute spectral calibration and residual artifact correction.

Architectures:
    - DirectConvPINN: Predicts log-space PCA coefficients to reconstruct the continuous 
      spectral calibration ratio R(lambda) mapping Gaia BP/RP to CALSPEC.
    - BlueWiggleNet: Lightweight CNN predicting fine pixel-level multiplicative correction 
      ratios to mitigate high-frequency blue-region spectral wiggles.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn


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
        """
        Initialize DirectConvPINN architecture and register PCA buffers.

        Parameters
        ----------
        input_dim : int
            Number of spectral flux channels in input spectra.
        pca_basis : numpy.ndarray
            Orthonormal PCA basis components array.
        pca_mean : numpy.ndarray
            Mean logarithmic calibration ratio vector.
        num_params : int, default=3
            Number of auxiliary stellar physical parameters.
        hidden_dim : int, default=128
            Dimension of dense hidden layers in the MLP network.
        """
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
            nn.AdaptiveAvgPool1d(128)  # Retain 128 spatial bins to capture broad continuum shape
        )

        total_features = (128 * 32) + num_params

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


class BlueWiggleNet(nn.Module):
    """
    1D Convolutional network for high-frequency pixel-level spectral wiggle correction.

    Predicts localized, wavelength-dependent multiplicative correction ratios 
    in exponential space to resolve fine blue-region spectral artifacts.

    Parameters
    ----------
    input_dim : int
        Number of blue spectral channels in input sub-spectrum.
    num_params : int, default=3
        Number of auxiliary stellar physical parameters.
    hidden_dim : int, default=128
        Dimension of dense hidden layers in the MLP network.
    """

    def __init__(self, input_dim: int, num_params: int = 3, hidden_dim: int = 128) -> None:
        """
        Initialize BlueWiggleNet architecture.

        Parameters
        ----------
        input_dim : int
            Number of spectral channels.
        num_params : int, default=3
            Number of auxiliary physical parameters.
        hidden_dim : int, default=128
            Dimension of dense hidden layers.
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Feature extractor without pooling to preserve pixel-level resolution
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten()
        )
        
        total_features = (16 * input_dim) + num_params
        
        self.mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)  # Predicts direct pixel-wise correction
        )
        
        # Initialize weights and biases to zero so correction starts neutral (exp(0) = 1)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_blue: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass predicting fine-grained pixel-level correction ratios.

        Parameters
        ----------
        x_blue : torch.Tensor
            Input blue spectral flux tensor of shape (batch_size, input_dim).
        params : torch.Tensor, optional
            Auxiliary stellar physical parameters tensor of shape (batch_size, num_params).

        Returns
        -------
        torch.Tensor
            Pixel-level multiplicative correction factors of shape (batch_size, input_dim).
        """
        scale = torch.max(torch.abs(x_blue), dim=1, keepdim=True)[0] + 1e-20
        x_norm = (x_blue / scale).unsqueeze(1)
        
        conv_feats = self.conv(x_norm)
        
        if params is not None:
            combined = torch.cat([conv_feats, params], dim=1)
        else:
            combined = conv_feats
        
        # Output in exponential space to guarantee strictly positive ratios
        return torch.exp(self.mlp(combined))