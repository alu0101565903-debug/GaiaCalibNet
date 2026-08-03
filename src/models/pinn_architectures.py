"""
Module: pinn_architectures.py

Physics-Informed Neural Network (PINN) architectures for spectral recalibration.

This module contains the definitions for different PINN models, including 
bicubic interpolation-based and Gaussian-smoothed residual networks. These 
architectures are specifically designed to predict the multiplicative correction 
factor R(lambda) required for accurately recalibrating Gaia BP/RP optical spectra 
to match CALSPEC ground-truth references.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSpectralPINN(nn.Module):
    """
    Base class for all spectral recalibration PINN models.

    This class serves as a blueprint for inheritance, ensuring that any 
    derived spectral PINN architectures share a common type hierarchy.
    """
    pass

class BiCubicSpectralPINN(BaseSpectralPINN):
    """
    Physics-Informed Neural Network based on Control Knots and Cubic Interpolation.

    This architecture predicts a smooth, C1-continuous recalibration curve. 
    Instead of predicting a correction for every wavelength bin directly, it 
    predicts values at a sparse set of control knots. These knots are then 
    upsampled to the full spectral resolution using bicubic interpolation. 
    This structural constraint prevents the network from overfitting to local 
    high-frequency noise (such as absorption lines).

    Attributes:
    - input_dim (int): The number of wavelength bins in the full input spectrum.
    - num_knots (int): The number of control knots (parameters) used to construct 
        the recalibration curve. Defaults to 20.
    - net (nn.Sequential): The multilayer perceptron predicting the delta shifts 
        at the control knots.
    """
    def __init__(self, input_dim, num_knots=20):
        """
        Initializes the BiCubicSpectralPINN architecture.

        Args:
            input_dim (int): The total number of spectral pixels/wavelengths.
            num_knots (int, optional): The number of control points. A lower number 
                increases smoothness. Defaults to 20.
        """
        super(BiCubicSpectralPINN, self).__init__()
        self.input_dim = input_dim
        self.num_knots = num_knots
        
        # Network responsible for predicting the correction solely at the control knots
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_knots)
        )

        # Zero initialization in the final layer to ensure it starts at R = 1.0 (identity mapping)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """
        Executes the forward pass to predict the recalibration factor R.

        Args:
            x (torch.Tensor): The input uncalibrated spectra of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The predicted multiplicative correction factor R(lambda) 
            of shape (batch_size, input_dim), safely clamped between 0.3 and 3.0.
        """
        # 1. Input normalization (by max flux per star) to ensure numerical stability
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        
        # 2. Control knots prediction -> Shape: (batch_size, num_knots)
        delta_knots = self.net(x_norm)
        
        # 3. Reshape to 4D to apply 2D bicubic interpolation (with height=1)
        # Resulting shape: (batch_size, 1, 1, num_knots)
        delta_4d = delta_knots.unsqueeze(1).unsqueeze(2)
        
        # 4. Smooth cubic interpolation to the full spectrum size (input_dim)
        delta_interp = F.interpolate(
            delta_4d, 
            size=(1, self.input_dim), 
            mode="bicubic", 
            align_corners=True
        )
        
        # 5. Revert to shape (batch_size, input_dim)
        delta_R = delta_interp.squeeze(2).squeeze(1)
        
        # 6. Scale the spectrum
        R = 1.0 + delta_R
        return torch.clamp(R, min=0.3, max=3.0)

    def prepare_for_lbfgs(self):
        """
        Prepares the model for the L-BFGS optimization phase.
        Sets the network to evaluation mode to stabilize normalization layers 
        and prevent unwanted stochastic behaviors during line searches.
        """
        self.eval()


class PCASpectralPINN(nn.Module):
    """
    PCA-based Physics-Informed Neural Network.
    
    This network predicts ONLY the PCA coefficients (c_i) for a given input 
    spectrum, functioning equivalently to a polynomial coefficient predictor.
    """
    def __init__(self, input_dim, pca_basis, pca_mean=None):
        """
        Args:
        - input_dim (int): Number of pixels/wavelengths (e.g., 343).
        - pca_basis (Tensor or ndarray): PCA basis of shape (num_components, input_dim).
        - pca_mean (Tensor or ndarray, optional): PCA mean of shape (input_dim,).
        """
        super(PCASpectralPINN, self).__init__()
        self.input_dim = input_dim
        
        # Save the PCA basis as a frozen buffer
        if not isinstance(pca_basis, torch.Tensor):
            pca_basis = torch.tensor(pca_basis, dtype=torch.float32)
        self.register_buffer('pca_basis', pca_basis.float())
        self.num_components = self.pca_basis.shape[0]

        # Save the PCA mean as a buffer
        if pca_mean is None:
            pca_mean = torch.ones(input_dim, dtype=torch.float32)
        elif not isinstance(pca_mean, torch.Tensor):
            pca_mean = torch.tensor(pca_mean, dtype=torch.float32)
        self.register_buffer('pca_mean', pca_mean.float())
        
        # Network in charge of predicting only the c_i (output = num_components)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_components)
        )

        # Zero initialization: at the start of training, the network predicts c_i = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """
        Args:
        - x (torch.Tensor): Input spectra (batch_size, input_dim).

        Returns:
        - torch.Tensor: Predicted c_i coefficients (batch_size, num_components).
        """
        # Normalization by the maximum flux per star for stability
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        
        return self.net(x_norm)

    def reconstruct(self, pca_coeffs):
        """
        Reconstructs the R(lambda) curve from the predicted coefficients.
        Useful for evaluation and inference outside the loss function.
        
        Args:
        - pca_coeffs (torch.Tensor): Coefficients (batch_size, num_components).
            
        Returns:
        - torch.Tensor: R correction factors (batch_size, input_dim).
        """
        delta_R = torch.matmul(pca_coeffs, self.pca_basis)
        R = self.pca_mean + delta_R
        return torch.clamp(R, min=0.3, max=3.0)

    def prepare_for_lbfgs(self):
        """
        Prepares the model for the L-BFGS optimization phase.
        Sets the network to evaluation mode to stabilize normalization layers 
        and prevent unwanted stochastic behaviors during line searches.
        """
        self.eval()

class GaussianSpectralPINN(BaseSpectralPINN):
    """
    Residual PINN architecture utilizing an internal 1D Gaussian filter.

    This network predicts the correction factor R(lambda) by evaluating a 
    Gaussian-smoothed version of the input spectrum. By embedding the 
    smoothing operation directly within the forward pass as a non-trainable 
    layer, it structurally prevents the dense layers from overfitting to 
    high-frequency noise, forcing the output R to respond only to the 
    underlying macroscopic spectral shape.

    Attributes:
    - net (nn.Sequential): The fully connected neural network predicting 
        the correction factor from the smoothed input.
    - kernel_padding (int): The padding required to maintain sequence length 
        during 1D convolution.
    """
    def __init__(self, input_dim):
        super(GaussianSpectralPINN, self).__init__()
        
        # 1. Dense neural network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, input_dim)
        )
        
        # Zero initialization in the last layer to start with R(lambda) ≈ 1.0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # 2. 1D Gaussian filter configuration
        kernel_size = 7  # Smoothing window size (must be an odd number)
        sigma = 1.5      # Standard deviation / smoothing strength
        
        # Create the Gaussian curve
        x_coord = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        g = torch.exp(-(x_coord ** 2) / (2 * sigma ** 2))
        g = g / g.sum()  # Normalize to maintain the flux scale
        
        # Save the kernel as a buffer (PyTorch will automatically move it to GPU/CPU with the model)
        # Shape required by F.conv1d: (out_channels, in_channels, length) -> (1, 1, kernel_size)
        self.register_buffer("gaussian_kernel", g.view(1, 1, kernel_size))
        self.kernel_padding = kernel_size // 2

    def forward(self, x):
        """
        Executes the forward pass applying Gaussian smoothing prior to evaluation.

        Args:
        - x (torch.Tensor): The input uncalibrated spectra of shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: The predicted multiplicative correction factor R(lambda) 
            of shape (batch_size, input_dim), clamped between 0.3 and 3.0.
        """
        # A. Local normalization by maximum flux
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        
        # B. 1D Gaussian smoothing
        # F.conv1d requires a 3D tensor: (batch_size, channels, length)
        x_norm_3d = x_norm.unsqueeze(1) 
        x_smoothed = F.conv1d(x_norm_3d, self.gaussian_kernel, padding=self.kernel_padding)
        x_smoothed = x_smoothed.squeeze(1) # Revert to shape (batch_size, length)

        # C. The network predicts delta_R evaluating ONLY the SMOOTHED spectrum
        delta_R = self.net(x_smoothed)
        
        # D. Final correction factor R
        R = 1.0 + delta_R
        return torch.clamp(R, min=0.3, max=3.0)

    def prepare_for_lbfgs(self):
        """
        Prepares the model for the L-BFGS optimization phase.
        """
        self.eval()


# =====================================================================
# FACTORY REGISTRY & INSTANTIATION
# =====================================================================

MODEL_REGISTRY = {
    "residual_bicubic": BiCubicSpectralPINN,
    "residual_gaussian": GaussianSpectralPINN
}

def get_pinn_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to dynamically instantiate PINN models by their string identifier.

    Args:
    - model_name (str): The name of the model architecture to instantiate. 
        Must be a valid key in `MODEL_REGISTRY`.
    - **kwargs: Arbitrary keyword arguments passed directly to the model's constructor 
        (e.g., `input_dim`, `num_knots`).

    Returns:
    - nn.Module: An initialized instance of the requested PyTorch model.

    Raises:
    - ValueError: If the requested `model_name` is not found in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not supported. Available options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)