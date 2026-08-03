"""
Module: physics_informed_losses.py

This module defines custom physics-informed loss functions for training 
neural networks in spectral recalibration tasks. These loss functions 
integrate data-driven terms with physically motivated constraints, such as 
integrated flux conservation and shape smoothness penalties, to ensure 
that the generated recalibration factors remain physically plausible.
"""

import torch
from torch import nn

class BiCubicPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for Spectral Recalibration.
    
    This loss function is designed to stably train a Physics-Informed Neural 
    Network (PINN) by incorporating the following mechanisms:
    - Uses mean flux normalization to avoid numerical instabilities commonly 
      found near deep absorption lines.
    - Integrates an L2 regularization term on the recalibration factor to 
      maintain smoothness and eliminate non-physical oscillations.
    - Utilizes stable, fixed weighting coefficients (avoiding adaptive weight collapse).

    Attributes:
    - lambda_flux (float): The weighting coefficient for the flux conservation loss.
    - lambda_smooth (float): The weighting coefficient for the smoothness (L2) loss.
    """
    def __init__(self, lambda_flux: float = 1.0, lambda_smooth: float = 1e-3):
        """
        Initializes the BiCubicPhysicsInformedLoss module.

        Args:
        - lambda_flux (float, optional): Weight for the integrated flux loss. Defaults to 1.0.
        - lambda_smooth (float, optional): Weight for the smoothness penalty. Defaults to 1e-3.
        """
        super(BiCubicPhysicsInformedLoss, self).__init__()
        self.lambda_flux = lambda_flux
        self.lambda_smooth = lambda_smooth

    def forward(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor):
        """
        Computes the total physics-informed loss for a given forward pass.

        Args:
            model (nn.Module): The neural network model being trained.
            X (torch.Tensor): The input spectra (e.g., original Gaia spectra).
            Y (torch.Tensor): The target ground-truth spectra (e.g., CALSPEC).

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The final aggregated loss value.
                - loss_flux (torch.Tensor): The integrated flux conservation component.
                - loss_smooth (torch.Tensor): The smoothness regularization component.
        """
        # 1. Predict the recalibration factor and compute the corrected spectrum
        R_pred = model(X)
        X_cal = X * R_pred

        # 2. Mean flux per star (Mathematical stabilizer to avoid division by zero)
        Y_mean = torch.mean(Y, dim=1, keepdim=True) + 1e-8

        # 3. Data Loss (Stably scaled Mean Squared Error)
        loss_data = torch.mean(((X_cal - Y) / Y_mean) ** 2)

        # 4. Integrated Flux Conservation Loss
        X_cal_integrated = torch.sum(X_cal, dim=1)
        Y_integrated = torch.sum(Y, dim=1)
        loss_flux = torch.mean(((X_cal_integrated - Y_integrated) / Y_integrated) ** 2)

        # 5. L2 Smoothness Regularization on R(lambda)
        diff_R = R_pred[:, 1:] - R_pred[:, :-1]
        loss_smooth = torch.mean(diff_R ** 2)

        # 6. Fixed Weighted Total Loss
        total_loss = loss_data + (self.lambda_flux * loss_flux) + (self.lambda_smooth * loss_smooth)

        return total_loss, loss_flux, loss_smooth


class PCAPhysicsInformedLoss(nn.Module):
    """
    Definitive physics-informed loss function for PCA-based spectral recalibration.
    
    Key features:
    1. Relative Loss: Prevents vanishing gradients caused by the absolute flux scale (1e-14).
    2. Pipeline Compatibility: Takes (model, x_input, y_target) and returns the 
       (total_loss, loss_flux, loss_smooth) tuple required by `train_2stage_fixed`.
    3. Frozen Buffers: Keeps `pca_basis` and `pca_mean` synchronized across CPU/GPU 
       without updating them during backpropagation.
    """
    def __init__(self, pca_basis, pca_mean):
        super(PCAPhysicsInformedLoss, self).__init__()
        
        # Convert to PyTorch tensors if they are numpy arrays
        if not isinstance(pca_basis, torch.Tensor):
            pca_basis = torch.tensor(pca_basis, dtype=torch.float32)
        if not isinstance(pca_mean, torch.Tensor):
            pca_mean = torch.tensor(pca_mean, dtype=torch.float32)
            
        # Register basis and mean as buffers (non-trainable parameters that move with the device)
        self.register_buffer('pca_basis', pca_basis.float())
        self.register_buffer('pca_mean', pca_mean.float())

    def forward(self, model, x_input, y_target):
        """
        Args:
        - model (nn.Module): The PCASpectralPINN model.
        - x_input (torch.Tensor): Observed Gaia BP/RP spectra (batch_size, input_dim).
        - y_target (torch.Tensor): CALSPEC reference spectra (batch_size, input_dim).
            
        Returns:
        - tuple: (total_loss, loss_flux, loss_smooth)
        """
        # 1. Extract the predicted c_i coefficients from the network
        pca_coeffs = model(x_input)
        
        # 2. Physical reconstruction of the correction curve: R(lambda) = Mean + sum(c_i * PC_i)
        delta_R = torch.matmul(pca_coeffs, self.pca_basis)
        R = self.pca_mean + delta_R
        R = torch.clamp(R, min=0.3, max=3.0)
        
        # 3. Apply the multiplicative correction to the Gaia flux
        calibrated_spec = x_input * R
        
        # 4. RELATIVE LOSS (Percentage Error)
        # Normalize by y_target to make the error independent of the magnitude scale (~1e-14)
        rel_error = (calibrated_spec - y_target) / (y_target + 1e-15)
        loss_flux = torch.mean(rel_error ** 2)
        
        # 5. Dummy smoothness loss to maintain signature compatibility with the 2-stage pipeline
        loss_smooth = torch.tensor(0.0, device=loss_flux.device)
        
        total_loss = loss_flux
        
        return total_loss, loss_flux, loss_smooth

    
class GaussianPhysicsInformedLoss(nn.Module):
    """
    Calculates three physically-motivated loss components on a relative scale.
    
    This variant uses a Huber (Smooth L1) loss for the data fitting component 
    to handle outliers gracefully. Furthermore, it enforces shape preservation 
    by penalizing both the first derivative (Total Variation) and the second 
    derivative (Curvature) of the predicted recalibration curve.

    Attributes:
        huber (nn.SmoothL1Loss): The PyTorch Smooth L1 loss function used 
            for the data consistency term.
    """
    def __init__(self):
        """
        Initializes the GaussianPhysicsInformedLoss module.
        """
        super(GaussianPhysicsInformedLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(beta=0.01)

    def forward(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor, Sigma: torch.Tensor = None):
        """
        Computes the physics-informed loss components.

        Args:
        - model (nn.Module): The neural network model predicting the recalibration factor.
        - X (torch.Tensor): The input uncalibrated spectra.
        - Y (torch.Tensor): The reference ground-truth spectra.
        - Sigma (torch.Tensor, optional): The associated uncertainty/variance of the 
            input data. If None, the target Y is used as the scaling factor. Defaults to None.

        Returns:
        tuple: A tuple containing the three individual loss components:
        - loss_data (torch.Tensor): The relative data consistency loss.
        - loss_flux (torch.Tensor): The relative integrated flux conservation loss.
        - loss_grad (torch.Tensor): The combined shape preservation (gradient) loss.
        """
        eps = 1e-12
        
        R_pred = model(X)
        X_calibrated = X * R_pred
        
        # Component A: Data Loss
        if Sigma is not None:
            diff_relative = (X_calibrated - Y) / (Sigma + eps)
        else:
            diff_relative = (X_calibrated - Y) / (Y + eps)
            
        loss_data = self.huber(diff_relative, torch.zeros_like(diff_relative))

        # Component B: Relative Integrated Flux Conservation
        flux_calibrated = torch.sum(X_calibrated, dim=1)
        flux_calspec = torch.sum(Y, dim=1)
        loss_flux = torch.mean(((flux_calibrated - flux_calspec) / (flux_calspec + eps)) ** 2)

        # Component C: Shape Preservation (penalty on the 1st and 2nd derivatives)
        # 1. Penalize high-frequency noise (differences between adjacent pixels)
        d1_R = R_pred[:, 1:] - R_pred[:, :-1]
        loss_tv = torch.mean(torch.abs(d1_R)) # Use L1 (abs) or L2 (**2) norm
        
        # 2. Original curvature penalty (second derivative)
        d2_R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        loss_curv = torch.mean(d2_R ** 2)

        # Multipliers (adjust lambda_tv if the resulting curve becomes too flat)
        lambda_tv = 5.0  
        lambda_curv = 1.0

        loss_grad = (loss_tv * lambda_tv) + (loss_curv * lambda_curv)

        return loss_data, loss_flux, loss_grad