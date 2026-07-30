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
    Definitive Physics-Informed Loss for Spectral Recalibration.
    
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

        # Component C: Shape Preservation (Penalización de 1ra y 2da derivada)
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