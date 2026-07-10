"""
Module: losses.py
Report Section: 6.5 Physics-Informed Loss Function (L_flux & L_grad)

This module implements mathematically rigorous Physics-Informed Loss (PIL) criteria 
for stellar spectrophotometric recalibration. It builds upon a robust Huber baseline 
by integrating a parameterized Riemann sum for total flux conservation and physical 
finite differences for local spectral gradient preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss (PIL) engine for stellar spectra prediction (Section 6.5).
    
    Combines a robust Huber Loss baseline with two astrophysical regularizations:
    - L_flux: Total Integrated Flux Conservation via Riemann Sum integration.
    - L_grad: Local Gradient Preservation via physical Finite Differences (dF/dlambda).
    """
    def __init__(self, wavelength_grid: torch.Tensor, alpha: float = 0.1, beta: float = 0.1, delta: float = 1.0):
        """
        Args:
        - wavelength_grid (torch.Tensor): Homogeneous wavelength vector (Lambda).
        - alpha (float): Scalar weight factor for flux conservation (L_flux).
        - beta (float): Scalar weight factor for line gradient preservation (L_grad).
        - delta (float): Threshold parameter governing the Huber loss transitions.
        """
        super().__init__()
        
        # Register buffers so tensors automatically migrate across devices (CPU/GPU) with the model
        self.register_buffer("wavelength_grid", wavelength_grid.detach().clone())
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        
        # Calculate localized Delta Lambda steps (Delta\lambda_i) for precise physical scaling
        self.register_buffer("delta_lambda", self.wavelength_grid[1:] - self.wavelength_grid[:-1])
        
        # Pad the final interval boundaries to preserve consistent dimensional arrays during integration
        last_step = self.delta_lambda[-1].unsqueeze(0)
        self.register_buffer("delta_lambda_padded", torch.cat([self.delta_lambda, last_step]))

    def compute_decoupled_components(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Computes and returns the three physical loss terms in an unweighted state.
        Essential structure for self-adaptive training weighting strategies.

        Returns:
            tuple: (loss_huber, loss_flux, loss_grad)
        """
        # 1. Base Statistical Loss (Huber) - Robust against instrument noise artifacts
        loss_huber = F.huber_loss(y_pred, y_true, delta=self.delta, reduction="mean")
        
        # 2. Flux Conservation Loss via formal Riemann Sum Integration
        integrated_flux_true = torch.sum(y_true * self.delta_lambda_padded, dim=1)
        integrated_flux_pred = torch.sum(y_pred * self.delta_lambda_padded, dim=1)
        loss_flux = torch.mean((integrated_flux_true - integrated_flux_pred) ** 2)
        
        # 3. Gradient Preservation Loss using true physical derivatives (dF / d\lambda)
        grad_true = (y_true[:, 1:] - y_true[:, :-1]) / self.delta_lambda
        grad_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / self.delta_lambda
        loss_grad = torch.mean((grad_true - grad_pred) ** 2)
        
        return loss_huber, loss_flux, loss_grad

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass returning the single composite scalar loss.
        Optimized for static hyperparameter training loops.
        """
        loss_huber, loss_flux, loss_grad = self.compute_decoupled_components(y_pred, y_true)
        return loss_huber + (self.alpha * loss_flux) + (self.beta * loss_grad)



# Functional Wrappers for Training Loop Compatibility

def adaptive_physics_wrapper(model, X_batch, Y_batch, loss_instance: PhysicsInformedLoss):
    """
    Interface wrapper linking the Neural Network forward pass outputs directly 
    to the decoupled loss component evaluator required by adaptive training loops.
    """
    Y_pred = model(X_batch)
    return loss_instance.compute_decoupled_components(Y_pred, Y_batch)


def compute_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes standard baseline Mean Squared Error."""
    return F.mse_loss(y_pred, y_true, reduction="mean")


def compute_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes standard baseline Mean Absolute Error."""
    return F.l1_loss(y_pred, y_true, reduction="mean")