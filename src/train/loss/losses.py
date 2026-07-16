"""
Physics-informed loss definitions for Gaia-CALSPEC spectrophotometric recalibration.

This module provides a physics-aware composite loss for training spectral
recalibration models. The complete loss is built from:
- Huber pointwise residuals on normalized spectra,
- relative integrated flux conservation,
- gradient-based shape matching,
- curvature-based smoothness regularization.

The implementation normalizes spectra internally to avoid numerical underflow
when flux values are extremely small, and supports optional noise-weighted
Huber residuals for uncertainty-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    """
    Composite loss for spectral recalibration training.

    Computes a weighted sum of:
    - normalized Huber residuals,
    - relative integrated flux conservation,
    - gradient-based shape consistency,
    - curvature smoothness.

    Normalization prevents numerical issues with very small spectral fluxes,
    and optional noise variance can be supplied to weight residuals.
    """
    def __init__(self, wavelength_grid: torch.Tensor, alpha: float = 1.0, beta: float = 1.0, gamma: float = 5.0, delta: float = 1.0):
        super().__init__()
        # Store the wavelength grid as a persistent tensor that moves with the model.
        self.register_buffer("wavelength_grid", wavelength_grid.detach().clone())
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Delta wavelength values are used to compute quadrature weights for
        # flux integration. This supports irregular spectral grids.
        delta_lambda = self.wavelength_grid[1:] - self.wavelength_grid[:-1]
        self.register_buffer("delta_lambda", delta_lambda)

        # Trapezoidal integration weights for the full spectral channel grid.
        weights = torch.zeros_like(self.wavelength_grid)
        weights[0] = 0.5 * delta_lambda[0]
        weights[-1] = 0.5 * delta_lambda[-1]
        weights[1:-1] = 0.5 * (delta_lambda[:-1] + delta_lambda[1:])
        self.register_buffer("integration_weights", weights)

    def compute_decoupled_components(self, y_pred: torch.Tensor, y_true: torch.Tensor, noise_variance: torch.Tensor = None):
        """Compute each loss component independently for analysis and combination.

        The method returns decomposed losses so the training loop can inspect
        model behavior and weight the contributions differently.
        """
        # ---------------------------------------------------------------------
        # KEY STEP: normalize spectra internally to O(1) scale.
        # This prevents squared-error terms from collapsing to extremely small
        # magnitudes when flux values are on the order of 10^-12.
        # ---------------------------------------------------------------------
        scale = torch.mean(y_true, dim=1, keepdim=True) + 1e-12
        y_pred_norm = y_pred / scale
        y_true_norm = y_true / scale
        
        # 1. Base statistical Huber loss on normalized spectra.
        #    This penalizes pointwise flux deviations robustly while remaining
        #    stable for outliers.
        if noise_variance is not None:
            sigma_norm = torch.sqrt(noise_variance + 1e-12) / scale
            normalized_residuals = (y_pred_norm - y_true_norm) / sigma_norm
            abs_res = torch.abs(normalized_residuals)
            loss_huber = torch.where(
                abs_res < self.delta,
                0.5 * (normalized_residuals ** 2),
                self.delta * (abs_res - 0.5 * self.delta),
            ).mean()
        else:
            loss_huber = F.huber_loss(y_pred_norm, y_true_norm, delta=self.delta, reduction="mean")
        
        # 2. Flux conservation loss. This compares integrated flux over the
        #    wavelength grid, enforcing global energy consistency.
        integrated_flux_true = torch.sum(y_true * self.integration_weights.unsqueeze(0), dim=1)
        integrated_flux_pred = torch.sum(y_pred * self.integration_weights.unsqueeze(0), dim=1)
        loss_flux = torch.mean(
            ((integrated_flux_true - integrated_flux_pred) / (integrated_flux_true + 1e-12)) ** 2
        )
        
        # 3. Shape/gradient loss on normalized spectrum.
        #    Matches first-order spectral slope behavior rather than raw values.
        grad_true = (y_true_norm[:, 1:] - y_true_norm[:, :-1])
        grad_pred = (y_pred_norm[:, 1:] - y_pred_norm[:, :-1])
        loss_grad = torch.mean((grad_true - grad_pred) ** 2)
        
        # 4. Curvature smoothness loss on normalized spectrum.
        #    Penalizes high-frequency wiggles in the predicted shape.
        tv_loss = torch.mean((y_pred_norm[:, 1:] - y_pred_norm[:, :-1]) ** 2)
        curv_loss = torch.mean((y_pred_norm[:, 2:] - 2 * y_pred_norm[:, 1:-1] + y_pred_norm[:, :-2]) ** 2)
        loss_smooth = tv_loss + curv_loss
        
        return loss_huber, loss_flux, loss_grad, loss_smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, noise_variance: torch.Tensor = None) -> torch.Tensor:
        """Compute the weighted sum of all loss components used for training."""
        loss_huber, loss_flux, loss_grad, loss_smooth = self.compute_decoupled_components(y_pred, y_true, noise_variance)
        # Combine the component losses using configured weights.
        return loss_huber + (self.alpha * loss_flux) + (self.beta * loss_grad) + (self.gamma * loss_smooth)


def adaptive_physics_wrapper(model, X_batch, Y_batch, loss_instance: PhysicsInformedLoss, noise_variance: torch.Tensor = None):
    """Compute loss metrics for adaptive training or diagnostics.

    This helper evaluates the model and returns separate reported quantities
    without performing a backward pass. It is useful for training regimes where
    gradient-based shape penalties may be monitored independently.
    """
    Y_pred = model(X_batch)
    loss_huber, loss_flux, loss_grad, loss_smooth = loss_instance.compute_decoupled_components(Y_pred, Y_batch, noise_variance)
    
    # Combine the two shape-based terms into a single shape loss component.
    combined_shape_loss = loss_grad + (loss_instance.gamma * loss_smooth)
    
    return loss_huber, loss_flux, combined_shape_loss


def compute_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the standard mean squared error between prediction and target."""
    return F.mse_loss(y_pred, y_true, reduction="mean")


def compute_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the standard mean absolute error between prediction and target."""
    return F.l1_loss(y_pred, y_true, reduction="mean")