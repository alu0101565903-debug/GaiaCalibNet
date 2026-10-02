"""Define physics-informed losses for spectral calibration models.

The losses combine instrumental uncertainty weighting, spectral smoothness,
and regularization terms.

Loss classes
------------
- ``DirectPhysicsLoss`` combines flux residual, curvature, and PCA regularization.
- ``BluePhysicsLoss`` regularizes fine blue-region corrections around unity.

Notes
-----
Report context: Section 4.7, Physics-Informed Loss Functions.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pinn_architecture import DirectConvPINN


def _normalized_inverse_uncertainty_weights(errors: torch.Tensor) -> torch.Tensor:
    """Return per-spectrum inverse-sigma weights without assuming flux units."""
    eps = torch.finfo(errors.dtype).tiny
    errors_safe = errors.abs().clamp_min(eps)
    reference = errors_safe.median(dim=1, keepdim=True).values.clamp_min(eps)
    relative_errors = (errors_safe / reference).clamp_min(1e-6)
    weights = relative_errors.reciprocal()
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(eps)


class DirectPhysicsLoss(nn.Module):
    """
    Physics-informed loss function for absolute global spectral calibration.

    Combines an uncertainty-weighted Smooth L1 flux loss with a second-derivative 
    curvature penalty for spectral smoothness and L2 regularization on PCA coefficients.

    Parameters
    ----------
    lambda_smooth : float, default=1e-4
        Weighting factor for the second-derivative curvature smoothness loss.
    lambda_reg : float, default=1e-4
        Weighting factor for L2 regularization on predicted PCA coefficients.
    """

    def __init__(self, lambda_smooth: float = 1e-4, lambda_reg: float = 1e-4) -> None:
        """
        Initialize DirectPhysicsLoss parameters.

        Parameters
        ----------
        lambda_smooth : float, default=1e-4
            Smoothness loss hyperparameter.
        lambda_reg : float, default=1e-4
            PCA regularization hyperparameter.
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg

    def forward(
        self, 
        model: DirectConvPINN, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor, 
        e_input: torch.Tensor, 
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total weighted physics loss for global spectral calibration.

        Parameters
        ----------
        x_input : torch.Tensor
            Input flux tensor of shape (batch_size, input_dim).
        y_target : torch.Tensor
            Target CALSPEC flux tensor of shape (batch_size, input_dim).
        e_input : torch.Tensor
            Input flux error uncertainty tensor of shape (batch_size, input_dim).
        params : torch.Tensor, optional
            Auxiliary stellar physical parameters tensor of shape (batch_size, num_params).

        Returns
        -------
        torch.Tensor
            Scalar composite loss value.
        """
        pca_coeffs = model(x_input, params)
        R_pred = model.reconstruct(pca_coeffs)
        y_pred = x_input * R_pred
        
        # 1. Apply instrumental uncertainty weighting.
        weights = _normalized_inverse_uncertainty_weights(e_input)
        
        scale = torch.max(torch.abs(y_target), dim=1, keepdim=True)[0] + 1e-20
        rel_err = (y_pred - y_target) / scale
        
        # 2. Weight the relative flux residual.
        loss_flux = F.smooth_l1_loss(rel_err * weights, torch.zeros_like(rel_err), beta=0.01)

        # 3. Penalize second-difference curvature to encourage smoothness.
        diff2 = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        loss_smooth = torch.mean(diff2 ** 2)
        
        # 4. Apply L2 regularization to the PCA coefficients.
        loss_reg = torch.mean(pca_coeffs ** 2)

        return loss_flux + (self.lambda_smooth * loss_smooth) + (self.lambda_reg * loss_reg)


class BluePhysicsLoss(nn.Module):
    """
    Physics-informed loss function for localized blue-region wiggle correction.

    Optimizes fine pixel-wise multiplicative correction factors against clamped 
    uncertainties, penalizing deviations from a neutral baseline (1.0) and high-frequency wiggles.

    Parameters
    ----------
    lambda_reg : float, default=1e-3
        Weighting factor for L2 regularization towards the neutral identity baseline (1.0).
    lambda_smooth : float, default=1e-4
        Weighting factor for second-derivative curvature penalty.
    """

    def __init__(self, lambda_reg: float = 1e-3, lambda_smooth: float = 1e-4) -> None:
        """
        Initialize BluePhysicsLoss parameters.

        Parameters
        ----------
        lambda_reg : float, default=1e-3
            Neutral-baseline regularization hyperparameter.
        lambda_smooth : float, default=1e-4
            Smoothness loss hyperparameter.
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_smooth = lambda_smooth

    def forward(
        self, 
        R_blue_pred: torch.Tensor, 
        y_target_blue: torch.Tensor, 
        y_pred_base_blue: torch.Tensor, 
        e_input_blue: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute localized loss for fine blue-region spectral correction.

        Parameters
        ----------
        R_blue_pred : torch.Tensor
            Predicted multiplicative blue correction factors of shape (batch_size, blue_channels).
        y_target_blue : torch.Tensor
            Target CALSPEC blue-region flux tensor of shape (batch_size, blue_channels).
        y_pred_base_blue : torch.Tensor
            Base PINN model predicted blue-region flux tensor of shape (batch_size, blue_channels).
        e_input_blue : torch.Tensor
            Input flux uncertainty tensor for blue channels of shape (batch_size, blue_channels).

        Returns
        -------
        torch.Tensor
            Scalar composite loss value.
        """
        # Final flux prediction is base prediction multiplied by fine blue correction
        y_pred_final = y_pred_base_blue * R_blue_pred
        
        # Weight each star with its own uncertainty profile. A batch-wide cap
        # changes the objective when validation contains only one held-out star.
        weights = _normalized_inverse_uncertainty_weights(e_input_blue)
        
        scale = torch.max(torch.abs(y_target_blue), dim=1, keepdim=True)[0] + 1e-20
        rel_err = (y_pred_final - y_target_blue) / scale
        
        loss_flux = F.smooth_l1_loss(rel_err * weights, torch.zeros_like(rel_err), beta=0.01)
        
        # L2 regularization towards neutral identity factor (1.0) rather than zero
        loss_reg = torch.mean((R_blue_pred - 1.0) ** 2)
        
        # Second-difference curvature penalty to suppress high-frequency oscillations
        diff2 = R_blue_pred[:, 2:] - 2 * R_blue_pred[:, 1:-1] + R_blue_pred[:, :-2]
        loss_smooth = torch.mean(diff2 ** 2)
        
        return loss_flux + (self.lambda_reg * loss_reg) + (self.lambda_smooth * loss_smooth)