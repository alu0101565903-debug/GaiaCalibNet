"""
Module: physics_informed_losses.py

Physics-Informed Custom Loss Functions for Spectral Recalibration.

This module defines custom loss functions for training Physics-Informed Neural 
Networks (PINNs) in Gaia BP/RP spectral recalibration tasks. These loss objectives 
combine relative data-driven terms with physically motivated regularization priors, 
including integrated flux conservation, total variation, curvature penalties, 
and coefficient magnitude regularization.

Loss Architectures:
1. BiCubicPhysicsInformedLoss:
   Combines normalized flux error, total integrated flux conservation, and 
   first-derivative (L2) smoothness penalties on predicted $R(\lambda)$ curves.
2. GaussianPhysicsInformedLoss:
   Multi-term relative loss employing Smooth L1 (Huber) data fitting, relative 
   integrated flux conservation, and combined TV/curvature shape preservation penalties.
3. PCAPhysicsInformedLoss:
   Evaluates relative percentage errors between calibrated spectra and ground-truth 
   references using frozen PCA basis vectors.
4. PCAParamsPhysicsInformedLoss:
   Extended PCA loss supporting physical stellar parameter conditioning, 
   second-derivative curvature penalties, and L2 regularization on higher-order 
   PCA components ($c_i, i \ge 1$).
5. UndegradedPCAPhysicsInformedLoss:
   Loss function tailored for dual-branch PINNs, applying Smooth L1 fitting on 
   scaled flux, second-derivative curvature constraints strictly on the Chebyshev global 
   branch, and $L_1$ regularization on local residual peak corrections.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BiCubicPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for Control-Knot BiCubic Spectral Recalibration.

    Enforces data consistency using local mean flux normalization, penalizes total 
    integrated flux discrepancies, and applies a first-order difference penalty 
    on predicted correction profiles $R(\lambda)$.

    Parameters
    - lambda_flux (float, default=1.0): Weighting coefficient for integrated flux conservation penalty.
    - lambda_smooth (float, default=1e-3): Weighting coefficient for first-derivative smoothness penalty.
    """
    def __init__(self, lambda_flux: float = 1.0, lambda_smooth: float = 1e-3) -> None:
        super(BiCubicPhysicsInformedLoss, self).__init__()
        self.lambda_flux = lambda_flux
        self.lambda_smooth = lambda_smooth

    def forward(
        self, 
        model: nn.Module, 
        X: torch.Tensor, 
        Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total physics-informed loss for a given forward pass.

        Parameters:
        - model (nn.Module): The neural network model being trained.
        - X (torch.Tensor): The input spectra (e.g., original Gaia spectra).
        - Y (torch.Tensor): The target ground-truth spectra (e.g., CALSPEC).

        Returns: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
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
    Multi-Component Relative Loss Function using Robust Huber Fitting.

    Combines Huber (Smooth L1) data consistency, relative integrated flux 
    conservation, and shape-preserving Total Variation (TV) + curvature penalties.

    Attributes:
    - huber (nn.SmoothL1Loss): The PyTorch Smooth L1 loss function used 
            for the data consistency term.
    """
    def __init__(self) -> None:
        super(GaussianPhysicsInformedLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(beta=0.01)

    def forward(
        self, 
        model: nn.Module, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        Sigma: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the physics-informed loss components.

        Parameters:
        - model (nn.Module): The neural network model predicting the recalibration factor.
        - X (torch.Tensor): The input uncalibrated spectra.
        - Y (torch.Tensor): The reference ground-truth spectra.
        - Sigma (torch.Tensor, optional): The associated uncertainty/variance of the 
            input data. If None, the target Y is used as the scaling factor. Defaults to None.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the three individual loss components:
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

    
class PCAPhysicsInformedLoss(nn.Module):
    """
    Definitive physics-informed loss function for PCA-based spectral recalibration.
    
    Key features:
    1. Relative Loss: Prevents vanishing gradients caused by the absolute flux scale (1e-14).
    2. Pipeline Compatibility: Takes (model, x_input, y_target) and returns the 
       (total_loss, loss_flux, loss_smooth) tuple required by `train_2stage_fixed`.
    3. Frozen Buffers: Keeps "pca_basis" and "pca_mean" synchronized across CPU/GPU 
       without updating them during backpropagation.
    
    Parameters
    - pca_basis : Union[torch.Tensor, np.ndarray]
        PCA basis matrix of shape `(num_components, input_dim)`.
    - pca_mean : Union[torch.Tensor, np.ndarray]
        PCA mean residual vector of shape `(input_dim,)`.
    """
    def __init__(
        self, 
        pca_basis: Union[torch.Tensor, np.ndarray], 
        pca_mean: Union[torch.Tensor, np.ndarray]
    ) -> None:
        super(PCAPhysicsInformedLoss, self).__init__()
        
        # Convert to PyTorch tensors if they are numpy arrays
        if not isinstance(pca_basis, torch.Tensor):
            pca_basis = torch.tensor(pca_basis, dtype=torch.float32)
        if not isinstance(pca_mean, torch.Tensor):
            pca_mean = torch.tensor(pca_mean, dtype=torch.float32)
            
        # Register basis and mean as buffers (non-trainable parameters that move with the device)
        self.register_buffer("pca_basis", pca_basis.float())
        self.register_buffer("pca_mean", pca_mean.float())

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute relative flux loss for PCA coefficient prediction models.

        Parameters:
        - model (nn.Module): The PCASpectralPINN model.
        - x_input (torch.Tensor): Observed Gaia BP/RP spectra (batch_size, input_dim).
        - y_target (torch.Tensor): CALSPEC reference spectra (batch_size, input_dim).
            
        Returns: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : Relative flux MSE loss tensor.
            - loss_flux : Relative flux error component.
            - loss_smooth : Zero tensor maintaining training interface compatibility.
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

    
class PCAParamsPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for Parameter-Conditioned PCA PINNs.

    Extends PCA loss by evaluating curvature (second-derivative) penalties 
    and applying selective $L_2$ regularization on higher-order PCA coefficients 
    ($c_i, i \ge 1$) to enforce minimal distortion from identity scaling ($R \approx 1$).

    Parameters
    - pca_basis : Union[torch.Tensor, np.ndarray]
        PCA component basis matrix of shape `(num_components, input_dim)`.
    - pca_mean : Union[torch.Tensor, np.ndarray]
        PCA mean residual vector of shape `(input_dim,)`.
    - lambda_smooth : float, default=1e-3
        Weighting factor for second-derivative curvature penalty.
    - lambda_reg : float, default=1e-4
        Weighting factor for $L_2$ penalty on non-zero PCA components.
    """
    def __init__(
        self, 
        pca_basis: Union[torch.Tensor, np.ndarray], 
        pca_mean: Union[torch.Tensor, np.ndarray], 
        lambda_smooth: float = 1e-3, 
        lambda_reg: float = 1e-4
    ) -> None:
        super(PCAParamsPhysicsInformedLoss, self).__init__()
        
        # Convert to PyTorch tensors if they are numpy arrays
        if not isinstance(pca_basis, torch.Tensor):
            pca_basis = torch.tensor(pca_basis, dtype=torch.float32)
        if not isinstance(pca_mean, torch.Tensor):
            pca_mean = torch.tensor(pca_mean, dtype=torch.float32)
            
        # Register basis and mean as buffers (non-trainable parameters that move with the device)
        self.register_buffer("pca_basis", pca_basis.float())
        self.register_buffer("pca_mean", pca_mean.float())

        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor, 
        stellar_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss with curvature and coefficient prior penalties.

        Parameters
        - model (nn.Module): The PCASpectralPINN model.
        - x_input (torch.Tensor): Observed Gaia BP/RP spectra (batch_size, input_dim).
        - y_target (torch.Tensor): CALSPEC reference spectra (batch_size, input_dim).
        - stellar_params (torch.Tensor): Normalized stellar parameters (batch_size, num_params).
        
        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            A tuple containing:
            - total_loss : Aggregated weighted loss tensor.
            - loss_flux : Clamped relative flux error component.
            - loss_smooth : Second-derivative curvature penalty component.
        """
        
        # 1. Extract the predicted c_i coefficients from the network
        pca_coeffs = model(x_input, stellar_params)
        
        # 2. Physical reconstruction of the correction curve: R(lambda) = Mean + sum(c_i * PC_i)
        delta_R = torch.matmul(pca_coeffs, self.pca_basis)
        R = self.pca_mean + delta_R
        R = torch.clamp(R, min=0.3, max=3.0)
        
        # 3. Apply the multiplicative correction to the Gaia flux
        calibrated_spec = x_input * R
        
        # 4. RELATIVE LOSS (Percentage Error)
        # Extrar maximun value to avoid division by zero in regions of deep absorption lines
        y_max = torch.max(y_target, dim=1, keepdim=True)[0]
        # Normalize by y_max to make the error independent of the magnitude scale (~1e-14)
        denominator = torch.clamp(y_target, min=0.05 * y_max)
        epsilon = 0.01 * y_max
        rel_error = (calibrated_spec - y_target) / (y_max + epsilon)
        loss_flux = torch.mean(rel_error ** 2)
        
        # 5. Dummy smoothness loss to maintain signature compatibility with the 2-stage pipeline
        #loss_smooth = torch.tensor(0.0, device=loss_flux.device)
        dR = R[:, 1:] - R[:, :-1]
        ddR = dR[:, 1:] - dR[:, :-1]
        loss_smooth = torch.mean(ddR ** 2)

        # 6. Selective Regularization: Exclude c_0 to allow full continuum level adjustment
        if pca_coeffs.shape[1] > 1:
            loss_reg = torch.mean(pca_coeffs[:, 1:] ** 2)
        else:
            loss_reg = torch.tensor(0.0, device=pca_coeffs.device)

        total_loss = loss_flux + (self.lambda_smooth * loss_smooth) + (self.lambda_reg * loss_reg)
        
        return total_loss, loss_flux, loss_smooth


class UndegradedPCAPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for Dual-Branch Undegraded PINNs.

    Evaluates Smooth L1 reconstruction error on scaled flux, constrains 
    continuum curvature strictly on the Chebyshev global branch, and applies $L_1$ 
    regularization to local residual outputs to prevent continuum distortion.

    Parameters:
    - curvature_weight (float, default=0.1): Weight factor for Chebyshev continuum curvature penalty.
    - local_reg_weight (float, default=0.005): Weight factor for $L_1$ norm regularization on local peak residuals.
    """
    def __init__(
        self, 
        curvature_weight: float = 0.1, 
        local_reg_weight: float = 0.005
    ) -> None:
        super().__init__()
        self.curvature_weight = curvature_weight
        self.local_reg_weight = local_reg_weight

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes multi-component loss for dual-branch (Chebyshev + local residual) PINNs.

        Parameters:
        - model (nn.Module): Undegraded PINN model instance.
        - x_input (torch.Tensor): Uncalibrated spectra tensor of shape `(batch_size, input_dim)`.
        - y_target (torch.Tensor): CALSPEC reference spectra tensor of shape `(batch_size, input_dim)`.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss (torch.Tensor): Aggregated loss value.
            - loss_flux (torch.Tensor): Smooth L1 flux reconstruction error.
            - loss_curvature (torch.Tensor): Curvature penalty evaluated on the global continuum branch.
        """
        out = model(x_input)
        global_coeffs, local_residual = out
        
        R_total = model.reconstruct(out)
        calibrated_spec = x_input * R_total
        
        scale_y = torch.median(y_target, dim=1, keepdim=True)[0] + 1e-30
        y_norm = y_target / scale_y
        cal_norm = calibrated_spec / scale_y

        # Direct error calculation to capture high-frequency features
        loss_flux = F.smooth_l1_loss(cal_norm, y_norm, beta=0.05)

        # Curvature is evaluated exclusively on the Chebyshev global continuum branch
        R_global = torch.matmul(global_coeffs, model.pca_basis)
        d2R = R_global[:, 2:] - 2 * R_global[:, 1:-1] + R_global[:, :-2]
        loss_curvature = torch.mean(d2R ** 2)

        # Regularization to prevent local peak corrections from distorting the continuum
        #loss_local_reg = torch.mean(local_residual ** 2)
        loss_local_reg = torch.mean(torch.abs(local_residual))

        total_loss = loss_flux + (self.curvature_weight * loss_curvature) + (self.local_reg_weight * loss_local_reg)

        return total_loss, loss_flux, loss_curvature