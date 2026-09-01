"""
Module: physics_informed_losses.py
Report Section: Chapter 5 Data Processing & Feature Engineering / Optimization

Physics-Informed Custom Loss Functions for Spectral Recalibration.

This module defines custom PyTorch loss functions for training Physics-Informed 
Neural Networks (PINNs) in Gaia BP/RP spectral recalibration tasks. These objectives 
combine relative data-driven fidelity terms with physically motivated regularization priors, 
including integrated flux conservation, total variation, higher-order curvature penalties, 
and selective coefficient magnitude regularization.

Loss Architectures:
1. BiCubicPhysicsInformedLoss:
   Combines normalized flux error, total integrated flux conservation, and 
   first-derivative (L2) smoothness penalties on predicted R(lambda) curves.
2. GaussianPhysicsInformedLoss:
   Multi-term relative loss employing Smooth L1 (Huber) data fitting, relative 
   integrated flux conservation, and combined TV/curvature shape preservation penalties.
3. PCAPhysicsInformedLoss:
   Evaluates relative percentage errors between calibrated spectra and ground-truth 
   references using frozen PCA basis vectors.
4. PCAParamsPhysicsInformedLoss:
   Extended PCA loss supporting physical stellar parameter conditioning, 
   integrated flux anchor constraints, and curvature penalties.
5. SobolevPINNLoss:
   Sobolev-space objective combining Huber reconstruction error, first-derivative 
   spectral line alignment, and instrumental response curvature regularization.
6. AdaptivePhysicsLoss:
   Wavelength-weighted adaptive physics loss with UV/IR region emphasis 
   and curvature smoothing penalties.
7. UltimateSpectralPINNLoss:
   Comprehensive multi-objective loss integrating Huber reconstruction, Sobolev derivative 
   alignment, bolometric integrated flux anchoring, and second-derivative response curvature regularization.
8. UndegradedChebyshevPhysicsInformedLoss:
   Loss function tailored for dual-branch PINNs, applying Smooth L1 fitting on 
   scaled flux, second-derivative curvature constraints strictly on the Chebyshev global 
   branch, and L1 regularization on local residual peak corrections.
9. UndegradedPCAPhysicsInformedLoss:
   Loss function for PCA-based undegraded PINNs, balancing Smooth L1 reconstruction 
   with selective L2 penalization on higher-order coefficients to suppress high-frequency noise.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# =====================================================================
# 1. BICUBIC CONTROL-KNOT LOSS
# =====================================================================

class BiCubicPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for Control-Knot BiCubic Spectral Recalibration.

    Enforces data consistency using local mean flux normalization, penalizes total 
    integrated flux discrepancies, and applies a first-order difference penalty 
    on predicted correction profiles R(lambda).

    Parameters
    ----------
    lambda_flux : float, default=1.0
        Weighting coefficient for the integrated flux conservation penalty.
    lambda_smooth : float, default=1e-3
        Weighting coefficient for the first-derivative smoothness penalty.
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
        Compute the total physics-informed loss for a given forward pass.

        Parameters
        ----------
        model : nn.Module
            The neural network model being trained.
        X : torch.Tensor
            The input spectra tensor (e.g., original Gaia spectra).
        Y : torch.Tensor
            The target ground-truth spectra tensor (e.g., CALSPEC).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : The final aggregated weighted loss tensor.
            - loss_flux : The integrated flux conservation component.
            - loss_smooth : The smoothness regularization component.
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


# =====================================================================
# 2. GAUSSIAN / HUBER LOSS
# =====================================================================

class GaussianPhysicsInformedLoss(nn.Module):
    """
    Multi-Component Relative Loss Function using Robust Huber Fitting.

    Combines Huber (Smooth L1) data consistency, relative integrated flux 
    conservation, and shape-preserving Total Variation (TV) + curvature penalties.

    Attributes
    ----------
    huber : nn.SmoothL1Loss
        The PyTorch Smooth L1 loss function used for the data consistency term.
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
        Compute the multi-component physics-informed loss.

        Parameters
        ----------
        model : nn.Module
            The neural network model predicting the recalibration factor.
        X : torch.Tensor
            The input uncalibrated spectra.
        Y : torch.Tensor
            The reference ground-truth spectra.
        Sigma : torch.Tensor, optional
            The associated uncertainty/variance of the input data. 
            If None, the target Y is used as the scaling factor. Defaults to None.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the individual loss components:
            - loss_data : Relative data consistency loss (Huber).
            - loss_flux : Relative integrated flux conservation loss.
            - loss_grad : Combined shape preservation gradient/curvature loss.
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
        loss_tv = torch.mean(torch.abs(d1_R))  # Use L1 (abs) or L2 (**2) norm
        
        # 2. Original curvature penalty (second derivative)
        d2_R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        loss_curv = torch.mean(d2_R ** 2)

        # Multipliers (adjust lambda_tv if the resulting curve becomes too flat)
        lambda_tv = 5.0  
        lambda_curv = 1.0

        loss_grad = (loss_tv * lambda_tv) + (loss_curv * lambda_curv)

        return loss_data, loss_flux, loss_grad


# =====================================================================
# 3. STANDARD PCA LOSS
# =====================================================================

class PCAPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for Standard PCA-based Spectral Recalibration.
    
    Key Features:
    1. Relative Error: Evaluates percentage error to prevent vanishing gradients 
       caused by absolute physical flux scales (~1e-14).
    2. Interface Compatibility: Returns the `(total_loss, loss_flux, loss_smooth)` 
       tuple required by training pipeline orchestrators.
    3. Frozen Buffers: Keeps `pca_basis` and `pca_mean` synchronized across CPU/GPU 
       devices without participating in gradient backpropagation.

    Parameters
    ----------
    pca_basis : Union[torch.Tensor, np.ndarray]
        PCA component basis matrix of shape `(num_components, input_dim)`.
    pca_mean : Union[torch.Tensor, np.ndarray]
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

        Parameters
        ----------
        model : nn.Module
            The PCASpectralPINN model instance.
        x_input : torch.Tensor
            Observed Gaia BP/RP spectra tensor of shape `(batch_size, input_dim)`.
        y_target : torch.Tensor
            CALSPEC reference spectra tensor of shape `(batch_size, input_dim)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : Relative flux MSE loss tensor.
            - loss_flux : Relative percentage error component.
            - loss_smooth : Zero tensor placeholder maintaining training interface compatibility.
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


# =====================================================================
# 4. PARAMETER-CONDITIONED PCA LOSS & VARIANTS
# =====================================================================

class PCAParamsPhysicsInformedLoss(nn.Module):
    """
    Parameter-Conditioned PCA Physics-Informed Loss Function.

    Evaluates relative L1 flux error, enforces integrated total flux scaling anchors, 
    and applies second-derivative curvature smoothness penalties.

    Parameters
    ----------
    lambda_smooth : float, default=10.0
        Weighting coefficient for the second-derivative curvature penalty.
    lambda_reg : float, default=1e-4
        Regularization weight parameter.
    **kwargs
        Additional keyword arguments for backward compatibility.
    """
    def __init__(
        self, 
        lambda_smooth: float = 10.0, 
        lambda_reg: float = 1e-4, 
        **kwargs
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor, 
        params: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total physics-informed loss for parameter-conditioned models.

        Parameters
        ----------
        model : nn.Module
            Model predicting PCA expansion coefficients.
        x_input : torch.Tensor
            Observed input spectra tensor of shape `(batch_size, input_dim)`.
        y_target : torch.Tensor
            Reference CALSPEC spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Auxiliary stellar parameter features of shape `(batch_size, num_params)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : Aggregated loss value.
            - loss_flux : Relative L1 spectral flux error.
            - loss_smooth : Second-derivative curvature smoothness penalty.
        """
        pca_coeffs = model(x_input, params)
        R_pred = model.reconstruct(pca_coeffs)
        
        calibrated_spec = x_input * R_pred
        
        # 1. Spectral loss (Relative L1 norm to suppress outlier sensitivity near absorption lines)
        loss_flux = torch.mean(torch.abs(calibrated_spec - y_target) / (y_target + 1e-12))
        
        # 2. SCALE ANCHOR (Prevents vertical intensity collapse)
        # Forces total integrated flux prediction to match target
        area_pred = torch.sum(calibrated_spec, dim=1)
        area_target = torch.sum(y_target, dim=1)
        loss_anchor = torch.mean(torch.abs(area_pred - area_target) / (area_target + 1e-12))
        
        # 3. Curvature smoothing (Strict L1 norm on second derivative)
        d2_R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        loss_smooth = torch.mean(torch.abs(d2_R))
        
        # Total: Apply strong weight multiplier (2.0) to scale anchor to ensure flux conservation
        total_loss = loss_flux + (2.0 * loss_anchor) + (self.lambda_smooth * loss_smooth)
        
        return total_loss, loss_flux, loss_smooth


class SobolevPINNLoss(nn.Module):
    """
    Sobolev-Space Physics-Informed Loss Function.

    Combines weighted Huber flux reconstruction loss, first-derivative (dY/dlambda) 
    spectral line alignment in Sobolev space, and second-derivative response curvature 
    regularization. Includes edge masking to de-weight boundary noise.

    Parameters
    ----------
    num_bins : int, default=343
        Number of spectral wavelength channels.
    lambda_smooth : float, default=0.05
        Weight coefficient for instrumental response curvature regularization.
    lambda_deriv : float, default=0.1
        Weight coefficient for the Sobolev first-derivative alignment term.
    huber_delta : float, default=0.02
        Threshold delta parameter for the Huber loss function.
    device : Union[str, torch.device], default='cpu'
        Target computational device for tensor allocation.
    """
    def __init__(
        self, 
        num_bins: int = 343, 
        lambda_smooth: float = 0.05, 
        lambda_deriv: float = 0.1, 
        huber_delta: float = 0.02, 
        device: Union[str, torch.device] = 'cpu'
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_deriv = lambda_deriv
        self.huber_delta = huber_delta
        
        wavelengths = torch.linspace(336.0, 1020.0, num_bins).to(device)
        self.edge_mask = torch.ones(1, num_bins).to(device)
        self.edge_mask[:, wavelengths < 380.0] = 0.3
        self.edge_mask[:, wavelengths > 950.0] = 0.3

    def forward(
        self, 
        model: nn.Module, 
        X: torch.Tensor, 
        Y_true: torch.Tensor, 
        params: Optional[torch.Tensor] = None, 
        x_err: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Sobolev loss terms for spectral recalibration.

        Parameters
        ----------
        model : nn.Module
            PINN model predicting recalibration parameters.
        X : torch.Tensor
            Uncalibrated input flux tensor of shape `(batch_size, input_dim)`.
        Y_true : torch.Tensor
            Target reference flux tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Auxiliary stellar parameters tensor.
        x_err : torch.Tensor, optional
            Measurement flux uncertainties for weighted reconstruction.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - total_loss : Aggregated Sobolev loss tensor.
            - recon_loss : Weighted Huber flux reconstruction error.
            - smooth_loss : Response curve curvature penalty.
        """
        scale = torch.max(X, dim=1, keepdim=True)[0] + 1e-12
        X_scaled = X / scale
        Y_true_scaled = Y_true / scale
        
        coeffs = model(X, params)
        R_pred = model.reconstruct(coeffs)
        Y_pred_scaled = X_scaled * R_pred

        # 1. Flux loss (Huber + optional error weighting)
        huber = F.huber_loss(Y_pred_scaled, Y_true_scaled, reduction='none', delta=self.huber_delta)
        if x_err is not None:
            rel_err = x_err / scale
            weights = 1.0 / (rel_err ** 2 + 1e-4)
            weights = weights / torch.mean(weights, dim=1, keepdim=True)
            recon_loss = torch.mean(huber * weights * self.edge_mask)
        else:
            recon_loss = torch.mean(huber * self.edge_mask)

        # 2. Sobolev Term: First-derivative flux alignment (dY/dlambda)
        dY_pred = Y_pred_scaled[:, 1:] - Y_pred_scaled[:, :-1]
        dY_true = Y_true_scaled[:, 1:] - Y_true_scaled[:, :-1]
        loss_deriv = F.l1_loss(dY_pred, dY_true)

        # 3. Instrumental response curve R(lambda) smoothing
        d2R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        smooth_loss = torch.mean(d2R ** 2)

        total_loss = recon_loss + (self.lambda_deriv * loss_deriv) + (self.lambda_smooth * smooth_loss)
        return total_loss, recon_loss, smooth_loss


class AdaptivePhysicsLoss(nn.Module):
    """
    Adaptive Physics-Informed Loss Function with Spectral Domain Weighting.

    Applies wavelength-dependent weighting factors to emphasize extreme UV (< 450 nm) 
    and IR (> 900 nm) spectral domains while penalizing curvature in R(lambda).

    Parameters
    ----------
    wavelengths : Union[np.ndarray, torch.Tensor], optional
        Wavelength array of shape `(input_dim,)`. If provided, UV/IR region weighting is applied.
    lambda_smooth : float, default=1e-1
        Weighting factor for the second-derivative curvature penalty.
    """
    def __init__(
        self, 
        wavelengths: Optional[Union[np.ndarray, torch.Tensor]] = None, 
        lambda_smooth: float = 1e-1
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        
        # Wavelength-dependent weighting (increases relevance in UV and IR regimes)
        if wavelengths is not None:
            if not isinstance(wavelengths, torch.Tensor):
                wl = torch.tensor(wavelengths, dtype=torch.float32)
            else:
                wl = wavelengths.float()
            weights = torch.ones_like(wl)
            weights[wl < 450.0] = 3.0   # Triple weight in UV/Blue band (< 450 nm)
            weights[wl > 900.0] = 2.0   # Double weight in IR band (> 900 nm)
            self.register_buffer("wl_weights", weights.unsqueeze(0))
        else:
            self.wl_weights = 1.0

    def forward(
        self, 
        model: nn.Module, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        params: Optional[torch.Tensor] = None, 
        x_err: Optional[torch.Tensor] = None,
        X_smooth: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adaptive physics loss with domain weighting.

        Parameters
        ----------
        model : nn.Module
            PINN model instance.
        X : torch.Tensor
            Uncalibrated spectra tensor of shape `(batch_size, input_dim)`.
        Y : torch.Tensor
            Ground-truth target spectra of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Stellar parameter inputs.
        x_err : torch.Tensor, optional
            Flux uncertainty estimates.
        X_smooth : torch.Tensor, optional
            Pre-smoothed baseline input to avoid propagating Gaia wiggles.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - total_loss : Combined adaptive loss value.
            - loss_flux : Weighted relative flux loss.
            - loss_smooth : Curvature regularization penalty.
        """
        # 1. Forward pass: predict PCA expansion coefficients
        coeffs = model(X, params)
        
        # 2. Reconstruct multiplicative factor R(lambda)
        R = model.reconstruct(coeffs)
        
        # 3. Apply to X_smooth (if provided) to prevent propagating Gaia spectral wiggles
        X_base = X_smooth if X_smooth is not None else X
        Y_pred = X_base * R
        
        # 4. Weighted relative flux loss
        if x_err is not None:
            weights = self.wl_weights / (x_err + 1e-8)
        else:
            weights = self.wl_weights / (Y + 1e-8)
            
        loss_flux = torch.mean(weights * torch.abs(Y_pred - Y))
        
        # 5. Second-derivative curvature regularization to enforce smooth R(lambda)
        d2R = R[:, 2:] - 2 * R[:, 1:-1] + R[:, :-2]
        loss_smooth = torch.mean(d2R ** 2)
        
        total_loss = loss_flux + self.lambda_smooth * loss_smooth
        return total_loss, loss_flux, loss_smooth


class UltimateSpectralPINNLoss(nn.Module):
    r"""
    Comprehensive Multi-Objective Physics-Informed Loss Function.

    Combines Huber flux reconstruction, Sobolev spectral derivative alignment, 
    bolometric integrated flux anchoring, and second-derivative response curve 
    curvature regularization into a unified training objective.

    Physical Guarantees:
    - The Sobolev term ensures local spectral lines are crisply reconstructed.
    - The Bolometric Anchor ensures globally that the star does not become 
      artificially brighter or fainter upon calibration.
    - The Curvature Penalty ($L_2$ norm on 2nd derivative of $R$) enforces 
      instrumental physics: throughput corrections vary smoothly across wavelengths.

    Parameters
    ----------
    num_bins : int, default=343
        Number of spectral wavelength channels.
    lambda_smooth : float, default=0.05
        Weight factor for the second-derivative curvature penalty.
    lambda_deriv : float, default=0.1
        Weight factor for Sobolev first-derivative line alignment.
    lambda_anchor : float, default=0.5
        Weight factor for integrated bolometric flux anchoring.
    huber_delta : float, default=0.02
        Threshold delta parameter for Huber loss calculation.
    device : Union[str, torch.device], default='cpu'
        Target computational device.
    """
    def __init__(
        self, 
        num_bins: int = 343, 
        lambda_smooth: float = 0.05, 
        lambda_deriv: float = 0.1, 
        lambda_anchor: float = 0.5, 
        huber_delta: float = 0.02, 
        device: Union[str, torch.device] = 'cpu'
    ) -> None:
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_deriv = lambda_deriv
        self.lambda_anchor = lambda_anchor
        self.huber_delta = huber_delta
        self.num_bins = num_bins
        
        wavelengths = torch.linspace(336.0, 1020.0, num_bins).to(device)
        self.edge_mask = torch.ones(1, num_bins).to(device)
        self.edge_mask[:, wavelengths < 380.0] = 0.3
        self.edge_mask[:, wavelengths > 950.0] = 0.3

    def forward(
        self, 
        model: nn.Module, 
        X: torch.Tensor, 
        Y_true: torch.Tensor, 
        params: Optional[torch.Tensor] = None, 
        x_err: Optional[torch.Tensor] = None, 
        *args, 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute multi-objective physics-informed loss.

        Parameters
        ----------
        model : nn.Module
            PINN model predicting spectral calibration factors.
        X : torch.Tensor
            Uncalibrated spectra tensor of shape `(batch_size, input_dim)`.
        Y_true : torch.Tensor
            Reference CALSPEC spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Auxiliary stellar parameters tensor.
        x_err : torch.Tensor, optional
            Flux measurement error uncertainties.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - total_loss : Aggregated loss value.
            - recon_loss : Weighted Huber flux reconstruction error.
            - loss_deriv : Sobolev first-derivative alignment loss.
            - loss_anchor : Bolometric integrated flux anchor loss.
            - smooth_loss : Response curve curvature penalty.
        """
        scale = torch.max(X, dim=1, keepdim=True)[0] + 1e-22
        X_scaled = X / scale
        Y_true_scaled = Y_true / scale
        
        coeffs = model(X_scaled, params)
        
        # KEY CORRECTION: Compute R_pred directly from PCA basis to prevent zero-division instabilities.
        log_delta = model.pca_mean_log + torch.matmul(coeffs, model.pca_basis)
        log_delta = torch.clamp(log_delta, min=-2.0, max=2.0)  # Clamped to match inference bounds
        R_pred = torch.exp(log_delta)
        
        Y_pred = X * R_pred
        Y_pred_scaled = Y_pred / scale

        huber = F.huber_loss(Y_pred_scaled, Y_true_scaled, reduction='none', delta=self.huber_delta)
        if x_err is not None:
            rel_err = x_err / scale
            weights = 1.0 / (rel_err ** 2 + 1e-4)
            weights = weights / torch.mean(weights, dim=1, keepdim=True)
            recon_loss = torch.mean(huber * weights * self.edge_mask)
        else:
            recon_loss = torch.mean(huber * self.edge_mask)

        dY_pred = Y_pred_scaled[:, 1:] - Y_pred_scaled[:, :-1]
        dY_true = Y_true_scaled[:, 1:] - Y_true_scaled[:, :-1]
        loss_deriv = torch.mean(torch.abs(dY_pred - dY_true) * self.edge_mask[:, :-1])

        area_pred = torch.sum(Y_pred_scaled, dim=1)
        area_true = torch.sum(Y_true_scaled, dim=1)
        loss_anchor = F.huber_loss(area_pred, area_true, reduction='mean', delta=self.huber_delta * self.num_bins)

        # Use R_pred directly for second-derivative evaluation to prevent NaN propagation
        d2R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        smooth_loss = torch.mean(d2R ** 2)

        total_loss = recon_loss + (self.lambda_deriv * loss_deriv) + (self.lambda_anchor * loss_anchor) + (self.lambda_smooth * smooth_loss)
        
        return total_loss, recon_loss, loss_deriv, loss_anchor, smooth_loss


# =====================================================================
# 5. DUAL-BRANCH CHEBYSHEV LOSS (UNDEGRADED)
# =====================================================================

class UndegradedChebyshevPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for Dual-Branch Undegraded PINNs.

    Evaluates Smooth L1 reconstruction error on median-scaled flux, enforces 
    continuum curvature penalties strictly on the Chebyshev global branch, and applies 
    L1 regularization to local residual peak outputs to prevent continuum distortion.

    Parameters
    ----------
    curvature_weight : float, default=0.1
        Weight factor for the Chebyshev global continuum curvature penalty.
    local_reg_weight : float, default=0.005
        Weight factor for the L1 norm regularization on local residual peak corrections.
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
        y_target: torch.Tensor,
        params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute multi-component loss for dual-branch (Chebyshev + local residual) PINNs.

        Parameters
        ----------
        model : nn.Module
            Undegraded dual-branch PINN model instance.
        x_input : torch.Tensor
            Uncalibrated spectra tensor of shape `(batch_size, input_dim)`.
        y_target : torch.Tensor
            CALSPEC reference spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Auxiliary stellar parameters tensor of shape `(batch_size, num_params)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : Aggregated multi-objective loss value.
            - loss_flux : Smooth L1 scaled flux reconstruction error.
            - loss_curvature : Curvature penalty evaluated strictly on the global branch.
        """
        out = model(x_input, params)
        global_coeffs, local_residual = out
        
        R_total = model.reconstruct(out)
        calibrated_spec = x_input * R_total

        # Robust median scaling for normalization
        scale_y = torch.median(y_target, dim=1, keepdim=True)[0] + 1e-30
        y_norm = y_target / scale_y
        cal_norm = calibrated_spec / scale_y

        # Data Consistency: Smooth L1 Loss to capture high-frequency line details
        loss_flux = F.smooth_l1_loss(cal_norm, y_norm, beta=0.05)

        # Curvature is evaluated exclusively on the Chebyshev global continuum branch
        R_global = torch.matmul(global_coeffs, model.cheb_basis)
        d2R = R_global[:, 2:] - 2 * R_global[:, 1:-1] + R_global[:, :-2]
        loss_curvature = torch.mean(d2R ** 2)

        # L1 Regularization to prevent local peak corrections from warping the continuum baseline
        loss_local_reg = torch.mean(torch.abs(local_residual))

        total_loss = loss_flux + (self.curvature_weight * loss_curvature) + (self.local_reg_weight * loss_local_reg)

        return total_loss, loss_flux, loss_curvature


# =====================================================================
# 6. PCA UNDEGRADED LOSS
# =====================================================================

class UndegradedPCAPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function for PCA-based Undegraded PINNs.

    Evaluates the Smooth L1 reconstruction error on quantile-scaled flux. 
    Applies selective L2 regularization specifically on higher-order PCA coefficients 
    to suppress high-frequency noise injection, allowing the mean component and first 
    N basis functions to drive the core calibration.

    Parameters
    ----------
    lambda_reg : float, default=1e-4
        Regularization weight for higher-order PCA coefficients.
    unpenalized_comps : int, default=4
        Number of initial PCA components left unpenalized (preserving physical continuum/low-order trends).
    """
    def __init__(self, lambda_reg: float = 1e-4, unpenalized_comps: int = 4) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.unpenalized_comps = unpenalized_comps

    def forward(
        self, 
        model: nn.Module, 
        x_input: torch.Tensor, 
        y_target: torch.Tensor,
        params: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss for undegraded PCA recalibration models.

        Parameters
        ----------
        model : nn.Module
            Undegraded PCA PINN model instance.
        x_input : torch.Tensor
            Observed input spectra tensor of shape `(batch_size, input_dim)`.
        y_target : torch.Tensor
            CALSPEC reference spectra tensor of shape `(batch_size, input_dim)`.
        params : torch.Tensor, optional
            Auxiliary stellar parameters tensor of shape `(batch_size, num_params)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - total_loss : Aggregated loss value.
            - loss_flux : Smooth L1 scaled flux reconstruction error.
            - loss_reg : Coefficient regularization penalty on higher-order components.
        """
        # 1. Predict PCA coefficients and reconstruct high-res R(lambda)
        coeffs = model(x_input, params)
        R_total = model.reconstruct(coeffs)
        
        # 2. Apply multiplicative correction
        calibrated_spec = x_input * R_total
        
        # 3. Stable Scaling using 99th percentile to prevent outlier instabilities
        scale_y = torch.quantile(y_target, 0.99, dim=1, keepdim=True) + 1e-15
        y_norm = y_target / scale_y
        cal_norm = calibrated_spec / scale_y

        # 4. Flux Reconstruction Error (Smooth L1)
        loss_flux = F.smooth_l1_loss(cal_norm, y_norm, beta=0.05)

        # 5. Selective L2 Regularization 
        # Penalize higher-order components (noise), leaving the first N components (continuum/physics) free
        if coeffs.shape[1] > self.unpenalized_comps:
            loss_reg = torch.mean(coeffs[:, self.unpenalized_comps:] ** 2)
        else:
            loss_reg = torch.tensor(0.0, device=coeffs.device)

        # 6. Combined Total Loss
        total_loss = loss_flux + (self.lambda_reg * loss_reg)

        return total_loss, loss_flux, loss_reg