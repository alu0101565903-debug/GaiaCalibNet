"""
Module: losses.py
Report Section: 6.5 Physics-Informed Loss Function (L_flux & L_grad)

Refactored with C^infinity Log-Cosh loss for L-BFGS stability, 
Trapezoidal Integration for flux conservation, and 2nd-order Central 
Differences for spectral gradient matching.
"""

import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss (PIL) engine optimized for L-BFGS second-order optimization.
    
    Combines a smooth Log-Cosh baseline with two astrophysical regularizations:
    - L_flux: Total Integrated Flux Conservation via Trapezoidal Integration.
    - L_grad: Local Shape/Gradient Preservation via 2nd-Order Central Differences.
    """
    def __init__(self, wavelength_grid: torch.Tensor, alpha: float = 0.1, beta: float = 0.1, scale_logcosh: float = 1.0):
        """
        Args:
        - wavelength_grid (torch.Tensor): Homogeneous wavelength vector (Lambda).
        - alpha (float): Weight factor for flux conservation (L_flux).
        - beta (float): Weight factor for line gradient preservation (L_grad).
        - scale_logcosh (float): Scaling factor to control the transition to linear loss.
        """
        super().__init__()
        
        self.register_buffer("wavelength_grid", wavelength_grid.detach().clone())
        self.alpha = alpha
        self.beta = beta
        self.scale_logcosh = scale_logcosh
        
        # Calculate Delta Lambda step (dl is constant for a homogeneous grid)
        dl = (wavelength_grid[-1] - wavelength_grid[0]) / (len(wavelength_grid) - 1)
        self.register_buffer("dl", dl)
        
        # 1. TRAPEZOIDAL INTEGRATION WEIGHTS: [0.5 * dl, dl, dl, ..., dl, 0.5 * dl]
        integration_weights = torch.ones_like(wavelength_grid)
        integration_weights[0] = 0.5
        integration_weights[-1] = 0.5
        integration_weights = integration_weights * dl
        self.register_buffer("integration_weights", integration_weights)

    def compute_decoupled_components(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Computes the three physical loss terms in an unweighted state.
        Guarantees C^infinity continuity for all components.
        """
        # 1. Log-Cosh Loss: Smooth, robust alternative to Huber for L-BFGS stability
        diff = y_pred - y_true
        # Log-cosh formula: log(cosh(x)) = x - log(2) + log(1 + exp(-2x)) for numerical stability
        # We apply scaling to allow a parameterized transition from MSE to MAE
        scaled_diff = diff / self.scale_logcosh
        loss_huber_smooth = torch.mean(
            self.scale_logcosh * (scaled_diff + torch.nn.functional.softplus(-2.0 * scaled_diff) - torch.log(torch.tensor(2.0)))
        )
        
        # 2. Flux Conservation via Trapezoidal Integration
        # More accurate area approximation than raw Riemann sums
        integrated_flux_true = torch.sum(y_true * self.integration_weights.unsqueeze(0), dim=1)
        integrated_flux_pred = torch.sum(y_pred * self.integration_weights.unsqueeze(0), dim=1)
        
        # Relative squared error protects against scaling imbalances between stars
        loss_flux = torch.mean(((integrated_flux_true - integrated_flux_pred) / (integrated_flux_true + 1e-8)) ** 2)
        
        # 3. 2nd-Order Central Differences for physical spectral gradient matching
        # Central diff: (y_{i+1} - y_{i-1}) / 2. We omit division by 2*dl for scale stability,
        # but the physics of the slopes is fully preserved.
        grad_true = y_true[:, 2:] - y_true[:, :-2]
        grad_pred = y_pred[:, 2:] - y_pred[:, :-2]
        loss_grad = torch.mean((grad_true - grad_pred) ** 2)
        
        return loss_huber_smooth, loss_flux, loss_grad

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning composite scalar loss."""
        loss_base, loss_flux, loss_grad = self.compute_decoupled_components(y_pred, y_true)
        return loss_base + (self.alpha * loss_flux) + (self.beta * loss_grad)


# --- WRAPPERS FOR ADAPTIVE TRAINING LOOPS ---

def scaled_physics_wrapper(m, x, y, loss_inst):
    """
    Computes loss components and scales them prior to reduction for L-BFGS visibility.
    """
    base_loss, flux, grad = loss_inst.compute_decoupled_components(m(x), y)
    return base_loss, flux * loss_inst.alpha, grad * loss_inst.beta
         

def adaptive_physics_wrapper(model, X_batch, Y_batch, loss_instance: PhysicsInformedLoss):
    """
    Evaluates decoupled raw loss terms for adaptive weight updates.
    """
    Y_pred = model(X_batch)
    return loss_instance.compute_decoupled_components(Y_pred, Y_batch)