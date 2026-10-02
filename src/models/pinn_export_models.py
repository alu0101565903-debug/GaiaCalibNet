"""Provide PyTorch wrappers for model export and inference.

The wrappers combine calibration models or expose precomputed cross-validation
predictions through a consistent inference interface.

Classes
-------
- ``EnsembleExportModel`` combines a base PINN and blue-region correction model.
- ``DirectExportModel`` exposes aggregated LOOCV predictions.

Notes
-----
Report context: Section 4.6, Model Export and Inference Wrappers.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class EnsembleExportModel(nn.Module):
    """
    Ensemble export wrapper combining global PINN predictions with fine blue-region corrections.

    Parameters
    ----------
    base_model : torch.nn.Module
        Base PINN model predicting global continuum calibration coefficients.
    blue_model : torch.nn.Module
        Specialized model predicting pixel-wise blue-region correction ratios.
    blue_mask : numpy.ndarray or torch.Tensor
        Boolean mask indicating the blue spectral channel region.
    wave_grid : numpy.ndarray or torch.Tensor
        Wavelength evaluation grid in nanometers.
    """

    def __init__(
        self, 
        base_model: nn.Module, 
        blue_model: nn.Module, 
        blue_mask: Union[np.ndarray, torch.Tensor], 
        wave_grid: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Initialize the ensemble export model.

        Parameters
        ----------
        base_model : torch.nn.Module
            Base global calibration model.
        blue_model : torch.nn.Module
            Blue-region wiggle correction model.
        blue_mask : numpy.ndarray or torch.Tensor
            Boolean mask for blue spectral channels.
        wave_grid : numpy.ndarray or torch.Tensor
            Evaluation wavelength grid array.
        """
        super().__init__()
        self.base_model = base_model
        self.blue_model = blue_model
        self.register_buffer("blue_mask", torch.tensor(blue_mask, dtype=torch.bool))

        # Smooth transition mask (optional, to blend the boundary around 450 nm)
        transition_weights = torch.ones_like(torch.tensor(wave_grid, dtype=torch.float32))
        # Logistic or linear decay can be applied around the cutoff pixel index

    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass producing composite calibration ratios.

        Parameters
        ----------
        x : torch.Tensor
            Input spectral flux tensor of shape (batch_size, input_dim).
        params : torch.Tensor, optional
            Auxiliary stellar physical parameters tensor of shape (batch_size, num_params).

        Returns
        -------
        torch.Tensor
            Fused calibration ratio R(lambda) of shape (batch_size, input_dim).
        """
        # 1. Predict the global continuum.
        coeffs = self.base_model(x, params)
        R_base = self.base_model.reconstruct(coeffs)

        # 2. Predict the fine-scale blue-region correction.
        x_blue = x[:, self.blue_mask]
        R_blue_fine = self.blue_model(x_blue, params)

        # 3. Combine the spectral predictions.
        R_final = R_base.clone()
        R_final[:, self.blue_mask] = R_base[:, self.blue_mask] * R_blue_fine

        return R_final


class DirectExportModel(nn.Module):
    """
    Export wrapper returning precomputed aggregated LOOCV predictions.

    Maintains pipeline interface compatibility during evaluation and export steps.

    Parameters
    ----------
    target_device : torch.device
        PyTorch execution device (CPU/GPU) where reference tensors reside.
    R_pred_ref : numpy.ndarray
        Precomputed reference calibration ratios array.
    """

    def __init__(self, target_device: torch.device, R_pred_ref: np.ndarray) -> None:
        """
        Initialize DirectExportModel with precomputed reference values.

        Parameters
        ----------
        target_device : torch.device
            PyTorch target execution device.
        R_pred_ref : numpy.ndarray
            Precomputed reference calibration ratio array.
        """
        super().__init__()
        self.target_device = target_device
        self.register_buffer(
            "R_pred_ref", 
            torch.tensor(R_pred_ref, dtype=torch.float32, device=target_device)
        )
        self.dummy_param = nn.Parameter(torch.empty(0, device=target_device))

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning precomputed reference predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input flux tensor (unused, maintained for interface compatibility).
        params : torch.Tensor, optional
            Auxiliary physical parameters tensor (unused).

        Returns
        -------
        torch.Tensor
            Precomputed calibration ratio reference tensor.
        """
        return self.R_pred_ref

    def reconstruct(self, coeffs_or_R: torch.Tensor) -> torch.Tensor:
        """
        Identity reconstruction method for interface compatibility.

        Parameters
        ----------
        coeffs_or_R : torch.Tensor
            Input coefficients or calibration ratio tensor.

        Returns
        -------
        torch.Tensor
            Unmodified input tensor.
        """
        return coeffs_or_R