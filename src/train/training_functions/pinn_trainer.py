"""
Module: pinn_trainer.py
Report Section: 4.8 Cross-Validation & Model Training Strategy

This module provides the core training routine for a single cross-validation fold, 
executing a two-phase optimization procedure: primary base network training followed 
by localized blue-region wiggle correction model optimization.

Functions:
    - train_fold: Executes the sequential two-stage training loop for a single fold.
"""

from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def train_fold(
    model: nn.Module,
    blue_model: nn.Module,
    criterion_base: nn.Module,
    criterion_blue: nn.Module,
    optimizer_base: torch.optim.Optimizer,
    optimizer_blue: torch.optim.Optimizer,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    P_train: torch.Tensor,
    E_train: torch.Tensor,
    blue_mask: Union[np.ndarray, torch.Tensor],
    epochs_adam: int,
    epochs_blue: int,
    X_val: torch.Tensor = None,
    Y_val: torch.Tensor = None,
    P_val: torch.Tensor = None,
    E_val: torch.Tensor = None,
    return_history: bool = False,
) -> Union[Tuple[nn.Module, nn.Module], Tuple[nn.Module, nn.Module, Dict[str, np.ndarray]]]:
    """
    Execute a two-phase training sequence for a single cross-validation fold.

    Phase 1 optimizes the primary global PINN base model using a Cosine Annealing 
    learning rate schedule. Phase 2 freezes base predictions and trains the specialized 
    blue-region correction network on residual channel features.

    Parameters
    ----------
    model : torch.nn.Module
        Base PINN module for global continuum calibration.
    blue_model : torch.nn.Module
        Specialized network for localized blue-region wiggle correction.
    criterion_base : torch.nn.Module
        Loss function instance for base model optimization.
    criterion_blue : torch.nn.Module
        Loss function instance for blue model optimization.
    optimizer_base : torch.optim.Optimizer
        Optimizer instance for the base model.
    optimizer_blue : torch.optim.Optimizer
        Optimizer instance for the blue correction model.
    X_train : torch.Tensor
        Input flux training tensor of shape (batch_size, input_dim).
    Y_train : torch.Tensor
        Target CALSPEC flux training tensor of shape (batch_size, input_dim).
    P_train : torch.Tensor
        Auxiliary stellar physical parameters tensor of shape (batch_size, num_params).
    E_train : torch.Tensor
        Input flux uncertainty tensor of shape (batch_size, input_dim).
    blue_mask : numpy.ndarray or torch.Tensor
        Boolean mask identifying blue spectral channels.
    epochs_adam : int
        Number of training epochs for Phase 1 (base model).
    epochs_blue : int
        Number of training epochs for Phase 2 (blue correction model).

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module]
        Tuple containing the optimized (model, blue_model) instances. When
        ``return_history`` is true, a third item contains per-epoch losses.
    """
    # --- PHASE 1: Train Primary Base Model ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, T_max=epochs_adam)
    model.train()

    history = {
        "base_train": [],
        "base_val": [],
        "blue_train": [],
        "blue_val": [],
    }

    for _ in range(epochs_adam):
        optimizer_base.zero_grad()
        loss_base = criterion_base(model, X_train, Y_train, E_train, P_train)
        loss_base.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_base.step()
        scheduler.step()

        history["base_train"].append(loss_base.detach().item())
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                history["base_val"].append(
                    criterion_base(model, X_val, Y_val, E_val, P_val).item()
                )
            model.train()

    # --- PREPARE INTERMEDIATE DATA FOR PHASE 2 ---
    model.eval()
    with torch.no_grad():
        R_base_pred = model.reconstruct(model(X_train, P_train))
        Y_pred_base = X_train * R_base_pred

        X_train_blue = X_train[:, blue_mask]
        Y_train_blue = Y_train[:, blue_mask]
        E_train_blue = E_train[:, blue_mask]
        Y_pred_base_blue = Y_pred_base[:, blue_mask]

    # --- PHASE 2: Train Specialized Blue Correction Model ---
    blue_model.train()
    for _ in range(epochs_blue):
        optimizer_blue.zero_grad()
        R_blue_fine = blue_model(X_train_blue, P_train)
        loss_blue = criterion_blue(R_blue_fine, Y_train_blue, Y_pred_base_blue, E_train_blue)
        loss_blue.backward()
        optimizer_blue.step()

        history["blue_train"].append(loss_blue.detach().item())
        if X_val is not None:
            blue_model.eval()
            with torch.no_grad():
                R_val_base = model.reconstruct(model(X_val, P_val))
                Y_val_base = X_val * R_val_base
                X_val_blue = X_val[:, blue_mask]
                Y_val_blue = Y_val[:, blue_mask]
                E_val_blue = E_val[:, blue_mask]
                Y_val_base_blue = Y_val_base[:, blue_mask]
                R_val_fine = blue_model(X_val_blue, P_val)
                history["blue_val"].append(
                    criterion_blue(R_val_fine, Y_val_blue, Y_val_base_blue, E_val_blue).item()
                )
            blue_model.train()

    if return_history:
        return model, blue_model, {key: np.asarray(value, dtype=np.float64) for key, value in history.items()}
    return model, blue_model