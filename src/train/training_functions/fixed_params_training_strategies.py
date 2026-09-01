"""
Module: fixed_params_training_strategies.py
Report Section: Chapter 7 Training Strategies & Hyperparameter Optimization

Two-Stage Physics-Informed Optimization Engine for Spectral Recalibration.

This module implements a hybrid two-stage optimization strategy for Physics-Informed 
Neural Networks (PINNs) applied to Gaia spectrophotometric recalibration. The strategy 
combines stochastic first-order gradient descent (Adam) for rapid global trajectory 
exploration with quasi-Newton second-order full-batch optimization (L-BFGS) for 
high-precision asymptotic convergence near physical minima.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_adam_fixed(
    model: nn.Module, 
    epochs: int, 
    dataloader: DataLoader, 
    physics_criterion: nn.Module, 
    lr: float = 0.001
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a Physics-Informed Neural Network using the Adam optimizer (Stage 1).

    Executes mini-batch first-order stochastic gradient descent over a specified 
    number of epochs. Flexibly handles variable-length input batch structures 
    and multi-term physics loss components while clipping gradient norms to 
    prevent exploding gradients.

    Parameters
    ----------
    model : nn.Module
        The PyTorch PINN model to be trained.
    epochs : int
        Number of full training epochs to perform.
    dataloader : DataLoader
        DataLoader providing mini-batches of training spectra, targets, and auxiliary parameters.
    physics_criterion : nn.Module
        Physics-informed loss function returning a tuple of loss components.
    lr : float, default=0.001
        Learning rate for the Adam optimizer.

    Returns
    -------
    Tuple[nn.Module, Dict[str, List[float]]]
        A tuple containing:
        - model : The trained PyTorch model instance.
        - history : Dictionary mapping loss component names to per-epoch average float values:
            - 'total_loss': Aggregated objective loss.
            - 'recon_loss': Scaled flux reconstruction error.
            - 'loss_deriv': Sobolev first-derivative alignment loss.
            - 'loss_anchor': Integrated flux anchor loss.
            - 'smooth_loss': Second-derivative response curve smoothness penalty.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm(range(epochs), desc=">>> Stage 1: Adam Training", unit="epoch")

    history: Dict[str, List[float]] = {
        "total_loss": [], 
        "recon_loss": [], 
        "loss_deriv": [], 
        "loss_anchor": [], 
        "smooth_loss": []
    }

    for epoch in pbar:
        epoch_losses = {k: 0.0 for k in history.keys()}

        for batch in dataloader:
            # Flexible batch unpacking to support optional auxiliary parameter tensors
            if len(batch) == 5:
                X_batch, Y_batch, params_batch, err_batch, X_smooth_batch = [b.to(device) for b in batch]
            elif len(batch) == 4:
                X_batch, Y_batch, params_batch, err_batch = [b.to(device) for b in batch]
                X_smooth_batch = None
            elif len(batch) == 3:
                X_batch, Y_batch, params_batch = [b.to(device) for b in batch]
                err_batch, X_smooth_batch = None, None
            else:
                X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
                params_batch, err_batch, X_smooth_batch = None, None, None

            optimizer.zero_grad()

            # Flexible unpacking for varying loss criteria returns
            losses = physics_criterion(
                model, X_batch, Y_batch, 
                params=params_batch, x_err=err_batch
            )
            
            # Map returns dynamically to avoid tuple unpacking errors
            total_loss = losses[0]
            recon_loss = losses[1] if len(losses) > 1 else torch.tensor(0.0, device=device)
            
            if len(losses) == 5:
                loss_deriv, loss_anchor, smooth_loss = losses[2], losses[3], losses[4]
            elif len(losses) == 3:
                smooth_loss = losses[2]
                loss_deriv = torch.tensor(0.0, device=device)
                loss_anchor = torch.tensor(0.0, device=device)
            else:
                loss_deriv = torch.tensor(0.0, device=device)
                loss_anchor = torch.tensor(0.0, device=device)
                smooth_loss = torch.tensor(0.0, device=device)

            if torch.isnan(total_loss):
                tqdm.write(f" [!] NaN detected at epoch {epoch}. Skipping batch step.")
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses["total_loss"] += total_loss.item()
            epoch_losses["recon_loss"] += recon_loss.item()
            epoch_losses["loss_deriv"] += loss_deriv.item()
            epoch_losses["loss_anchor"] += loss_anchor.item()
            epoch_losses["smooth_loss"] += smooth_loss.item()

        num_batches = len(dataloader)
        for k in history.keys():
            history[k].append(epoch_losses[k] / num_batches)

        if epoch % 10 == 0:
            pbar.set_postfix({"Total Loss": f"{history['total_loss'][-1]:.4e}"})

    return model, history


def train_lbfgs_fixed(
    model: nn.Module, 
    X_full: torch.Tensor, 
    Y_full: torch.Tensor, 
    params_full: Optional[torch.Tensor],
    err_full: Optional[torch.Tensor],
    X_smooth_full: Optional[torch.Tensor],
    physics_criterion: nn.Module, 
    steps: int = 10
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Refine a Physics-Informed Neural Network using L-BFGS optimization (Stage 2).

    Applies second-order quasi-Newton optimization (L-BFGS with Strong Wolfe 
    line search) on full-batch dataset tensors to converge tightly to physical local minima.

    Parameters
    ----------
    model : nn.Module
        The PyTorch PINN model pre-trained by Stage 1 (Adam).
    X_full : torch.Tensor
        Complete uncalibrated input spectra tensor of shape `(N, num_channels)`.
    Y_full : torch.Tensor
        Complete target reference spectra tensor of shape `(N, num_channels)`.
    params_full : torch.Tensor, optional
        Complete normalized stellar parameters tensor of shape `(N, num_params)`.
    err_full : torch.Tensor, optional
        Complete flux error measurement tensor of shape `(N, num_channels)`.
    X_smooth_full : torch.Tensor, optional
        Complete pre-smoothed continuous input spectra tensor.
    physics_criterion : nn.Module
        Physics-informed loss function returning a tuple of loss components.
    steps : int, default=10
        Number of full-batch L-BFGS optimization iterations to perform.

    Returns
    -------
    Tuple[nn.Module, Dict[str, List[float]]]
        A tuple containing:
        - model : The refined PyTorch model instance.
        - history : Dictionary mapping loss component names to per-step float values:
            - 'total_loss': Aggregated objective loss.
            - 'recon_loss': Scaled flux reconstruction error.
            - 'loss_deriv': Sobolev first-derivative alignment loss.
            - 'loss_anchor': Integrated flux anchor loss.
            - 'smooth_loss': Second-derivative response curve smoothness penalty.
    """
    device = next(model.parameters()).device
    X_full, Y_full = X_full.to(device), Y_full.to(device)
    if params_full is not None:
        params_full = params_full.to(device)
    if err_full is not None:
        err_full = err_full.to(device)

    model.eval()

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=20, 
        history_size=10, line_search_fn="strong_wolfe"
    )

    pbar = tqdm(total=steps, desc=">>> Stage 2: L-BFGS Refinement", unit="step")

    if hasattr(model, "prepare_for_lbfgs"):
        getattr(model, "prepare_for_lbfgs")()

    history: Dict[str, List[float]] = {
        "total_loss": [], 
        "recon_loss": [], 
        "loss_deriv": [], 
        "loss_anchor": [], 
        "smooth_loss": []
    }
    step_losses: Dict[str, float] = {}

    def closure() -> torch.Tensor:
        optimizer.zero_grad()

        losses = physics_criterion(
            model, X_full, Y_full, 
            params=params_full, x_err=err_full
        )

        total_loss = losses[0]
        recon_loss = losses[1] if len(losses) > 1 else torch.tensor(0.0, device=device)
        
        if len(losses) == 5:
            loss_deriv, loss_anchor, smooth_loss = losses[2], losses[3], losses[4]
        elif len(losses) == 3:
            smooth_loss = losses[2]
            loss_deriv = torch.tensor(0.0, device=device)
            loss_anchor = torch.tensor(0.0, device=device)
        else:
            loss_deriv = torch.tensor(0.0, device=device)
            loss_anchor = torch.tensor(0.0, device=device)
            smooth_loss = torch.tensor(0.0, device=device)

        if total_loss.requires_grad:
            total_loss.backward()

        step_losses["total_loss"] = total_loss.item()
        step_losses["recon_loss"] = recon_loss.item()
        step_losses["loss_deriv"] = loss_deriv.item()
        step_losses["loss_anchor"] = loss_anchor.item()
        step_losses["smooth_loss"] = smooth_loss.item()

        return total_loss

    for _ in range(steps):
        loss_val = optimizer.step(closure)

        for k in history.keys():
            history[k].append(step_losses[k])
        
        pbar.update(1)
        val_display = loss_val.item() if loss_val is not None else step_losses["total_loss"]
        pbar.set_postfix({"Loss": f"{val_display:.4e}"})

    pbar.close()
    return model, history


def train_2stage_fixed(
    model: nn.Module, 
    dataloader: DataLoader, 
    X_full: torch.Tensor, 
    Y_full: torch.Tensor, 
    params_full: Optional[torch.Tensor],
    err_full: Optional[torch.Tensor],
    X_smooth_full: Optional[torch.Tensor],
    physics_criterion: nn.Module, 
    epochs_adam: int = 1000, 
    steps_lbfgs: int = 10
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Orchestrate the complete two-stage PINN training pipeline (Adam + L-BFGS).

    Combines stochastic first-order optimization (Stage 1: Adam) for global trajectory 
    exploration with full-batch quasi-Newton optimization (Stage 2: L-BFGS) for high-precision 
    asymptotic parameter refinement.

    Parameters
    ----------
    model : nn.Module
        The PyTorch PINN model to be trained.
    dataloader : DataLoader
        DataLoader providing mini-batches for Stage 1 Adam optimization.
    X_full : torch.Tensor
        Full uncalibrated spectra tensor for Stage 2 L-BFGS optimization.
    Y_full : torch.Tensor
        Full ground-truth reference spectra tensor for Stage 2 L-BFGS optimization.
    params_full : torch.Tensor, optional
        Full normalized stellar parameters tensor for Stage 2 L-BFGS optimization.
    err_full : torch.Tensor, optional
        Full measurement uncertainties tensor for Stage 2 L-BFGS optimization.
    X_smooth_full : torch.Tensor, optional
        Full pre-smoothed input spectra tensor for Stage 2 L-BFGS optimization.
    physics_criterion : nn.Module
        Physics-informed loss function returning a tuple of loss terms.
    epochs_adam : int, default=1000
        Number of training epochs for Stage 1 (Adam).
    steps_lbfgs : int, default=10
        Number of optimization steps for Stage 2 (L-BFGS).

    Returns
    -------
    Tuple[nn.Module, Dict[str, List[float]]]
        A tuple containing:
        - model : The fully trained and refined PyTorch PINN model instance.
        - history_combined : Concatenated loss history dictionary containing metrics 
          across all Adam epochs and L-BFGS refinement steps.
    """
    print("=" * 60)
    print("[7.1] EXECUTING DEFINITIVE 2-STAGE TRAINING PIPELINE")
    print("=" * 60)

    model, history_adam = train_adam_fixed(
        model, epochs=epochs_adam, dataloader=dataloader, 
        physics_criterion=physics_criterion, lr=0.001
    )

    model, history_lbfgs = train_lbfgs_fixed(
        model, X_full=X_full, Y_full=Y_full, params_full=params_full,
        err_full=err_full, X_smooth_full=X_smooth_full,
        physics_criterion=physics_criterion, steps=steps_lbfgs
    )

    history_combined = {
        key: history_adam[key] + history_lbfgs[key] 
        for key in history_adam.keys()
    }

    print("\n-> Training pipeline successfully completed.")
    return model, history_combined