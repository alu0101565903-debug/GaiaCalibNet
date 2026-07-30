"""
Module: fixed_training_strategies.py
Report Section: Chapter 7 Training Strategies & Hyperparameter Optimization

This module implements the definitive 2-stage fixed training strategy 
(Adam optimizer followed by L-BFGS refinement) for Physics-Informed Neural 
Networks (PINNs). This specific pipeline is designed to operate deterministically 
without explicitly injecting standard deviation or observational uncertainty arrays 
into the loss computation, relying instead on stable, fixed physical constraints.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_adam_fixed(model: torch.nn.Module, epochs: int, dataloader: DataLoader, physics_criterion: torch.nn.Module, lr: float = 0.001) -> torch.nn.Module:
    """
    Stage 1: Stable Adam training using fixed physics constraints.

    This function performs the initial training phase using the Adam optimizer. 
    Adam provides robust, first-order gradient-based optimization to quickly 
    navigate the loss landscape and locate a viable local minimum. It applies 
    gradient clipping to prevent exploding gradients.

    Args:
    - model (torch.nn.Module): The neural network model to be trained.
    - epochs (int): The total number of training epochs for this stage.
    - dataloader (DataLoader): PyTorch DataLoader providing batched training data.
    - physics_criterion (torch.nn.Module): The custom physics-informed loss 
        function (e.g., BiCubicPhysicsInformedLoss).
    - lr (float, optional): The learning rate for the Adam optimizer. Defaults to 0.001.

    Returns:
    - torch.nn.Module: The model with updated weights after Stage 1 training.
    """

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm(range(epochs), desc=">>> Stage 1: Adam Training", unit="epoch")

    for epoch in pbar:
        epoch_loss = 0.0

        for batch in dataloader:
            # Extract only the uncalibrated spectra (X) and reference spectra (Y) from the batch
            X_batch = batch[0].to(device)
            Y_batch = batch[1].to(device)

            optimizer.zero_grad()

            # Forward pass and compute the combined physics-informed loss components
            total_loss, loss_flux, loss_smooth = physics_criterion(model, X_batch, Y_batch)

            # Robust error handling: Check for numerical instabilities
            if torch.isnan(total_loss):
                tqdm.write(f" [!] NaN detected at epoch {epoch}. Skipping batch step.")
                continue

            total_loss.backward()

            # Clip gradients to prevent gradient explosion (max_norm = 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        # Update progress bar every 10 epochs for cleaner output
        if epoch % 10 == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            pbar.set_postfix({"Total Loss": f"{avg_epoch_loss:.4e}"})

    return model

def train_lbfgs_fixed(model: torch.nn.Module, X_full: torch.Tensor, Y_full: torch.Tensor, physics_criterion: torch.nn.Module, steps: int=10)-> torch.nn.Module:
    """
    Stage 2: Refinement stage using the L-BFGS optimizer.

    Following the Adam optimization, this stage uses L-BFGS (a quasi-Newton, 
    second-order optimization algorithm) to fine-tune the network weights. 
    L-BFGS is highly effective at finding precise local minima once Adam has 
    brought the parameters into the correct basin of attraction. This stage 
    processes the entire dataset at once (full-batch) rather than in mini-batches.

    Args:
    - model (torch.nn.Module): The neural network model, pre-trained by Stage 1.
    - X_full (torch.Tensor): The complete dataset of uncalibrated spectra.
    - Y_full (torch.Tensor): The complete dataset of target reference spectra.
    - physics_criterion (torch.nn.Module): The custom physics-informed loss function.
    - steps (int, optional): The number of top-level L-BFGS optimization steps 
        to perform. Defaults to 10.

    Returns:
    - torch.nn.Module: The fully optimized and refined model.
    """
    device = next(model.parameters()).device
    X_full, Y_full = X_full.to(device), Y_full.to(device)

    # Initialize L-BFGS with standard parameters for deep learning refinement
    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=0.1, 
        max_iter=20, 
        history_size=10, 
        line_search_fn="strong_wolfe"
    )

    pbar = tqdm(total=steps, desc=">>> Stage 2: L-BFGS Refinement", unit="step")

    # Optional hook: Prepare specific model parameters for L-BFGS if required
    # (e.g., freezing certain layers or adjusting batch norm statistics)
    if hasattr(model, "prepare_for_lbfgs"):
        model.prepare_for_lbfgs()

    def closure():
        """
        Closure function required by L-BFGS to re-evaluate the loss 
        multiple times per step during the line search process.
        """
        optimizer.zero_grad()
        total_loss, _, _ = physics_criterion(model, X_full, Y_full)

        if total_loss.requires_grad:
            total_loss.backward()
            return total_loss

    for _ in range(steps):
        loss_val = optimizer.step(closure)
        pbar.update(1)
        pbar.set_postfix({"Loss": f"{loss_val.item():.4e}"})

    pbar.close()
    return model

def train_2stage_fixed(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    X_full: torch.Tensor, 
    Y_full: torch.Tensor, 
    physics_criterion: torch.nn.Module, 
    epochs_adam: int = 1000, 
    steps_lbfgs: int = 10
) -> torch.nn.Module:
    """
    Executes the complete, definitive 2-stage training pipeline (Adam + L-BFGS).

    This function acts as the primary orchestrator for the training process. 
    It sequentially invokes the robust Adam training phase followed by the 
    high-precision L-BFGS refinement phase.

    Args:
    - model (torch.nn.Module): The uninitialized neural network model.
    - dataloader (DataLoader): Batched DataLoader for Stage 1 (Adam).
    - X_full (torch.Tensor): Full uncalibrated dataset tensor for Stage 2 (L-BFGS).
    - Y_full (torch.Tensor): Full reference dataset tensor for Stage 2 (L-BFGS).
    - physics_criterion (torch.nn.Module): The loss function computing physical constraints.
    - epochs_adam (int, optional): Number of epochs for the Adam optimizer. Defaults to 1000.
    - steps_lbfgs (int, optional): Number of steps for the L-BFGS optimizer. Defaults to 10.

    Returns:
    - torch.nn.Module: The fully trained and calibrated PINN model ready for inference.
    """
    print("=" * 60)
    print("[7.1] EXECUTING DEFINITIVE 2-STAGE TRAINING PIPELINE")
    print("=" * 60)

    # Stage 1: Broad optimization using Adam
    model = train_adam_fixed(
        model, 
        epochs=epochs_adam, 
        dataloader=dataloader, 
        physics_criterion=physics_criterion,
        lr=0.001
    )

    # Stage 2: Fine-tuning refinement using L-BFGS
    model = train_lbfgs_fixed(
        model,
        X_full=X_full,
        Y_full=Y_full, 
        physics_criterion=physics_criterion,
        steps=steps_lbfgs
    )

    print("\n-> Training pipeline successfully completed.")
    return model