"""
Module: adp_training_strategies.py
Report Section: Chapter 7 Training Strategies & Hyperparameter Optimization

This module implements the robust 2-Stage Adaptive Training. It utilizes 
Self-Adaptive Weights equipped with log-barrier penalties to dynamically balance 
the competing gradients of data regression, flux conservation, and gradient preservation.
"""

import torch
from torch import nn
from tqdm import tqdm

def train_adam_adaptive(model, epochs, dataloader, physics_criterion, lr=0.001):
    """
    Stage 1: Adaptive training using Adam optimizer with stable log-barrier penalties.
    Dynamically balances the Data, Flux, and Gradient loss components.
    """
    device = next(model.parameters()).device
    
    # 1. Initialize adaptive weights for Data (Huber), Flux Conservation, and Gradient
    log_w_data = nn.Parameter(torch.tensor(0.0, device=device))
    log_w_flux = nn.Parameter(torch.tensor(0.0, device=device))
    log_w_grad = nn.Parameter(torch.tensor(0.0, device=device))
    log_weights = [log_w_data, log_w_flux, log_w_grad]

    # Combine model parameters and the dynamically learned weights in the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + log_weights, lr=lr)
    
    model.train()
    pbar = tqdm(range(epochs), desc=">>> Stage 1: Adaptive Adam", unit="epoch")

    for epoch in pbar:
        epoch_loss = 0.0
        
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()

            # --- A. Clamp weights prior to exponential (Critical for numerical stability) ---
            with torch.no_grad():
                for lw in log_weights:
                    lw.clamp_(-5.0, 5.0)

            # Transform log weights to positive scalar coefficients
            w_data, w_flux, w_grad = torch.exp(log_w_data), torch.exp(log_w_flux), torch.exp(log_w_grad)
            
            # --- B. Compute physical loss components ---
            loss_data, loss_flux, loss_grad = physics_criterion(model, X_batch, Y_batch)

            # CORRECCIÓN CRÍTICA: Añadimos el castigo logarítmico para evitar que los pesos colapsen a 0
            penalty = log_w_data + log_w_flux + log_w_grad
            total_loss = (w_data * loss_data) + (w_flux * loss_flux) + (w_grad * loss_grad) - penalty
            
            # --- C. Safety validation to prevent exploding gradients ---
            if torch.isnan(total_loss):
                tqdm.write(f" [!] NaN detected at epoch {epoch}. Skipping batch step.")
                continue

            total_loss.backward()
            
            # --- D. Dual Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(log_weights, max_norm=0.1) 
            
            optimizer.step()
            epoch_loss += total_loss.item()

        if epoch % 10 == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            pbar.set_postfix({
                "Loss": f"{avg_epoch_loss:.4e}",
                "W_Data": f"{w_data.item():.2f}",
                "W_Flux": f"{w_flux.item():.2f}",
                "W_Grad": f"{w_grad.item():.2f}"
            })

    return model, log_weights


def train_lbfgs_adaptive(model, X_full, Y_full, log_weights, physics_criterion, steps=10):
    """
    Stage 2: Refinement stage using L-BFGS optimizer for high-precision convergence.
    """
    device = next(model.parameters()).device
    X_full, Y_full = X_full.to(device), Y_full.to(device)
    
    optimizer = torch.optim.LBFGS(
        list(model.parameters()) + log_weights, 
        lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
    )

    max_iter_per_step = 20
    pbar = tqdm(total=steps * max_iter_per_step, desc=">>> Stage 2: L-BFGS Refinement", unit="it")
    
    def closure():
        # --- E. Enforce strict clamp inside closure ---
        with torch.no_grad():
            for lw in log_weights:
                lw.clamp_(-5.0, 5.0)
        
        optimizer.zero_grad()
        
        loss_data, loss_flux, loss_grad = physics_criterion(model, X_full, Y_full)

        w_data, w_flux, w_grad = torch.exp(log_weights[0]), torch.exp(log_weights[1]), torch.exp(log_weights[2])
        
        # CORRECCIÓN CRÍTICA: Mantenemos el castigo logarítmico también en la optimización de segundo orden
        penalty = log_weights[0] + log_weights[1] + log_weights[2]
        total_loss = (w_data * loss_data) + (w_flux * loss_flux) + (w_grad * loss_grad) - penalty
        
        if total_loss.requires_grad:
            total_loss.backward()
        
        pbar.update(1)
        pbar.set_postfix({
            "Loss": f"{total_loss.item():.4e}",
            "Weights": f"({w_data.item():.1f}, {w_flux.item():.1f}, {w_grad.item():.1f})"
        })
        return total_loss
    
    model.train()
    for _ in range(steps):
        optimizer.step(closure)

    pbar.close()    
    return model


def train_2stage_adaptive(model, dataloader, X_full, Y_full, physics_criterion, epochs_adam=1000, steps_lbfgs=10):
    """
    Executes the complete 2-stage adaptive training pipeline.
    """
    print("=" * 60)
    print("[7.1] EXECUTING 2-STAGE ADAPTIVE TRAINING PIPELINE")
    print("=" * 60)
    
    # Stage 1: Adam
    model, log_params = train_adam_adaptive(
        model, 
        epochs=epochs_adam, 
        dataloader=dataloader, 
        physics_criterion=physics_criterion
    )
    
    # Stage 2: L-BFGS
    model = train_lbfgs_adaptive(
        model, 
        X_full=X_full, 
        Y_full=Y_full, 
        log_weights=log_params, 
        physics_criterion=physics_criterion, 
        steps=steps_lbfgs
    )
    
    print("\n-> Training pipeline successfully completed.")
    return model