"""
Module: adp_training_strategies.py
Report Section: Chapter 7 Training Strategies & Hyperparameter Optimization

Refactored to guarantee L-BFGS second-order convergence by freezing 
adaptive weights in Stage 2 and applying out-of-graph gradient projections in Stage 1.
"""

import torch
from torch import nn
from tqdm import tqdm

def train_adam_adaptive(model, epochs, dataloader, physics_criterion, lr=0.001):
    """
    Stage 1: Adaptive training using Adam.
    Dynamically balances Data, Flux, and Gradient loss components.
    """
    device = next(model.parameters()).device
    
    # Initialization: exp(-(-2)) = exp(2) ≈ 7.38 multiplier
    log_w_data = nn.Parameter(torch.tensor(-2.0, device=device))
    log_w_flux = nn.Parameter(torch.tensor(-2.0, device=device))
    log_w_grad = nn.Parameter(torch.tensor(-2.0, device=device))
    log_weights = [log_w_data, log_w_flux, log_w_grad]

    optimizer = torch.optim.Adam(list(model.parameters()) + log_weights, lr=lr)
    
    model.train()
    pbar = tqdm(range(epochs), desc=">>> Stage 1: Adaptive Adam", unit="epoch")

    for epoch in pbar:
        epoch_loss = 0.0
        
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            
            # --- A. Compute physical loss components ---
            loss_data, loss_flux, loss_grad = physics_criterion(model, X_batch, Y_batch)

            # --- B. Adaptive Multi-Objective Equation ---
            loss_data_adaptive = (torch.exp(-log_w_data) * loss_data) + log_w_data
            loss_flux_adaptive = (torch.exp(-log_w_flux) * loss_flux) + log_w_flux
            loss_grad_adaptive = (torch.exp(-log_w_grad) * loss_grad) + log_w_grad
            
            total_loss = loss_data_adaptive + loss_flux_adaptive + loss_grad_adaptive

            if torch.isnan(total_loss):
                tqdm.write(f" [!] NaN detected at epoch {epoch}. Skipping batch step.")
                continue

            total_loss.backward()
            
            # --- C. Dual Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(log_weights, max_norm=0.1) 
            
            optimizer.step()

            # --- D. Post-Step Projection (Safe Clamping Outside Autograd Graph) ---
            with torch.no_grad():
                # Data weight can go high (up to exp(10) ≈ 22000)
                log_w_data.clamp_(-10.0, 10.0) 
                # Physics weights strictly capped at exp(5) ≈ 148 and minimum exp(-5) ≈ 0.006
                log_w_flux.clamp_(-5.0, 5.0)   
                log_w_grad.clamp_(-5.0, 5.0)

            epoch_loss += total_loss.item()

        if epoch % 10 == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            pbar.set_postfix({
                "Loss": f"{avg_epoch_loss:.4e}",
                "W_Data": f"{torch.exp(-log_w_data).item():.2f}",
                "W_Flux": f"{torch.exp(-log_w_flux).item():.2f}",
                "W_Grad": f"{torch.exp(-log_w_grad).item():.2f}"
            })

    return model, log_weights


def train_lbfgs_adaptive(model, X_full, Y_full, log_weights, physics_criterion, steps=10):
    """
    Stage 2: Refinement stage using L-BFGS.
    Optimizes network parameters ONLY on a static, pre-balanced physical landscape.
    """
    device = next(model.parameters()).device
    X_full, Y_full = X_full.to(device), Y_full.to(device)
    
    # 1. Freeze adaptive log weights (Do not pass them to the L-BFGS optimizer!)
    for lp in log_weights:
        lp.requires_grad = False
        
    # Extract static effective multipliers from Stage 1 convergence
    w_data_eff = torch.exp(-log_weights[0]).item()
    w_flux_eff = torch.exp(-log_weights[1]).item()
    w_grad_eff = torch.exp(-log_weights[2]).item()

    # L-BFGS optimizes only neural network parameters
    optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
    )

    max_iter_per_step = 20
    pbar = tqdm(total=steps * max_iter_per_step, desc=">>> Stage 2: L-BFGS Refinement", unit="it")
    
    def closure():
        optimizer.zero_grad()
        
        # Evaluate raw components
        loss_data, loss_flux, loss_grad = physics_criterion(model, X_full, Y_full)

        # Static weighted multi-objective landscape (Log-barrier constants omitted as they don't affect gradients)
        total_loss = (w_data_eff * loss_data) + (w_flux_eff * loss_flux) + (w_grad_eff * loss_grad)

        if total_loss.requires_grad:
            total_loss.backward()
        
        pbar.update(1)
        pbar.set_postfix({
            "Loss": f"{total_loss.item():.4e}",
            "Weights": f"({w_data_eff:.1f}, {w_flux_eff:.1f}, {w_grad_eff:.1f})"
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
    
    # Stage 1: Exploration and weight balancing with Adam
    model, log_params = train_adam_adaptive(
        model, 
        epochs=epochs_adam, 
        dataloader=dataloader, 
        physics_criterion=physics_criterion
    )
    
    # Stage 2: High-precision Quasi-Newton Fine-Tuning with L-BFGS
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