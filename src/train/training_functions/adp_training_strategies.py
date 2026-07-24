"""
Module: adp_training_strategies.py
Report Section: Chapter 7 Training Strategies & Hyperparameter Optimization

This module provides the 2-stage adaptive training strategy used by the PINN
pipeline. It first trains the model with Adam while learning task-specific
uncertainty weights, then refines the solution with a deterministic L-BFGS
optimizer.

The adaptive scheme dynamically balances the competing gradients of:
- data regression,
- flux conservation,
- shape/gradient preservation.

This design enables robust convergence for physics-informed spectral
recalibration models.
"""

import torch
from torch import nn
from tqdm import tqdm

def train_adam_adaptive(model, epochs, dataloader, physics_criterion, lr=0.001):
    """
    Stage 1: Adaptive training using Adam optimizer with mathematically rigorous
    homoscedastic uncertainty weighting (exp(-s) * L + s).
    """
    device = next(model.parameters()).device
    
    # 1. Initialize adaptive log-variance parameters (s = log(sigma^2))
    # Starting at 0.0 means initial weight exp(-0) = 1.0
    log_var_data = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    log_var_flux = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    log_var_grad = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    log_vars = [log_var_data, log_var_flux, log_var_grad]

    # Combine model parameters and the dynamically learned weights in the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + log_vars, lr=lr)
    
    model.train()
    pbar = tqdm(range(epochs), desc=">>> Stage 1: Adaptive Adam", unit="epoch")

    for epoch in pbar:
        epoch_loss = 0.0
        
        for batch in dataloader:
            # Support optional noise variance tensors when the dataloader provides them.
            if len(batch) == 3:
                X_batch, Y_batch, Sigma_batch = [b.to(device) for b in batch]
            else:
                X_batch, Y_batch = batch[0].to(device), batch[1].to(device)
                Sigma_batch = None

            optimizer.zero_grad()

            # --- A. Clamp log-variances to ensure numerical stability ---
            with torch.no_grad():
                for lv in log_vars:
                    lv.clamp_(-5.0, 5.0)

            # --- B. Compute physical loss components ---
            loss_data, loss_flux, loss_grad = physics_criterion(model, X_batch, Y_batch, Sigma_batch)

            # --- C. Rigorous Multi-Task Uncertainty Formulation (Kendall et al.) ---
            # w = exp(-s), penalty = s
            w_data = torch.exp(-log_var_data)
            w_flux = torch.exp(-log_var_flux)
            w_grad = torch.exp(-log_var_grad)
            
            total_loss = (w_data * loss_data + log_var_data) + \
                         (w_flux * loss_flux + log_var_flux) + \
                         (w_grad * loss_grad + log_var_grad)
            
            # --- D. Safety validation ---
            if torch.isnan(total_loss):
                tqdm.write(f" [!] NaN detected at epoch {epoch}. Skipping batch step.")
                continue

            total_loss.backward()
            
            # --- E. Dual Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(log_vars, max_norm=0.1) 
            
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

    return model, log_vars


def train_lbfgs_adaptive(model, X_full, Y_full, log_vars, physics_criterion, Sigma_full=None, steps=10):
    """
    Stage 2: Refinement stage using L-BFGS optimizer for high-precision convergence.
    """
    device = next(model.parameters()).device
    X_full, Y_full = X_full.to(device), Y_full.to(device)
    if Sigma_full is not None:
        Sigma_full = Sigma_full.to(device)
    
    optimizer = torch.optim.LBFGS(
        list(model.parameters()) + log_vars, 
        lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
    )

    pbar = tqdm(total=steps, desc=">>> Stage 2: L-BFGS Refinement", unit="step")
    
    # Critical improvement: disable stochastic behavior (Dropout) so that
    # the deterministic L-BFGS optimizer can compute consistent Hessians.
    model.prepare_for_lbfgs()
    
    def closure():
        with torch.no_grad():
            for lv in log_vars:
                lv.clamp_(-5.0, 5.0)
        
        optimizer.zero_grad()
        
        loss_data, loss_flux, loss_grad = physics_criterion(model, X_full, Y_full, Sigma_full)

        # Use the same uncertainty-weighted multi-task formulation as in Adam.
        w_data = torch.exp(-log_vars[0])
        w_flux = torch.exp(-log_vars[1])
        w_grad = torch.exp(-log_vars[2])
        
        total_loss = (w_data * loss_data + log_vars[0]) + \
                     (w_flux * loss_flux + log_vars[1]) + \
                     (w_grad * loss_grad + log_vars[2])
        
        if total_loss.requires_grad:
            total_loss.backward()
            
        return total_loss
    
    for _ in range(steps):
        # L-BFGS may invoke closure() multiple times within a single optimizer
        # step. Update the progress bar once per completed optimization step.
        loss_val = optimizer.step(closure)
        pbar.update(1)
        pbar.set_postfix({"Loss": f"{loss_val.item():.4e}"})

    pbar.close()    
    return model


def train_2stage_adaptive(model, dataloader, X_full, Y_full, physics_criterion, Sigma_full=None, epochs_adam=1000, steps_lbfgs=10):
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
        physics_criterion=physics_criterion,
        lr=0.001
    )
    
    # Stage 2: L-BFGS
    model = train_lbfgs_adaptive(
        model, 
        X_full=X_full, 
        Y_full=Y_full, 
        log_vars=log_params, 
        physics_criterion=physics_criterion,
        Sigma_full=Sigma_full,
        steps=steps_lbfgs
    )
    
    print("\n-> Training pipeline successfully completed.")
    return model