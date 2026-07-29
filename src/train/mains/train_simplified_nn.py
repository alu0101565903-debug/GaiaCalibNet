"""
Module: train_simplified_nn.py

Residual Physics-Informed Neural Network (PINN) Spectral Recalibration Module
This module implements a Residual PINN designed to predict a multiplicative correction 
factor R(lambda) for recalibrating Gaia BP/RP optical spectra using CALSPEC references.

Key Architecture:
- Predicts residual deviations delta_R around an initial baseline (R = 1.0 + delta_R).
- Applies relative physics-informed loss constraints (data fidelity, integrated flux conservation, 
  and shape preservation).
- Evaluates performance against a classic per-star Chebyshev polynomial baseline.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from numpy.polynomial import Chebyshev

from src.train.training_functions.adp_training_strategies import train_2stage_adaptive


# =====================================================================
# 1. PINN MODEL ARCHITECTURE (RESIDUAL + NORMALIZED)
# =====================================================================
class SpectralPINN(nn.Module):
    """
    Residual PINN designed to predict the multiplicative correction factor R(lambda).
    """
    def __init__(self, input_dim):
        super(SpectralPINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.02),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.02),
            nn.Linear(64, input_dim)
        )
        
        # Zero-initialization for the final layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # 1. Local dynamic normalization
        max_val = torch.max(x, dim=1, keepdim=True)[0] + 1e-15
        x_norm = x / max_val
        
        # 2. Residual deviation inference delta_R
        delta_R = self.net(x_norm)
        
        # 3. Final multiplicative factor
        R = 1.0 + delta_R
        return torch.clamp(R, min=0.3, max=3.0)

    def prepare_for_lbfgs(self):
        """Prepares the model for L-BFGS optimization mode."""
        self.eval()


# =====================================================================
# 2. RELATIVE PHYSICS-INFORMED LOSS FUNCTION
# =====================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Calculates the three physically-motivated loss components on a relative scale
    (original version without Total Variation penalty or curvature weighting).
    """
    def __init__(self):
        super(PhysicsInformedLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(beta=0.01)

    def forward(self, model, X, Y, Sigma=None):
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

        # Component C: Shape Preservation (Simple discrete 2nd derivative)
        d2_R = R_pred[:, 2:] - 2 * R_pred[:, 1:-1] + R_pred[:, :-2]
        loss_grad = torch.mean(d2_R ** 2)

        return loss_data, loss_flux, loss_grad


# =====================================================================
# 3. VISUALIZATION AND EVALUATION (PINN vs POLYNOMIAL BASELINE)
# =====================================================================
def visualize_predictions(model, X_tensor, Y_tensor, output_dir, num_samples=5, poly_deg=5):
    """
    Generates plots comparing PINN calibration against a classic 
    Chebyshev polynomial baseline fitted to the Y/X ratio per star.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        R_pred_tensor = model(X_tensor)
        X_calibrated_pinn = X_tensor * R_pred_tensor
    
    X_np = X_tensor.cpu().numpy()
    Y_np = Y_tensor.cpu().numpy()
    X_cal_pinn_np = X_calibrated_pinn.cpu().numpy()
    
    num_channels = X_np.shape[1]
    wave = np.linspace(330, 1050, num_channels)
    x_cheb_domain = np.linspace(-1, 1, num_channels)
    
    samples_to_plot = min(num_samples, X_np.shape[0])
    
    print(f"\n[>] Generating {samples_to_plot} blind validation plots (PINN vs Chebyshev Baseline Degree {poly_deg}) in: {output_dir}")
    
    for i in range(samples_to_plot):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        eps = 1e-15
        
        # --- Classic Polynomial Baseline ---
        ratio_ideal = Y_np[i] / (X_np[i] + eps)
        cheb_fit = Chebyshev.fit(x_cheb_domain, ratio_ideal, deg=poly_deg)
        R_baseline_i = cheb_fit(x_cheb_domain)
        X_cal_baseline_i = X_np[i] * R_baseline_i
        
        # ==========================================
        # Top Panel: Absolute Flux
        # ==========================================
        ax1.plot(wave, Y_np[i], label="CALSPEC Reference (Y)", color='black', alpha=0.85, linewidth=1.2)
        ax1.plot(wave, X_np[i], label="Original Gaia BP/RP (X)", color="#4f88f3", alpha=0.6, linewidth=1.2)
        ax1.plot(wave, X_cal_baseline_i, label=f"Classical Baseline (Polynomial deg={poly_deg})", color='#00cf49', alpha=0.9, linewidth=1.5, linestyle='--')
        ax1.plot(wave, X_cal_pinn_np[i], label="PINN Prediction (X * R)", color='#e63946', alpha=0.8, linewidth=1.5)
        
        ax1.set_title(f"Blind Test PINN vs Baseline | Test Star Index: {i}")
        ax1.set_ylabel("Absolute Flux")
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend()
        
        # ==========================================
        # Bottom Panel: Percentage Residual (%)
        # ==========================================
        #res_original_pct = ((X_np[i] - Y_np[i]) / (Y_np[i] + eps)) * 100.0
        #res_baseline_pct = ((X_cal_baseline_i - Y_np[i]) / (Y_np[i] + eps)) * 100.0
        #res_pinn_pct = ((X_cal_pinn_np[i] - Y_np[i]) / (Y_np[i] + eps)) * 100.0
        
        mean_flux = np.mean(Y_np[i])
        res_original_pct = ((X_np[i] - Y_np[i]) / mean_flux) * 100.0
        res_baseline_pct = ((X_cal_baseline_i - Y_np[i]) / mean_flux) * 100.0
        res_pinn_pct = ((X_cal_pinn_np[i] - Y_np[i]) / mean_flux) * 100.0
        

        # Ideal zero-error reference line
        ax2.axhline(0.0, color='black', linestyle='-', linewidth=1.2, alpha=0.85)
        
        ax2.plot(wave, res_original_pct, label="Original Gaia Residual", color="#4f88f3", alpha=0.5, linewidth=1.0)
        ax2.plot(wave, res_baseline_pct, label=f"Baseline Residual (deg={poly_deg})", color="#00cf49", alpha=0.8, linewidth=1.5, linestyle='--')
        ax2.plot(wave, res_pinn_pct, label="PINN Residual", color='#e63946', alpha=0.9, linewidth=1.5)
        
        ax2.set_xlabel("Approximate Wavelength [nm]")
        ax2.set_ylabel("Residual (%)")
        
        # Fixed display limit between -20% and +20%
        ax2.set_ylim(-20, 20)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(loc="upper right", fontsize="small")
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(output_dir, f"star_comparison_test_{i:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print("[✓] Plots successfully generated.")


# =====================================================================
# 4. MAIN TRAINING PIPELINE
# =====================================================================
def run_training(
    x_path="data/processed/X.npy",
    y_path="data/processed/Y.npy",
    output_model_dir="results/models/simplified_nn",
    vis_dir="results/models/simplified_nn/comparison",
    test_size=0.2,
    random_state=42,
    poly_deg=5,
    epochs_adam=500,
    steps_lbfgs=15,
    max_batch_size=32,
    num_vis_samples=None
):
    """
    Runs the full Residual PINN training pipeline.

    Parameters:
    - x_path (str): Path to input spectra numpy file (X).
    - y_path (str): Path to target spectra numpy file (Y).
    - output_model_dir (str): Directory to save the trained joblib model.
    - vis_dir (str): Directory to save test evaluation plots.
    - test_size (float): Fraction of samples reserved for the blind test split (default: 0.2).
    - random_state (int): Random seed for train/test split reproducibility (default: 42).
    - poly_deg (int): Degree of the Chebyshev polynomial for the evaluation baseline (default: 5).
    - epochs_adam (int): Number of Adam optimization epochs (default: 500).
    - steps_lbfgs (int): Number of L-BFGS optimization steps (default: 15).
    - max_batch_size (int): Maximum batch size for the DataLoader (default: 32).
    - num_vis_samples (int or None): Number of test sample plots to generate. If None, plots all test samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running PINN training on device: {device}")

    # --- Data Loading ---
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Verify that '{x_path}' and '{y_path}' are located in the correct path.")

    X_np = np.load(x_path)
    Y_np = np.load(y_path)

    # --- 80/20 Split (Train / Test) ---
    X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
        X_np, Y_np, test_size=test_size, random_state=random_state
    )
    
    print(f"[>] Training samples: {X_train_np.shape[0]}")
    print(f"[>] Blind test samples: {X_test_np.shape[0]}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32)
    
    # DataLoader preparation ONLY with training data
    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
    batch_size = min(max_batch_size, len(dataset_train))
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    X_full_train = X_train_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)

    # --- Model Initialization ---
    num_channels = X_train_tensor.shape[1]
    model = SpectralPINN(input_dim=num_channels).to(device)
    physics_criterion = PhysicsInformedLoss()

    # --- Training (Completely ignores Test set) ---
    trained_model = train_2stage_adaptive(
        model=model,
        dataloader=dataloader,
        X_full=X_full_train,
        Y_full=Y_full_train,
        physics_criterion=physics_criterion,
        Sigma_full=None,
        epochs_adam=epochs_adam,
        steps_lbfgs=steps_lbfgs
    )

    # --- Save Model in JOBLIB Format ---
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_spectral_recalibration.joblib")
    
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "input_dim": num_channels,
    }
    joblib.dump(export_model, save_path)
    
    print(f"\n[✓] Model successfully saved (Joblib) to: {save_path}")

    # --- Generate Visualizations EXCLUSIVELY with Test Set ---
    samples_to_plot = len(X_test_tensor) if num_vis_samples is None else num_vis_samples
    visualize_predictions(
        model=trained_model, 
        X_tensor=X_test_tensor.to(device), 
        Y_tensor=Y_test_tensor.to(device), 
        output_dir=vis_dir, 
        num_samples=samples_to_plot,
        poly_deg=poly_deg
    )


if __name__ == "__main__":
    
    run_training(
            x_path="data/processed/X.npy",
            y_path="data/processed/Y.npy",
            output_model_dir="results/models/pinn",
            vis_dir="results/models/simplified_nn/comparison",
            test_size=0.2,
            random_state=42,
            poly_deg=5,
            epochs_adam=500,
            steps_lbfgs=15,
            max_batch_size=15,
            num_vis_samples=None
        )
    """
    run_training(
        x_path="data/processed/nn/blue/X.npy",
        y_path="data/processed/nn/blue/Y.npy",
        output_model_dir="results/models/simplified_nn/blue",
        vis_dir="results/models/simplified_nn/blue/comparison",
        test_size=0.2,
        random_state=42,
        poly_deg=5,
        epochs_adam=500,
        steps_lbfgs=15,
        max_batch_size=15,
        num_vis_samples=None
    )
    """

    """
    run_training(
            x_path="data/processed/nn/solar/X.npy",
            y_path="data/processed/nn/solar/Y.npy",
            output_model_dir="results/models/simplified_nn/solar",
            vis_dir="results/models/simplified_nn/solar/comparison",
            test_size=0.2,
            random_state=42,
            poly_deg=5,
            epochs_adam=500,
            steps_lbfgs=15,
            max_batch_size=15,
            num_vis_samples=None
        )
    """