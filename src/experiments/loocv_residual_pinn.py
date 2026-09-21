"""
Module: loocv_residual_pinn.py
Report Section: 5.0 Machine Learning Calibration Framework

Leave-One-Out Cross-Validation (LOOCV) Pipeline for Residual PINN.

This module executes leave-one-out cross-validation for a residual Physics-Informed 
Neural Network (PINN) architecture. The model learns to predict the absolute continuous 
spectral response function R(lambda) mapping low-resolution Gaia XP spectra directly to 
degraded CALSPEC physical reference targets without relying on external baseline ratios.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut

from GaiaXPcorrection.GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
from src.models.pinn_architecture import BlueWiggleNet, DirectConvPINN
from src.models.pinn_export_models import DirectExportModel, EnsembleExportModel
from src.train.loss.physics_loss import BluePhysicsLoss, DirectPhysicsLoss
from src.train.training_functions.pinn_trainer import train_fold
from src.utils.reproducibility_utils import set_seed
from src.utils.stellar_loader import load_stellar_parameters
from src.utils.visualize_models_performance.export_utils import export_test_predictions
from src.utils.visualize_models_performance.metrics_eval import evaluate_test_set_from_file
from src.utils.visualize_models_performance.plot_model_comparison import (
      generate_plots_from_file,
      plot_training_history,
)


def run_loocv(
      x_path: Union[str, Path], 
      y_path: Union[str, Path], 
      x_err_path: Union[str, Path], 
      params_path: Union[str, Path],
      output_model_dir: Union[str, Path], 
      random_state: int = 42,
      num_components: int = 20, 
      lambda_smooth: float = 1e-4, 
      lambda_reg: float = 1e-4, 
      epochs_adam: int = 400,
      epochs_blue: int = 150
) -> None:
      """
      Execute Leave-One-Out Cross-Validation for Residual PINN model.

      Parameters
      ----------
      x_path : str or Path
            Path to input spectra NumPy file (X.npy).
      y_path : str or Path
            Path to target spectra NumPy file (Y.npy).
      x_err_path : str or Path
            Path to flux errors NumPy file (X_error.npy).
      params_path : str or Path
            Directory containing stellar parameter NumPy arrays.
      output_model_dir : str or Path
            Target directory to output evaluation metrics, plots, and array binaries.
      random_state : int, default=42
            Random seed for reproducibility.
      num_components : int, default=20
            Number of PCA basis components used for spectral reconstruction.
      lambda_smooth : float, default=1e-4
            Smoothness loss hyperparameter weight.
      lambda_reg : float, default=1e-4
            L2 regularization hyperparameter weight.
      epochs_adam : int, default=400
            Number of training epochs per CV fold.
      epochs_blue : int, default=150
            Number of training epochs for the blue wiggle model.

      Returns
      -------
      None
      """
      set_seed(random_state)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      out_dir = Path(output_model_dir).resolve()
      out_dir.mkdir(parents=True, exist_ok=True)

      X_np, Y_np = np.load(x_path), np.load(y_path)
      X_err_np = np.load(x_err_path)
      params_np = load_stellar_parameters(params_path)

      data_dir = Path(params_path)
      ids_np = np.load(data_dir / "gaia_source_ids.npy")
      g_mags_raw_np = np.load(data_dir / "G_mags.npy")
      wave_grid = np.load(data_dir / "wave_grid.npy")

      n_samples, num_channels = X_np.shape
      num_stellar_params = params_np.shape[1]
      print(f"[>] Initiating Residual LOOCV on {n_samples} samples. Device: {device}")

      R_pred_all_np = np.zeros_like(X_np)
      base_train_history = []
      base_val_history = []
      blue_train_history = []
      blue_val_history = []
      loo = LeaveOneOut()

      for fold, (train_idx, test_idx) in enumerate(loo.split(X_np)):
            val_i = test_idx[0]
            print(f"[{val_i+1}/{n_samples}] Training fold (inferring on star index {val_i})...", end="\r")

            X_train, Y_train = X_np[train_idx], Y_np[train_idx]
            P_train, E_train = params_np[train_idx], X_err_np[train_idx]
            X_test, P_test = X_np[test_idx], params_np[test_idx]

            # -------------------------------------------------------------
            # ABSOLUTE CALIBRATION PCA (Log-space mapping)
            # -------------------------------------------------------------
            R_train = Y_train / np.clip(X_train, 1e-16, None)
            
            # Usa el log directo, SIN gaussian_filter1d
            R_train_clipped = np.clip(R_train, 0.1, 15.0)
            R_train_log = np.log(R_train_clipped)
            R_train_log_smooth = gaussian_filter1d(R_train_log, sigma=0.5, axis=1)
            pca = PCA(n_components=num_components, random_state=random_state)
            pca.fit(R_train_log_smooth) # Usa los datos suavizados

            pca_basis_np = pca.components_
            pca_mean_np = pca.mean_

            if fold == 0:
                np.save(out_dir / "pca_basis.npy", pca_basis_np)
                np.save(out_dir / "pca_mean.npy", pca_mean_np)

            # Transfer fold data to compute device
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
            P_train_t = torch.tensor(P_train, dtype=torch.float32).to(device)
            E_train_t = torch.tensor(E_train, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            Y_val_t = torch.tensor(Y_np[test_idx], dtype=torch.float32).to(device)
            P_val_t = torch.tensor(P_test, dtype=torch.float32).to(device)
            E_val_t = torch.tensor(X_err_np[test_idx], dtype=torch.float32).to(device)

            model = DirectConvPINN(
                  input_dim=num_channels, 
                  pca_basis=pca_basis_np, 
                  pca_mean=pca_mean_np,
                  num_params=num_stellar_params
            ).to(device)

            optimizer_base = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Instancia limpia sin pesos extra
            criterion_base = DirectPhysicsLoss(
                  lambda_smooth=lambda_smooth, 
                  lambda_reg=lambda_reg
            ).to(device)

            blue_mask = wave_grid < 450.0
            num_blue_channels = np.sum(blue_mask)
            
            blue_model = BlueWiggleNet(input_dim=num_blue_channels).to(device)
            optimizer_blue = torch.optim.Adam(blue_model.parameters(), lr=1e-3)
            criterion_blue = BluePhysicsLoss().to(device)

            trained_model, trained_blue_model, fold_history = train_fold(
                model=model,
                blue_model=blue_model,
                criterion_base=criterion_base,
                criterion_blue=criterion_blue,
                optimizer_base=optimizer_base,
                optimizer_blue=optimizer_blue,
                X_train=X_train_t,
                Y_train=Y_train_t,
                P_train=P_train_t,
                E_train=E_train_t,
                blue_mask=blue_mask,
                epochs_adam=epochs_adam,
                epochs_blue=epochs_blue,
                X_val=X_val_t,
                Y_val=Y_val_t,
                P_val=P_val_t,
                E_val=E_val_t,
                return_history=True,
            )
            base_train_history.append(fold_history["base_train"])
            base_val_history.append(fold_history["base_val"])
            blue_train_history.append(fold_history["blue_train"])
            blue_val_history.append(fold_history["blue_val"])

            trained_model.eval()
            trained_blue_model.eval()
            
            ensemble = EnsembleExportModel(trained_model, trained_blue_model, blue_mask, wave_grid)

            with torch.no_grad():
                # Extraemos directamente R_pred usando el modelo ensamblado
                R_pred = ensemble(X_val_t, P_val_t)
                
                # Guardamos la ratio R
                R_pred_all_np[val_i] = R_pred.cpu().numpy()

                if fold == 0:
                    torch.save(ensemble.state_dict(), str(out_dir / "modelo_residual_pinn.pth"))

                # Liberar memoria explícitamente
                del model, blue_model, criterion_base, criterion_blue, trained_model, trained_blue_model, ensemble
                del X_train_t, Y_train_t, P_train_t, E_train_t, X_val_t, Y_val_t, P_val_t, E_val_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                  
      print("\n\n[>] Residual LOOCV Completed. Exporting metrics...")

      history_filepath = out_dir / "training_loss_history.npz"
      np.savez_compressed(
            history_filepath,
            base_train=np.asarray(base_train_history),
            base_val=np.asarray(base_val_history),
            blue_train=np.asarray(blue_train_history),
            blue_val=np.asarray(blue_val_history),
      )
      plot_training_history(history_filepath, out_dir / "training_loss_phases.pdf")

      

      X_all_t = torch.tensor(X_np, dtype=torch.float32).to(device)
      Y_all_t = torch.tensor(Y_np, dtype=torch.float32).to(device)
      E_all_t = torch.tensor(X_err_np, dtype=torch.float32).to(device)
      P_all_t = torch.tensor(params_np, dtype=torch.float32).to(device)

      ### Huang Baseline Correction ###
      print("\n[>] Calculating Huang et al. Baseline for the full dataset...")
      df_huang_input = pd.DataFrame({
            'source_id': ids_np.flatten(), 
            'flux': list(X_np),                              
            'flux_error': list(X_err_np)                
      })
      g_mags_series = pd.Series(g_mags_raw_np.flatten())

      flux_out = Gaia_Correction_V1.correction_df(
            df_huang_input, 
            g_mags_series, 
            have_error=True, 
            Truncation=True
      )

      X_huang_np = np.vstack(flux_out['flux_cor'].values)
      X_huang_tensor = torch.tensor(X_huang_np, dtype=torch.float32).to(device)

      G_mags_all_t = torch.tensor(g_mags_raw_np, dtype=torch.float32).to(device)

      results_filepath = str(out_dir / "loocv_aggregated_predictions.npz")

      export_test_predictions(
            model=DirectExportModel(device, R_pred_all_np),
            X_tensor=X_all_t,
            Y_tensor=Y_all_t,
            X_err_tensor=E_all_t,
            output_filepath=results_filepath,
            params_tensor=P_all_t,
            wave=wave_grid,
            G_mags_tensor=G_mags_all_t,
            X_huang_tensor=X_huang_tensor   # Set to it to None if Huang baseline is not used
      )

      evaluate_test_set_from_file(
            input_filepath=results_filepath, 
            output_dir=str(out_dir), 
            num_knots=num_components
      )

      vis_dir = str(out_dir / "comparison_eps_plots")
      generate_plots_from_file(
            input_filepath=results_filepath, 
            output_dir=vis_dir, 
            num_samples=n_samples
      )

      print(f"\n[✓] Finished. Metrics and plots saved to: {out_dir}")


if __name__ == "__main__":
      OUTPUT_MODEL_DIR = f"results/models/loocv_residual_pinn"
      
      run_loocv(
            x_path=f"data/processed/nn_data/X.npy",
            y_path=f"data/processed/nn_data/Y.npy",
            x_err_path=f"data/processed/nn_data/X_error.npy",
            params_path=f"data/processed/nn_data",
            output_model_dir=OUTPUT_MODEL_DIR,
            random_state=42,
            num_components=20,
            lambda_smooth=1e-4,
            lambda_reg=1e-4,
            epochs_adam=600,
            epochs_blue=150
      )