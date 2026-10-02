"""Train and evaluate a residual PINN with leave-one-out cross-validation.

The model predicts a continuous spectral response function that maps Gaia XP
spectra to degraded CALSPEC reference spectra without an external baseline ratio.

Notes
-----
Report context: Section 5.0, Machine Learning Calibration Framework.
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneOut
from tqdm.auto import tqdm

from src.models.pinn_architecture import BlueWiggleNet, DirectConvPINN
from src.models.pinn_export_models import DirectExportModel, EnsembleExportModel
from src.train.loss.physics_loss import BluePhysicsLoss, DirectPhysicsLoss
from src.train.training_functions.pinn_trainer import train_fold
from src.utils.processing.pinn_data import apply_huang_correction, fit_log_ratio_pca, load_pinn_dataset
from src.utils.reproducibility_utils import set_seed
from src.utils.processing.stellar_loader import standardize_stellar_parameters
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
      epochs_adam: int = 600,
      epochs_blue: int = 200,
      use_ruwe: bool = True,
      lr_base: float = 1e-3,
      lr_blue: float = 1e-3,
      weight_decay_base: float = 1e-4,
      weight_decay_blue: float = 0.0,
      lambda_reg_blue: float = 1e-3,
      lambda_smooth_blue: float | None = 1e-4,
      reset_seed_each_fold: bool = True,
) -> None:
      """
      Execute LOOCV using the selected seed-42 configuration by default.

      Defaults match candidate 12 from the 24-configuration hyperparameter search.

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
      epochs_adam : int, default=600
            Number of training epochs per CV fold.
      epochs_blue : int, default=150
            Number of training epochs for the blue wiggle model.
      use_ruwe : bool, default=True
            Include RUWE alongside BP-RP and G in both networks.
      lr_base, lr_blue : float, default=1e-3
            Adam learning rates for the base and blue networks.
      weight_decay_base, weight_decay_blue : float, default=1e-4, 0
            Adam weight decay for each network.
      lambda_reg_blue : float, default=1e-3
            Blue correction penalty towards a neutral ratio of one.
      lambda_smooth_blue : float or None, default=1e-4
            Blue curvature penalty; None uses lambda_smooth.
      reset_seed_each_fold : bool, default=True
            Reset the training seed in each fold, as in the hyperparameter search.

      Returns
      -------
      None
      """
      set_seed(random_state)
      if lambda_smooth_blue is None:
            lambda_smooth_blue = lambda_smooth
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      out_dir = Path(output_model_dir).resolve()
      out_dir.mkdir(parents=True, exist_ok=True)

      X_np, Y_np = np.load(x_path), np.load(y_path)
      X_err_np = np.load(x_err_path)
      _, _, _, params_raw, wave_grid, g_mags_raw_np, ids_np = load_pinn_dataset(
            params_path, standardize_params=False
      )
      if not use_ruwe:
            params_raw = params_raw[:, :2]
      parameter_names = ["bp_rp", "G", "ruwe"] if use_ruwe else ["bp_rp", "G"]

      n_samples, num_channels = X_np.shape
      num_stellar_params = params_raw.shape[1]
      cv_indices = np.arange(n_samples)
      np.save(out_dir / "loocv_indices.npy", cv_indices)
      print(
            f"[>] Initiating Residual LOOCV on {len(cv_indices)} samples. Device: {device}"
      )

      R_pred_cv_np = np.zeros_like(X_np[cv_indices])
      params_cv_np = np.empty_like(params_raw[cv_indices])
      fold_means, fold_stds, fold_ruwe_medians = [], [], []
      (out_dir / "loocv_config.json").write_text(json.dumps({
            "normalization": "fit on training stars separately in each fold",
            "ruwe_imputation": "median of observed training RUWE before normalization" if use_ruwe else None,
            "parameter_names": parameter_names, "use_ruwe": use_ruwe,
            "random_state": random_state, "num_components": num_components,
            "epochs_adam": epochs_adam, "epochs_blue": epochs_blue,
            "lambda_smooth": lambda_smooth, "lambda_reg": lambda_reg,
            "weight_decay": weight_decay_base,
            "lr_base": lr_base, "lr_blue": lr_blue,
            "weight_decay_base": weight_decay_base, "weight_decay_blue": weight_decay_blue,
            "lambda_reg_blue": lambda_reg_blue, "lambda_smooth_blue": lambda_smooth_blue,
            "reset_seed_each_fold": reset_seed_each_fold,
      }, indent=2), encoding="utf-8")
      base_train_history = []
      base_val_history = []
      blue_train_history = []
      blue_val_history = []
      loo = LeaveOneOut()

      fold_iterator = tqdm(
            loo.split(cv_indices),
            total=len(cv_indices),
            desc="LOOCV folds",
            unit="fold",
            position=0,
      )
      for fold, (train_local_idx, val_local_idx) in enumerate(fold_iterator):
            if reset_seed_each_fold:
                  set_seed(random_state, verbose=False)
            train_idx = cv_indices[train_local_idx]
            val_idx = cv_indices[val_local_idx]
            val_i = val_idx[0]
            fold_iterator.set_postfix(star=val_i)

            params_np, param_mean, param_std, ruwe_median = standardize_stellar_parameters(
                  params_raw, train_idx
            )
            fold_means.append(param_mean)
            fold_stds.append(param_std)
            if use_ruwe:
                  fold_ruwe_medians.append(ruwe_median)
            params_cv_np[val_local_idx[0]] = params_np[val_i]
            X_train, Y_train = X_np[train_idx], Y_np[train_idx]
            P_train, E_train = params_np[train_idx], X_err_np[train_idx]
            X_test, P_test = X_np[val_idx], params_np[val_idx]

            # Fit the PCA basis for the absolute calibration ratio.
            pca_basis_np, pca_mean_np = fit_log_ratio_pca(
                  X_train,
                  Y_train,
                  num_components=num_components,
                  random_state=random_state,
            )

            if fold == 0:
                  np.save(out_dir / "pca_basis_first_fold.npy", pca_basis_np)
                  np.save(out_dir / "pca_mean_first_fold.npy", pca_mean_np)
                  np.save(out_dir / "stellar_mean_first_fold.npy", param_mean)
                  np.save(out_dir / "stellar_std_first_fold.npy", param_std)
                  if use_ruwe:
                        np.save(out_dir / "ruwe_imputation_median_first_fold.npy", ruwe_median)

            # Transfer fold data to compute device
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
            P_train_t = torch.tensor(P_train, dtype=torch.float32).to(device)
            E_train_t = torch.tensor(E_train, dtype=torch.float32).to(device)
            X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            Y_val_t = torch.tensor(Y_np[val_idx], dtype=torch.float32).to(device)
            P_val_t = torch.tensor(P_test, dtype=torch.float32).to(device)
            E_val_t = torch.tensor(X_err_np[val_idx], dtype=torch.float32).to(device)

            model = DirectConvPINN(
                  input_dim=num_channels, 
                  pca_basis=pca_basis_np, 
                  pca_mean=pca_mean_np,
                  num_params=num_stellar_params
            ).to(device)

            optimizer_base = torch.optim.Adam(model.parameters(), lr=lr_base, weight_decay=weight_decay_base)

            # Instancia limpia sin pesos extra
            criterion_base = DirectPhysicsLoss(
                  lambda_smooth=lambda_smooth, 
                  lambda_reg=lambda_reg
            ).to(device)

            blue_mask = wave_grid < 450.0
            num_blue_channels = np.sum(blue_mask)
            
            blue_model = BlueWiggleNet(input_dim=num_blue_channels, num_params=num_stellar_params).to(device)

            if fold == 0:
                tqdm.write(f"BlueWiggleNet architecture:\n{blue_model.mlp}")

            optimizer_blue = torch.optim.Adam(blue_model.parameters(), lr=lr_blue, weight_decay=weight_decay_blue)
            criterion_blue = BluePhysicsLoss(lambda_reg=lambda_reg_blue, lambda_smooth=lambda_smooth_blue).to(device)

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
                  progress_position=1,
                  progress_leave=False,
            )
            base_train_history.append(fold_history["base_train"])
            base_val_history.append(fold_history["base_val"])
            blue_train_history.append(fold_history["blue_train"])
            blue_val_history.append(fold_history["blue_val"])

            trained_model.eval()
            trained_blue_model.eval()
            
            ensemble = EnsembleExportModel(trained_model, trained_blue_model, blue_mask, wave_grid)

            with torch.no_grad():
                # Predict R_pred directly with the assembled model.
                R_pred = ensemble(X_val_t, P_val_t)
                
                # Store the predicted correction ratio.
                R_pred_cv_np[val_local_idx[0]] = R_pred.cpu().numpy()[0]

                if fold == 0:
                              torch.save(ensemble.state_dict(), str(out_dir / "model_ensemble_state.pth"))

                # Release fold-specific objects before the next iteration.
                del model, blue_model, criterion_base, criterion_blue, trained_model, trained_blue_model, ensemble
                del X_train_t, Y_train_t, P_train_t, E_train_t, X_val_t, Y_val_t, P_val_t, E_val_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                  
      print("\n\n[>] Residual LOOCV Completed. Exporting metrics...")

      np.savez_compressed(
            out_dir / "fold_normalization.npz", mean=np.asarray(fold_means),
            std=np.asarray(fold_stds),
            validation_indices=cv_indices,
            parameter_names=np.asarray(parameter_names),
            **({"ruwe_imputation_median": np.asarray(fold_ruwe_medians),
                "ruwe_missing": np.isnan(params_raw[:, 2])} if use_ruwe else {}),
      )
      history_filepath = out_dir / "training_loss_history.npz"
      np.savez_compressed(
            history_filepath,
            base_train=np.asarray(base_train_history),
            base_val=np.asarray(base_val_history),
            blue_train=np.asarray(blue_train_history),
            blue_val=np.asarray(blue_val_history),
      )
      plot_training_history(history_filepath, out_dir / "training_loss_phases.pdf")

      

      X_cv_t = torch.tensor(X_np[cv_indices], dtype=torch.float32).to(device)
      Y_cv_t = torch.tensor(Y_np[cv_indices], dtype=torch.float32).to(device)
      E_cv_t = torch.tensor(X_err_np[cv_indices], dtype=torch.float32).to(device)
      P_cv_t = torch.tensor(params_cv_np, dtype=torch.float32).to(device)

      ### Huang Baseline Correction ###
      print("\n[>] Calculating Huang et al. Baseline for the full dataset...")
      X_huang_np = apply_huang_correction(
            X_np,
            X_err_np,
            ids_np,
            g_mags_raw_np,
      )
      X_huang_cv_tensor = torch.tensor(X_huang_np[cv_indices], dtype=torch.float32).to(device)

      G_mags_cv_t = torch.tensor(g_mags_raw_np[cv_indices], dtype=torch.float32).to(device)

      results_filepath = str(out_dir / "loocv_aggregated_predictions.npz")

      export_test_predictions(
            model=DirectExportModel(device, R_pred_cv_np),
            X_tensor=X_cv_t,
            Y_tensor=Y_cv_t,
            X_err_tensor=E_cv_t,
            output_filepath=results_filepath,
            params_tensor=P_cv_t,
            wave=wave_grid,
            G_mags_tensor=G_mags_cv_t,
            X_huang_tensor=X_huang_cv_tensor
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
            num_samples=len(cv_indices),
            include_gaussian=False
      )

      print(f"\n[✓] Finished. Metrics and plots saved to: {out_dir}")


if __name__ == "__main__":
      # Winning search configuration: candidate 12, training seed 42.
      OUTPUT_MODEL_DIR = "results/models/loocv_eps_residual_pinn"
      
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
            epochs_blue=200,
            use_ruwe=True,
            lr_base=1e-3,
            lr_blue=1e-3,
            weight_decay_base=1e-4,
            weight_decay_blue=0.0,
            lambda_reg_blue=1e-3,
            lambda_smooth_blue=1e-4,
            reset_seed_each_fold=True,
      )
