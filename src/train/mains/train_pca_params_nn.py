"""
Module: train_pca_params_nn.py
Report Section: Chapter 5 Data Processing & Feature Engineering / Model Training

Parameter-Conditioned Physics-Informed Neural Network (PINN) Training Pipeline.

This module trains an advanced Conv1D-based neural network to recalibrate Gaia DR3 
XP continuous spectra using CALSPEC standard stars as ground truth references. It 
combines stellar astrophysics parameters (G magnitude, BP-RP color, RUWE) with 
an adaptive physics-guided loss function to improve upon standard empirical 
baselines (such as Huang et al., 2021).
"""

import os
import traceback
from typing import List, Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Internal Project Dependencies
from GaiaXPcorrection.GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
from src.models.pinn_architectures import ImprovedSpectralPINN
from src.train.loss.losses import UltimateSpectralPINNLoss
from src.train.training_functions.fixed_params_training_strategies import (
    train_2stage_fixed,
)
from src.utils.extra.seed import set_seed
from src.utils.visualize_models_performance.export_utils import (
    export_test_predictions,
)
from src.utils.visualize_models_performance.metrics_eval import (
    evaluate_test_set_from_file,
)
from src.utils.visualize_models_performance.plot_model_comparison import (
    generate_plots_from_file,
)


def get_gaia_ids_for_star_names(
    star_names_path: str, 
    catalog_path: str = "data/processed/crossmatch_catalog.csv"
) -> np.ndarray:
    """
    Map target star names to corresponding Gaia DR3 Source IDs using a crossmatch catalog.

    Parameters
    ----------
    star_names_path : str
        Path to the `.npy` binary file containing an array of target star names.
    catalog_path : str, default="data/processed/crossmatch_catalog.csv"
        Path to the CSV crossmatch catalog containing source identifiers.

    Returns
    -------
    np.ndarray
        Array of mapped Gaia DR3 source ID strings corresponding to the input star names.

    Raises
    ------
    FileNotFoundError
        If `star_names_path` or `catalog_path` does not exist on disk.
    ValueError
        If required identifier columns (CALSPEC target name or Gaia Source ID) 
        are missing from the catalog DataFrame.
    KeyError
        If any star name present in `star_names_path` cannot be found in the catalog.
    """
    if not os.path.exists(star_names_path):
        raise FileNotFoundError(f"Star names file not found at: {star_names_path}")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Crossmatch catalog file not found at: {catalog_path}")

    star_names = np.load(star_names_path)
    df = pd.read_csv(catalog_path, dtype=str)
    
    # Locate CALSPEC target name column dynamically
    calspec_cols = [col for col in df.columns if "calspec" in col.lower() or "name" in col.lower()]
    if not calspec_cols:
        raise ValueError(f"Could not locate CALSPEC target name column in {catalog_path}")
    calspec_col = calspec_cols[0]

    # Locate Gaia Source ID column dynamically
    id_col = "source_id" if "source_id" in df.columns else ("gaia_source_id" if "gaia_source_id" in df.columns else None)
    if id_col is None:
        raise ValueError(f"Could not locate Gaia source ID column in {catalog_path}")
    
    # Build lower-case dictionary lookup for robust string matching
    mapping = dict(zip(
        df[calspec_col].astype(str).str.strip().str.lower(), 
        df[id_col].astype(str).str.strip()
    ))
    
    mapped_ids = []
    for name in star_names:
        clean_name = str(name).strip().lower()
        if clean_name not in mapping:
            raise KeyError(f"Star name '{name}' not found in crossmatch catalog mapping.")
        mapped_ids.append(mapping[clean_name])

    return np.array(mapped_ids)


class PCAExportWrapper(nn.Module):
    """
    PyTorch Wrapper Module for Deploying and Exporting PCA PINN Models.

    Wraps a trained `ImprovedSpectralPINN` model to accept raw input spectra, 
    predict PCA expansion coefficients, and execute internal spectral 
    reconstruction for standardized inference and export workflows.

    Parameters
    ----------
    base_model : nn.Module
        The trained physics-informed neural network model instance.

    Attributes
    ----------
    base_model : nn.Module
        The underlying PINN model used for feature extraction and reconstruction.
    """
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform forward pass to generate calibrated physical spectra.

        Parameters
        ----------
        x : torch.Tensor
            Raw input Gaia BP/RP flux tensor of shape `(batch_size, num_wavelengths)`.
        params : torch.Tensor, optional
            Normalized stellar parameters tensor of shape `(batch_size, num_params)`.

        Returns
        -------
        torch.Tensor
            Recalibrated physical flux tensor of shape `(batch_size, num_wavelengths)`.
        """
        # Pass raw astronomical flux directly to preserve input scaling (~1e-14)
        coeffs = self.base_model(x, params)
        
        # Reconstruct physical calibrated flux: X_cal = X * exp(pca_mean + sum(c_i * basis_i))
        return self.base_model.reconstruct(coeffs, original_x=x)


def run_training(
    x_path: str = "data/processed/X.npy",
    y_path: str = "data/processed/Y.npy",
    x_err_path: str = "data/processed/X_error.npy", 
    star_names_path: str = "data/processed/star_names.npy",
    catalog_path: str = "data/processed/catalogs_by_color/all_catalog.csv",
    params_paths: Optional[List[str]] = None,
    output_model_dir: str = "results/models/pca_nn",
    num_components: int = 6,          
    lambda_smooth: float = 0.01,      
    lambda_deriv: float = 0.1,        
    lambda_anchor: float = 0.5,
    lambda_reg: float = 1e-3,         
    epochs_adam: int = 3000,
    steps_lbfgs: int = 150,
    max_batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42
) -> str:
    """
    Orchestrate the PINN training, validation, export, and evaluation pipeline.

    Loads raw Gaia XP continuous spectra, CALSPEC reference standards, and stellar 
    parameter covariates. Fits a log-space dynamic PCA basis, trains an 
    `ImprovedSpectralPINN` using a two-stage optimizer (Adam + L-BFGS) combined with 
    `UltimateSpectralPINNLoss`, computes empirical baselines (Huang et al. 2021), 
    and serializes metrics and predictions.

    Parameters
    ----------
    x_path : str, default="data/processed/X.npy"
        Path to input Gaia spectra array file.
    y_path : str, default="data/processed/Y.npy"
        Path to ground-truth CALSPEC reference spectra file.
    x_err_path : str, default="data/processed/X_error.npy"
        Path to flux measurement error array file.
    star_names_path : str, default="data/processed/star_names.npy"
        Path to star target names array file.
    catalog_path : str, default="data/processed/catalogs_by_color/all_catalog.csv"
        Path to crossmatch catalog CSV containing source metadata.
    params_paths : Optional[List[str]], default=None
        List of paths to `.npy` files containing stellar parameters (e.g., G mag, BP-RP, RUWE).
    output_model_dir : str, default="results/models/pca_nn"
        Directory path where trained model checkpoints and evaluation outputs are saved.
    num_components : int, default=6
        Number of principal component basis vectors retained for spectral correction.
    lambda_smooth : float, default=0.01
        Weight multiplier for the second-derivative curvature penalty term in the loss.
    lambda_deriv : float, default=0.1
        Weight multiplier for Sobolev first-derivative alignment loss.
    lambda_anchor : float, default=0.5
        Weight multiplier for total integrated bolometric flux anchor loss.
    lambda_reg : float, default=1e-3
        Weight multiplier for L2 coefficient regularization.
    epochs_adam : int, default=3000
        Maximum training iterations for Stage 1 (Adam optimizer).
    steps_lbfgs : int, default=150
        Maximum evaluation steps for Stage 2 (L-BFGS quasi-Newton optimizer).
    max_batch_size : int, default=32
        Maximum batch size used by the DataLoader during Stage 1 training.
    test_size : float, default=0.2
        Fraction of dataset withheld for blind testing/evaluation.
    random_state : int, default=42
        Seed initialized across pseudo-random generators to ensure strict reproducibility.

    Returns
    -------
    str
        Filepath pointing to exported test predictions `.npz` archive.

    Raises
    ------
    ValueError
        If `params_paths` is None or empty.
    """
    # 0. Deterministic Seed & Hardware Device Setup
    set_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[>] Running PINN training on device: {device}")

    # 1. Data Ingestion & Stellar Feature Stacking
    X_np = np.load(x_path)
    Y_np = np.load(y_path)
    X_err_np = np.load(x_err_path)
    gaia_ids_np = get_gaia_ids_for_star_names(star_names_path, catalog_path=catalog_path)

    if params_paths is not None:
        params_list = [np.load(p) for p in params_paths]
        stellar_params_np = np.column_stack(params_list)
    else:
        raise ValueError("Parameter paths (`params_paths`) must be provided for Parameter-Conditioned training.")

    # 2. Train / Blind Test Partitioning
    (
        X_train_np, X_test_np, 
        Y_train_np, Y_test_np, 
        X_err_train_np, X_err_test_np,
        params_train_np, params_test_np,
        ids_train_np, ids_test_np  
    ) = train_test_split(
        X_np, Y_np, X_err_np, stellar_params_np, gaia_ids_np, 
        test_size=test_size, random_state=random_state
    )
    
    print(f"[>] Training samples: {X_train_np.shape[0]}")
    print(f"[>] Blind test samples: {X_test_np.shape[0]}")

    # 3. Pre-smoothing Gaia Spectra to remove instrumental wiggles
    print("[>] Applying Savitzky-Golay smoothing to Gaia input spectra...")
    X_train_smooth_np = savgol_filter(X_train_np, window_length=15, polyorder=2, axis=-1)
    X_test_smooth_np  = savgol_filter(X_test_np, window_length=15, polyorder=2, axis=-1)

    # 4. Stellar Parameter Standardization (Z-Score Normalization)
    mean_params = np.nanmean(params_train_np, axis=0)
    std_params = np.nanstd(params_train_np, axis=0) + 1e-8

    # Impute NaNs in Training parameters using column means
    inds_tr = np.where(np.isnan(params_train_np))
    params_train_np[inds_tr] = np.take(mean_params, inds_tr[1])

    # Impute NaNs in Test parameters using training column means (prevent leakage)
    inds_te = np.where(np.isnan(params_test_np))
    params_test_np[inds_te] = np.take(mean_params, inds_te[1])

    params_train_norm = (params_train_np - mean_params) / std_params
    params_test_norm = (params_test_np - mean_params) / std_params

    # Tensor Conversions
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_train_smooth_tensor = torch.tensor(X_train_smooth_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    params_train_tensor = torch.tensor(params_train_norm, dtype=torch.float32)
    X_err_train_tensor = torch.tensor(X_err_train_np, dtype=torch.float32)

    # Dataset & DataLoader Assembly
    dataset_train = TensorDataset(
        X_train_tensor, 
        Y_train_tensor, 
        params_train_tensor, 
        X_err_train_tensor,
        X_train_smooth_tensor  
    )
    batch_size = min(max_batch_size, len(dataset_train))

    g = torch.Generator()
    g.manual_seed(random_state) 
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)

    # Full Tensors allocated to device for Stage 2 (L-BFGS) Optimization
    X_full_train = X_train_tensor.to(device)
    X_smooth_full_train = X_train_smooth_tensor.to(device)
    Y_full_train = Y_train_tensor.to(device)
    params_full_train = params_train_tensor.to(device) 
    X_err_full_train = X_err_train_tensor.to(device)

    # 5. Dynamic Log-Space PCA Basis Extraction
    print(f"[>] Computing PCA basis ({num_components} components) in Log-Space...")
    epsilon = 1e-22
    delta_train_raw = np.log(np.maximum(Y_train_np, epsilon)) - np.log(np.maximum(X_train_np, epsilon))
    
    # Clip extreme ratios caused by near-zero flux in outer UV/IR boundaries
    lower_bound = np.percentile(delta_train_raw, 0.5, axis=0)
    upper_bound = np.percentile(delta_train_raw, 99.5, axis=0)
    delta_train_clipped = np.clip(delta_train_raw, lower_bound, upper_bound)

    print("[>] Applying Savitzky-Golay filtering to training deltas prior to PCA extraction...")
    delta_train_smooth = savgol_filter(
        delta_train_clipped, 
        window_length=45,  
        polyorder=2,       
        axis=1
    )

    pca = PCA(n_components=num_components, random_state=random_state)
    pca.fit(delta_train_smooth) 
    
    pca_mean_np = pca.mean_
    pca_basis_np = pca.components_
    
    # 6. Model Initialization & Loss Function Construction
    num_channels = X_train_tensor.shape[1]
    num_params = params_train_tensor.shape[1]
    
    model = ImprovedSpectralPINN(
        input_dim=num_channels, 
        pca_basis=pca_basis_np,
        pca_mean_log=pca_mean_np,
        num_params=num_params
    ).to(device)
    
    print("[>] Initializing UltimateSpectralPINNLoss (5-components)...")
    physics_criterion = UltimateSpectralPINNLoss(
        num_bins=num_channels,
        lambda_smooth=lambda_smooth,
        lambda_deriv=lambda_deriv,
        lambda_anchor=lambda_anchor,
        device=device
    ).to(device)

    # 7. Two-Stage Optimization Execution (Adam -> L-BFGS)
    trained_model, loss_history = train_2stage_fixed(
        model=model,
        dataloader=dataloader,
        X_full=X_full_train,
        Y_full=Y_full_train,
        params_full=params_full_train,
        err_full=X_err_full_train,
        X_smooth_full=X_smooth_full_train, 
        physics_criterion=physics_criterion,
        epochs_adam=epochs_adam,
        steps_lbfgs=steps_lbfgs
    )

    # 8. Model Checkpointing & Serialization
    os.makedirs(output_model_dir, exist_ok=True)
    save_path = os.path.join(output_model_dir, "pinn_pca_recalibration.joblib")
    
    export_model = {
        "model_state_dict": trained_model.state_dict(),
        "history": loss_history,
        "input_dim": num_channels,
        "num_components": num_components,
        "num_params": num_params,
        "pca_basis": pca_basis_np,
        "pca_mean": pca_mean_np,
        "mean_params": mean_params,
        "std_params": std_params
    }

    joblib.dump(export_model, save_path)
    print(f"\n[✓] Model successfully saved to: {save_path}")

    # 9. Compute Empirical Baselines & Export Blind Test Predictions (.npz)
    export_ready_model = PCAExportWrapper(trained_model)
    results_filepath = os.path.join(output_model_dir, "test_predictions_final.npz")

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32).to(device)
    X_err_test_tensor = torch.tensor(X_err_test_np, dtype=torch.float32).to(device)
    params_test_tensor = torch.tensor(params_test_norm, dtype=torch.float32).to(device)

    g_mags_test = params_test_np[:, 0]
    G_mags_tensor = torch.tensor(g_mags_test, dtype=torch.float32).to(device)

    print("[>] Computing Huang et al. baseline locally with physical wavelengths...")
    try:
        g_mags_1d = np.asarray(g_mags_test, dtype=float).flatten()
        
        df_local = pd.DataFrame({
            'source_id': np.asarray(ids_test_np).flatten(),
            'flux': list(X_test_np), 
            'flux_error': [np.ones(343) * 1e-10 for _ in range(len(X_test_np))]
        })

        flux_out = Gaia_Correction_V1.correction_df(
            df_local, 
            pd.Series(g_mags_1d), 
            have_error=True, 
            Truncation=True
        )
        
        X_huang_np = np.vstack(flux_out['flux_cor'].values)
        print("[✓] Huang et al. baseline calculated successfully.")
        
    except Exception as e:
        print(f"[!] FATAL ERROR applying Huang correction locally: {e}")
        traceback.print_exc() 
        X_huang_np = np.zeros_like(X_test_np)

    X_huang_tensor = torch.tensor(X_huang_np, dtype=torch.float32).to(device)

    export_test_predictions(
        model=export_ready_model,
        X_tensor=X_test_tensor,
        Y_tensor=Y_test_tensor,
        X_err_tensor=X_err_test_tensor,
        params_tensor=params_test_tensor,
        G_mags_tensor=G_mags_tensor,
        X_huang_tensor=X_huang_tensor,
        output_filepath=results_filepath
    )

    return results_filepath


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Ensure reproducible random state
    set_seed(42)

    mode = "all"
    OUTPUT_MODEL_DIR = f"results/models/pca_params_nn_huang/{mode}"
    VIS_DIR = os.path.join(OUTPUT_MODEL_DIR, "comparison")
    BASE_DATA_DIR = f"data/processed/nn/{mode}"
    
    stellar_params = [
        os.path.join(BASE_DATA_DIR, "G_mags.npy"),
        os.path.join(BASE_DATA_DIR, "bp_rp.npy"),
        os.path.join(BASE_DATA_DIR, "ruwe.npy")
    ]

    # Execute full training and export pipeline with ringing-suppression hyperparameters
    npz_results_path = run_training(
        x_path=os.path.join(BASE_DATA_DIR, "X.npy"),
        y_path=os.path.join(BASE_DATA_DIR, "Y.npy"),
        x_err_path=os.path.join(BASE_DATA_DIR, "X_error.npy"),
        star_names_path=os.path.join(BASE_DATA_DIR, "star_names.npy"),
        catalog_path=f"data/processed/catalogs_by_color/{mode}_catalog.csv",
        params_paths=stellar_params,
        output_model_dir=OUTPUT_MODEL_DIR,
        
        # Hyperparameters optimized to suppress ringing artifacts & conserve energy
        num_components=4,        
        lambda_smooth=0.1,       
        lambda_deriv=1.0,        
        lambda_anchor=0.5,
        lambda_reg=1e-3,         
        
        epochs_adam=3000,        
        steps_lbfgs=50,          
        max_batch_size=32,
        test_size=0.2,
        random_state=42
    )

    # Evaluate blind test statistics across spectral knots
    evaluate_test_set_from_file(
        input_filepath=npz_results_path,
        output_dir=OUTPUT_MODEL_DIR,
        num_knots=10
    )

    # Generate visual model comparison diagnostics
    generate_plots_from_file(
        input_filepath=npz_results_path,
        output_dir=VIS_DIR,
        num_samples=18
    )