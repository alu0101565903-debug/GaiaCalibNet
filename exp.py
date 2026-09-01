from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import norm

import gaiaxpy
from GaiaXPcorrection.GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1


# =================================================================
# 1. ARQUITECTURAS PINN (Extraídas de loocv_residual_pinn.py)
# =================================================================

class DenseResidualPINN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_params: int = 3, 
        num_coarse_bins: int = 60, 
        max_correction: float = 0.8, 
        hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.num_coarse_bins = num_coarse_bins
        self.max_correction = max_correction
        
        total_features = num_coarse_bins + num_params
        self.mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(1)
        x_coarse = F.adaptive_avg_pool1d(x_in, self.num_coarse_bins).flatten(1)
        combined = torch.cat([x_coarse, params], dim=1)
        raw_delta = self.mlp(combined)
        bounded_delta = self.max_correction * torch.tanh(raw_delta / self.max_correction)
        return 1.0 + bounded_delta


class Conv1DResidualPINN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_params: int = 3, 
        max_correction: float = 0.8
    ) -> None:
        super().__init__()
        self.max_correction = max_correction
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 + num_params, 256),
            nn.GELU(),
            nn.Linear(256, input_dim)
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(1)
        conv_features = self.conv_blocks(x_in).flatten(1)
        combined = torch.cat([conv_features, params], dim=1)
        raw_delta = self.fc(combined)
        bounded_delta = self.max_correction * torch.tanh(raw_delta / self.max_correction)
        return 1.0 + bounded_delta


# =================================================================
# 2. FUNCIONES DE CARGA E INFERENCIA
# =================================================================

def load_pinn_model(weights_path: str | Path, architecture: str, input_dim: int, device: str = "cpu"):
    """
    Carga el modelo PINN preentrenado utilizando los pesos guardados (sin reentrenar).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos en: {weights_path}")
    
    if architecture.lower() == "dense":
        model = DenseResidualPINN(input_dim=input_dim)
    else:
        model = Conv1DResidualPINN(input_dim=input_dim)
        
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model


def fetch_and_correct_target(source_id: int, g_mag: float, bp_rp: float, ruwe: float, wavelength_grid: np.ndarray):
    """
    Obtiene el espectro de Gaia para el source_id dado y aplica la corrección de GaiaXPcalibration (Huang et al.).
    """
    print(f"[>] Descargando espectro de Gaia para ID: {source_id}...")
    calibrated_spectra, sampling = gaiaxpy.calibrate([source_id], sampling=wavelength_grid)
    
    X_orig = calibrated_spectra['flux'].iloc[0]
    X_err = calibrated_spectra['flux_error'].iloc[0]
    
    print(f"[>] Aplicando corrección de GaiaXPcalibration (Huang et al.)...")
    df_input = pd.DataFrame({
        'source_id': [source_id],
        'flux': [X_orig],
        'flux_error': [X_err]
    })
    g_mags_series = pd.Series([g_mag])
    
    flux_out = Gaia_Correction_V1.correction_df(
        df_input, g_mags_series, have_error=True, Truncation=True
    )
    X_calib = np.vstack(flux_out['flux_cor'].values)[0]
    
    # Normalizar espectro y preparar parámetros estelares para la PINN
    X_norm = X_orig / (np.max(X_orig) + 1e-30)
    params = np.array([[bp_rp, g_mag, ruwe]], dtype=np.float32)
    
    return X_orig, X_err, X_calib, X_norm, params


# =================================================================
# 3. VISUALIZACIÓN (Estructura de 3 paneles)
# =================================================================

def plot_target_comparison(
    wave: np.ndarray,
    X_orig: np.ndarray,
    X_pinn: np.ndarray,
    X_calib: np.ndarray,
    Y_ref: np.ndarray = None,
    source_id: int | str = "Unknown",
    save_path: str | Path = None
) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[2, 1], wspace=0.06, hspace=0.1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)

    # --- Panel Superior: Flujo Absoluto ---
    if Y_ref is not None:
        ax1.plot(wave, Y_ref, label="CALSPEC Reference ($Y$)", color="black", alpha=0.85, linewidth=2)
    ax1.plot(wave, X_orig, label="Original Gaia BP/RP ($X$)", color="#4f88f3", alpha=0.6, linewidth=1.5)
    ax1.plot(wave, X_calib, label="GaiaXPcalibration (Huang)", color="#00cf49", alpha=0.9, linewidth=1.5, linestyle="--")
    ax1.plot(wave, X_pinn, label=r"PINN Prediction ($X \cdot R$)", color="#e63946", alpha=0.8, linewidth=1.5)

    ax1.set_title(f"Recalibration Comparison | Gaia Source ID: {source_id}")
    ax1.set_ylabel("Absolute Flux")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- Panel Inferior Izquierdo: Residuales (%) ---
    base_spectrum = Y_ref if Y_ref is not None else X_orig
    peak_flux = np.max(base_spectrum) + 1e-30

    res_original_pct = ((X_orig - base_spectrum) / peak_flux) * 100.0
    res_calib_pct = ((X_calib - base_spectrum) / peak_flux) * 100.0
    res_pinn_pct = ((X_pinn - base_spectrum) / peak_flux) * 100.0

    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85)
    if Y_ref is not None:
        ax2.plot(wave, res_original_pct, label="Original Gaia Residual", color="#4f88f3", alpha=0.75, linewidth=1.0)
    ax2.plot(wave, res_calib_pct, label="GaiaXPcalib Residual", color="#00cf49", alpha=0.8, linewidth=1.5, linestyle="--")
    ax2.plot(wave, res_pinn_pct, label="PINN Residual", color="#e63946", alpha=0.9, linewidth=1.5)

    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("Residual (%)")
    ax2.set_ylim(-20, 20)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize="small")

    # --- Panel Inferior Derecho: Histograma de Densidad y Gaussiana ---
    bins = np.linspace(-20, 20, 40)
    if Y_ref is not None:
        ax3.hist(res_original_pct, bins=bins, orientation="horizontal", color="#4f88f3", alpha=0.3, density=True)
    ax3.hist(res_calib_pct, bins=bins, orientation="horizontal", color="#00cf49", alpha=0.3, density=True)
    ax3.hist(res_pinn_pct, bins=bins, orientation="horizontal", color="#e63946", alpha=0.6, density=True)

    mu, std = norm.fit(res_pinn_pct)
    y_vals = np.linspace(-20, 20, 100)
    p = norm.pdf(y_vals, mu, std)
    ax3.plot(p, y_vals, "k--", linewidth=1.5, label=rf"PINN Fit: $\mu$={mu:.1f}, $\sigma$={std:.1f}")

    ax3.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85)
    ax3.set_xlabel("Density")
    ax3.tick_params(labelleft=False)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize="x-small", loc="upper right")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Gráfica guardada en: {save_path}")
    
    plt.show()


# =================================================================
# 4. EJECUCIÓN PRINCIPAL
# =================================================================

def main():
    GAIA_SOURCE_ID = 3811802956277083904  # Cambia por tu ID de Gaia objetivo
    G_MAG_VAL = 12.5                      # Magnitud G de la estrella
    BP_RP_VAL = 1.0                       # Color BP-RP estelar
    RUWE_VAL = 1.0                        # Parámetro RUWE de Gaia
    
    WEIGHTS_PATH = "results/models/loocv_residual_pinn/all/conv1d_bins60_smooth0.05_anchor0.0/best_model.pth" # Ajusta la ruta a tus pesos
    ARCHITECTURE = "conv1d"               # "dense" o "conv1d" según corresponda
    OUTPUT_PLOT = f"plots/inference_{GAIA_SOURCE_ID}.png"
    
    wave_grid = np.linspace(336.0, 1020.0, 343)
    num_channels = len(wave_grid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Obtener espectro original, corregido por Huang y normalizado
    X_orig, X_err, X_calib, X_norm, params = fetch_and_correct_target(
        GAIA_SOURCE_ID, G_MAG_VAL, BP_RP_VAL, RUWE_VAL, wave_grid
    )

    # 2. Cargar modelo PINN con pesos preentrenados
    print(f"[>] Cargando modelo PINN desde {WEIGHTS_PATH}...")
    model = load_pinn_model(WEIGHTS_PATH, architecture=ARCHITECTURE, input_dim=num_channels, device=device)

    # 3. Inferencia (sin reentrenar)
    print(f"[>] Ejecutando inferencia con el modelo PINN...")
    x_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)
    params_tensor = torch.tensor(params, dtype=torch.float32).to(device)

    with torch.no_grad():
        r_pred = model(x_tensor, params_tensor)
        r_pred_np = r_pred.squeeze(0).cpu().numpy()
        X_pinn = X_orig * r_pred_np

    # 4. Graficar resultados
    print(f"[>] Generando gráfico comparativo...")
    plot_target_comparison(
        wave=wave_grid,
        X_orig=X_orig,
        X_pinn=X_pinn,
        X_calib=X_calib,
        Y_ref=None,  # Pasa un array con CALSPEC real si lo tienes disponible para este ID
        source_id=GAIA_SOURCE_ID,
        save_path=OUTPUT_PLOT
    )


if __name__ == "__main__":
    main()