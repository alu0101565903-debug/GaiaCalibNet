"""
Module: plot_metric_limitation_analysis.py
Path: src/experiments/metrics/plot_metric_limitation_analysis.py

Purpose:
    Generates a comparative figure showing how standard global metrics 
    (MSE, R2) are blind to critical local failures (like edge divergences), 
    and proves how alternative custom metrics successfully expose them.
"""

import os
import matplotlib.pyplot as plt
from src.experiments.metrics.helpers import simulate_spectra, evaluate_predictions

def generate_metric_limitation_plot(out_dir: str = "results/plots/metrics_analysis"):
    """
    Generates and saves the side-by-side metric limitation analysis plot.

    Args:
    - out_dir (str): Output directory where the final PNG image is saved.
    """
    print("=" * 70)
    print("[GENERATING METRIC LIMITATION ANALYSIS PLOT]")
    print("=" * 70)

    # 1. Simulate single-spectrum case (n_samples = 1)
    wavelengths, y_true, y_pred_A, y_pred_B, _ = simulate_spectra(n_samples=1)

    # 2. Evaluate performance using the centralized helper
    metrics_A = evaluate_predictions(wavelengths, y_true, y_pred_A)
    metrics_B = evaluate_predictions(wavelengths, y_true, y_pred_B)

    # 3. Plot configuration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    c_true = "#1f77b4"  # Scientific blue
    c_pred = "#e31a1c"  # Alarming red

    # --- PANEL 1: Model A (Edge Failure) ---
    print("-> Plotting Model A (Edge Divergence Case)...")
    ax1.plot(wavelengths, y_true, label="Target Ground Truth (CALSPEC)", color=c_true, linewidth=2.5)
    ax1.plot(wavelengths, y_pred_A, label="Model A Reconstruction", color=c_pred, linestyle="--", linewidth=2.0)
    
    # Shade the edge failure zone (< 360 nm)
    ax1.axvspan(330, 360, color="yellow", alpha=0.15, label="Edge Divergence Zone (< 360 nm)")
    ax1.set_title("Model A: Localized Spectral Edge Divergence", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Wavelength [nm]", fontsize=11)
    ax1.set_ylabel("Normalized Flux", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10)
    
    text_A = (
        f"Standard Metrics (Insensitive to Edges):\n"
        f"  • MSE: {metrics_A['mse']:.5f} (Optimal)\n"
        f"  • $R^2$: {metrics_A['r2']:.4f} (Optimal)\n\n"
        f"Alternative Metrics (Exposes Failure):\n"
        f"  • Blue Band Err: {metrics_A['blue_err']:.4f}\n"
        f"  • MAPE: {metrics_A['mape']:.2f}%\n"
        f"  • Pseudo-$\\chi^2$: {metrics_A['pseudo_chi2']:.2f} (Critical)"
    )
    ax1.text(620, 0.15, text_A, fontsize=10.5, 
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.9))

    # --- PANEL 2: Model B (Noisy Baseline) ---
    print("-> Plotting Model B (Noisy but Consistent Case)...")
    ax2.plot(wavelengths, y_true, label="Target Ground Truth (CALSPEC)", color=c_true, linewidth=2.5)
    ax2.plot(wavelengths, y_pred_B, label="Model B Reconstruction", color=c_pred, linestyle="--", alpha=0.8, linewidth=1.5)
    
    ax2.set_title("Model B: Uniform Distributed Instrumental Noise", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Wavelength [nm]", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)
    
    text_B = (
        f"Standard Metrics (Penalizes Noise):\n"
        f"  • MSE: {metrics_B['mse']:.5f} (Suboptimal)\n"
        f"  • $R^2$: {metrics_B['r2']:.4f} (Suboptimal)\n\n"
        f"Alternative Metrics (Validates Continuum):\n"
        f"  • Blue Band Err: {metrics_B['blue_err']:.4f}\n"
        f"  • MAPE: {metrics_B['mape']:.2f}%\n"
        f"  • Pseudo-$\\chi^2$: {metrics_B['pseudo_chi2']:.2f} (Resolved)"
    )
    ax2.text(620, 0.15, text_B, fontsize=10.5, 
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()

    # Save visualization to disk
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "spectral_metrics_blindspot_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print("-" * 70)
    print(f"[+] Plot saved to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    generate_metric_limitation_plot()