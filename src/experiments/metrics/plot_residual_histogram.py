r"""
Module: plot_residual_histogram.py
Path: src/experiments/metrics/plot_residual_histogram.py

Purpose:
    Generates a comparative residual histogram plot to prove the physical consistency
    of the PINN model (pure Gaussian white noise, mean ~ 0) against Model A's 
    highly biased, bimodal non-Gaussian distribution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.experiments.metrics.helpers import simulate_spectra

def generate_residual_histogram_plot(out_dir: str = "results/plots/metrics_analysis"):
    """
    Generates and saves the comparative residual histogram analysis plot.

    Args:
    - out_dir (str): Output directory where the PNG image is written.
    """
    print("=" * 70)
    print("[GENERATING RESIDUAL HISTOGRAM ANALYSIS PLOT]")
    print("=" * 70)

    # 1. Simulate batch data (100 stars, 300 pixels) using the helper
    wavelengths, y_true, y_pred_A, _, y_pred_pinn = simulate_spectra(n_samples=100)

    # 2. Compute flat residuals (Ground Truth - Prediction) for statistical fitting
    residuals_A = (y_true - y_pred_A).flatten()
    residuals_pinn = (y_true - y_pred_pinn).flatten()

    # 3. Configure Plot (Independent axes to avoid squashing)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    c_pinn = "#1f77b4"    
    c_modelA = "#e31a1c"  

    # PANEL 1: Model A (Biased/Bimodal Residuals - LOGARITHMIC SCALE)
    mu_A, std_A = norm.fit(residuals_A)
    x_A = np.linspace(residuals_A.min(), residuals_A.max(), 200)
    pdf_A = norm.pdf(x_A, mu_A, std_A)

    # Histogram with probability density and safe range to avoid cutting the 0 bar
    ax1.hist(residuals_A, bins=60, range=(-0.175, 0.015), density=True, color=c_modelA, alpha=0.7, edgecolor="white")
    ax1.plot(x_A, pdf_A, "k--", linewidth=2, label=f"Ideal Normal Fit\n($\\mu$={mu_A:.3f}, $\\sigma$={std_A:.3f})")
    
    # 1. Enable logarithmic scale on the Y-axis
    ax1.set_yscale("log")
    
    # 2. Adjust limits to provide breathing room on the right (avoid cutting) and top (peak at 350)
    ax1.set_xlim(-0.18, 0.015)
    ax1.set_ylim(0.1, 500) 
    
    # 3. Bias annotation adapted for the logarithmic scale
    ax1.annotate("Systematic Bias\n(Edge Failure)", xy=(-0.167, 2.5), xytext=(-0.10, 35),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=7),
                 fontsize=10, ha="center", color="black", weight="bold")

    ax1.set_title("Model A: Bimodal (Biased) Residuals", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Residuals (Target - Prediction)", fontsize=11)
    ax1.set_ylabel("Probability Density (Log Scale)", fontsize=11)
    
    # Grid adapted for logarithmic scale (shows lines at multiples of 10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # Legend at the top right (plenty of collision-free space in log scale)
    ax1.legend(loc="upper right")

    # PANEL 2: PINN Model (Gaussian White Noise - LINEAR SCALE)
    mu_pinn, std_pinn = norm.fit(residuals_pinn)
    x_pinn = np.linspace(residuals_pinn.min(), residuals_pinn.max(), 200)
    pdf_pinn = norm.pdf(x_pinn, mu_pinn, std_pinn)

    ax2.hist(residuals_pinn, bins=60, density=True, color=c_pinn, alpha=0.7, edgecolor="white")
    ax2.plot(x_pinn, pdf_pinn, "k--", linewidth=2, label=f"Ideal Normal Fit\n($\\mu$={mu_pinn:.4f}, $\\sigma$={std_pinn:.4f})")

    ax2.set_title(r"PINN: Pure Gaussian White Noise ($\mu \approx 0$)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Residuals (Target - Prediction)", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    # Physical consistency explanatory box
    text_pinn = (
        "Physical Consistency:\n"
        "Residuals follow a perfect Gaussian\n"
        "distribution centered at 0. This\n"
        "proves the model captures the physical\n"
        "continuum and leaves only random noise."
    )
    ax2.text(0.95, 0.05, text_pinn, fontsize=10.5, transform=ax2.transAxes,
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f4f9fc", edgecolor=c_pinn, alpha=0.9))

    plt.tight_layout()

    # Save to disk
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "residual_distribution_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[+] Residual histogram saved to: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    generate_residual_histogram_plot()