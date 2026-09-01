"""
Module: plot_model_comparison.py
Report Section: Chapter 8 Model Evaluation & Visualization

Visualization module for the Residual Physics-Informed Neural Network (PINN).

This script generates quantitative and qualitative validation plots by loading 
exported `.npz` inference archives. It compares uncalibrated Gaia spectra, 
CALSPEC ground-truth references, baseline predictions (Huang et al. 2024), 
and PINN recalibrated predictions across individual stellar targets.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def generate_plots_from_file(
    input_filepath: str | Path, 
    output_dir: str | Path, 
    num_samples: int = 5
) -> None:
    """
    Read inference results from a compressed `.npz` archive and generate validation plots.

    Extracts the wavelength grid, original uncalibrated spectra, reference spectra, 
    baseline predictions, and PINN predictions from the archive. For each target sample, 
    it generates a three-panel figure:
    1. Top Left: Absolute flux comparison across wavelength (shares X-axis with bottom left).
    2. Bottom Left: Percentage residual relative to the reference spectrum peak flux.
    3. Bottom Right: Horizontal density histogram of residuals with a fitted theoretical Gaussian curve.

    Parameters
    ----------
    input_filepath : str | Path
        Full path to the `.npz` archive containing exported test predictions.
    output_dir : str | Path
        Destination directory where generated validation PNG figures will be saved. 
        Created automatically if it does not exist.
    num_samples : int, default=5
        Maximum number of samples to plot starting from index 0. If the dataset 
        contains fewer samples than requested, all available samples are plotted.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the `.npz` file specified by `input_filepath` does not exist on disk.
    KeyError
        If mandatory data arrays (`wave`, `X_orig`, `Y_ref`, `X_pinn`) are missing from the archive.
    """
    input_path = Path(input_filepath)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found at: {input_path}")

    data = np.load(input_path)
    
    required_keys = {"wave", "X_orig", "Y_ref", "X_pinn"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise KeyError(f"Missing required arrays in `.npz` archive: {missing_keys}")

    wave = data["wave"]
    X_orig = data["X_orig"]
    Y_ref = data["Y_ref"]
    X_pinn = data["X_pinn"]
    X_huang = data["X_huang"] if "X_huang" in data.files else None

    output_path.mkdir(parents=True, exist_ok=True)
    samples_to_plot = min(num_samples, X_orig.shape[0])

    print(f"\n[>] Generating {samples_to_plot} figures from archive: {input_path}")

    for i in range(samples_to_plot):
        # Create figure with GridSpec layout for main plots and side residual histogram
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(
            2, 2, 
            width_ratios=[2.5, 1], 
            height_ratios=[2, 1], 
            wspace=0.06, 
            hspace=0.1
        )

        # Subplot assignments ensuring X-axis alignment between ax1 and ax2
        ax1 = fig.add_subplot(gs[0, 0])                             # Top Left: Absolute Flux
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)                 # Bottom Left: Residuals
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)                 # Bottom Right: Density Histogram

        # ---------------------------------------------------------
        # Top Panel: Absolute Flux Comparison
        # ---------------------------------------------------------
        ax1.plot(wave, Y_ref[i], label="CALSPEC Reference ($Y$)", color="black", alpha=0.85, linewidth=2)
        ax1.plot(wave, X_orig[i], label="Original Gaia BP/RP ($X$)", color="#4f88f3", alpha=0.6, linewidth=1.5)

        if X_huang is not None and np.any(X_huang[i] > 0):
            ax1.plot(
                wave, X_huang[i], 
                label="Huang et al. 2024 Baseline", 
                color="#00cf49", alpha=0.9, linewidth=1.5, linestyle="--"
            )

        ax1.plot(wave, X_pinn[i], label=r"PINN Prediction ($X \cdot R$)", color="#e63946", alpha=0.8, linewidth=1.5)

        ax1.set_title(f"Blind Test PINN vs Baseline | Star Index: {i:03d}")
        ax1.set_ylabel("Absolute Flux")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        # Hide top panel X-axis labels to prevent overlap with the bottom panel
        plt.setp(ax1.get_xticklabels(), visible=False)

        # ---------------------------------------------------------
        # Bottom Left Panel: Percentage Residual (%)
        # ---------------------------------------------------------
        peak_flux = np.max(Y_ref[i]) + 1e-30
        res_original_pct = ((X_orig[i] - Y_ref[i]) / peak_flux) * 100.0
        res_pinn_pct = ((X_pinn[i] - Y_ref[i]) / peak_flux) * 100.0

        ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85)
        ax2.plot(wave, res_original_pct, label="Original Gaia Residual", color="#4f88f3", alpha=0.75, linewidth=1.0)

        if X_huang is not None and np.any(X_huang[i] != 0):
            res_baseline_pct = ((X_huang[i] - Y_ref[i]) / peak_flux) * 100.0
            ax2.plot(
                wave, res_baseline_pct, 
                label="Huang Baseline Residual", 
                color="#00cf49", alpha=0.8, linewidth=1.5, linestyle="--"
            )

        ax2.plot(wave, res_pinn_pct, label="PINN Residual", color="#e63946", alpha=0.9, linewidth=1.5)

        ax2.set_xlabel("Wavelength [nm]")
        ax2.set_ylabel("Residual (%)")
        ax2.set_ylim(-20, 20)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="upper right", fontsize="small")

        # ---------------------------------------------------------
        # Bottom Right Panel: Residual Histogram with Gaussian Fit
        # ---------------------------------------------------------
        bins = np.linspace(-20, 20, 40)

        # Draw background density histograms for context
        ax3.hist(res_original_pct, bins=bins, orientation="horizontal", color="#4f88f3", alpha=0.3, density=True)
        if X_huang is not None and np.any(X_huang[i] != 0):
            ax3.hist(res_baseline_pct, bins=bins, orientation="horizontal", color="#00cf49", alpha=0.3, density=True)
            
        # Draw density histogram for PINN residuals
        ax3.hist(res_pinn_pct, bins=bins, orientation="horizontal", color="#e63946", alpha=0.6, density=True)

        # Fit Gaussian distribution to PINN residuals
        mu, std = norm.fit(res_pinn_pct)

        # Generate theoretical Gaussian curve
        y_vals = np.linspace(-20, 20, 100)
        p = norm.pdf(y_vals, mu, std)

        # Overlay theoretical Gaussian fit
        ax3.plot(p, y_vals, "k--", linewidth=1.5, label=rf"Fit: $\mu$={mu:.1f}, $\sigma$={std:.1f}")

        ax3.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85)
        ax3.set_xlabel("Density")
        
        # Hide left Y-axis tick labels (shared with ax2)
        ax3.tick_params(labelleft=False)
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(fontsize="x-small", loc="upper right")

        # ---------------------------------------------------------
        # Save Figure
        # ---------------------------------------------------------
        save_file = output_path / f"star_comparison_test_{i:03d}.png"
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[✓] Visualization complete. {samples_to_plot} plots saved to: {output_path}")


if __name__ == "__main__":
    generate_plots_from_file(
        input_filepath="results/test_predictions_final.npz", 
        output_dir="plots/validation_figures",
        num_samples=5
    )