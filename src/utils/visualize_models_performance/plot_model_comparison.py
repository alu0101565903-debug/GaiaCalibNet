"""
Module: plot_model_comparison.py
Report Section: Chapter 8 Model Evaluation & Visualization

Visualization module for the Residual Physics-Informed Neural Network (PINN).

This script generates quantitative and qualitative validation plots by loading 
exported `.npz` inference archives. It compares uncalibrated Gaia spectra, 
CALSPEC ground-truth references, baseline predictions (Huang et al. 2024), 
and PINN recalibrated predictions across individual stellar targets.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from src.utils.metrics.alternative_metrics import calculate_peak_normalized_mae


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

    if wave.ndim != 1 or X_orig.ndim != 2 or Y_ref.ndim != 2 or X_pinn.ndim != 2:
        raise ValueError("wave must be 1D and spectral arrays must be 2D")
    if any(array.shape != Y_ref.shape for array in (X_orig, X_pinn)):
        raise ValueError("X_orig, Y_ref, and X_pinn must have the same shape")
    if wave.size != Y_ref.shape[1]:
        raise ValueError("wave length must match the spectral channel count")

    # Summarize the blind-test residual distribution at every wavelength.
    peak_flux = np.nanmax(Y_ref, axis=1, keepdims=True)
    peak_flux = np.where(np.abs(peak_flux) > 1e-30, peak_flux, 1e-30)
    residuals = {
        "Original Gaia": (X_orig - Y_ref) / peak_flux * 100.0,
        "PINN": (X_pinn - Y_ref) / peak_flux * 100.0,
    }
    if X_huang is not None and X_huang.shape == Y_ref.shape and np.any(X_huang):
        residuals["Huang baseline"] = (X_huang - Y_ref) / peak_flux * 100.0

    summary_fig, summary_ax = plt.subplots(figsize=(11, 5.5))
    colors = {"Original Gaia": "#4f88f3", "Huang baseline": "#00a83b", "PINN": "#e63946"}
    for label, residual in residuals.items():
        median = np.nanmedian(residual, axis=0)
        lower, upper = np.nanpercentile(residual, [25, 75], axis=0)
        summary_ax.fill_between(wave, lower, upper, color=colors[label], alpha=0.12)
        summary_ax.plot(wave, median, color=colors[label], linewidth=0.8, label=f"{label} median")

    summary_ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.8)
    summary_ax.set_xlim(336, 1020)
    summary_ax.set_xlabel("Wavelength [nm]")
    summary_ax.set_ylabel("Median relative residual (%)")
    summary_ax.set_title("Median spectral residuals with interquartile range")
    summary_ax.grid(True, linestyle="--", alpha=0.35)
    summary_ax.legend(loc="best")
    summary_fig.savefig(output_path / "wavelength_residuals.pdf", dpi=150, bbox_inches="tight")
    plt.close(summary_fig)

    if X_huang is None or X_huang.shape != Y_ref.shape or not np.any(X_huang):
        raise ValueError("A valid X_huang array is required to generate baseline_boxplot.pdf")

    pn_mae_values = [
        [calculate_peak_normalized_mae(Y_ref[i], predictions[i]) for i in range(Y_ref.shape[0])]
        for predictions in (X_orig, X_huang, X_pinn)
    ]
    boxplot_fig, boxplot_ax = plt.subplots(figsize=(8, 5.5))
    boxplot = boxplot_ax.boxplot(
        pn_mae_values,
        tick_labels=["Original Gaia", "Huang Baseline", "PINN"],
        patch_artist=True,
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(boxplot["boxes"], ["#4f88f3", "#00a83b", "#e63946"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)

    boxplot_ax.set_xlabel("Model")
    boxplot_ax.set_ylabel("PN-MAE (%)")
    boxplot_ax.set_title("Peak-normalized MAE by model")
    boxplot_ax.grid(axis="y", linestyle="--", alpha=0.35)
    boxplot_fig.savefig(output_path / "baseline_boxplot.pdf", dpi=150, bbox_inches="tight")
    plt.close(boxplot_fig)

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


def plot_training_history(input_filepath: str | Path, output_filepath: str | Path) -> None:
    """Plot LOOCV training and held-out validation losses for both phases."""
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    if not input_path.exists():
        raise FileNotFoundError(f"Training history not found at: {input_path}")

    data = np.load(input_path)
    required_keys = {"base_train", "base_val", "blue_train", "blue_val"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise KeyError(f"Missing training history arrays: {missing_keys}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (axes[0], data["base_train"], data["base_val"], 600, "Base PINN (Phase 1)"),
        (axes[1], data["blue_train"], data["blue_val"], 150, "BlueWiggleNet (Phase 2)"),
    )
    for axis, train_history, val_history, epoch_limit, title in panels:
        train_history = np.atleast_2d(train_history)
        val_history = np.atleast_2d(val_history)
        train_median = np.nanmedian(train_history, axis=0)
        val_median = np.nanmedian(val_history, axis=0)
        train_low, train_high = np.nanpercentile(train_history, [25, 75], axis=0)
        val_low, val_high = np.nanpercentile(val_history, [25, 75], axis=0)
        train_epochs = np.arange(1, train_median.size + 1)
        val_epochs = np.arange(1, val_median.size + 1)

        axis.fill_between(train_epochs, train_low, train_high, color="#4f88f3", alpha=0.12)
        axis.fill_between(val_epochs, val_low, val_high, color="#e63946", alpha=0.12)
        axis.plot(train_epochs, np.clip(train_median, 1e-16, None), color="#4f88f3", linewidth=0.9, label="Training")
        axis.plot(val_epochs, np.clip(val_median, 1e-16, None), color="#e63946", linewidth=0.9, label="Validation")
        axis.set_xlim(0, epoch_limit)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(title)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
        axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_plots_from_file(
        input_filepath="results/test_predictions_final.npz", 
        output_dir="plots/validation_figures",
        num_samples=5
    )