"""
Script: test_alternative_metrics.py
Report Section:

Description:
    A lightweight, deterministic experiment to quickly demonstrate and verify
    how custom astronomical metrics capture localized spectral failures that
    standard statistical metrics (MSE, R2) easily miss.
"""

from src.experiments.metrics_performance.helpers import simulate_spectra, evaluate_predictions

def run_experiment():
    """
    Executes the quick terminal-based metric validation experiment.
    """
    print("============================================================")
    print(" RUNNING SPECTRAL METRICS EXPERIMENT")
    print("============================================================\n")

    # 1. Simulate single spectra (n_samples = 1) using the helper
    wavelengths, y_true, y_pred_A, y_pred_B, _ = simulate_spectra(n_samples=1)

    # 2. Evaluate performance
    metrics_A = evaluate_predictions(wavelengths, y_true, y_pred_A)
    metrics_B = evaluate_predictions(wavelengths, y_true, y_pred_B)

    # 3. Print Results
    print("--> EVALUATING MODEL A (Edge failure at blue wavelengths)")
    print(f"MSE Standard: {metrics_A['mse']:.5f}")
    print(f"R^2 Score:    {metrics_A['r2']:.5f}")
    print(f"MAPE (%):     {metrics_A['mape']:.2f}%")
    print(f"Pseudo-Chi2:  {metrics_A['pseudo_chi2']:.2f}")
    print(f"Blue Band Err:{metrics_A['blue_err']:.4f}\n")

    print("--> EVALUATING MODEL B (Uniform instrumental noise)")
    print(f"MSE Standard: {metrics_B['mse']:.5f}")
    print(f"R^2 Score:    {metrics_B['r2']:.5f}")
    print(f"MAPE (%):     {metrics_B['mape']:.2f}%")
    print(f"Pseudo-Chi2:  {metrics_B['pseudo_chi2']:.2f}")
    print(f"Blue Band Err:{metrics_B['blue_err']:.4f}\n")

    print("============================================================")
    print("CONCLUSION:")
    print("Model A might look acceptable using MSE or R^2 because the central")
    print("stellar peak dominates the global average, but alternative metrics")
    print("like MAPE and Pseudo-Chi2 instantly reveal the critical blue edge failure.")


if __name__ == "__main__":
    run_experiment()