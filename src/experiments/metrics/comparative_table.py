"""
Module: comparative_table.py
Path: src/experiments/metrics/comparative_table.py

Purpose:
    Generates a compact comparative table (Markdown, LaTeX, and high-DPI PNG)
    contrasting standard statistical metrics against domain-specific diagnostics 
    across Model A, Model B, and the Physics-Informed Neural Network (PINN).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.experiments.metrics.helpers import simulate_spectra, evaluate_predictions

if __name__ == "__main__": 

    # 1. Simulate batch data (100 synthetic stars with 300 pixels each)
    wavelengths, y_true, y_pred_A, y_pred_B, y_pred_pinn = simulate_spectra(n_samples=100)

    # 2. Evaluate metrics using the centralized helper module
    metrics_A = evaluate_predictions(wavelengths, y_true, y_pred_A)
    metrics_B = evaluate_predictions(wavelengths, y_true, y_pred_B)
    metrics_pinn = evaluate_predictions(wavelengths, y_true, y_pred_pinn)

    # 3. Structure the consolidated DataFrame
    summary_data = {
        "Model Architecture": [
            "Model A (Edge Failure Baseline)",
            "Model B (Uniform Noise Baseline)",
            "Our Physics-Informed NN (PINN)"
        ],
        "MSE (x10^-3)": [
            round(metrics_A["mse"] * 1000, 4), 
            round(metrics_B["mse"] * 1000, 4), 
            round(metrics_pinn["mse"] * 1000, 4)
        ],
        "R2 Score": [
            round(metrics_A["r2"], 4), 
            round(metrics_B["r2"], 4), 
            round(metrics_pinn["r2"], 4)
        ],
        "MAPE (%)": [
            round(metrics_A["mape"], 2), 
            round(metrics_B["mape"], 2), 
            round(metrics_pinn["mape"], 2)
        ],
        "Blue Band Err": [
            round(metrics_A["blue_err"], 4), 
            round(metrics_B["blue_err"], 4), 
            round(metrics_pinn["blue_err"], 4)
        ],
        "Pseudo-Chi2": [
            round(metrics_A["pseudo_chi2"], 2), 
            round(metrics_B["pseudo_chi2"], 2), 
            round(metrics_pinn["pseudo_chi2"], 2)
        ],
    }
    df = pd.DataFrame(summary_data)

    # EXPORTING TO TEXT FORMATS (Stdout)
    print("\n=== MARKDOWN TABLE FORMAT (For GitHub / PRs) ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))

    print("\n=== LATEX TABLE FORMAT (For your TFG / Thesis Document) ===")
    print(df.to_latex(index=False, float_format="%.4f", caption="Model Comparison with Custom Metrics"))

    # EXPORTING TO SCIENTIFIC PUBLICATION-GRADE IMAGE
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparative_table.png")

    fig, ax = plt.subplots(figsize=(15, 2.8), dpi=300)
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_column_width(col=list(range(len(df.columns))))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Apply aesthetic styling to the Matplotlib table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        
        if row == 0:
            # Table Header styling
            cell.set_text_props(weight='bold', color='#111111')
            cell.set_facecolor('#f2f5f8')
            cell.set_linewidth(1.5)
        else:
            # Highlight our PINN model row to emphasize superior scientific performance
            if row == 3:
                cell.set_text_props(weight='bold', color='#0f4c81')
                cell.set_facecolor('#f4f9fc')
                cell.set_linewidth(1.0)
            else:
                cell.set_facecolor('#ffffff')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nDone — Image generated and saved: {output_path}")