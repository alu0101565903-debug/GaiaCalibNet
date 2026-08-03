"""
Module: check_pca.py
Spectral Correction Factor PCA Variance Analysis.

This script computes the cumulative explained variance ratio for Principal Component
Analysis (PCA) applied to spectral correction factors (R = Y / X). It determines the 
optimal number of principal components needed to capture a specified target variance 
threshold (e.g., 99.9%) and generates a visualization plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_pca_variance(x_path: str, y_path: str, target_variance: float = 0.999) -> None:
    """
    Analyzes the cumulative explained variance of PCA applied to spectral 
    correction factors (R = Y / X) to determine the optimal number of components.

    Args:
    - x_path (str): Path to the input spectra NumPy file (X.npy).
    - y_path (str): Path to the target spectra NumPy file (Y.npy).
    - target_variance (float, optional): Target cumulative variance fraction (e.g., 0.999). 
        Defaults to 0.999.
    """
    print("Loading data...")
    X = np.load(x_path)
    Y = np.load(y_path)
    
    # Calculate the ideal correction factor R = Y / X
    # Add 1e-15 to prevent division by zero
    R = Y / (X + 1e-15)
    
    # The maximum possible number of components is the minimum between 
    # the number of samples (stars) and the number of pixels (wavelengths, e.g., 343)
    max_components = min(R.shape[0], R.shape[1])
    
    print(f"Calculating PCA up to {max_components} components...")
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(R)
    
    # Calculate cumulative sum of explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the exact number of components required to hit the target variance (e.g., 99.9%)
    # argmax returns the first index where condition is met; +1 handles 0-based indexing
    optimal_components = np.argmax(cumulative_variance >= target_variance) + 1
    
    print(f"\n--- RESULTS ---")
    print(f"To explain {target_variance * 100:.2f}% of variance, you need: {optimal_components} components.")
    
    # --- Plotting cumulative variance ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_variance) + 1), 
        cumulative_variance, 
        marker='o', 
        linestyle='-', 
        markersize=4
    )
    
    # Reference thresholds
    plt.axhline(y=0.99, color='r', linestyle='--', label='99.0% Variance')
    plt.axhline(y=0.999, color='g', linestyle='--', label='99.9% Variance')
    plt.axvline(
        x=optimal_components, 
        color='k', 
        linestyle=':', 
        label=f'{optimal_components} components'
    )
    
    plt.title('Cumulative Explained Variance of Correction Factor (R)')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance')
    plt.grid(True, alpha=0.5)
    plt.legend()
    
    # Save output plot
    plt.savefig("pca_variance_analysis.png", dpi=300, bbox_inches='tight')
    print("Plot saved as 'pca_variance_analysis.png'.")

if __name__ == "__main__":
    # Use paths for your solar dataset (or blue/red datasets)
    analyze_pca_variance(
        x_path="data/processed/nn/solar/X.npy",
        y_path="data/processed/nn/solar/Y.npy",
        target_variance=0.999  # Modify to 0.995 if 0.999 requires too many components
    )