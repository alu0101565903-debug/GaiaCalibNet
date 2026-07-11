"""
Module: baselines.py
Report Section: 6.2 Classical Baseline Models (Baseline)

This module initializes the non-neural classical machine learning baseline models 
using Scikit-Learn. These models act as a quantitative benchmark to evaluate 
whether high-capacity deep learning network architectures are mathematically 
justified for resolving Gaia DR3 instrumental systematic errors.
"""

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

def get_baseline_models(random_seed: int = 42) -> dict:
    """
    Initializes and returns a dictionary of classical ML baseline models.
    
    Since SVR does not natively support multi-output regression (predicting 
    300 channels at once), it is wrapped in a MultiOutputRegressor.

    Parameters
    - random_seed (int): Seed for reproducibility across experiments.

    Returns
    - dict
        A dictionary containing instantiated, untrained Scikit-Learn models.
    """
    models = {
        # Ridge Regression: Evaluates multicollinearity and structural linear mappings
        "Ridge_Regression": Ridge(alpha=1.0, random_state=random_seed),
        
        # Random Forest: Evaluates non-linear relationships and feature importance
        "Random_Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=random_seed,
            n_jobs=-1  # Use all available CPU cores
        ),
        
        # SVR (RBF Kernel): Projects parameters into higher-dimensional Hilbert spaces
        "SVR": MultiOutputRegressor(
            SVR(kernel='rbf', C=1.0, epsilon=0.1), 
            n_jobs=-1) # Concurrently fit independent wavelength models
    }
    
    return models
