"""
Module: baselines.py
Report Section: 6.2 Classical Baseline Models (Baseline)

This module initializes the non-neural classical machine learning baseline models 
using Scikit-Learn. These models act as a quantitative benchmark to evaluate 
whether high-capacity deep learning network architectures are mathematically 
justified for resolving Gaia DR3 instrumental systematic errors.
"""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
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
    
    # L2 regularization grid to search (10^-3 .. 10^3) for RidgeCV.
    alphas_to_search = np.logspace(-3, 3, 13)

    models = {
        # Dummy Regressor: a zero-skill baseline that always predicts the mean
        # spectrum. Useful to confirm that any positive R^2 from other models
        # corresponds to learned physical signal rather than data bias.
        "Zero_Skill_Mean": DummyRegressor(strategy="mean"),

        # RidgeCV: L2-regularized linear model. Automatically selects the best
        # "alpha" from "alphas_to_search" using internal cross-validation.
        "Ridge_Regression": RidgeCV(
            alphas=alphas_to_search
        ),

        # Random Forest: non-linear ensemble baseline; captures interactions
        # and tends to be robust to overfitting when tuned conservatively.
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_seed,
            n_jobs=-1,  # use all available CPU cores
        ),

        # Support Vector Regressor (RBF kernel) wrapped for multi-output.
        # SVR is a strong non-linear baseline; the wrapper enables predicting
        # the full spectrum (multi-channel) by fitting one estimator per target.
        "SVR": MultiOutputRegressor(
            SVR(kernel="rbf", C=10.0, epsilon=0.01),
            n_jobs=-1,
        ),
    }
    
    return models