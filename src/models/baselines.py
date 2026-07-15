"""
Module: baselines.py
Report Section: 6.2 Classical Baseline Models (Baseline)

This module initializes the non-neural classical machine learning baseline models 
using Scikit-Learn. These models act as a quantitative benchmark to evaluate 
whether high-capacity deep learning network architectures are mathematically 
justified for resolving Gaia DR3 instrumental systematic errors.
"""

import numpy as np
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
    
    # Rango de regularización L2 (alphas) a explorar analíticamente (de 10^-3 a 10^3)
    alphas_to_search = np.logspace(-3, 3, 13)
    
    models = {
        # RidgeCV: Resuelve la matriz (X^T X + alpha I)^-1 X^T Y
        # Evalúa automáticamente el mejor 'alpha' mediante validación cruzada interna (LOOCV).
        "Ridge_Regression": RidgeCV(
            alphas=alphas_to_search, 
            store_cv_values=False
        ),
        
        # Random Forest: Evalúa relaciones no lineales y es robusto al sobreajuste
        "Random_Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=random_seed,
            n_jobs=-1  # Utiliza todos los hilos físicos del procesador
        ),
        
        # SVR (RBF Kernel): Proyecta parámetros en espacios de Hilbert de dimensión infinita
        "SVR": MultiOutputRegressor(
            SVR(kernel='rbf', C=10.0, epsilon=0.01), # Ligero ajuste heurístico para flujos estelares
            n_jobs=-1
        ) 
    }
    
    return models