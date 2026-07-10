"""
Module: neural_net.py
Report Section: 6.3 Deep Learning Neural Network Architecture (MLP - Multiplicative Residual)

This module defines the Multi-Layer Perceptron (MLP) architecture customized for 
the Gaia-CALSPEC spectrophotometric recalibration pipeline. It implements a 
multiplicative physics-inspired residual mapping: instead of predicting absolute fluxes,
the network outputs a positive correction vector applied directly to the source Gaia grid.
"""

import torch
import torch.nn as nn

class SpectralRecalibrationMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture tailored for Gaia-CALSPEC Spectral Recalibration.
    
    Architecture (Multiplicative Residual Design):
    - Input: Gaia DR3 observed dataset (either the 300-channel aligned grid vectors 
        or the isolated 5 scalar astrometric/photometric features).
    - Hidden layers: Dynamically defined by hidden_dims with dropout regularization.
    - Output: Multiplicative correction vector mapped strictly to positive bounds via Softplus,
        yielding F_pred = F_Gaia * Correction_Factor.
    """
    def __init__(self, input_dim=300, output_dim=300, hidden_dims=[256, 512, 256], 
                 activation=nn.GELU(), dropout_rate=0.2):
        """
        Args:
        - input_dim (int): Dimension of the input features (default: 300 for aligned grids).
        - output_dim (int): Dimension of the output spectrum (default: 300 channels).
        - hidden_dims (list): List containing the number of neurons per hidden layer.
        - activation (nn.Module): Instantiated PyTorch activation function.
        - dropout_rate (float): Dropout probability to handle variance in compact catalogs.
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        # Dynamically build hidden layers following your structural design
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            # Opcional: Descomenta la siguiente línea si buscas mayor estabilidad con L-BFGS:
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim 

        # Final linear regression layer to project onto the 300 spectral channels
        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)
        
        # Enforce that the multiplicative correction factor is strictly positive
        self.positive_activation = nn.Softplus()

    def forward(self, x, gaia_features=None):
        """
        Performs the multiplicative forward pass of the model.
        Maps inputs to a correction factor vector and applies it to the original Gaia spectrum.
        """
        if gaia_features is None:
            inputs = x
        else:
            inputs = torch.cat([x, gaia_features], dim=1)
            
        # 1. Compute the raw output from the dynamic linear layers
        raw_output = self.net(inputs)
        
        # 2. BIFURCACIÓN INTELIGENTE DE MODOS
        # Comparamos la dimensión de la entrada 'x' con la salida 'raw_output'
        if x.shape[1] == raw_output.shape[1]:
            # MODO SPECTRUM (300D == 300D)
            # Map raw outputs to a strictly positive scale (Correction factor centered at 1.0)
            correction_factor = self.positive_activation(raw_output) - 0.6931 + 1.0
            
            # Apply the multiplicative correction physics: F_pred = F_Gaia * Correction
            f_predicted = x * correction_factor
            return f_predicted
            
        else:
            # MODO FEATURES (5D != 300D)
            # Como 'x' son 5 parámetros estelares, no hay un espectro base de 300D para multiplicar.
            # La red genera y devuelve el flujo directamente.
            return raw_output

# Compatibility aliases to prevent breakage in legacy scripts that call the class differently
CalspecFluxMLP = SpectralRecalibrationMLP
PINN_MLP = SpectralRecalibrationMLP