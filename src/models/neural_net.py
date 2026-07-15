"""
Module: neural_net.py
Report Section: 6.3 Deep Learning Neural Network Architecture (MLP - Multiplicative Residual)

This module defines the Multi-Layer Perceptron (MLP) architecture customized for 
the Gaia-CALSPEC spectrophotometric recalibration pipeline. It implements a 
multiplicative physics-inspired residual mapping: instead of predicting absolute fluxes,
the network outputs a positive correction vector applied directly to the source Gaia grid.
"""

import math
import torch
import torch.nn as nn

class SpectralRecalibrationMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture tailored for Gaia-CALSPEC Spectral Recalibration.
    
    Architecture (Multiplicative Residual Design):
    - Input: Gaia DR3 observed dataset (either the 300-channel aligned grid vectors 
        or the isolated 5 scalar astrometric/photometric features).
    - Hidden layers: Dynamically defined by hidden_dims with LayerNorm & dropout regularization.
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

        # Dynamically build hidden layers following the structural design
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            # IMPROVEMENT: Use LayerNorm instead of BatchNorm to preserve stability
            # of the Hessian during optimization with L-BFGS.
            layers.append(nn.LayerNorm(h_dim))
            layers.append(activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
                
            current_dim = h_dim 

        # Final linear regression layer to project onto the 300 spectral channels
        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)
        
        # Enforce that the multiplicative correction factor (or flux) is strictly positive
        self.positive_activation = nn.Softplus()
        
        # IMPROVEMENT: Exact mathematical constant used to remove initialization bias.
        self.ln_2 = math.log(2.0)

    def forward(self, x, gaia_features=None):
        """
        Performs the forward pass of the model.
        Spectrum Mode: Maps inputs to a correction factor vector and applies it to the original Gaia spectrum.
        Features Mode: Maps parameters directly to a positive flux space.
        """
        if gaia_features is None:
            inputs = x
        else:
            inputs = torch.cat([x, gaia_features], dim=1)
            
        # 1. Compute the raw output from the dynamic linear layers
        raw_output = self.net(inputs)
        
        # 2. SMART MODE BRANCHING
        if x.shape[1] == raw_output.shape[1]:
            # SPECTRUM MODE (300D == 300D)
            # Map raw outputs to a strictly positive scale (correction factor centered at 1.0)
            correction_factor = self.positive_activation(raw_output) - self.ln_2 + 1.0
            
            # Apply the multiplicative correction physics: F_pred = F_Gaia * Correction
            f_predicted = x * correction_factor
            return f_predicted
            
        else:
            # FEATURES MODE (5D != 300D)
            # IMPROVEMENT: Force the Softplus output so the directly predicted flux
            # from features is strictly greater than zero.
            f_predicted_features = self.positive_activation(raw_output)
            return f_predicted_features

