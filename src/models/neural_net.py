"""
Module: neural_net.py
Report Section: 6.3 Deep Learning Neural Network Architecture (MLP - Multiplicative/Additive Residual)

This module defines the Multi-Layer Perceptron (MLP) architecture customized for 
the Gaia-CALSPEC spectrophotometric recalibration pipeline. It implements a 
robust physics-inspired residual mapping compatible with second-order optimizers.
"""

import torch
import torch.nn as nn

class SpectralRecalibrationMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture tailored for Gaia-CALSPEC Spectral Recalibration.
    Compatible with Adam, L-BFGS, and robust scale-invariant physics constraints.
    """
    def __init__(self, input_dim=300, output_dim=300, hidden_dims=[256, 512, 256], 
                 activation=nn.GELU(), dropout_rate=0.2, residual_type="multiplicative"):
        """
        Args:
            - input_dim (int): Dimension of the input features (e.g., 300 for 300D spectrum).
            - output_dim (int): Dimension of the output target (e.g., 300 channels).
            - hidden_dims (list): Neurons per hidden layer.
            - activation (nn.Module): Smooth activation function (GELU/Tanh/SiLU).
            - dropout_rate (float): Dropout probability.
            - residual_type (str): "multiplicative" (default) for physics correction, "additive" for normalized residuals.
        """
        super().__init__()
        self.residual_type = residual_type
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim

        # Dynamic construction using LayerNorm instead of BatchNorm for L-BFGS stability
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim 

        # Final linear projection layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Sigmoid bounds the correction factor strictly between 0.1 and 10.0 
        # to prevent catastrophic scaling errors while keeping the center at 1.0
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gaia_features=None):
        """
        Performs the forward pass. Automatically unpacks high-dimensional inputs 
        into spectrum and features for the residual logic based on output_dim.
        """
        cutoff = self.output_dim
        
        # --- DYNAMIC UNPACKING LOGIC ---
        if x.shape[1] > cutoff and gaia_features is None:
            base_x = x[:, :cutoff]  # Isolate the pure spectrum for physics constraints
            inputs = x              # Full dimensions (285) passed to the network
        elif gaia_features is not None:
            base_x = x
            inputs = torch.cat([x, gaia_features], dim=1)
        else:
            base_x = x
            inputs = x
            
        # 1. Compute the raw output from the dense network
        raw_output = self.net(inputs)
        
        # 2. Operational modes (compare against base_x, NOT against x)
        if base_x.shape[1] == raw_output.shape[1]:
            # --- SPECTRUM MODE (300D == 300D) ---
            if self.residual_type == "multiplicative":
                # USAR UNA DE LAS SIGUIENTES OPCIONES (aseguran corrección = 1.0 en el inicio):
                # Opción A (Rango 0.1x a 5.0x):
                correction_factor = 0.1 + 4.9 * self.sigmoid(raw_output - 1.4917)
                
                # Opción B (Rango 0.1x a 1.6x - Recomendada si la calibración es fina):
                # correction_factor = 0.1 + 1.5 * self.sigmoid(raw_output + 0.405)
                
                return base_x * correction_factor
            else:
                # Additive residual
                return base_x + raw_output
        else:
            # --- FEATURE MODE ---
            return raw_output