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
                 activation=nn.GELU(), dropout_rate=0.2, residual_type="additive"):
        """
        Args:
            - input_dim (int): Dimension of the input features (e.g., 285 for 280D spectrum + 5D features).
            - output_dim (int): Dimension of the output target (e.g., 280 channels).
            - hidden_dims (list): Neurons per hidden layer.
            - activation (nn.Module): Smooth activation function (GELU/Tanh/SiLU).
            - dropout_rate (float): Dropout probability for Stage 1.
            - residual_type (str): "additive" for standardized data, "multiplicative" for raw physical flux.
        """
        super().__init__()
        self.residual_type = residual_type
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim

        # Dynamic construction using LayerNorm instead of BatchNorm
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            # LayerNorm is crucial for L-BFGS stability and does not break PINN Autograd
            layers.append(nn.LayerNorm(h_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = h_dim 

        # Final linear projection layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Positive activation for the strict multiplicative case
        self.positive_activation = nn.Softplus()

    def forward(self, x, gaia_features=None):
        """
        Performs the forward pass. Automatically unpacks high-dimensional inputs 
        into spectrum and features for the residual logic based on output_dim.
        """
        # We use self.output_dim instead of a hardcoded value (e.g., 300) to
        # dynamically support different resolutions or truncated data (e.g., 280).
        cutoff = self.output_dim
        
        # --- DYNAMIC UNPACKING LOGIC ---
        if x.shape[1] > cutoff and gaia_features is None:
            base_x = x[:, :cutoff]  # Isolate the pure spectrum for physics constraints (e.g., first 280)
            inputs = x              # Full dimensions (e.g., 285) are passed to the network
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
            # --- SPECTRUM MODE (e.g., 280D == 280D) ---
            if self.residual_type == "multiplicative":
                # Softplus centered at 1.0 for neutral corrections when raw_output is 0.
                correction_factor = self.positive_activation(raw_output) - 0.6931 + 1.0
                return base_x * correction_factor
            else:
                return base_x + raw_output
        else:
            # --- FEATURE MODE ---
            return raw_output