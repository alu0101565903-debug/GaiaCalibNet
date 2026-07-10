"""
Module: neural_net.py
Report Section: 6.3 Deep Learning Neural Network Architecture (MLP)

This module defines the Multi-Layer Perceptron (MLP) architecture customized for 
the Gaia-CALSPEC spectrophotometric recalibration pipeline. The structural layers 
are built dynamically via parameter arrays to facilitate optimization sweeps.
"""

import torch
import torch.nn as nn

class SpectralRecalibrationMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture tailored for Gaia-CALSPEC Spectral Recalibration.
    
    Architecture:
    - Input: Gaia DR3 observed dataset (either the 300-channel aligned grid vectors 
        or the isolated 5 scalar astrometric/photometric features).
    - Hidden layers: Dynamically defined by hidden_dims with dropout regularization.
    - Output: Predicted continuous absolute flux mapped onto the 300-channel Lambda grid.
    """
    def __init__(self, input_dim=300, output_dim=300, hidden_dims=[128, 256, 128], 
                 activation=nn.GELU(), dropout_rate=0.2):
        """
        Args:
        - input_dim (int): Dimension of the input features (default: 300 for aligned grids).
        - output_dim (int): Dimension of the output spectrum (default: 300 channels).
        - hidden_dims (list): List containing the number of neurons per hidden layer.
                                Example: [256, 512, 256] for a deep funnel architecture.
        - activation (nn.Module): Instantiated PyTorch activation function.
        - dropout_rate (float): Dropout probability to handle variance in compact catalogs.
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        # Dynamically build hidden layers following your structural design
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            # Set the current output dimension as the input for the next layer
            current_dim = h_dim 

        # Final linear regression layer to project onto the 300 spectral channels
        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, gaia_features=None):
        """
        Performs the forward pass of the model. 
        Automatically handles a single matrix or concatenates separate inputs on the fly.

        Args:
        - x (torch.Tensor): Primary input tensor (e.g., Gaia observed continuous spectrum).
        - gaia_features (torch.Tensor, optional): Secondary astrometric/photometric scalar 
            feature matrix (5 columns). Defaults to None.

        Returns:
        - torch.Tensor: Predicted continuous absolute flux density profile.
        """
        if gaia_features is None:
            # Input data is already unified or only the primary matrix is processed
            inputs = x
        else:
            # Combined input: concatenate the spectrum with scalar features along dim=1
            inputs = torch.cat([x, gaia_features], dim=1)
            
        return self.net(inputs)


# Compatibility aliases to prevent breakage in legacy scripts that call the class differently
CalspecFluxMLP = SpectralRecalibrationMLP
PINN_MLP = SpectralRecalibrationMLP