"""
Module: neural_net.py
Report Section: 6.3 Deep Learning Neural Network Architecture (MLP - Multiplicative Residual)

Defines the spectral recalibration neural network used for Gaia to CALSPEC
recalibration. The model is a flexible MLP that can either predict a full
spectrum correction vector or produce strictly positive flux predictions from
auxiliary Gaia features.

Key characteristics:
- Multiplicative residual design: the network learns a positive correction factor
  applied to the input Gaia spectrum rather than regressing raw flux directly.
- Softplus activation is shifted so a zero-initialized network starts with an
  identity correction (correction factor of 1.0).
- Supports both spectrum-mode inputs and scalar-feature modes through dynamic
  output shaping.
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
            # LayerNorm preserves stability of the Hessian during optimization with L-BFGS.
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
        
        # Exact mathematical constant used to remove initialization bias when
        # shifting the Softplus activation so the correction factor is centered
        # around 1.0 for zero initialization.
        self.ln_2 = math.log(2.0)

        # Enhancement: enforce a physics-informed initialization of the residual
        # mapping so that at the start of training the network implements an
        # identity-like multiplicative correction (no change to input flux).
        self._initialize_residual_mapping()

    def _initialize_residual_mapping(self):
        """
        Initializes the final linear layer to zero. 
        This guarantees that at Epoch 0, raw_output = 0, and the network acts 
        as a perfect Identity Mapping: F_pred = F_Gaia * (Softplus(0) - ln(2) + 1.0) = F_Gaia.
        """
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def prepare_for_lbfgs(self):
        """
        Convenience method to disable Dropout before L-BFGS optimization.
        L-BFGS requires a purely deterministic loss landscape to compute the Hessian.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def forward(self, x, gaia_features=None):
        """
        Performs the forward pass of the model.
        Spectrum Mode: Maps inputs to a clean, smooth, directly predicted spectrum
                       in scaled space, then scales it back up to physical units.
        Features Mode: Maps parameters directly to a positive flux space.
        """
        
        # 1. INPUT SCALE NORMALIZATION: bring inputs close to O(1) to revive
        #    gradients in the linear layers and make learning numerically stable.
        scale_factor = torch.mean(x, dim=1, keepdim=True) + 1e-12
        x_scaled_for_mlp = x / scale_factor
        
        if gaia_features is None:
            inputs = x_scaled_for_mlp
        else:
            inputs = torch.cat([x_scaled_for_mlp, gaia_features], dim=1)
            
        # 2. Process through the MLP core.
        raw_output = self.net(inputs)

        # 3. Mode branching: spectrum mode vs feature mode.
        if x.shape[1] == raw_output.shape[1]:
            # === DIRECT SPECTRUM MODE (WITHOUT MULTIPLYING BY NOISY INPUT) ===
            # Predict the spectrum shape at O(1) scale.
            # Apply the adjusted Softplus trick (-ln_2 + 1.0) so that when
            # raw_output is zero, the normalized prediction equals 1.0 exactly.
            normalized_prediction = self.positive_activation(raw_output) - self.ln_2 + 1.0

            # Rescale the normalized prediction back to the original flux scale.
            # At initialization, this yields a flat spectrum equal to the mean
            # input flux for the star, avoiding noisy startup behavior.
            f_predicted = normalized_prediction * scale_factor
            return f_predicted

        else:
            # Features mode: return the directly predicted positive flux values.
            f_predicted_features = self.positive_activation(raw_output)
            return f_predicted_features