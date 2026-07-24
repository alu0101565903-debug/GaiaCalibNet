"""
Module: neural_net.py
Report Section: 6.3 Deep Learning Neural Network Architecture (MLP - Multiplicative Residual)

Defines the spectral recalibration neural network used for Gaia to CALSPEC
recalibration. The model is a flexible MLP that can either predict a full
spectrum correction vector or produce strictly positive flux predictions from
auxiliary Gaia features.

Key characteristics:
- Multiplicative residual design: the network learns a positive correction factor
  applied to the smoothed input Gaia spectrum to avoid propagating high-frequency noise.
- Softplus activation is shifted so a zero-initialized network starts with an
  identity correction (correction factor of 1.0).
- Custom 1D smoothing with replicate padding ensures physical spectral continuity
  without boundary drop-off artifacts.
- Supports both spectrum-mode inputs and scalar-feature modes through dynamic
  output shaping.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralRecalibrationMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture tailored for Gaia-CALSPEC Spectral Recalibration.
    
    Architecture (Multiplicative Residual Design):
    - Input: Gaia DR3 observed dataset (either the 300-channel aligned grid vectors 
        or the isolated 5 scalar astrometric/photometric features).
    - Hidden layers: Dynamically defined by hidden_dims with LayerNorm & dropout regularization.
    - Output: Multiplicative correction vector mapped strictly to positive bounds via Softplus,
        yielding F_pred = F_Gaia_smoothed * Correction_Factor_smoothed.
    """
    def __init__(self, input_dim=300, output_dim=300, hidden_dims=[64, 32, 64], 
                 activation=nn.GELU(), dropout_rate=0.2, kernel_size=25):
        """
        Args:
        - input_dim (int): Dimension of the input features (default: 300 for aligned grids).
        - output_dim (int): Dimension of the output spectrum (default: 300 channels).
        - hidden_dims (list): List containing the number of neurons per hidden layer.
        - activation (nn.Module): Instantiated PyTorch activation function.
        - dropout_rate (float): Dropout probability to handle variance in compact catalogs.
        - kernel_size (int): Kernel size for 1D smoothing filter to enforce spectral continuity.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0, "kernel_size must be an odd number to maintain tensor dimensions."

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
        as a perfect Identity Mapping: 
        F_pred = F_Gaia_smoothed * (Softplus(0) - ln(2) + 1.0) = F_Gaia_smoothed.
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

    def _smooth_tensor(self, t):
        """
        Applies a 1D average pooling smoothing filter while preventing boundary
        drop-off artifacts by using "replicate" padding instead of zero-padding.
        
        Args:
        - t (torch.Tensor): 2D tensor of shape (batch_size, channels).
        
        Returns:
        - torch.Tensor: Smoothed 2D tensor of the same shape.
        """
        # Reshape tensor to 3D format for 1D pooling: (Batch, Channels=1, Length)
        t_3d = t.unsqueeze(1)
        pad_size = self.kernel_size // 2
        
        # Pad with edge replication to maintain physical values at the boundaries
        t_padded = F.pad(t_3d, (pad_size, pad_size), mode="replicate")
        
        # Apply average pooling (CRITICAL: stride=1 maintains the 300 channels)
        t_smoothed = F.avg_pool1d(t_padded, kernel_size=self.kernel_size, stride=1)
        
        # Return to 2D shape: (Batch, Length)
        return t_smoothed.squeeze(1)

    def forward(self, x, gaia_features=None):
        """
        Performs the forward pass of the model.
        Spectrum Mode: Isolates the stellar continuum and applies a learned, smoothed 
                       multiplicative correction to reconstruct the physical flux.
        Features Mode: Maps parameters directly to a positive flux space.
        """
        
        # 1. INPUT SCALE NORMALIZATION: bring inputs close to O(1) to revive
        #    gradients in the linear layers and make learning numerically stable.
        if x.shape[1] == self.output_dim:
            # Spectrum mode: normalize continuum by mean flux
            scale_factor = torch.mean(x, dim=1, keepdim=True) + 1e-12
            x_scaled_for_mlp = x / scale_factor
        else:
            # Features mode: already standardized by StandardScaler in training script
            x_scaled_for_mlp = x
        
        if gaia_features is None:
            inputs = x_scaled_for_mlp
        else:
            inputs = torch.cat([x_scaled_for_mlp, gaia_features], dim=1)
            
        # 2. Process through the MLP core.
        raw_output = self.net(inputs)

        # 3. Mode branching: spectrum mode vs feature mode.
        if x.shape[1] == raw_output.shape[1]:
            # === TRUE RESIDUAL SPECTRUM MODE ===
            # Predict the unconstrained correction factor.
            # Apply the adjusted Softplus trick (-ln_2 + 1.0) so that when
            # raw_output is zero, the normalized correction equals 1.0 exactly.
            correction_factor = self.positive_activation(raw_output) - self.ln_2 + 1.0

            # Smooth the correction factor to enforce instrumental response continuity
            c_smoothed = self._smooth_tensor(correction_factor)
            
            # Smooth the input spectrum to isolate the stellar continuum (Denoising)
            x_smoothed = self._smooth_tensor(x)

            # True multiplicative residual combination: Continuum * Correction
            f_predicted = c_smoothed * x_smoothed
            return f_predicted

        else:
            # Features mode: return the directly predicted positive flux values.
            f_predicted_features = self.positive_activation(raw_output)
            return f_predicted_features