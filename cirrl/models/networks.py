"""
Neural network building blocks for CIRRL models

This module contains the fundamental neural network components used in DPA:
- Activation functions
- Stochastic layers
- Residual blocks
- Stochastic neural networks
"""

import torch
import torch.nn as nn
from typing import Optional


class OnlyRelu(nn.Module):
    """
    ReLU-based variance regularizer.
    
    This ensures positive variance by applying ReLU with an epsilon offset.
    
    Args:
        epsilon: Minimum variance value
        regularize: Whether to apply additional regularization (unused)
    """
    def __init__(self, epsilon: Optional[float] = None, regularize: bool = False):
        super().__init__()
        self.eps = epsilon
        self.reg = regularize
    
    def __repr__(self):
        return f'OnlyRelu(epsilon={self.eps}, regularize={self.reg})'
    
    def forward(self, logvar):
        """
        Transform log variance to positive variance.
        
        Args:
            logvar: Log variance tensor
            
        Returns:
            Positive variance tensor
        """
        sigma = self.eps + nn.ReLU()(logvar)
        return sigma


def get_act_func(name: Optional[str]):
    """
    Get activation function by name.
    
    Args:
        name: Activation function name ('relu', 'sigmoid', 'tanh', 'softmax', 'leaky')
        
    Returns:
        PyTorch activation module or None
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "leaky":
        return nn.LeakyReLU(0.01)
    else:
        return None


class StoLayer(nn.Module):
    """
    Stochastic layer with optional noise injection.
    
    This layer adds Gaussian noise to the input before applying a linear transformation,
    enabling stochastic computation in the network.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        noise_dim: Noise dimension (0 for deterministic)
        add_bn: Whether to add batch normalization
        add_ln: Whether to add layer normalization
        add_do: Dropout rate (None for no dropout)
        out_act: Output activation function name
        noise_std: Standard deviation of injected noise
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 noise_dim: int = 100,
                 add_bn: bool = True, 
                 add_ln: bool = False, 
                 add_do: Optional[float] = None,
                 out_act: Optional[str] = None, 
                 noise_std: float = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.add_ln = add_ln
        self.noise_std = noise_std
        
        # Build layer
        layer = [nn.Linear(in_dim + noise_dim, out_dim)]
        
        if add_bn:
            layer += [nn.BatchNorm1d(out_dim)]
        elif add_ln:
            layer += [nn.LayerNorm(out_dim)]
        
        if add_do is not None:
            layer += [nn.Dropout(add_do)]
        
        self.layer = nn.Sequential(*layer)
        
        # Output activation
        if out_act == "softmax" and out_dim == 1:
            out_act = "sigmoid"
        self.out_act = get_act_func(out_act)
    
    def forward(self, x):
        """Forward pass with noise injection."""
        eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
        out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        
        if self.out_act is not None:
            out = self.out_act(out)
        
        return out


class StoResBlock(nn.Module):
    """
    Stochastic residual block with skip connections.
    
    Implements a residual block with noise injection, enabling both
    stochastic computation and gradient flow through skip connections.
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (defaults to dim)
        out_dim: Output dimension (defaults to dim)
        noise_dim: Noise dimension
        add_bn: Whether to add batch normalization
        add_ln: Whether to add layer normalization
        out_act: Output activation function
        add_do: Dropout rate
    """
    def __init__(self, 
                 dim: int = 100, 
                 hidden_dim: Optional[int] = None,
                 out_dim: Optional[int] = None, 
                 noise_dim: int = 100,
                 add_bn: bool = True, 
                 add_ln: bool = False,
                 out_act: Optional[str] = None, 
                 add_do: Optional[float] = None):
        super().__init__()
        self.noise_dim = noise_dim
        self.add_do = add_do if add_do is not None else 0.0
        
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim
        
        # First layer
        self.layer1 = [nn.Linear(dim + noise_dim, hidden_dim)]
        self.add_bn = add_bn
        self.add_ln = add_ln
        
        if add_bn:
            self.layer1.append(nn.BatchNorm1d(hidden_dim))
        elif add_ln:
            self.layer1.append(nn.LayerNorm(hidden_dim))
        
        self.layer1.append(nn.Dropout(self.add_do))
        self.layer1.append(nn.LeakyReLU(0.01))
        self.layer1 = nn.Sequential(*self.layer1)
        
        # Second layer
        self.layer2 = nn.Linear(hidden_dim + noise_dim, out_dim)
        
        if add_bn and out_act == "leaky":  # For intermediate blocks
            self.layer2 = nn.Sequential(*[self.layer2, nn.BatchNorm1d(out_dim)])
        elif add_ln and out_act == "leaky":
            self.layer2 = nn.Sequential(*[self.layer2, nn.LayerNorm(out_dim)])
        
        # Skip connection projection if dimensions change
        if out_dim != dim:
            self.layer3 = nn.Linear(dim, out_dim)
        
        self.dim = dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        
        # Output activation
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        elif out_act == "leaky":
            self.out_act = nn.LeakyReLU(0.01)
        else:
            self.out_act = None
    
    def forward(self, x):
        """Forward pass with residual connection."""
        if self.noise_dim > 0:
            # First layer with noise
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
            out = self.layer1(torch.cat([x, eps], dim=1))
            
            # Second layer with noise
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device)
            out = self.layer2(torch.cat([out, eps], dim=1))
        else:
            out = self.layer2(self.layer1(x))
        
        # Skip connection
        if self.out_dim != self.dim:
            out2 = self.layer3(x)
            out = out + out2
        else:
            out += x
        
        # Output activation
        if self.out_act is not None:
            out = self.out_act(out)
        
        return out


class StoNet(nn.Module):
    """
    Stochastic neural network with flexible architecture.
    
    Can be configured as either a standard feed-forward network or
    a residual network with skip connections.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        num_layer: Number of layers
        hidden_dim: Hidden layer dimension
        noise_dim: Noise dimension (0 for deterministic)
        add_bn: Whether to use batch normalization
        add_ln: Whether to use layer normalization
        out_act: Output activation function
        resblock: Whether to use residual blocks
        add_do: Dropout rate
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 num_layer: int = 2,
                 hidden_dim: int = 100,
                 noise_dim: int = 100, 
                 add_bn: bool = True, 
                 add_ln: bool = False,
                 out_act: Optional[str] = None, 
                 resblock: bool = False,
                 add_do: Optional[float] = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.add_ln = add_ln
        self.add_do = add_do if add_do is not None else 0.0
        
        # Output activation
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        elif out_act == "leaky":
            self.out_act = nn.LeakyReLU(0.01)
        else:
            self.out_act = None
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print(f"Number of layers must be even for residual blocks. Changed to {num_layer}")
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        
        self.resblock = resblock
        self.num_layer = num_layer
        
        # Build network
        if self.resblock:
            if self.num_blocks == 1:
                self.net = StoResBlock(
                    dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                    noise_dim=noise_dim, add_bn=add_bn, add_ln=add_ln,
                    out_act=out_act, add_do=add_do
                )
            else:
                self.input_layer = StoResBlock(
                    dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                    noise_dim=noise_dim, add_bn=add_bn, add_ln=add_ln,
                    out_act="leaky", add_do=add_do
                )
                self.inter_layer = nn.Sequential(*[
                    StoResBlock(
                        dim=hidden_dim, noise_dim=noise_dim, 
                        add_bn=add_bn, add_ln=add_ln,
                        out_act="leaky", add_do=add_do
                    ) for _ in range(self.num_blocks - 2)
                ])
                self.out_layer = StoResBlock(
                    dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                    noise_dim=noise_dim, add_bn=add_bn, add_ln=add_ln,
                    out_act=out_act, add_do=add_do
                )
        else:
            # Standard feed-forward architecture
            self.input_layer = StoLayer(
                in_dim=in_dim, out_dim=hidden_dim, noise_dim=noise_dim,
                add_bn=add_bn, add_ln=add_ln, out_act="leaky", add_do=add_do
            )
            self.inter_layer = nn.Sequential(*[
                StoLayer(
                    in_dim=hidden_dim, out_dim=hidden_dim, noise_dim=noise_dim,
                    add_bn=add_bn, add_ln=add_ln, out_act="leaky", add_do=add_do
                ) for _ in range(num_layer - 2)
            ])
            self.out_layer = StoLayer(
                in_dim=hidden_dim, out_dim=out_dim, noise_dim=noise_dim,
                add_bn=False, add_ln=False, out_act=out_act, add_do=add_do
            )
    
    def forward(self, x):
        """Forward pass through the network."""
        if self.num_blocks == 1:
            return self.net(x)
        else:
            return self.out_layer(self.inter_layer(self.input_layer(x)))
    
    def predict(self, x, target=["mean"], sample_size: int = 100):
        """
        Point prediction using multiple samples.
        
        Args:
            x: Input tensor
            target: Prediction targets ('mean', 'median', or quantile values)
            sample_size: Number of samples for estimation
            
        Returns:
            Predicted values
        """
        from cirrl.utils.data import vectorize
        
        if self.noise_dim == 0:
            sample_size = 1
        
        samples = self.sample(x=x, sample_size=sample_size, expand_dim=True)
        
        if not isinstance(target, list):
            target = [target]
        
        results = []
        extremes = []
        
        for t in target:
            if t == "mean":
                results.append(samples.mean(dim=len(samples.shape) - 1))
            else:
                if t == "median":
                    t = 0.5
                assert isinstance(t, float)
                results.append(samples.quantile(t, dim=len(samples.shape) - 1))
                if min(t, 1 - t) * sample_size < 10:
                    extremes.append(t)
        
        if len(extremes) > 0:
            print(f"Warning: quantile estimates at {extremes} with sample_size={sample_size} may be inaccurate")
        
        if len(results) == 1:
            return results[0]
        else:
            return results
    
    def sample_onebatch(self, x, sample_size: int = 100, expand_dim: bool = True):
        """
        Generate samples for one batch.
        
        Args:
            x: Input tensor (batch_size, in_dim)
            sample_size: Number of samples per input
            expand_dim: Whether to expand sample dimension
            
        Returns:
            Samples tensor
        """
        data_size = x.size(0)
        
        with torch.no_grad():
            # Repeat input for multiple samples
            x_rep = x.repeat(sample_size, 1)
            samples = self.forward(x=x_rep).detach()
        
        if not expand_dim or sample_size == 1:
            return samples
        else:
            expand_dim_idx = len(samples.shape)
            samples = samples.unsqueeze(expand_dim_idx)
            samples = list(torch.split(samples, data_size))
            samples = torch.cat(samples, dim=expand_dim_idx)
            return samples
    
    def sample_batch(self, x, sample_size: int = 100, 
                    expand_dim: bool = True, batch_size: Optional[int] = None):
        """
        Generate samples with mini-batching for memory efficiency.
        
        Args:
            x: Input tensor
            sample_size: Number of samples per input
            expand_dim: Whether to expand sample dimension
            batch_size: Batch size for processing
            
        Returns:
            Samples tensor
        """
        from cirrl.utils.data import make_dataloader
        
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, batch_size=batch_size, shuffle=False)
            samples = []
            for (x_batch,) in test_loader:
                samples.append(self.sample_onebatch(x_batch, sample_size, expand_dim))
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim)
        
        return samples
    
    def sample(self, x, sample_size: int = 100, 
              expand_dim: bool = True, verbose: bool = True):
        """
        Generate samples with automatic batch size adjustment.
        
        Args:
            x: Input tensor
            sample_size: Number of samples per input
            expand_dim: Whether to expand sample dimension
            verbose: Whether to print memory warnings
            
        Returns:
            Samples tensor
        """
        batch_size = x.shape[0]
        
        while True:
            try:
                samples = self.sample_batch(x, sample_size, expand_dim, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if verbose:
                        print(f"Out of memory; reducing batch size to {batch_size}")
                else:
                    raise e
        
        return samples
