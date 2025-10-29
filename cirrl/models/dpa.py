"""
Distributional Principal Autoencoder (DPA) Model

This module implements the DPA model for learning causal representations
across multiple environments with distributional robustness.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Optional, List, Union
from cirrl.models.networks import StoNet, OnlyRelu


class DPAmodel(nn.Module):
    """
    DPA neural network model with encoder and decoder.
    
    This model learns hierarchical latent representations using stochastic
    encoders and decoders, with optional environment conditioning.
    
    Args:
        data_dim: Input data dimension
        latent_dim: Latent representation dimension
        out_dim: Output dimension (defaults to data_dim)
        condition_dim: Conditioning variable dimension (for environments)
        num_layer: Number of layers
        num_layer_enc: Number of encoder layers (defaults to num_layer)
        hidden_dim: Hidden layer dimension
        noise_dim: Noise dimension for stochasticity
        dist_enc: Encoder distribution type ('deterministic' or 'stochastic')
        dist_dec: Decoder distribution type
        priorvar: Prior variance module
        resblock: Whether to use residual blocks
        encoder_k: Whether to embed k in encoder
        bn_enc: Batch normalization in encoder
        bn_dec: Batch normalization in decoder
        ln_enc: Layer normalization in encoder
        ln_dec: Layer normalization in decoder
        out_act: Output activation function
        totalvar: Whether to use full covariance
        linear: Whether to use linear models
        lin_dec: Whether decoder is linear
        lin_bias: Whether to include bias in linear models
    """
    def __init__(self, 
                 data_dim: int = 2, 
                 latent_dim: int = 10, 
                 out_dim: Optional[int] = None,
                 condition_dim: Optional[int] = None,
                 num_layer: int = 3, 
                 num_layer_enc: Optional[int] = None,
                 hidden_dim: int = 500, 
                 noise_dim: int = 100,
                 dist_enc: str = "deterministic", 
                 dist_dec: str = "stochastic",
                 priorvar: Optional[nn.Module] = None,
                 resblock: bool = False,
                 encoder_k: bool = False, 
                 bn_enc: bool = False, 
                 bn_dec: bool = False,
                 ln_enc: bool = False, 
                 ln_dec: bool = False,
                 out_act: Optional[str] = None,
                 totalvar: bool = False,
                 linear: bool = False, 
                 lin_dec: bool = True, 
                 lin_bias: bool = True):
        super().__init__()
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        
        if out_dim is None:
            out_dim = data_dim
        self.out_dim = out_dim
        
        self.condition_dim = condition_dim
        self.num_layer = num_layer
        
        if num_layer_enc is None:
            num_layer_enc = num_layer
        self.num_layer_enc = num_layer_enc
        
        self.bn_enc = bn_enc
        self.bn_dec = bn_dec
        self.ln_enc = ln_enc
        self.ln_dec = ln_dec
        self.hidden_dim = hidden_dim
        
        noise_dim_enc = 0 if dist_enc == "deterministic" else noise_dim
        noise_dim_dec = 0 if dist_dec == "deterministic" else noise_dim
        
        self.noise_dim = noise_dim
        self.noise_dim_enc = noise_dim_enc
        self.noise_dim_dec = noise_dim_dec
        self.dist_enc = dist_enc
        self.dist_dec = dist_dec
        self.out_act = out_act
        self.linear = linear
        self.lin_dec = lin_dec
        self.encoder_k = encoder_k
        self.priorvar = priorvar
        self.totalvar = totalvar
        
        # Build encoder
        self.encoder = StoNet(
            data_dim, latent_dim, num_layer_enc, hidden_dim, 
            noise_dim_enc, bn_enc, ln_enc, resblock=resblock
        )
        
        # Build decoder
        self.decoder = StoNet(
            latent_dim, out_dim, num_layer, hidden_dim,
            noise_dim_dec, bn_dec, ln_dec, out_act, resblock
        )
        
        # Build prior network (if conditioning is used)
        if condition_dim is not None:
            self.prior = StoNet(
                condition_dim, hidden_dim, num_layer_enc - 1,
                hidden_dim, noise_dim_enc, bn_enc, ln_enc, resblock=resblock
            )
            
            if bn_enc:
                self.priormean = nn.Sequential(
                    nn.LeakyReLU(0.01), 
                    nn.BatchNorm1d(hidden_dim), 
                    nn.Linear(hidden_dim, latent_dim)
                )
                self.priorlogvar = nn.Sequential(
                    nn.LeakyReLU(0.01), 
                    nn.BatchNorm1d(hidden_dim), 
                    nn.Linear(hidden_dim, latent_dim)
                )
            elif ln_enc:
                self.priormean = nn.Sequential(
                    nn.LeakyReLU(0.01), 
                    nn.LayerNorm(hidden_dim), 
                    nn.Linear(hidden_dim, latent_dim)
                )
                self.priorlogvar = nn.Sequential(
                    nn.LeakyReLU(0.01), 
                    nn.LayerNorm(hidden_dim), 
                    nn.Linear(hidden_dim, latent_dim)
                )
            else:
                self.priormean = nn.Sequential(
                    nn.LeakyReLU(0.01), 
                    nn.Linear(hidden_dim, latent_dim)
                )
                self.priorlogvar = nn.Sequential(
                    nn.LeakyReLU(0.01), 
                    nn.Linear(hidden_dim, latent_dim)
                )
            
            if self.totalvar:
                self.nondiagonal(lower_diag=False)
        
        if self.encoder_k:
            self.k_embed_layer = nn.Linear(self.latent_dim, self.data_dim * 2)
    
    def nondiagonal(self, lower_diag: bool = True):
        """
        Setup for full covariance matrix estimation.
        
        Args:
            lower_diag: Whether to use lower triangular parameterization
        """
        cov_entries = int(self.latent_dim ** 2)
        
        if lower_diag:
            self.tri_idx_row, self.tri_idx_col = torch.tril_indices(
                row=self.latent_dim, col=self.latent_dim
            )
            idx = (self.tri_idx_row != self.tri_idx_col)
            self.tri_idx_row = self.tri_idx_row[idx]
            self.tri_idx_col = self.tri_idx_col[idx]
            cov_entries = len(self.tri_idx_col)
        
        if self.bn_enc:
            self.l2_Lvals = nn.Sequential(
                nn.LeakyReLU(0.01), 
                nn.BatchNorm1d(self.hidden_dim), 
                nn.Linear(self.hidden_dim, cov_entries)
            )
        elif self.ln_enc:
            self.l2_Lvals = nn.Sequential(
                nn.LeakyReLU(0.01), 
                nn.LayerNorm(self.hidden_dim), 
                nn.Linear(self.hidden_dim, cov_entries)
            )
        else:
            self.l2_Lvals = nn.Sequential(
                nn.LeakyReLU(0.01), 
                nn.Linear(self.hidden_dim, cov_entries)
            )
    
    def get_k_embedding(self, k: int, x: Optional[torch.Tensor] = None):
        """
        Get latent level embedding.
        
        Args:
            k: Latent level
            x: Input tensor (optional)
            
        Returns:
            k embedding tensor
        """
        k_emb = torch.ones(1, self.latent_dim)
        k_emb[:, k:].zero_()
        
        if x is not None:
            k_emb = k_emb.to(x.device)
            gamma, beta = self.k_embed_layer(k_emb).chunk(2, dim=1)
            k_emb = gamma * x + beta
        
        return k_emb
    
    def encode(self, x, k: Optional[int] = None, mean: bool = True, 
              gen_sample_size: int = 100, in_training: bool = False):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            k: Latent level (None for full dimension)
            mean: Whether to return mean or sample
            gen_sample_size: Number of samples to generate
            in_training: Whether in training mode
            
        Returns:
            Latent representation
        """
        if k is None:
            k = self.latent_dim
        
        if self.encoder_k:
            x = self.get_k_embedding(k, x)
        
        if in_training:
            return self.encoder(x)
        
        if self.dist_enc == "deterministic":
            gen_sample_size = 1
        
        if mean:
            z = self.encoder.predict(x, sample_size=gen_sample_size)
        else:
            z = self.encoder.sample(x, sample_size=gen_sample_size)
            if gen_sample_size == 1:
                z = z.squeeze(len(z.shape) - 1)
        
        return z[:, :k]
    
    def reparam(self, mean, diagonal, L_vals: Optional[torch.Tensor] = None, 
                lower_diag: bool = True):
        """
        Reparameterization trick for sampling from Gaussian.
        
        Args:
            mean: Mean vector
            diagonal: Diagonal of covariance
            L_vals: Off-diagonal covariance values
            lower_diag: Whether using lower triangular parameterization
            
        Returns:
            Sampled tensor
        """
        device = mean.device
        
        if L_vals is not None:
            batch_size = mean.size(0)
            
            if lower_diag:
                L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=device)
                L[:, self.tri_idx_row, self.tri_idx_col] = L_vals
                L += torch.diag_embed(diagonal)
            else:
                L = L_vals.reshape(batch_size, self.latent_dim, self.latent_dim)
                L = torch.bmm(L, L.permute(0, 2, 1))
            
            eps = torch.randn((batch_size, self.latent_dim), device=device)
            sample = mean + torch.bmm(L, eps.unsqueeze(-1)).squeeze()
        else:
            eps = torch.randn(mean.shape, device=device)
            sample = mean + diagonal * eps
        
        return sample
    
    def priore(self, e, gen_sample_size: int = 100, double: bool = False, reg: bool = False):
        """
        Sample from environment-conditioned prior.
        
        Args:
            e: Environment indicator
            gen_sample_size: Number of samples
            double: Whether to return two samples
            reg: Whether to return regularization terms
            
        Returns:
            Sampled latent representations
        """
        raws = self.prior(e)
        mean = self.priormean(raws)
        logvar = self.priorlogvar(raws)
        L_vals = None
        
        if self.priorvar is not None:
            standard_deviation = self.priorvar.forward(logvar)
        else:
            standard_deviation = torch.exp(0.5 * logvar)
        
        if self.totalvar:
            L_vals = self.l2_Lvals(raws)
        
        if double:
            z_1 = self.reparam(mean=mean, diagonal=standard_deviation, 
                              L_vals=L_vals, lower_diag=False)
            z_2 = self.reparam(mean=mean, diagonal=standard_deviation, 
                              L_vals=L_vals, lower_diag=False)
            if reg:
                return z_1, z_2, standard_deviation
            else:
                return z_1, z_2
        else:
            z = self.reparam(mean=mean, diagonal=standard_deviation, 
                           L_vals=L_vals, lower_diag=False)
            if reg:
                return z, standard_deviation, L_vals
            else:
                return z
    
    def decode(self, z, c: Optional[torch.Tensor] = None, mean: bool = True, 
              gen_sample_size: int = 100):
        """
        Decode latent representation to data space.
        
        Args:
            z: Latent representation
            c: Conditioning variable (optional)
            mean: Whether to return mean or sample
            gen_sample_size: Number of samples
            
        Returns:
            Decoded tensor
        """
        if z.size(1) != self.latent_dim:
            z_ = torch.randn((z.size(0), self.latent_dim - z.size(1)), device=z.device)
            z = torch.cat([z, z_], dim=1)
        
        if c is not None:
            z = torch.cat([z, c], dim=1)
        
        if self.dist_enc == "deterministic":
            gen_sample_size = 1
        
        if mean:
            x = self.decoder.predict(z, sample_size=gen_sample_size)
        else:
            x = self.decoder.sample(z, sample_size=gen_sample_size)
        
        return x
    
    def forward(self, x, c: Optional[torch.Tensor] = None, k: Optional[int] = None,
                gen_sample_size: Optional[int] = None, return_latent: bool = False,
                device: Optional[torch.device] = None, double: bool = False):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            c: Conditioning variable
            k: Latent level
            gen_sample_size: Sample size for generation
            return_latent: Whether to return latent representation
            device: Device for computation
            double: Whether to return two reconstructions
            
        Returns:
            Reconstructed data (and latent if return_latent=True)
        """
        if k is None:
            k = self.latent_dim
        
        if self.encoder_k:
            x = self.get_k_embedding(k, x)
        
        if double:
            z = self.encode(x, in_training=True)
            if return_latent:
                z_ = z.clone()
            
            z1 = z.clone()
            x1 = self.decoder(z)
            z1[:, k:].normal_(0, 1)
            x2 = self.decoder(z1)
            
            if return_latent:
                return x1, x2, z_
            else:
                return x1, x2
        else:
            if x is not None and k > 0:
                z = self.encode(x, in_training=True)
                if return_latent:
                    z_ = z.clone()
                z[:, k:].normal_(0, 1)
            else:
                if return_latent:
                    z_ = self.encode(x, in_training=True)
                if gen_sample_size is None:
                    gen_sample_size = x.size(0)
                if device is None:
                    device = x.device if x is not None else torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                z = torch.randn((gen_sample_size, self.latent_dim), device=device)
            
            if self.condition_dim is not None and c is not None:
                z = torch.cat([z, c], dim=1)
            
            x = self.decoder(z)
            
            if return_latent:
                return x, z_
            else:
                return x


class DPA(object):
    """
    Distributional Principal Autoencoder trainer and interface.
    
    This class wraps the DPAmodel and provides training, evaluation,
    and inference functionality.
    
    Args:
        data_dim: Input data dimension
        latent_dims: List of latent dimensions for hierarchical learning
        num_layer: Number of layers in networks
        num_layer_enc: Number of encoder layers
        hidden_dim: Hidden layer dimension
        noise_dim: Noise dimension
        out_dim: Output dimension
        condition_dim: Conditioning dimension (for environments)
        linear: Whether to use linear models
        lin_dec: Whether decoder is linear
        lin_bias: Whether to include bias
        dist_enc: Encoder distribution type
        dist_dec: Decoder distribution type
        resblock: Whether to use residual blocks
        out_act: Output activation
        bn_enc, bn_dec: Batch normalization flags
        ln_enc, ln_dec: Layer normalization flags
        priorvar: Prior variance module
        totalvar: Whether to use full covariance
        encoder_k: Whether to embed k in encoder
        coef_match_latent: Coefficient for latent matching loss
        lr: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        standardize: Whether to standardize data
        beta: Beta parameter for energy loss
        l2: L2 regularization weight
        device: Computation device
        dim1, dim2: Image dimensions (if applicable)
        seed: Random seed
    """
    def __init__(self,
                 data_dim, latent_dims, num_layer=2, num_layer_enc=None,
                 hidden_dim=100, noise_dim=100, out_dim=None, condition_dim=None,
                 linear=False, lin_dec=True, lin_bias=False,
                 dist_enc="deterministic", dist_dec="stochastic", resblock=True,
                 out_act=None, bn_enc=False, bn_dec=False, ln_enc=False, ln_dec=False,
                 priorvar=None, totalvar=None, encoder_k=False, coef_match_latent=0,
                 lr=1e-4, num_epochs=500, batch_size=None, standardize=False, beta=1, l2=0.0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dim1=192, dim2=288, seed=222):
        super().__init__()
        
        self.data_dim = data_dim
        
        if not isinstance(latent_dims, list):
            latent_dims = [latent_dims]
        self.latent_dims = latent_dims
        self.latent_dim = latent_dims[0]
        self.num_levels = len(latent_dims)
        
        self.num_layer = num_layer
        self.num_layer_enc = num_layer_enc
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.out_dim = out_dim
        self.condition_dim = condition_dim
        self.linear = linear
        self.lin_dec = lin_dec
        self.lin_bias = lin_bias
        self.dist_enc = dist_enc
        self.dist_dec = dist_dec
        self.bn_enc = bn_enc
        self.bn_dec = bn_dec
        self.ln_enc = ln_enc
        self.ln_dec = ln_dec
        self.priorvar = priorvar
        self.totalvar = totalvar
        self.out_act = out_act
        self.dim1 = dim1
        self.dim2 = dim2
        self.l2 = l2
        
        self.encoder_k = encoder_k
        self.coef_match_latent = coef_match_latent
        self.match_latent = coef_match_latent > 0
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta
        
        if isinstance(device, str):
            if device in ["gpu", "cuda"]:
                device = torch.device("cuda")
            else:
                device = torch.device(device)
        self.device = device
        
        self.standardize = standardize
        self.x_mean = None
        self.x_std = None
        
        # Loss tracking
        self.loss_all_k = np.zeros(self.num_levels)
        self.loss_pred_all_k = np.zeros(self.num_levels)
        self.loss_var_all_k = np.zeros(self.num_levels)
        self.loss_all_k_test = np.zeros(self.num_levels)
        self.loss_pred_all_k_test = np.zeros(self.num_levels)
        self.loss_var_all_k_test = np.zeros(self.num_levels)
        self.energy_loss = None
        self.recon_mse = None
        
        # Set random seeds
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Build model
        self.model = DPAmodel(
            data_dim=data_dim, latent_dim=self.latent_dim, out_dim=self.out_dim,
            condition_dim=condition_dim, num_layer=num_layer, num_layer_enc=num_layer_enc,
            hidden_dim=hidden_dim, noise_dim=noise_dim, dist_enc=dist_enc, dist_dec=dist_dec,
            priorvar=priorvar, resblock=resblock, encoder_k=encoder_k,
            bn_enc=bn_enc, bn_dec=bn_dec, ln_enc=ln_enc, ln_dec=ln_dec, totalvar=totalvar,
            out_act=out_act, linear=linear, lin_dec=lin_dec, lin_bias=lin_bias
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def _standardize_data_and_record_stats(self, x, univar: bool = False):
        """Standardize data and record statistics."""
        self.x_mean = torch.mean(x, dim=0)
        
        if univar:
            self.x_std = x.std()
        else:
            self.x_std = torch.std(x, dim=0)
            self.x_std[self.x_std == 0] += 1e-5
        
        x_standardized = (x - self.x_mean) / self.x_std
        self.x_mean = self.x_mean.to(self.device)
        self.x_std = self.x_std.to(self.device)
        
        return x_standardized
    
    def standardize_data(self, x):
        """Standardize data using recorded statistics."""
        if self.standardize:
            return (x - self.x_mean) / self.x_std
        else:
            return x
    
    def unstandardize_data(self, x):
        """Reverse standardization."""
        if self.standardize:
            return x * self.x_std + self.x_mean
        else:
            return x
    
    @torch.no_grad()
    def encode(self, x, k: Optional[int] = None, mean: bool = True, 
              gen_sample_size: int = 100, in_training: bool = True):
        """Encode input to latent space."""
        self.eval_mode()
        
        if not in_training and self.standardize:
            x = self.standardize_data(x)
        
        z = self.model.encode(x, k, mean, gen_sample_size)
        self.train_mode()
        
        return z
    
    @torch.no_grad()
    def decode(self, z, c: Optional[torch.Tensor] = None, 
              mean: bool = True, gen_sample_size: int = 100):
        """Decode latent representation to data space."""
        self.eval_mode()
        
        samples = self.model.decode(z, c, mean, gen_sample_size)
        
        if self.standardize:
            samples = self.unstandardize_data(samples)
        
        self.train_mode()
        
        return samples
