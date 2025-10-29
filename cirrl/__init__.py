"""
CIRRL: Causal Invariant Representation Learning

A package for learning causal representations using Distributional Principal Autoencoder (DPA)
and Distributionally Robust Instrumental Regression (DRIG).
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easy access
from cirrl.models.dpa import DPA, OnlyRelu
from cirrl.estimators.drig import (
    est_drig,
    est_drig_gd_fast,
    est_drig_gd_batch,
    est_drig_gd_analytical_init,
    est_drig_gd_auto,
)
from cirrl.training.trainer import (
    train_cirrl_model,
    eval_mse,
    compare_latent_dimensions,
)
from cirrl.utils.data import (
    load_singlecell_data,
    MyDS,
    make_dataloader,
)

__all__ = [
    # Models
    "DPA",
    "OnlyRelu",
    # Estimators
    "est_drig",
    "est_drig_gd_fast",
    "est_drig_gd_batch",
    "est_drig_gd_analytical_init",
    "est_drig_gd_auto",
    # Training
    "train_cirrl_model",
    "eval_mse",
    "compare_latent_dimensions",
    # Data utilities
    "load_singlecell_data",
    "MyDS",
    "make_dataloader",
]
