# CIRRL Single-Cell Experiment Configuration

# Experiment metadata
experiment:
  name: "singlecell_cirrl"
  description: "CIRRL experiment on single-cell data with multiple environments"
  seed: 123
  output_dir: "results/"

# Data paths
data:
  train_path: "data/singlecell.pkl"
  test_path: "data/singlecelltest.pkl"
  separate_test_path: "data/testenvs_separate.pkl"

# DPA Model configuration
model:
  data_dim: 9
  latent_dims: [3]  # Can be a list for hierarchical learning: [2, 3, 5]
  hidden_dim: 400
  num_layer: 2
  num_layer_enc: null  # null means same as num_layer
  condition_dim: 11  # Number of environments
  
  # Architecture options
  resblock: true
  bn_enc: true  # Batch normalization in encoder
  bn_dec: true  # Batch normalization in decoder
  ln_enc: false  # Layer normalization in encoder
  ln_dec: false  # Layer normalization in decoder
  
  # Stochasticity
  noise_dim: 100
  dist_enc: "deterministic"
  dist_dec: "stochastic"
  
  # Prior configuration
  epsilon: 0.1  # For OnlyRelu variance regularizer
  totalvar: true  # Whether to use full covariance
  
  # Optimization
  lr: 1.0e-4
  l2: 0.0  # L2 regularization weight
  
  # Output
  out_dim: null  # null means same as data_dim
  out_act: null  # Output activation function

# Training configuration
training:
  epochs: 1000
  batch_size: null  # null means full batch
  
  # Loss weights
  alpha: 0.1  # Weight for GMM loss (prior matching)
  beta: 0  # Weight for regularization loss
  gamma: 5  # DRIG gamma for evaluation during training
  
  # Progressive training
  num_pro_epoch: 0  # Epochs for progressive training (0 = start with all levels)
  
  # Logging
  print_every: 100
  verbose: true
  
  # Model checkpointing
  save_model_every: null  # Save model every N epochs (null = don't save)
  save_recon_every: 0  # Save reconstructions every N epochs (0 = don't save)

# DRIG configuration
drig:
  method: "auto"  # Options: "closed_form", "fast", "batch", "analytical_init", "auto"
  gamma_values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Gamma values to test
  
  # Optimization (for gradient-based methods)
  iters: 10000
  lr: 1.0e-3
  device: "auto"  # "auto", "cpu", or "cuda"
  batch_size: null  # For batched method (null = auto)
  
  # Weights
  unif_weight: false  # Whether to use uniform environment weights
  
  # Output
  save_results: true
  results_file: "singlecell_drig_results.csv"

# Comparison experiments (optional)
comparison:
  enabled: false
  latent_dims_list: [2, 3, 5, 10]
  seeds: [123, 456, 789]
  epochs: 500  # Can use fewer epochs for comparison

# Visualization
visualization:
  enabled: true
  save_plots: true
  plot_format: "png"  # "png", "pdf", or "svg"
  dpi: 300
  
  plots:
    - "training_curves"
    - "loss_components"
    - "gamma_sweep"
    - "mse_distribution"
    - "latent_comparison"

# Hardware
hardware:
  device: "auto"  # "auto", "cpu", or "cuda"
  num_workers: 0  # Number of DataLoader workers
  pin_memory: true  # Pin memory for faster GPU transfer
