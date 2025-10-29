# CIRRL: Causal Invariant Representation Learning

Implementation of CIRRL (Distributional Principal Autoencoder with DRIG).

## Overview

CIRRL learns causal representations by combining:
- **DPA (Distributional Principal Autoencoder)**: Learns latent representations across multiple environments
- **DRIG (Distributionally Robust Instrumental Regression)**: Performs robust regression on learned representations

## Installation

```bash
git clone https://github.com/marinsola/CIRRL.git
cd CIRRL
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Project Structure

```
CIRRL/
├── cirrl/                      # Main package
│   ├── models/                 # Neural network models
│   │   ├── dpa.py             # DPA model implementation
│   │   └── networks.py        # Network layers and blocks
│   ├── estimators/            # Statistical estimators
│   │   └── drig.py            # DRIG implementations
│   ├── training/              # Training utilities
│   │   └── trainer.py         # Training loops and evaluation
│   └── utils/                 # Helper functions
│       ├── data.py            # Data loading/processing
│       └── metrics.py         # Evaluation metrics
├── experiments/               # Experiment scripts
│   ├── singlecell_experiment.py
│   └── configs/              # Configuration files
├── notebooks/                # Jupyter notebooks for analysis
└── data/                     # Data directory (not tracked)
```

## Quick Start

### Single-Cell Experiment

```python
from cirrl.models.dpa import DPA
from cirrl.estimators.drig import est_drig_gd_auto
from cirrl.training.trainer import train_cirrl_model
from cirrl.utils.data import load_singlecell_data

# Load data
X, Y, E, X_test, Y_test = load_singlecell_data('data/singlecell.pkl', 
                                                'data/singlecelltest.pkl')

# Initialize model
model = DPA(
    data_dim=9,
    latent_dims=[3],
    condition_dim=11,
    hidden_dim=400,
    lr=1e-4
)

# Train model
history = train_cirrl_model(
    model, X, Y, E, X_test, Y_test,
    alpha=0.1,
    beta=0,
    gamma=5,
    epochs=1000
)

# Extract latent representations and apply DRIG
z_train = model.encode(X, E)
z_test = model.encode(X_test)

# Prepare data for DRIG
train_data = prepare_drig_data(z_train, Y, E)
drig_coef = est_drig_gd_auto(train_data, gamma=5)

# Make predictions
y_pred = z_test @ drig_coef
```

## Usage Examples

### Training with Different Configurations

```python
# Compare different latent dimensions
from cirrl.training.trainer import compare_latent_dimensions

results = compare_latent_dimensions(
    X, Y, E, X_test, Y_test,
    latent_dims=[2, 3, 5, 10],
    seeds=[123, 456, 789],
    epochs=1000
)
```

### Using Optimized DRIG Estimators

```python
from cirrl.estimators.drig import (
    est_drig,              # Closed-form solution
    est_drig_gd_fast,      # Fast gradient descent
    est_drig_gd_batch,     # Mini-batch version
    est_drig_gd_auto       # Auto-select best method
)

# Auto-select based on data size
coef = est_drig_gd_auto(train_data, gamma=5, device='cuda')

# For large datasets, use batched version
coef = est_drig_gd_batch(train_data, gamma=5, batch_size=1024)

# With analytical initialization for faster convergence
coef = est_drig_gd_analytical_init(train_data, gamma=5, iters=5000)
```

## Performance Tips

1. **GPU Usage**: The code automatically uses CUDA if available
2. **Memory**: For large datasets, use `est_drig_gd_batch` with appropriate `batch_size`
3. **Convergence**: Use `est_drig_gd_analytical_init` for faster convergence on medium-sized datasets
4. **Device Selection**: Set `device='auto'` in DRIG estimators for automatic GPU/CPU selection

## Contact

For questions or issues, please open an issue on GitHub or contact [marin.sola@math.ethz.ch]
