# CIRRL Project Structure Guide

This document explains the complete project structure and how to organize your CIRRL repository.

## Complete Directory Structure

```
CIRRL/
│
├── README.md                      # Main project documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation file
├── .gitignore                     # Git ignore rules
├── LICENSE                        # License file (MIT recommended)
├── PROJECT_STRUCTURE.md          # This file
│
├── cirrl/                        # Main Python package
│   ├── __init__.py               # Package initialization
│   │
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   ├── dpa.py               # DPA model (from setup_singlecell.py)
│   │   └── networks.py          # Network layers (StoNet, StoResBlock, etc.)
│   │
│   ├── estimators/              # Statistical estimators
│   │   ├── __init__.py
│   │   └── drig.py             # DRIG implementations (optimized versions)
│   │
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py          # Training loops and evaluation
│   │
│   └── utils/                  # Helper functions
│       ├── __init__.py
│       ├── data.py            # Data loading/processing
│       └── metrics.py         # Evaluation metrics
│
├── experiments/                # Experiment scripts
│   ├── singlecell_experiment.py
│   ├── compare_methods.py
│   └── configs/               # Configuration files
│       └── singlecell_config.yaml
│
├── notebooks/                 # Jupyter notebooks
│   ├── quickstart.ipynb
│   ├── analysis.ipynb
│   └── visualization.ipynb
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_drig.py
│   ├── test_dpa.py
│   └── test_training.py
│
├── data/                     # Data directory (not tracked in git)
│   ├── .gitkeep
│   ├── singlecell.pkl
│   ├── singlecelltest.pkl
│   └── testenvs_separate.pkl
│
├── results/                  # Results directory (not tracked)
│   └── .gitkeep
│
└── docs/                    # Additional documentation
    ├── api.md
    ├── tutorials.md
    └── paper_references.md
```

## File Organization Plan

### Step 1: Extract from setup_singlecell.py

The current `setup_singlecell.py` contains multiple components that should be split:

**Move to `cirrl/models/networks.py`:**
- `OnlyRelu` class
- `StoLayer` class
- `StoResBlock` class
- `StoNet` class
- Helper functions: `get_act_func()`, `vectorize()`

**Move to `cirrl/models/dpa.py`:**
- `DPAmodel` class
- `DPA` class

**Move to `cirrl/utils/data.py`:**
- `MyDS` class
- `make_dataloader()` function

**Move to `cirrl/training/trainer.py`:**
- `train_one_iter()` function
- `eval_mse()` function
- `energy_loss_two_sample()` function
- `drig_est()` function (simple version)

**Move to `cirrl/estimators/drig.py`:**
- All DRIG estimator variants
- Test MSE functions

### Step 2: Clean Up Notebook

The Jupyter notebook should become a **clean experiment runner**, not contain implementation details:

**Keep in notebook:**
- Data loading calls
- Model initialization
- Training calls
- Visualization
- Result analysis

**Remove from notebook:**
- All class/function definitions
- Implementation details
- Replace with proper imports from cirrl package

### Step 3: Create Proper Package Structure

```bash
# Initialize empty __init__.py files
touch cirrl/__init__.py
touch cirrl/models/__init__.py
touch cirrl/estimators/__init__.py
touch cirrl/training/__init__.py
touch cirrl/utils/__init__.py
```

## Migration Steps

### 1. Set Up Repository Structure

```bash
# Create main directories
mkdir -p cirrl/{models,estimators,training,utils}
mkdir -p experiments/configs
mkdir -p notebooks
mkdir -p tests
mkdir -p data results docs

# Create placeholder files
touch data/.gitkeep
touch results/.gitkeep
```

### 2. Split setup_singlecell.py

Create separate files as outlined above. Here's the recommended order:

1. **First**: Create `cirrl/models/networks.py` with basic building blocks
2. **Second**: Create `cirrl/models/dpa.py` (depends on networks.py)
3. **Third**: Create `cirrl/utils/data.py` (standalone utilities)
4. **Fourth**: Create `cirrl/estimators/drig.py` (standalone)
5. **Fifth**: Create `cirrl/training/trainer.py` (depends on all above)

### 3. Update Imports

In each file, update imports to use the new package structure:

```python
# Example: In cirrl/models/dpa.py
from cirrl.models.networks import StoNet, OnlyRelu
from cirrl.utils.data import vectorize

# Example: In experiments/singlecell_experiment.py
from cirrl.models.dpa import DPA, OnlyRelu
from cirrl.estimators.drig import est_drig_gd_auto
from cirrl.training.trainer import train_cirrl_model
```

### 4. Create Clean Experiment Script

Transform the notebook into `experiments/singlecell_experiment.py` with:
- Clear sections
- Proper logging
- Command-line arguments support
- Results saving

### 5. Write Tests

Create unit tests in `tests/`:

```python
# tests/test_drig.py
import numpy as np
from cirrl.estimators.drig import est_drig

def test_drig_basic():
    # Generate simple test data
    # Test that DRIG runs without errors
    # Test output shape
    pass
```

## Installation After Migration

```bash
# Development installation
pip install -e .

# Regular installation
pip install .

# With development tools
pip install -e ".[dev]"
```

## Usage After Migration

```python
# Clean imports
from cirrl import DPA, OnlyRelu, train_cirrl_model, est_drig_gd_auto
from cirrl.utils.data import load_singlecell_data

# Load data
X, Y, E, X_test, Y_test = load_singlecell_data(...)

# Initialize model
dpa = DPA(data_dim=9, latent_dims=[3], ...)

# Train
history = train_cirrl_model(dpa, X, Y, E, X_test, Y_test, ...)

# Evaluate with DRIG
coef = est_drig_gd_auto(train_data, gamma=5)
```

## Configuration Files

Create `experiments/configs/singlecell_config.yaml`:

```yaml
# Model configuration
model:
  data_dim: 9
  latent_dims: [3]
  hidden_dim: 400
  num_layer: 2
  condition_dim: 11
  lr: 1e-4
  bn_enc: true
  bn_dec: true

# Training configuration
training:
  epochs: 1000
  batch_size: null
  alpha: 0.1
  beta: 0
  gamma: 5
  print_every: 100

# Data paths
data:
  train_path: "data/singlecell.pkl"
  test_path: "data/singlecelltest.pkl"
  separate_test_path: "data/testenvs_separate.pkl"

# DRIG configuration
drig:
  method: "auto"
  iters: 10000
  lr: 1e-3
  device: "auto"
```

## Git Workflow

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: Clean CIRRL structure"

# Create development branch
git checkout -b develop

# For features
git checkout -b feature/new-estimator
# ... make changes ...
git commit -m "Add new DRIG variant"
git checkout develop
git merge feature/new-estimator

# Push to remote
git remote add origin https://github.com/yourusername/CIRRL.git
git push -u origin main
```

## Documentation

Generate API documentation:

```bash
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Generate docs
cd docs
sphinx-quickstart
sphinx-apidoc -o . ../cirrl
make html
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=cirrl --cov-report=html

# Specific test file
pytest tests/test_drig.py
```

## Code Quality

```bash
# Format code
black cirrl/ experiments/ tests/

# Check style
flake8 cirrl/

# Type checking
mypy cirrl/
```

## Release Checklist

Before releasing a new version:

1. [ ] All tests pass
2. [ ] Documentation is up to date
3. [ ] CHANGELOG.md is updated
4. [ ] Version number bumped in setup.py and __init__.py
5. [ ] Git tag created
6. [ ] PyPI package built and uploaded (optional)

```bash
# Create release
git tag -a v0.1.0 -m "First release"
git push origin v0.1.0

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (optional)
twine upload dist/*
```

## Summary

This structure provides:
- **Clean separation of concerns**: Models, estimators, training, utilities
- **Easy testing**: Isolated components
- **Reusable code**: Package can be imported anywhere
- **Professional organization**: Follows Python best practices
- **Version control friendly**: Clear file structure
- **Easy to extend**: Add new estimators, models, experiments
- **Documentation ready**: Clear structure for docs

The migration from the monolithic notebook to this structure will make your code:
- More maintainable
- Easier to test
- Easier for others to use and contribute
- Ready for publication alongside a research paper
