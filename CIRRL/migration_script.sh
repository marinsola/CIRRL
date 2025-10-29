#!/bin/bash
# Migration script from notebook-based code to package structure

set -e  # Exit on error

echo "============================================"
echo "CIRRL Migration Script"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    print_error "setup.py not found. Please run this script from the project root directory."
    exit 1
fi

print_info "Starting migration..."

# Step 1: Create directory structure
print_info "Creating directory structure..."
mkdir -p cirrl/{models,estimators,training,utils}
mkdir -p experiments/configs
mkdir -p notebooks
mkdir -p tests
mkdir -p data
mkdir -p results
mkdir -p docs
mkdir -p scripts

# Create placeholder files
touch data/.gitkeep
touch results/.gitkeep

print_info "Directory structure created."

# Step 2: Create __init__.py files
print_info "Creating __init__.py files..."
touch cirrl/__init__.py
touch cirrl/models/__init__.py
touch cirrl/estimators/__init__.py
touch cirrl/training/__init__.py
touch cirrl/utils/__init__.py
touch tests/__init__.py

print_info "__init__.py files created."

# Step 3: Check for existing files
print_info "Checking for existing implementation files..."

if [ -f "setup_singlecell.py" ]; then
    print_warning "Found setup_singlecell.py - this file needs to be split manually"
    print_warning "Components should be moved to:"
    echo "  - cirrl/models/networks.py (OnlyRelu, StoLayer, StoResBlock, StoNet)"
    echo "  - cirrl/models/dpa.py (DPAmodel, DPA)"
    echo "  - cirrl/utils/data.py (MyDS, make_dataloader, data utilities)"
    echo "  - cirrl/training/trainer.py (train_one_iter, eval_mse, energy_loss_two_sample)"
    echo "  - cirrl/estimators/drig.py (drig_est and related functions)"
fi

if [ -f "CIRRL_git.ipynb" ]; then
    print_warning "Found CIRRL_git.ipynb - consider converting to:"
    echo "  - experiments/singlecell_experiment.py (main experiment script)"
    echo "  - notebooks/quickstart.ipynb (clean tutorial notebook)"
fi

# Step 4: Install package in development mode
print_info "Installing package in development mode..."
if command -v pip &> /dev/null; then
    pip install -e . 2>/dev/null || print_warning "Package installation failed. Install dependencies first."
else
    print_warning "pip not found. Please install the package manually with: pip install -e ."
fi

# Step 5: Create sample configuration file
print_info "Creating sample configuration file..."
if [ ! -f "experiments/configs/singlecell_config.yaml" ]; then
    cat > experiments/configs/singlecell_config.yaml << 'EOF'
# Sample CIRRL configuration
experiment:
  name: "singlecell_cirrl"
  seed: 123

data:
  train_path: "data/singlecell.pkl"
  test_path: "data/singlecelltest.pkl"

model:
  data_dim: 9
  latent_dims: [3]
  hidden_dim: 400
  lr: 1.0e-4

training:
  epochs: 1000
  alpha: 0.1
  beta: 0
  gamma: 5

drig:
  method: "auto"
  gamma_values: [0, 5, 10, 15]
EOF
    print_info "Sample config created at experiments/configs/singlecell_config.yaml"
fi

# Step 6: Create README if it doesn't exist
if [ ! -f "README.md" ]; then
    print_info "Creating README.md..."
    cat > README.md << 'EOF'
# CIRRL: Causal Invariant Representation Learning

Implementation of CIRRL for learning causal representations across multiple environments.

## Quick Start

```bash
# Install
pip install -e .

# Run experiment
python experiments/singlecell_experiment.py
```

See documentation for more details.
EOF
fi

# Step 7: Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    print_info "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Data and results
*.pkl
*.h5
data/
results/
logs/

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
EOF
fi

# Step 8: Run tests (if they exist)
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    print_info "Running tests..."
    if command -v pytest &> /dev/null; then
        pytest tests/ -v || print_warning "Some tests failed"
    else
        print_warning "pytest not found. Install with: pip install pytest"
    fi
fi

# Step 9: Summary
echo ""
echo "============================================"
echo "Migration Summary"
echo "============================================"
echo ""
print_info "Directory structure created"
print_info "Configuration files set up"
echo ""
echo "Next steps:"
echo "  1. Move code from setup_singlecell.py to appropriate modules"
echo "  2. Convert notebook to clean experiment script"
echo "  3. Update imports in all files"
echo "  4. Run tests: make test"
echo "  5. Format code: make format"
echo ""
echo "For detailed instructions, see PROJECT_STRUCTURE.md"
echo ""
print_info "Migration script completed!"
