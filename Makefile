# Makefile for CIRRL project

.PHONY: help install install-dev test lint format clean run-experiment docs

# Default target
help:
	@echo "CIRRL Project Makefile"
	@echo "======================"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install package in editable mode"
	@echo "  install-dev    - Install package with development dependencies"
	@echo "  test          - Run unit tests"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  lint          - Run code linting (flake8)"
	@echo "  format        - Format code with black"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  clean         - Remove build artifacts and cache files"
	@echo "  run-experiment - Run single-cell experiment"
	@echo "  docs          - Generate documentation"
	@echo "  all           - Format, lint, and test"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pytest pytest-cov black flake8 mypy sphinx sphinx-rtd-theme

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=cirrl --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	flake8 cirrl/ experiments/ tests/ --max-line-length=100 --ignore=E203,W503

format:
	black cirrl/ experiments/ tests/ --line-length=100

type-check:
	mypy cirrl/ --ignore-missing-imports

# Run quality checks
all: format lint test

# Cleaning
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/
	@echo "Cleaned up build artifacts and cache files"

# Run experiments
run-experiment:
	python experiments/singlecell_experiment.py

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation generated in docs/_build/html/index.html"

# Create necessary directories
setup-dirs:
	mkdir -p data results logs docs

# Download sample data (placeholder - adjust URL as needed)
download-data:
	@echo "Downloading sample data..."
	@echo "Note: Update this target with actual data URLs"
	mkdir -p data
	# wget -O data/singlecell.pkl https://your-data-url/singlecell.pkl
	# wget -O data/singlecelltest.pkl https://your-data-url/singlecelltest.pkl

# Check code style without modifying files
check-format:
	black cirrl/ experiments/ tests/ --check --line-length=100

# Run a quick check (format, lint, test)
check: check-format lint test

# Development workflow
dev-setup: install-dev setup-dirs
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"
