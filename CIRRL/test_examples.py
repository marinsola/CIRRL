"""
Unit tests for DRIG estimators
"""

import pytest
import numpy as np
import torch
from cirrl.estimators.drig import (
    est_drig,
    est_drig_gd_fast,
    est_drig_gd_analytical_init,
    est_drig_gd_auto,
    test_mse,
    test_mse_list,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    
    # True coefficients
    true_coef = np.array([1.5, -0.5, 2.0])
    n_envs = 3
    n_samples = 100
    n_features = 3
    
    # Generate data for multiple environments
    data_list = []
    for env in range(n_envs):
        X = np.random.randn(n_samples, n_features)
        # Add environment-specific noise
        y = X @ true_coef + np.random.randn(n_samples) * (0.1 + 0.1 * env)
        data = np.column_stack([X, y])
        data_list.append(data)
    
    return data_list, true_coef


def test_est_drig_closed_form(synthetic_data):
    """Test closed-form DRIG estimator."""
    data_list, true_coef = synthetic_data
    
    # Test with gamma=0 (observational only)
    coef_obs = est_drig(data_list, gamma=0.0, y_idx=-1)
    assert coef_obs.shape == true_coef.shape
    assert np.isfinite(coef_obs).all()
    
    # Test with gamma=1 (pooled)
    coef_pooled = est_drig(data_list, gamma=1.0, y_idx=-1)
    assert coef_pooled.shape == true_coef.shape
    assert np.isfinite(coef_pooled).all()
    
    # Test with intermediate gamma
    coef_drig = est_drig(data_list, gamma=0.5, y_idx=-1)
    assert coef_drig.shape == true_coef.shape
    assert np.isfinite(coef_drig).all()


def test_est_drig_gd_fast(synthetic_data):
    """Test fast gradient descent DRIG estimator."""
    data_list, true_coef = synthetic_data
    
    coef = est_drig_gd_fast(
        data_list, 
        gamma=0.5, 
        iters=1000, 
        lr=1e-2,
        device='cpu',
        verbose=False
    )
    
    assert coef.shape == true_coef.shape
    assert np.isfinite(coef).all()
    
    # Should be reasonably close to true coefficients
    error = np.linalg.norm(coef - true_coef)
    assert error < 1.0, f"Estimation error {error} is too large"


def test_est_drig_gd_analytical_init(synthetic_data):
    """Test DRIG with analytical initialization."""
    data_list, true_coef = synthetic_data
    
    coef = est_drig_gd_analytical_init(
        data_list,
        gamma=0.5,
        iters=500,  # Should need fewer iterations
        lr=1e-2,
        device='cpu',
        verbose=False
    )
    
    assert coef.shape == true_coef.shape
    assert np.isfinite(coef).all()


def test_est_drig_gd_auto(synthetic_data):
    """Test automatic DRIG method selection."""
    data_list, true_coef = synthetic_data
    
    coef = est_drig_gd_auto(
        data_list,
        gamma=0.5,
        device='cpu',
        verbose=False
    )
    
    assert coef.shape == true_coef.shape
    assert np.isfinite(coef).all()


def test_uniform_weights(synthetic_data):
    """Test DRIG with uniform environment weights."""
    data_list, true_coef = synthetic_data
    
    coef_weighted = est_drig(data_list, gamma=0.5, unif_weight=False)
    coef_uniform = est_drig(data_list, gamma=0.5, unif_weight=True)
    
    # Results should be different
    assert not np.allclose(coef_weighted, coef_uniform)


def test_test_mse(synthetic_data):
    """Test MSE computation."""
    data_list, true_coef = synthetic_data
    
    # Compute MSE with true coefficients
    mse = test_mse(data_list[0], true_coef, y_idx=-1)
    
    assert isinstance(mse, float)
    assert mse >= 0
    assert np.isfinite(mse)


def test_test_mse_list(synthetic_data):
    """Test MSE computation on multiple datasets."""
    data_list, true_coef = synthetic_data
    
    # Test without pooling
    mses = test_mse_list(data_list, true_coef, pooled=False)
    assert len(mses) == len(data_list)
    assert all(mse >= 0 for mse in mses)
    
    # Test with pooling
    mses_pooled = test_mse_list(data_list, true_coef, pooled=True)
    assert len(mses_pooled) == len(data_list) + 1
    
    # Test with stats only
    mean_mse, std_mse, max_mse = test_mse_list(
        data_list, true_coef, stats_only=True
    )
    assert mean_mse >= 0
    assert std_mse >= 0
    assert max_mse >= mean_mse


def test_gamma_range(synthetic_data):
    """Test DRIG with different gamma values."""
    data_list, _ = synthetic_data
    
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for gamma in gammas:
        coef = est_drig(data_list, gamma=gamma, y_idx=-1)
        assert np.isfinite(coef).all()


def test_edge_cases(synthetic_data):
    """Test edge cases."""
    data_list, _ = synthetic_data
    
    # Test with very small dataset
    small_data = [d[:10] for d in data_list]
    coef = est_drig(small_data, gamma=0.5, y_idx=-1)
    assert np.isfinite(coef).all()
    
    # Test with single environment
    single_env = [data_list[0]]
    coef = est_drig(single_env, gamma=0.5, y_idx=-1)
    assert np.isfinite(coef).all()


def test_convergence(synthetic_data):
    """Test that gradient descent converges."""
    data_list, _ = synthetic_data
    
    # Run with different iteration counts
    coef_short = est_drig_gd_fast(data_list, gamma=0.5, iters=100, device='cpu')
    coef_long = est_drig_gd_fast(data_list, gamma=0.5, iters=5000, device='cpu')
    
    # Longer training should give similar or better results
    # (this is a weak test, but checks for obvious issues)
    assert np.isfinite(coef_short).all()
    assert np.isfinite(coef_long).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_support(synthetic_data):
    """Test GPU computation if available."""
    data_list, _ = synthetic_data
    
    coef_gpu = est_drig_gd_fast(
        data_list,
        gamma=0.5,
        iters=100,
        device='cuda',
        verbose=False
    )
    
    assert np.isfinite(coef_gpu).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
