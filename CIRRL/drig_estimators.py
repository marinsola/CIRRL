"""
DRIG (Distributionally Robust Instrumental Regression) Estimators

This module provides multiple implementations of DRIG estimators optimized for different
data sizes and computational resources.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union


def est_drig(data: List[np.ndarray], 
             gamma: float, 
             y_idx: int = -1, 
             del_idx: Optional[int] = None, 
             unif_weight: bool = False) -> np.ndarray:
    """
    Closed-form DRIG estimator.
    
    Best for: Small to medium datasets where matrix inversion is feasible.
    
    Args:
        data: List of numpy arrays from all environments (first element is observational)
        gamma: DRIG hyperparameter (0 = observational only, 1 = pooled)
        y_idx: Index of the response variable
        del_idx: Index of variable to exclude
        unif_weight: Whether to use uniform weights across environments
        
    Returns:
        Estimated coefficients as numpy array
    """
    if del_idx is None:
        del_idx = y_idx
        
    m = len(data)
    
    # Compute weights
    if unif_weight:
        w = [1/m] * m
    else:
        w = [data[e].shape[0] for e in range(m)]
        w = [a/sum(w) for a in w]
    
    # Compute gram matrices
    gram_x = []  # E[XX^T]
    gram_xy = []  # E[XY]
    
    for e in range(m):
        data_e = data[e]
        n = data_e.shape[0]
        y = data_e[:, y_idx]
        x = np.delete(data_e, (y_idx, del_idx), 1)
        gram_x.append(x.T.dot(x) / n)
        gram_xy.append(x.T.dot(y) / n)
    
    # DRIG weighted combination
    G = (1 - gamma) * gram_x[0] + gamma * sum([a*b for a, b in zip(gram_x, w)])
    Z = (1 - gamma) * gram_xy[0] + gamma * sum([a*b for a, b in zip(gram_xy, w)])
    
    return np.linalg.inv(G).dot(Z)


def est_drig_gd_fast(data: List[np.ndarray],
                     gamma: float,
                     iters: int = 10000,
                     lr: float = 1e-3,
                     verbose: bool = False,
                     y_idx: int = -1,
                     del_idx: Optional[int] = None,
                     unif_weight: bool = False,
                     device: str = 'cpu') -> np.ndarray:
    """
    Fast gradient descent DRIG estimator with optimized PyTorch operations.
    
    Best for: Medium datasets (1K-50K samples) with GPU acceleration.
    
    Args:
        data: List of numpy arrays from all environments
        gamma: DRIG hyperparameter
        iters: Number of optimization iterations
        lr: Learning rate
        verbose: Whether to print progress
        y_idx: Index of response variable
        del_idx: Index of variable to exclude
        unif_weight: Whether to use uniform weights
        device: Device for computation ('cpu' or 'cuda')
        
    Returns:
        Estimated coefficients as numpy array
    """
    if del_idx is None:
        del_idx = y_idx
    
    m = len(data)
    
    # Pre-process all data once and move to device
    data_x = []
    data_y = []
    
    for e in range(m):
        data_e = data[e]
        y = torch.tensor(data_e[:, y_idx], dtype=torch.float32, device=device)
        x = torch.tensor(np.delete(data_e, (y_idx, del_idx), 1), dtype=torch.float32, device=device)
        data_x.append(x)
        data_y.append(y)
    
    # Initialize parameters
    b = torch.zeros(data_x[0].shape[1], dtype=torch.float32, device=device, requires_grad=True)
    
    # Pre-compute weights
    if unif_weight:
        w = torch.tensor([1/m] * m, dtype=torch.float32, device=device)
    else:
        weights = [data[e].shape[0] for e in range(m)]
        total_weight = sum(weights)
        w = torch.tensor([weight/total_weight for weight in weights], dtype=torch.float32, device=device)
    
    # Optimizer
    opt = torch.optim.Adam([b], lr=lr, eps=1e-8)
    
    for i in range(iters):
        opt.zero_grad()
        
        # Compute losses for each environment
        current_losses = []
        for e in range(m):
            pred = data_x[e] @ b
            loss_e = torch.mean((data_y[e] - pred) ** 2)
            current_losses.append(loss_e)
        
        # DRIG objective
        losses_tensor = torch.stack(current_losses)
        loss_min = torch.min(losses_tensor)
        weighted_loss = torch.sum(losses_tensor * w)
        
        loss = (1 - gamma) * loss_min + gamma * weighted_loss
        loss.backward()
        opt.step()
        
        if verbose and (i == 0 or (i + 1) % 100 == 0):
            print(f'Iter {i + 1}: Loss = {loss.item():.4f}')
    
    return b.cpu().detach().numpy()


def est_drig_gd_batch(data: List[np.ndarray],
                      gamma: float,
                      iters: int = 10000,
                      lr: float = 1e-3,
                      verbose: bool = False,
                      y_idx: int = -1,
                      del_idx: Optional[int] = None,
                      unif_weight: bool = False,
                      device: str = 'cpu',
                      batch_size: Optional[int] = None) -> np.ndarray:
    """
    Mini-batch gradient descent DRIG estimator for large datasets.
    
    Best for: Large datasets (>50K samples) where full-batch doesn't fit in memory.
    
    Args:
        data: List of numpy arrays from all environments
        gamma: DRIG hyperparameter
        iters: Number of optimization iterations
        lr: Learning rate
        verbose: Whether to print progress
        y_idx: Index of response variable
        del_idx: Index of variable to exclude
        unif_weight: Whether to use uniform weights
        device: Device for computation
        batch_size: Mini-batch size (None for full batch)
        
    Returns:
        Estimated coefficients as numpy array
    """
    if del_idx is None:
        del_idx = y_idx
    
    m = len(data)
    
    # Pre-process data
    processed_data = []
    for e in range(m):
        data_e = data[e]
        y = torch.tensor(data_e[:, y_idx], dtype=torch.float32, device=device)
        x = torch.tensor(np.delete(data_e, (y_idx, del_idx), 1), dtype=torch.float32, device=device)
        processed_data.append((x, y))
    
    # Initialize parameters
    feature_dim = processed_data[0][0].shape[1]
    b = torch.zeros(feature_dim, dtype=torch.float32, device=device, requires_grad=True)
    
    # Pre-compute weights
    if unif_weight:
        w = torch.tensor([1/m] * m, dtype=torch.float32, device=device)
    else:
        weights = [len(processed_data[e][1]) for e in range(m)]
        total_weight = sum(weights)
        w = torch.tensor([weight/total_weight for weight in weights], dtype=torch.float32, device=device)
    
    # Optimizer with adaptive learning rate
    opt = torch.optim.AdamW([b], lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1000, factor=0.8)
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 2000
    
    for i in range(iters):
        opt.zero_grad()
        
        # Compute losses for all environments
        env_losses = []
        for e in range(m):
            x_e, y_e = processed_data[e]
            
            if batch_size and len(y_e) > batch_size:
                # Mini-batch sampling
                idx = torch.randperm(len(y_e), device=device)[:batch_size]
                x_batch = x_e[idx]
                y_batch = y_e[idx]
            else:
                x_batch = x_e
                y_batch = y_e
            
            pred = x_batch @ b
            loss_e = torch.mean((y_batch - pred) ** 2)
            env_losses.append(loss_e)
        
        # DRIG objective
        losses_tensor = torch.stack(env_losses)
        loss_min = torch.min(losses_tensor)
        weighted_loss = torch.sum(losses_tensor * w)
        loss = (1 - gamma) * loss_min + gamma * weighted_loss
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([b], max_norm=1.0)
        
        opt.step()
        scheduler.step(loss)
        
        # Early stopping
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at iteration {i+1}")
            break
        
        if verbose and (i == 0 or (i + 1) % 100 == 0):
            print(f'Iter {i + 1}: Loss = {current_loss:.4f}, LR = {opt.param_groups[0]["lr"]:.2e}')
    
    return b.cpu().detach().numpy()


def est_drig_gd_analytical_init(data: List[np.ndarray],
                                gamma: float,
                                iters: int = 5000,
                                lr: float = 1e-3,
                                verbose: bool = False,
                                y_idx: int = -1,
                                del_idx: Optional[int] = None,
                                unif_weight: bool = False,
                                device: str = 'cpu') -> np.ndarray:
    """
    DRIG with analytical least squares initialization for faster convergence.
    
    Best for: Medium datasets where good initialization significantly reduces iterations.
    
    Args:
        data: List of numpy arrays from all environments
        gamma: DRIG hyperparameter
        iters: Number of optimization iterations (typically needs fewer than other methods)
        lr: Learning rate
        verbose: Whether to print progress
        y_idx: Index of response variable
        del_idx: Index of variable to exclude
        unif_weight: Whether to use uniform weights
        device: Device for computation
        
    Returns:
        Estimated coefficients as numpy array
    """
    if del_idx is None:
        del_idx = y_idx
    
    m = len(data)
    
    # Pre-process data
    data_x = []
    data_y = []
    
    for e in range(m):
        data_e = data[e]
        y = torch.tensor(data_e[:, y_idx], dtype=torch.float32, device=device)
        x = torch.tensor(np.delete(data_e, (y_idx, del_idx), 1), dtype=torch.float32, device=device)
        data_x.append(x)
        data_y.append(y)
    
    # Initialize with least squares solution from pooled data
    X_pooled = torch.cat(data_x, dim=0)
    y_pooled = torch.cat(data_y, dim=0)
    
    try:
        XtX = X_pooled.T @ X_pooled
        Xty = X_pooled.T @ y_pooled
        b_init = torch.linalg.solve(XtX + 1e-6 * torch.eye(XtX.shape[0], device=device), Xty)
    except:
        # Fallback to pseudo-inverse if singular
        b_init = torch.linalg.pinv(X_pooled) @ y_pooled
    
    b = torch.nn.Parameter(b_init.clone().detach())
    
    # Pre-compute weights
    if unif_weight:
        w = torch.tensor([1/m] * m, dtype=torch.float32, device=device)
    else:
        weights = [len(data_y[e]) for e in range(m)]
        total_weight = sum(weights)
        w = torch.tensor([weight/total_weight for weight in weights], dtype=torch.float32, device=device)
    
    # Optimizer with lower learning rate (closer to optimum)
    opt = torch.optim.Adam([b], lr=lr*0.1, eps=1e-8)
    
    for i in range(iters):
        opt.zero_grad()
        
        # Compute losses
        losses = []
        for e in range(m):
            pred = data_x[e] @ b
            loss_e = torch.mean((data_y[e] - pred) ** 2)
            losses.append(loss_e)
        
        losses_tensor = torch.stack(losses)
        loss_min = torch.min(losses_tensor)
        weighted_loss = torch.sum(losses_tensor * w)
        
        loss = (1 - gamma) * loss_min + gamma * weighted_loss
        loss.backward()
        opt.step()
        
        if verbose and (i == 0 or (i + 1) % 100 == 0):
            print(f'Iter {i + 1}: Loss = {loss.item():.4f}')
    
    return b.cpu().detach().numpy()


def est_drig_gd_auto(data: List[np.ndarray],
                     gamma: float,
                     iters: int = 10000,
                     lr: float = 1e-3,
                     verbose: bool = False,
                     y_idx: int = -1,
                     del_idx: Optional[int] = None,
                     unif_weight: bool = False,
                     device: str = 'auto') -> np.ndarray:
    """
    Automatically choose the best DRIG optimization strategy based on data size and hardware.
    
    This function analyzes the dataset size and available computational resources to select
    the most efficient estimation method.
    
    Args:
        data: List of numpy arrays from all environments
        gamma: DRIG hyperparameter
        iters: Number of optimization iterations
        lr: Learning rate
        verbose: Whether to print progress
        y_idx: Index of response variable
        del_idx: Index of variable to exclude
        unif_weight: Whether to use uniform weights
        device: Device for computation ('auto', 'cpu', or 'cuda')
        
    Returns:
        Estimated coefficients as numpy array
    """
    # Auto device selection
    if device == 'auto':
        if torch.cuda.is_available():
            total_samples = sum(len(d) for d in data)
            # Only use GPU for larger datasets
            device = 'cuda' if total_samples > 10000 else 'cpu'
        else:
            device = 'cpu'
    
    total_samples = sum(len(d) for d in data)
    
    if verbose:
        print(f"Auto-selecting DRIG method...")
        print(f"Device: {device}")
        print(f"Total samples: {total_samples}")
    
    # Choose method based on data size
    if total_samples > 50000:
        if verbose:
            print("Using batched gradient descent (large dataset)")
        return est_drig_gd_batch(data, gamma, iters, lr, verbose, y_idx, del_idx, 
                                unif_weight, device, batch_size=1024)
    elif total_samples > 10000:
        if verbose:
            print("Using analytical initialization (medium dataset)")
        return est_drig_gd_analytical_init(data, gamma, iters//2, lr, verbose, 
                                          y_idx, del_idx, unif_weight, device)
    else:
        if verbose:
            print("Using fast gradient descent (small dataset)")
        return est_drig_gd_fast(data, gamma, iters, lr, verbose, y_idx, del_idx, 
                               unif_weight, device)


def test_mse(data: np.ndarray, 
             b: np.ndarray, 
             y_idx: int = -1, 
             del_idx: Optional[int] = None) -> float:
    """
    Compute test MSE on a single dataset.
    
    Args:
        data: Test data array
        b: Coefficient vector
        y_idx: Index of response variable
        del_idx: Index of variable to exclude
        
    Returns:
        Mean squared error
    """
    if del_idx is None:
        del_idx = y_idx
    
    x = np.delete(data, (y_idx, del_idx), 1)
    y = data[:, y_idx]
    y_pred = x.dot(b)
    
    return ((y - y_pred)**2).mean()


def test_mse_list(data_list: List[np.ndarray],
                  b: np.ndarray,
                  pooled: bool = False,
                  stats_only: bool = False,
                  y_idx: int = -1,
                  del_idx: Optional[int] = None) -> Union[List[float], tuple]:
    """
    Test on multiple datasets and optionally return statistics.
    
    Args:
        data_list: List of test datasets
        b: Coefficient vector
        pooled: Whether to also compute MSE on pooled data
        stats_only: Whether to return only statistics (mean, std, max)
        y_idx: Index of response variable
        del_idx: Index of variable to exclude
        
    Returns:
        List of errors or tuple of (mean, std, max) if stats_only=True
    """
    if del_idx is None:
        del_idx = y_idx
    
    errors = []
    for i in range(len(data_list)):
        errors.append(test_mse(data_list[i], b, y_idx, del_idx))
    
    if pooled:
        errors.append(test_mse(np.concatenate(data_list), b, y_idx, del_idx))
    
    if stats_only:
        return np.mean(errors), np.std(errors), np.max(errors)
    else:
        return errors
