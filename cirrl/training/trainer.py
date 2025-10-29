"""
Training utilities for CIRRL models
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List


def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=True, weights=None):
    """
    Energy loss based on two samples (for stable estimation).
    
    Args:
        x0: Sample from true distribution
        x: Sample from estimated distribution
        xp: Another sample from estimated distribution
        x0p: Another sample from true distribution (optional)
        beta: Power parameter in energy score
        verbose: Whether to return loss components
        weights: Sample weights
        
    Returns:
        Loss tensor or tuple of (loss, s1, s2) if verbose=True
    """
    from torch.linalg import vector_norm
    from cirrl.utils.data import vectorize
    
    EPS = 0 if float(beta).is_integer() else 1e-5
    
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    
    if weights is None:
        weights = 1 / x0.size(0)
    
    if x0p is None:
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2 + \
             ((vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2
        s2 = ((vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) * weights).sum()
        loss = s1 - s2 / 2
    else:
        x0p = vectorize(x0p)
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta).sum() +
              (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta).sum() +
              (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta).sum() +
              (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta).sum()) / 4
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta).sum()
        s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta).sum()
        loss = s1 - s2 / 2 - s3 / 2
    
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss


def train_one_iter(dpa, x_batch, c_batch, k_max, alpha=1.0, beta=1.0, isotropic=False, reg=False):
    """
    Execute one training iteration.
    
    Args:
        dpa: DPA model
        x_batch: Batch of input data
        c_batch: Batch of environment indicators
        k_max: Maximum latent level to train
        alpha: Weight for GMM loss
        beta: Weight for regularization loss
        isotropic: Whether to use isotropic prior
        reg: Whether to use regularization on prior variance
        
    Returns:
        Tuple of (dpa_loss, gmm_loss, reg_loss) as floats
    """
    dpa.model.zero_grad()
    losses = []
    
    for k in range(k_max + 1):
        # Generate two samples from the model
        gen1, gen2, z = dpa.model(
            x=x_batch, 
            k=dpa.latent_dims[k], 
            c=None, 
            return_latent=True, 
            double=True
        )
        
        # Sample from prior
        if isotropic:
            z_1 = torch.randn(size=z.shape, device=dpa.device)
            z_2 = torch.randn(size=z.shape, device=dpa.device)
        else:
            prior_ = dpa.model.priore(c_batch, double=True, reg=reg)
            if reg:
                z_1, z_2, sigma = prior_
            else:
                z_1, z_2 = prior_
        
        # Compute losses
        loss_dpa, s1, s2 = energy_loss_two_sample(
            x_batch, gen1, gen2, beta=dpa.beta, verbose=True
        )
        loss_gmm, s1_, s2_ = energy_loss_two_sample(
            z, z_1, z_2, beta=dpa.beta, verbose=True
        )
        
        # Regularization loss (encourages unit norm latents)
        reg_loss = torch.mean(torch.linalg.norm(z, ord=2, dim=1) ** 2, 0)
        
        # Combined loss
        loss = loss_dpa + alpha * loss_gmm + beta * (1 / reg_loss)
        
        if reg:
            loss += 1 / (torch.sum(sigma.pow(2)) + 1e-04)
        
        # Track losses
        dpa.loss_all_k[k] += loss.item()
        dpa.loss_pred_all_k[k] += s1.item()
        dpa.loss_var_all_k[k] += s2.item()
        losses.append(loss)
    
    # Backpropagation
    total_loss = sum(losses)
    total_loss.backward()
    dpa.optimizer.step()
    
    return loss_dpa.item(), loss_gmm.item(), reg_loss.item()


def train_cirrl_model(dpa, X, Y, E, X_test, Y_test, 
                      alpha=0.1, beta=0, gamma=5, 
                      epochs=1000, batch_size=None,
                      print_every=10, verbose=True):
    """
    Train CIRRL model (DPA + DRIG).
    
    Args:
        dpa: DPA model instance
        X, Y, E: Training data (features, outcomes, environments)
        X_test, Y_test: Test data
        alpha: Weight for GMM loss
        beta: Weight for regularization
        gamma: DRIG gamma parameter for evaluation
        epochs: Number of training epochs
        batch_size: Batch size (None = full batch)
        print_every: Print frequency
        verbose: Whether to print progress
        
    Returns:
        Dictionary with training history
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    device = dpa.device
    
    # Use full batch if not specified
    if batch_size is None:
        batch_size = len(X)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X, E)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Move data to device
    X_tensor = X.to(device)
    Y_tensor = Y.to(device)
    E_tensor = E.to(device)
    X_test_tensor = X_test.to(device)
    Y_test_tensor = Y_test.to(device)
    
    n_envs = E.shape[1]
    
    # Training tracking
    train_mse_history = []
    test_mse_history = []
    dpa_loss_history = []
    gmm_loss_history = []
    reg_loss_history = []
    
    # Initial evaluation
    if verbose:
        n_in_ref = int(torch.sum(E[:, 0]))
        initial_train_mse, initial_test_mse = eval_mse(
            dpa, X_tensor, Y_tensor, n_in_ref, E_tensor,
            X_test_tensor, Y_test_tensor, gamma=gamma, printit=False
        )
        print(f'Initial MSE - Train: {initial_train_mse:.4f}, Test: {initial_test_mse:.4f}')
    
    # Training loop
    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    
    for epoch in iterator:
        dpa.model.train()
        dpa.loss_all_k = np.zeros(dpa.num_levels)
        dpa.loss_pred_all_k = np.zeros(dpa.num_levels)
        dpa.loss_var_all_k = np.zeros(dpa.num_levels)
        
        epoch_dpa_loss = []
        epoch_gmm_loss = []
        epoch_reg_loss = []
        
        for batch_idx, (x_batch, c_batch) in enumerate(train_dl):
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            
            # Progressive training: gradually increase latent dimension
            k_max = min(int((epoch + 1) // 1e-05), dpa.num_levels - 1)
            
            dpa_loss, gmm_loss, reg_loss = train_one_iter(
                dpa, x_batch, c_batch, k_max,
                alpha=alpha, beta=beta, isotropic=False
            )
            
            epoch_dpa_loss.append(dpa_loss)
            epoch_gmm_loss.append(gmm_loss)
            epoch_reg_loss.append(reg_loss)
        
        # Periodic evaluation
        if (epoch == 0 or (epoch + 1) % print_every == 0):
            n_in_ref = int(torch.sum(E[:, 0]))
            train_mse, test_mse = eval_mse(
                dpa, X_tensor, Y_tensor, n_in_ref, E_tensor,
                X_test_tensor, Y_test_tensor, gamma=gamma, 
                epoch=epoch+1, trainlist=train_mse_history,
                testlist=test_mse_history, printit=False
            )
            
            dpa_loss_history.append(np.mean(epoch_dpa_loss))
            gmm_loss_history.append(np.mean(epoch_gmm_loss))
            reg_loss_history.append(np.mean(epoch_reg_loss))
            
            if verbose:
                if epoch % 100 == 0:
                    print(f"\nEpoch {epoch+1}/{epochs}:")
                    print(f"  Train MSE: {train_mse:.4f}")
                    print(f"  Test MSE: {test_mse:.4f}")
                    print(f"  DPA Loss: {dpa_loss_history[-1]:.4f}")
                    print(f"  GMM Loss: {gmm_loss_history[-1]:.4f}")
    
    return {
        'train_mse': train_mse_history,
        'test_mse': test_mse_history,
        'dpa_loss': dpa_loss_history,
        'gmm_loss': gmm_loss_history,
        'reg_loss': reg_loss_history
    }


def eval_mse(dpa, X, Y, n_in_ref, E, X_test, Y_test, 
             gamma=2, epoch=None, trainlist=None, testlist=None, 
             printit=True):
    """
    Evaluate MSE using DRIG on learned representations.
    
    Args:
        dpa: Trained DPA model
        X, Y, E: Training data
        X_test, Y_test: Test data
        n_in_ref: Number of samples in reference environment
        gamma: DRIG gamma parameter
        epoch: Current epoch (for printing)
        trainlist: List to append train MSE
        testlist: List to append test MSE
        printit: Whether to print results
        
    Returns:
        Tuple of (train_mse, test_mse)
    """
    
    dpa.model.eval()
    
    with torch.no_grad():
        # Extract latent representations
        _, _, z = dpa.model(
            x=X.to(dpa.device), 
            k=dpa.latent_dims[0], 
            c=E.to(dpa.device), 
            return_latent=True, 
            double=True
        )
        
        _, _, z_test = dpa.model(
            x=X_test.to(dpa.device), 
            k=dpa.latent_dims[0], 
            c=E.to(dpa.device), 
            return_latent=True, 
            double=True
        )
    
    # Center representations
    center_of_z_ref = torch.mean(z[:n_in_ref], dim=0)
    z_centered = z.to('cpu') - center_of_z_ref.to('cpu')
    z_test_cen = z_test.to('cpu') - center_of_z_ref.to('cpu')
    
    center_of_y_ref = torch.mean(Y[:n_in_ref], dim=0)
    y_centered = Y.to('cpu') - center_of_y_ref.to('cpu')
    y_test_cen = Y_test.to('cpu') - center_of_y_ref.to('cpu')
    
    # Estimate DRIG coefficients
    drig_estimate = drig_est(z_centered, y_centered, gamma=gamma, m=5, E=E)
    
    # Compute predictions
    y_hat = z_centered @ drig_estimate
    y_test_hat = z_test_cen @ drig_estimate
    
    # Compute MSE
    trainloss = F.mse_loss(y_hat, y_centered).item()
    testloss = F.mse_loss(y_test_hat, y_test_cen).item()
    
    if trainlist is not None:
        trainlist.append(trainloss)
    if testlist is not None:
        testlist.append(testloss)
    
    dpa.model.train()
    
    if printit:
        if epoch is None:
            print(f'MSE - Train: {trainloss:.4f}, Test: {testloss:.4f}')
        else:
            print(f'Epoch {epoch} - MSE - Train: {trainloss:.4f}, Test: {testloss:.4f}')
    
    return trainloss, testloss


def drig_est(X, Y, gamma, m, E=None, device='cpu'):
    """
    DRIG estimator for latent representations.
    
    Args:
        X: Latent features tensor
        Y: Outcomes tensor
        gamma: DRIG hyperparameter
        m: Number of environments
        E: Environment indicators (optional)
        device: Device for computation
        
    Returns:
        Estimated coefficients
    """
    if E is not None:
        # Group by environment
        unique_rows, indices = torch.unique(E, dim=0, return_inverse=True)
        if device is not None:
            indices = indices.to(X.device)
        
        X_ = []
        Y_ = []
        for i in range(len(unique_rows)):
            groupx = X[indices == i]
            groupy = Y[indices == i]
            X_.append(groupx)
            Y_.append(groupy)
        
        X_, Y_ = X_[::-1], Y_[::-1]
        size = [len(y) for y in Y_]
    else:
        # Equal split
        size = [len(Y) // m] * m
        X_ = [X[s * idx : s * (idx + 1), :] for idx, s in enumerate(size)]
        Y_ = [Y[s * idx : s * (idx + 1)].unsqueeze(1) for idx, s in enumerate(size)]
    
    # Compute weights
    w_ = [s / X.shape[0] for s in size]
    
    # Compute gram matrices
    X_X = [(X_[e].t() @ X_[e]) / size[e] for e in range(len(X_))]
    X_Y = [(X_[e].t() @ Y_[e]) / size[e] for e in range(len(X_))]
    
    # DRIG weighted combination
    eSummands_x = torch.cat([
        (w_[e] * (X_X[e] - X_X[0])).unsqueeze(0) 
        for e in range(len(X_X))
    ], dim=0)
    
    eSummands_y = torch.cat([
        (w_[e] * (X_Y[e] - X_Y[0])).unsqueeze(0) 
        for e in range(len(X_Y))
    ], dim=0)
    
    eSum_x = torch.sum(eSummands_x, dim=0)
    eSum_y = torch.sum(eSummands_y, dim=0)
    
    M = X_X[0] + gamma * eSum_x
    v = X_Y[0] + gamma * eSum_y
    
    # Solve linear system
    try:
        return torch.inverse(M) @ v
    except torch.linalg.LinAlgError:
        return torch.linalg.pinv(M) @ v


def compare_latent_dimensions(X, Y, E, X_test, Y_test,
                              latent_dims_list=[2, 3, 5, 10],
                              seeds=[123, 456, 789],
                              epochs=1000,
                              alpha=0.1, beta=0, gamma=5,
                              verbose=True):
    """
    Compare CIRRL performance with different latent dimensions.
    
    Args:
        X, Y, E: Training data
        X_test, Y_test: Test data
        latent_dims_list: List of latent dimensions to try
        seeds: List of random seeds for multiple runs
        epochs: Number of training epochs
        alpha, beta, gamma: Training hyperparameters
        verbose: Whether to print progress
        
    Returns:
        List of result dictionaries for each latent dimension
    """
    from cirrl.models.dpa import DPA, OnlyRelu
    
    if verbose:
        print(f"Comparing latent dimensions: {latent_dims_list}")
        print(f"Using {len(seeds)} random seeds")
    
    all_results = []
    
    for latent_dim in latent_dims_list:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing latent dimension: {latent_dim}")
            print('='*60)
        
        latent_results = []
        
        for seed in seeds:
            if verbose:
                print(f"\n  Running with seed {seed}...")
            
            # Initialize model
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            SIGMA = OnlyRelu(epsilon=0.1)
            
            dpa = DPA(
                data_dim=X.shape[1],
                latent_dims=[latent_dim],
                num_layer=2,
                condition_dim=E.shape[1],
                lr=1e-4,
                hidden_dim=400,
                bn_enc=True,
                bn_dec=True,
                priorvar=SIGMA,
                resblock=True,
                totalvar=True,
                seed=seed
            )
            
            # Train model
            history = train_cirrl_model(
                dpa, X, Y, E, X_test, Y_test,
                alpha=alpha, beta=beta, gamma=gamma,
                epochs=epochs, verbose=False
            )
            
            latent_results.append({
                'seed': seed,
                'history': history,
                'model': dpa
            })
            
            if verbose and history['test_mse']:
                print(f"    Final test MSE: {history['test_mse'][-1]:.4f}")
        
        # Aggregate results
        test_mses = [r['history']['test_mse'][-1] 
                    for r in latent_results 
                    if r['history']['test_mse']]
        
        if verbose and test_mses:
            mean_mse = np.mean(test_mses)
            std_mse = np.std(test_mses)
            print(f"\n  Summary for latent dim {latent_dim}:")
            print(f"    Mean test MSE: {mean_mse:.4f} ± {std_mse:.4f}")
            print(f"    Min: {np.min(test_mses):.4f}, Max: {np.max(test_mses):.4f}")
        
        all_results.append({
            'latent_dim': latent_dim,
            'results': latent_results,
            'test_mses': test_mses
        })
    
    return all_results
