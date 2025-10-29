"""
Data loading and preprocessing utilities for CIRRL
"""

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional, Tuple


class MyDS(Dataset):
    """
    Custom Dataset for CIRRL training.
    
    Args:
        X: Feature tensor
        y: Target tensor (optional)
        e: Environment indicator tensor (optional)
        transform: Optional transform to apply
    """
    def __init__(self, X, y=None, e=None, transform=None):
        self.X = X
        self.y = y
        self.e = e
        self.transform = transform
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.e is not None and self.y is not None:
            return self.X[idx], self.y[idx], self.e[idx]
        elif self.e is not None:
            return self.X[idx], self.e[idx]
        elif self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def make_dataloader(x, y=None, batch_size=128, shuffle=True, num_workers=0):
    """
    Create a PyTorch DataLoader.
    
    Args:
        x: Feature tensor
        y: Target tensor (optional)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader object
    """
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    
    return dataloader


def load_singlecell_data(train_path: str, 
                        test_path: str,
                        separate_test_path: Optional[str] = None) -> Tuple:
    """
    Load single-cell datasets from pickle files.
    
    Args:
        train_path: Path to training data pickle file
        test_path: Path to test data pickle file
        separate_test_path: Path to separated test environments pickle file (optional)
        
    Returns:
        Tuple of (X, Y, E, X_test, Y_test[, test_environments])
        
    Expected data format:
        - Training data: numpy array of shape (n_samples, n_features + n_env_indicators + 1)
          where columns are [features(9), outcome(1), environment_indicators(11)]
        - Test data: numpy array of shape (n_samples, n_features + 1)
          where columns are [features(9), outcome(1)]
        - Separate test environments: list of numpy arrays
    """
    print(f"Loading training data from {train_path}...")
    with open(train_path, 'rb') as f:
        singlecell = pickle.load(f)
    
    # Parse training data
    # Columns: [0:9] = features, [9] = outcome, [10:] = environment indicators
    X = torch.from_numpy(singlecell[:, :9]).to(torch.float32)
    Y = torch.from_numpy(singlecell[:, 9]).to(torch.float32)
    E = torch.from_numpy(singlecell[:, 10:]).to(torch.float32)
    
    print(f"  Training samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Environments: {E.shape[1]}")
    
    print(f"\nLoading test data from {test_path}...")
    with open(test_path, 'rb') as f:
        singlecelltest = pickle.load(f)
    
    # Parse test data
    X_test = torch.from_numpy(singlecelltest[:, :-1]).squeeze(-1).to(torch.float32)
    Y_test = torch.from_numpy(singlecelltest[:, -1]).squeeze(-1).to(torch.float32)
    
    print(f"  Test samples: {X_test.shape[0]}")
    
    if separate_test_path is not None:
        print(f"\nLoading separate test environments from {separate_test_path}...")
        with open(separate_test_path, 'rb') as f:
            test_environments = pickle.load(f)
        
        # Convert to torch tensors
        test_environments = [
            torch.tensor(env, dtype=torch.float32) 
            for env in test_environments
        ]
        
        print(f"  Number of test environments: {len(test_environments)}")
        
        return X, Y, E, X_test, Y_test, test_environments
    
    return X, Y, E, X_test, Y_test


def standardize_data(X, mean=None, std=None, univar=False):
    """
    Standardize data to zero mean and unit variance.
    
    Args:
        X: Data tensor
        mean: Pre-computed mean (if None, compute from X)
        std: Pre-computed std (if None, compute from X)
        univar: Whether to use univariate scaling
        
    Returns:
        Tuple of (standardized_X, mean, std)
    """
    if mean is None:
        mean = torch.mean(X, dim=0)
    
    if std is None:
        if univar:
            std = X.std()
        else:
            std = torch.std(X, dim=0)
            std[std == 0] += 1e-5  # Avoid division by zero
    
    X_standardized = (X - mean) / std
    
    return X_standardized, mean, std


def unstandardize_data(X, mean, std):
    """
    Reverse standardization.
    
    Args:
        X: Standardized data tensor
        mean: Mean used for standardization
        std: Std used for standardization
        
    Returns:
        Original scale data tensor
    """
    return X * std + mean


def prepare_drig_data_from_latents(z, y, E):
    """
    Prepare latent representations for DRIG estimation.
    
    Args:
        z: Latent representations tensor (n_samples, latent_dim)
        y: Outcome tensor (n_samples,)
        E: Environment indicators tensor (n_samples, n_envs)
        
    Returns:
        List of numpy arrays, one per environment, with [z, y] concatenated
    """
    train_data = []
    n_envs = E.shape[1]
    
    for env_idx in range(n_envs):
        # Create mask for this environment
        env_vector = torch.eye(n_envs)[env_idx]
        mask = torch.all(E == env_vector, dim=1)
        
        # Extract data for this environment
        z_env = z[mask].numpy() if isinstance(z, torch.Tensor) else z[mask]
        y_env = y[mask].numpy() if isinstance(y, torch.Tensor) else y[mask]
        
        # Concatenate [z, y]
        z_y = np.concatenate([z_env, y_env.reshape(-1, 1)], axis=1)
        train_data.append(z_y)
    
    return train_data


def center_representations(z, y, E, ref_env_idx=0):
    """
    Center latent representations and outcomes based on reference environment.
    
    Args:
        z: Latent representations
        y: Outcomes
        E: Environment indicators
        ref_env_idx: Index of reference environment
        
    Returns:
        Tuple of (z_centered, y_centered, z_center, y_center)
    """
    # Find samples in reference environment
    ref_mask = E[:, ref_env_idx] == 1
    
    # Compute centers
    z_center = torch.mean(z[ref_mask], dim=0)
    y_center = torch.mean(y[ref_mask], dim=0)
    
    # Center data
    z_centered = z - z_center
    y_centered = y - y_center
    
    return z_centered, y_centered, z_center, y_center


def split_by_environment(X, y, E):
    """
    Split data by environment.
    
    Args:
        X: Features
        y: Outcomes
        E: Environment indicators (one-hot encoded)
        
    Returns:
        List of (X_env, y_env) tuples, one per environment
    """
    splits = []
    n_envs = E.shape[1]
    
    for env_idx in range(n_envs):
        mask = E[:, env_idx] == 1
        X_env = X[mask]
        y_env = y[mask]
        splits.append((X_env, y_env))
    
    return splits


def vectorize(x, multichannel=False):
    """
    Vectorize data of any shape.
    
    Args:
        x: Input tensor
        multichannel: Whether to keep multiple channels (in second dimension)
        
    Returns:
        Vectorized tensor of shape (n_samples, dim) or (n_samples, n_channels, dim)
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel:  # Single channel
            return x.reshape(x.shape[0], -1)
        else:  # Multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)


def get_environment_sizes(E):
    """
    Get the size of each environment.
    
    Args:
        E: Environment indicators (n_samples, n_envs)
        
    Returns:
        List of environment sizes
    """
    n_envs = E.shape[1]
    sizes = []
    
    for env_idx in range(n_envs):
        size = int(torch.sum(E[:, env_idx]))
        sizes.append(size)
    
    return sizes


def print_data_summary(X, Y, E, X_test=None, Y_test=None):
    """
    Print a summary of the dataset.
    
    Args:
        X: Training features
        Y: Training outcomes
        E: Environment indicators
        X_test: Test features (optional)
        Y_test: Test outcomes (optional)
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print("\nTraining Data:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Environments: {E.shape[1]}")
    
    env_sizes = get_environment_sizes(E)
    for i, size in enumerate(env_sizes):
        print(f"    Environment {i}: {size} samples ({size/X.shape[0]*100:.1f}%)")
    
    print(f"\n  Outcome statistics:")
    print(f"    Mean: {Y.mean():.4f}")
    print(f"    Std: {Y.std():.4f}")
    print(f"    Min: {Y.min():.4f}")
    print(f"    Max: {Y.max():.4f}")
    
    if X_test is not None and Y_test is not None:
        print("\nTest Data:")
        print(f"  Samples: {X_test.shape[0]}")
        print(f"  Features: {X_test.shape[1]}")
        
        print(f"\n  Outcome statistics:")
        print(f"    Mean: {Y_test.mean():.4f}")
        print(f"    Std: {Y_test.std():.4f}")
        print(f"    Min: {Y_test.min():.4f}")
        print(f"    Max: {Y_test.max():.4f}")
    
    print("="*60 + "\n")
