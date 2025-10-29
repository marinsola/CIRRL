"""
Single-Cell CIRRL Experiment

This script runs the complete CIRRL pipeline on single-cell data:
1. Load and preprocess data
2. Train DPA model to learn causal representations
3. Apply DRIG estimator on learned representations
4. Evaluate performance across multiple test environments
"""

import os
import pickle
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import CIRRL modules (adjust path as needed)
import sys
sys.path.append('..')

from cirrl.models.dpa import DPA, OnlyRelu
from cirrl.estimators.drig import est_drig_gd_analytical_init, test_mse_list
from cirrl.training.trainer import train_cirrl_model, eval_mse
from cirrl.utils.data import MyDS

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_singlecell_data(train_path, test_path, separate_test_path):
    """
    Load single-cell datasets.
    
    Args:
        train_path: Path to training data pickle file
        test_path: Path to pooled test data pickle file
        separate_test_path: Path to separated test environments pickle file
        
    Returns:
        Tuple of (X, Y, E, X_test, Y_test, test_environments)
    """
    print("Loading data...")
    
    # Load training data
    with open(train_path, 'rb') as f:
        singlecell = pickle.load(f)
    
    X = torch.from_numpy(singlecell[:, :9]).to(torch.float32)
    E = torch.from_numpy(singlecell[:, 10:]).to(torch.float32)
    Y = torch.from_numpy(singlecell[:, 9]).to(torch.float32)
    
    # Load test data
    with open(test_path, 'rb') as f:
        singlecelltest = pickle.load(f)
    
    X_test = torch.from_numpy(singlecelltest[:, :-1]).squeeze(-1).to(torch.float32)
    Y_test = torch.from_numpy(singlecelltest[:, -1]).squeeze(-1).to(torch.float32)
    
    # Load separate test environments
    with open(separate_test_path, 'rb') as f:
        test_environments = pickle.load(f)
    test_environments = [torch.tensor(env, dtype=torch.float32) for env in test_environments]
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Number of environments: {E.shape[1]}")
    print(f"Number of test environments: {len(test_environments)}")
    
    return X, Y, E, X_test, Y_test, test_environments


def train_dpa_model(X, Y, E, X_test, Y_test, config):
    """
    Train DPA model for representation learning.
    
    Args:
        X, Y, E: Training data
        X_test, Y_test: Test data
        config: Dictionary with model configuration
        
    Returns:
        Trained DPA model
    """
    print("\n" + "="*60)
    print("Training DPA Model")
    print("="*60)
    
    # Initialize model
    SIGMA = OnlyRelu(epsilon=config.get('epsilon', 0.1))
    
    dpa = DPA(
        data_dim=X.shape[1],
        latent_dims=config['latent_dims'],
        num_layer=config.get('num_layer', 2),
        condition_dim=E.shape[1],
        lr=config.get('lr', 1e-4),
        hidden_dim=config.get('hidden_dim', 400),
        bn_enc=config.get('bn_enc', True),
        bn_dec=config.get('bn_dec', True),
        ln_enc=config.get('ln_enc', False),
        ln_dec=config.get('ln_dec', False),
        priorvar=SIGMA,
        resblock=config.get('resblock', True),
        noise_dim=config.get('noise_dim', 100),
        totalvar=config.get('totalvar', True),
        seed=config.get('seed', 123),
        l2=config.get('l2', 0.0)
    )
    
    # Train model
    history = train_cirrl_model(
        dpa, X, Y, E, X_test, Y_test,
        alpha=config.get('alpha', 0.1),
        beta=config.get('beta', 0),
        gamma=config.get('gamma', 5),
        epochs=config.get('epochs', 1000),
        batch_size=len(X),
        print_every=config.get('print_every', 100),
        verbose=config.get('verbose', True)
    )
    
    return dpa, history


def extract_latent_representations(dpa, X, Y, E, X_test, Y_test):
    """
    Extract and center latent representations from trained DPA model.
    
    Args:
        dpa: Trained DPA model
        X, Y, E: Training data
        X_test, Y_test: Test data
        
    Returns:
        Tuple of (z_centered, y_centered, z_test_centered, y_test_centered)
    """
    print("\n" + "="*60)
    print("Extracting Latent Representations")
    print("="*60)
    
    dpa.model.eval()
    
    # Extract latent representations
    with torch.no_grad():
        _, _, z = dpa.model(x=X.to(dpa.device), k=dpa.latent_dims[0], 
                           c=E.to(dpa.device), return_latent=True, double=True)
        _, _, z_test = dpa.model(x=X_test.to(dpa.device), k=dpa.latent_dims[0], 
                                c=E.to(dpa.device), return_latent=True, double=True)
    
    # Find reference environment (first environment)
    n_in_ref = int(torch.sum(E[:, 0]))
    
    # Center representations
    center_of_z_ref = torch.mean(z[:n_in_ref], dim=0)
    z_centered = z.to('cpu').detach() - center_of_z_ref.to('cpu').detach()
    z_test_centered = z_test.to('cpu').detach() - center_of_z_ref.to('cpu').detach()
    
    # Center outcomes
    center_of_y_ref = torch.mean(Y[:n_in_ref], dim=0)
    y_centered = Y.to('cpu').detach() - center_of_y_ref.to('cpu').detach()
    y_test_centered = Y_test.to('cpu').detach() - center_of_y_ref.to('cpu').detach()
    
    print(f"Latent dimension: {z_centered.shape[1]}")
    print(f"Reference environment size: {n_in_ref}")
    
    return z_centered, y_centered, z_test_centered, y_test_centered


def prepare_drig_data(z_centered, y_centered, E):
    """
    Prepare data in format required by DRIG estimator.
    
    Args:
        z_centered: Centered latent representations
        y_centered: Centered outcomes
        E: Environment indicators
        
    Returns:
        List of numpy arrays, one per environment
    """
    train_data = []
    
    for env_idx in range(E.shape[1]):
        # Select samples from this environment
        mask = torch.all(E == torch.eye(E.shape[1])[env_idx], dim=1)
        z_env = z_centered[mask].numpy()
        y_env = y_centered[mask].numpy()
        
        # Concatenate [z, y]
        z_y = np.concatenate([z_env, y_env.reshape(-1, 1)], axis=1)
        train_data.append(z_y)
    
    print(f"Prepared {len(train_data)} training environments")
    for i, data in enumerate(train_data):
        print(f"  Environment {i}: {data.shape[0]} samples")
    
    return train_data


def run_drig_analysis(train_data, z_test_centered, y_test_centered, 
                     gamma_values, use_analytical_init=True):
    """
    Run DRIG estimator with different gamma values.
    
    Args:
        train_data: List of training data arrays per environment
        z_test_centered: Test latent representations
        y_test_centered: Test outcomes
        gamma_values: Array of gamma values to try
        use_analytical_init: Whether to use analytical initialization
        
    Returns:
        DataFrame with results
    """
    print("\n" + "="*60)
    print("Running DRIG Analysis")
    print("="*60)
    
    # Prepare test data (one sample per row)
    test_data = []
    for idx in range(len(z_test_centered)):
        z_y = np.expand_dims(
            np.hstack([z_test_centered[idx].numpy(), y_test_centered[idx].numpy()]), 
            axis=0
        )
        test_data.append(z_y)
    
    num_test_envs = len(test_data)
    results = []
    
    # Try different gamma values
    for gamma in tqdm(gamma_values, desc="Testing gamma values"):
        # Estimate DRIG coefficients
        if use_analytical_init:
            coef = est_drig_gd_analytical_init(train_data, gamma=gamma, y_idx=-1)
        else:
            from cirrl.estimators.drig import est_drig_gd_fast
            coef = est_drig_gd_fast(train_data, gamma=gamma, y_idx=-1, iters=3000, device='cpu')
        
        # Compute test MSEs
        mses = test_mse_list(test_data, coef, y_idx=-1)
        
        # Store results
        for mse in mses:
            results.append({
                'method': 'CIRRL',
                'gamma': gamma,
                'test_mse': mse
            })
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nResults Summary:")
    summary = results_df.groupby('gamma').agg({
        'test_mse': ['mean', 'median', 'std', 'min', 'max']
    }).round(4)
    print(summary)
    
    return results_df


def plot_results(results_df, save_path=None):
    """
    Create visualizations of DRIG results.
    
    Args:
        results_df: DataFrame with results
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Mean MSE vs Gamma
    summary = results_df.groupby('gamma').agg({
        'test_mse': ['mean', 'std']
    })
    summary.columns = ['mean', 'std']
    
    axes[0, 0].errorbar(summary.index, summary['mean'], yerr=summary['std'],
                       marker='o', capsize=5, capthick=2, linewidth=2)
    axes[0, 0].set_xlabel('Gamma', fontsize=12)
    axes[0, 0].set_ylabel('Test MSE', fontsize=12)
    axes[0, 0].set_title('Mean Test MSE vs Gamma', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Median MSE vs Gamma
    median_mse = results_df.groupby('gamma')['test_mse'].median()
    axes[0, 1].plot(median_mse.index, median_mse.values, marker='s', 
                   linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Gamma', fontsize=12)
    axes[0, 1].set_ylabel('Median Test MSE', fontsize=12)
    axes[0, 1].set_title('Median Test MSE vs Gamma', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution of MSE for different gammas (violin plot)
    gamma_subset = results_df['gamma'].unique()[::len(results_df['gamma'].unique())//5]
    data_subset = results_df[results_df['gamma'].isin(gamma_subset)]
    
    sns.violinplot(data=data_subset, x='gamma', y='test_mse', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Gamma', fontsize=12)
    axes[1, 0].set_ylabel('Test MSE', fontsize=12)
    axes[1, 0].set_title('MSE Distribution Across Test Environments', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Heatmap of MSE across gammas
    pivot_data = results_df.pivot_table(
        values='test_mse', 
        index=results_df.index % (len(results_df) // len(results_df['gamma'].unique())),
        columns='gamma',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_data.iloc[:20], cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'MSE'})
    axes[1, 1].set_xlabel('Gamma', fontsize=12)
    axes[1, 1].set_ylabel('Test Sample', fontsize=12)
    axes[1, 1].set_title('MSE Heatmap (First 20 Samples)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    """
    Main execution function.
    """
    # Configuration
    config = {
        'latent_dims': [3],
        'hidden_dim': 400,
        'num_layer': 2,
        'lr': 1e-4,
        'bn_enc': True,
        'bn_dec': True,
        'epochs': 1000,
        'alpha': 0.1,
        'beta': 0,
        'gamma': 5,
        'seed': 123,
        'print_every': 100,
        'verbose': True
    }
    
    # Data paths (adjust as needed)
    train_path = 'data/singlecell.pkl'
    test_path = 'data/singlecelltest.pkl'
    separate_test_path = 'data/testenvs_separate.pkl'
    
    # Step 1: Load data
    X, Y, E, X_test, Y_test, test_environments = load_singlecell_data(
        train_path, test_path, separate_test_path
    )
    
    # Step 2: Train DPA model
    dpa, history = train_dpa_model(X, Y, E, X_test, Y_test, config)
    
    # Step 3: Extract latent representations
    z_centered, y_centered, z_test_centered, y_test_centered = extract_latent_representations(
        dpa, X, Y, E, X_test, Y_test
    )
    
    # Step 4: Prepare data for DRIG
    train_data = prepare_drig_data(z_centered, y_centered, E)
    
    # Step 5: Run DRIG analysis with different gamma values
    gamma_values = np.linspace(0, 15, 16)
    results_df = run_drig_analysis(
        train_data, z_test_centered, y_test_centered, 
        gamma_values, use_analytical_init=True
    )
    
    # Step 6: Visualize results
    plot_results(results_df, save_path='results/singlecell_results.png')
    
    # Step 7: Save results
    results_df.to_csv('results/singlecell_results.csv', index=False)
    print("\nResults saved to results/singlecell_results.csv")
    
    # Find best gamma
    best_gamma = results_df.groupby('gamma')['test_mse'].median().idxmin()
    best_mse = results_df.groupby('gamma')['test_mse'].median().min()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best gamma: {best_gamma}")
    print(f"Best median MSE: {best_mse:.4f}")
    print("="*60)
    
    return dpa, results_df, history


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run experiment
    dpa, results, history = main()
