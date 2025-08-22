#!/usr/bin/env python3
"""
Training Functions for Experiment 2b: Aggregator Validation

Provides consistent training functions that match the baseline methodology
for both GAM and MLP models across different ground truth targets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pipeline.core.aggregator_training import GAMAggregator, MLPTrainer, compute_metrics, load_training_config, determine_training_scale


def calculate_variance_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Calculate variance and correlation statistics for ground truth analysis.
    
    Args:
        X: Judge scores matrix (n_samples, n_judges)
        y: Ground truth scores
        
    Returns:
        Dictionary with variance statistics
    """
    # Mean judge score for each sample
    judge_mean = np.mean(X, axis=1)
    
    # Calculate correlation between mean judge score and ground truth
    correlation = np.corrcoef(judge_mean, y)[0, 1] if len(np.unique(y)) > 1 else 0.0
    
    # Calculate residual standard deviation from linear fit
    if len(np.unique(judge_mean)) > 1:
        # Linear regression residuals
        coeffs = np.polyfit(judge_mean, y, 1)
        y_pred_linear = np.polyval(coeffs, judge_mean)
        residuals = y - y_pred_linear
        std_from_fit = np.std(residuals)
    else:
        std_from_fit = np.std(y)
    
    return {
        'variance': float(np.var(y)),
        'std': float(np.std(y)),
        'correlation_with_judge_mean': float(correlation),
        'std_from_linear_fit': float(std_from_fit),
        'range': float(np.max(y) - np.min(y)),
        'mean': float(np.mean(y))
    }


def train_gam_baseline_config(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_seed: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Train GAM model using baseline configuration.
    
    Args:
        X: Judge scores matrix
        y: Ground truth scores
        test_size: Test set fraction
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with training results
    """
    print(f"🧠 Training GAM model (normalize={normalize})...")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    # Optional normalization
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train GAM with baseline configuration
    # Use parameters similar to baseline GAM tuning
    try:
        from pygam import LinearGAM, s
        
        # Create GAM with splines for each feature (similar to GAMAggregator but fixed)
        n_features = X_train.shape[1]
        terms = s(0, n_splines=10, lam=0.6)
        for i in range(1, n_features):
            terms = terms + s(i, n_splines=10, lam=0.6)
        gam_model = LinearGAM(terms)
        gam_model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = gam_model.predict(X_train)
        y_test_pred = gam_model.predict(X_test)
        
        # Calculate metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)
        
        # Calculate variance statistics
        variance_stats = calculate_variance_stats(X, y)
        
        print(f"✅ GAM training complete - R²: {test_metrics['r2']:.4f}")
        
        return {
            'model': gam_model,
            'scaler': scaler,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'variance_stats': variance_stats,
            'config': {
                'n_splines': 10,
                'lam': 0.6,
                'normalize': normalize,
                'test_size': test_size,
                'random_seed': random_seed
            }
        }
        
    except Exception as e:
        print(f"❌ GAM training failed: {e}")
        return {
            'error': str(e),
            'variance_stats': calculate_variance_stats(X, y)
        }


def train_mlp_baseline_config(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_seed: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Train MLP model using baseline configuration.
    
    Args:
        X: Judge scores matrix
        y: Ground truth scores
        test_size: Test set fraction
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with training results
    """
    print(f"🤖 Training MLP model (normalize={normalize})...")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    # Optional normalization
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    try:
        # Load training config and determine scale
        training_config = load_training_config()
        scale = determine_training_scale(len(X_train))
        mlp_config = training_config["mlp_training"].get(scale, training_config["mlp_training"]["medium_scale"])
        
        # Train MLP with baseline configuration
        mlp_trainer = MLPTrainer(
            hidden_dim=mlp_config["hidden_dim"],
            learning_rate=mlp_config["learning_rate"],
            batch_size=min(mlp_config["batch_size"], max(2, len(X_train) // 2)),
            n_epochs=mlp_config["n_epochs"],
            early_stopping_patience=mlp_config.get("early_stopping_patience", 15)
        )
        
        # Train with early stopping using validation set
        # Split training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_seed
        )
        
        train_losses, val_losses = mlp_trainer.fit(X_tr, y_tr, X_val, y_val)
        
        # Make predictions
        y_train_pred = mlp_trainer.predict(X_train)
        y_test_pred = mlp_trainer.predict(X_test)
        
        # Calculate metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)
        
        # Calculate variance statistics
        variance_stats = calculate_variance_stats(X, y)
        
        print(f"✅ MLP training complete - R²: {test_metrics['r2']:.4f}")
        
        return {
            'model': mlp_trainer,
            'scaler': scaler,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'variance_stats': variance_stats,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                **mlp_config,
                'scale': scale,
                'normalize': normalize,
                'test_size': test_size,
                'random_seed': random_seed
            }
        }
        
    except Exception as e:
        print(f"❌ MLP training failed: {e}")
        return {
            'error': str(e),
            'variance_stats': calculate_variance_stats(X, y)
        }


def train_both_models(
    X: np.ndarray, 
    y: np.ndarray, 
    target_name: str,
    test_size: float = 0.2, 
    random_seed: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Train both GAM and MLP models for a specific target.
    
    Args:
        X: Judge scores matrix
        y: Ground truth scores
        target_name: Name of the target (for logging)
        test_size: Test set fraction
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with results from both models
    """
    print(f"\n🎯 Training models for {target_name} target...")
    print(f"   Data: {len(X)} samples, {X.shape[1]} judges")
    print(f"   Target stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}, var={np.var(y):.3f}")
    
    # Ensure we have sufficient aligned data
    min_samples = min(len(X), len(y))
    if min_samples < len(X) or min_samples < len(y):
        X = X[:min_samples]
        y = y[:min_samples]
        print(f"   Aligned to {min_samples} samples")
    
    # Train GAM
    gam_results = train_gam_baseline_config(X, y, test_size, random_seed, normalize)
    
    # Train MLP
    mlp_results = train_mlp_baseline_config(X, y, test_size, random_seed, normalize)
    
    # Combine results
    results = {
        'target_name': target_name,
        'data_stats': {
            'n_samples': len(X),
            'n_judges': X.shape[1],
            'target_mean': float(np.mean(y)),
            'target_std': float(np.std(y)),
            'target_variance': float(np.var(y)),
            'target_range': float(np.max(y) - np.min(y))
        },
        'gam': gam_results,
        'mlp': mlp_results
    }
    
    # Summary comparison
    if 'test_metrics' in gam_results and 'test_metrics' in mlp_results:
        gam_r2 = gam_results['test_metrics']['r2']
        mlp_r2 = mlp_results['test_metrics']['r2']
        print(f"\n📊 {target_name} Results Summary:")
        print(f"   GAM R²: {gam_r2:.4f}")
        print(f"   MLP R²: {mlp_r2:.4f}")
        print(f"   Best: {'GAM' if gam_r2 > mlp_r2 else 'MLP'} ({max(gam_r2, mlp_r2):.4f})")
        
        results['summary'] = {
            'best_model': 'gam' if gam_r2 > mlp_r2 else 'mlp',
            'best_r2': float(max(gam_r2, mlp_r2)),
            'gam_r2': float(gam_r2),
            'mlp_r2': float(mlp_r2),
            'difference': float(abs(gam_r2 - mlp_r2))
        }
    
    return results


def validate_results_consistency(results: Dict[str, Any]) -> bool:
    """
    Validate that training results are consistent and reasonable.
    
    Args:
        results: Training results dictionary
        
    Returns:
        True if results pass validation
    """
    checks_passed = []
    
    # Check R² values are reasonable
    for model_type in ['gam', 'mlp']:
        if model_type in results and 'test_metrics' in results[model_type]:
            r2 = results[model_type]['test_metrics']['r2']
            checks_passed.append(-1.0 <= r2 <= 1.0)  # R² should be reasonable
    
    # Check variance statistics are calculated
    for model_type in ['gam', 'mlp']:
        if model_type in results and 'variance_stats' in results[model_type]:
            variance_stats = results[model_type]['variance_stats']
            required_keys = ['variance', 'correlation_with_judge_mean', 'std_from_linear_fit']
            checks_passed.append(all(key in variance_stats for key in required_keys))
    
    # Check data alignment
    if 'data_stats' in results:
        checks_passed.append(results['data_stats']['n_samples'] > 0)
        checks_passed.append(results['data_stats']['n_judges'] == 10)  # Should have 10 judges
    
    all_checks_passed = all(checks_passed)
    
    if not all_checks_passed:
        print(f"⚠️  Validation warning: Some consistency checks failed")
        print(f"   Checks passed: {sum(checks_passed)}/{len(checks_passed)}")
    
    return all_checks_passed


if __name__ == "__main__":
    # Test the training functions
    print("🧪 Testing training functions...")
    
    # Create dummy data for testing
    np.random.seed(42)
    X_test = np.random.rand(100, 10) * 4  # Judge scores 0-4
    y_test = np.random.rand(100) * 10     # Human scores 0-10
    
    print("🎯 Testing with dummy data...")
    results = train_both_models(X_test, y_test, "test_target")
    
    # Validate results
    is_valid = validate_results_consistency(results)
    
    print(f"\n✅ Training functions test {'passed' if is_valid else 'failed'}!")