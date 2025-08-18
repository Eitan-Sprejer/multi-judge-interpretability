#!/usr/bin/env python3
"""
Simplified Persona Poisoning Experiment Runner
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from typing import Dict, List
import logging
from datetime import datetime
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simple MLP model (copied to avoid import issues)
class SingleLayerMLP(nn.Module):
    def __init__(self, n_judges: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(n_judges, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

# Import troll generator directly from same directory
sys.path.append(str(Path(__file__).parent))
from troll_generator import TrollPersona

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def contaminate_arrays_simple(X: np.ndarray, y: np.ndarray, rate: float, strategy: str = "inverse") -> tuple:
    """Simple contamination of arrays (same logic as before but with arrays)."""
    if rate == 0:
        return X.copy(), y.copy()
    
    n_contaminate = int(len(X) * rate)
    troll = TrollPersona(strategy)
    
    # Deep copy to avoid modifying originals
    X_contaminated = X.copy()
    y_contaminated = y.copy()
    
    # Randomly select samples to contaminate (same seed for reproducibility)
    np.random.seed(42)
    contaminate_indices = np.random.choice(len(X), n_contaminate, replace=False)
    
    for idx in contaminate_indices:
        original_score = float(y_contaminated[idx])
        judge_scores = list(X_contaminated[idx])  # Convert to list for troll
        
        # Generate troll rating
        troll_rating = troll.generate_rating(original_score, judge_scores)
        
        # Update only the human score, not judge scores
        y_contaminated[idx] = float(troll_rating)
    
    logger.info(f"Contaminated {n_contaminate}/{len(X)} samples ({rate*100:.0f}%)")
    return X_contaminated, y_contaminated


def contaminate_dataset_simple(df: pd.DataFrame, rate: float, strategy: str = "inverse") -> pd.DataFrame:
    """Simple contamination of dataset."""
    # Deep copy dataframe to avoid issues
    df_contaminated = df.copy(deep=True)
    df_contaminated['is_contaminated'] = False
    
    if rate == 0:
        return df_contaminated
    
    n_contaminate = int(len(df) * rate)
    troll = TrollPersona(strategy)
    
    # Randomly select samples to contaminate
    np.random.seed(42)
    contaminate_indices = np.random.choice(len(df), n_contaminate, replace=False)
    
    for idx in contaminate_indices:
        original_score = float(df_contaminated.iloc[idx]['human_feedback_score'])
        judge_scores = list(df_contaminated.iloc[idx]['judge_scores'])  # Ensure it's a list
        
        # Generate troll rating
        troll_rating = troll.generate_rating(original_score, judge_scores)
        
        # Update the dataframe - only modify the score, not the judge scores
        df_contaminated.at[idx, 'human_feedback_score'] = float(troll_rating)
        df_contaminated.at[idx, 'is_contaminated'] = True
    
    logger.info(f"Contaminated {n_contaminate}/{len(df)} samples ({rate*100:.0f}%)")
    return df_contaminated


def train_simple_model(X_train, y_train, X_test, y_test, epochs=100):
    """Train a simple MLP model."""
    n_features = X_train.shape[1]
    model = SingleLayerMLP(n_judges=n_features, hidden_dim=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate works better
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t).squeeze()
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_t).squeeze()
                test_loss = criterion(test_pred, y_test_t)
            logger.info(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().numpy()
    
    metrics = compute_metrics(y_test, y_pred)
    return model, metrics


def run_contamination_experiment(data_path: str, rates: List[float], strategy: str = "inverse"):
    """Run the full contamination experiment."""
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    # Ensure we have the required columns - handle both 'scores' and 'judge_scores'
    if 'judge_scores' not in df.columns:
        if 'scores' in df.columns:
            df['judge_scores'] = df['scores']
        else:
            raise ValueError("Dataset must have 'judge_scores' or 'scores' column")
    
    # Handle human feedback score extraction
    if 'human_feedback_score' not in df.columns:
        if 'human_feedback' in df.columns:
            # Extract score from human_feedback dict
            df['human_feedback_score'] = df['human_feedback'].apply(
                lambda x: x.get('score', x.get('average_score', 5.0)) if isinstance(x, dict) else 5.0
            )
            logger.info("Extracted human_feedback_score from human_feedback column")
        else:
            raise ValueError("Dataset must have 'human_feedback_score' or 'human_feedback' column")
    
    # Remove rows with missing values
    df = df.dropna(subset=['human_feedback_score'])
    
    # Filter out rows where judge_scores don't have the right length
    df = df[df['judge_scores'].apply(lambda x: len(x) == 10 if isinstance(x, list) else False)]
    logger.info(f"Loaded {len(df)} samples (after removing missing values and invalid scores)")
    
    # Prepare data arrays (same as hyperparameter tuning)
    X = np.array(df['judge_scores'].tolist())
    y = np.array(df['human_feedback_score'].values, dtype=np.float32)
    
    # Use RANDOM split with same seed as hyperparameter tuning (42)
    X_clean, X_test_raw, y_clean, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Apply normalization (same as hyperparameter tuning)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test_raw.reshape(-1, 10)).astype(np.float32)
    y_test = y_test_raw.astype(np.float32)
    
    logger.info(f"Using RANDOM split and NORMALIZATION (same as hyperparameter tuning)")
    logger.info(f"Train size: {len(X_clean)}, Test size: {len(X_test)}")
    
    results = {}
    
    for rate in rates:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing contamination rate: {rate*100:.0f}%")
        logger.info(f"{'='*50}")
        
        # Create contaminated training data from clean arrays
        X_train_raw, y_train_raw = contaminate_arrays_simple(X_clean.copy(), y_clean.copy(), rate, strategy)
        
        # Apply same normalization to training data (fit on contaminated training data)
        scaler_train = StandardScaler()
        X_train = scaler_train.fit_transform(X_train_raw).astype(np.float32)
        y_train = y_train_raw.astype(np.float32)
        
        # Re-normalize test data with training scaler for fair comparison
        X_test_normalized = scaler_train.transform(X_test_raw).astype(np.float32)
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test_normalized.shape}")
        
        # Train model
        model, metrics = train_simple_model(X_train, y_train, X_test_normalized, y_test)
        
        logger.info(f"Results for {rate*100:.0f}% contamination:")
        logger.info(f"  R² Score: {metrics['r2']:.3f}")
        logger.info(f"  MSE: {metrics['mse']:.3f}")
        logger.info(f"  MAE: {metrics['mae']:.3f}")
        
        results[rate] = metrics
    
    # Analyze degradation
    logger.info(f"\n{'='*50}")
    logger.info("PERFORMANCE DEGRADATION ANALYSIS")
    logger.info(f"{'='*50}")
    
    baseline_r2 = results[0.0]['r2'] if 0.0 in results else max(r['r2'] for r in results.values())
    
    for rate in sorted(results.keys()):
        degradation = baseline_r2 - results[rate]['r2']
        logger.info(f"{rate*100:3.0f}% contamination: R²={results[rate]['r2']:.3f} (degradation: {degradation:+.3f})")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple Persona Poisoning Test")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.25],
                       help="Contamination rates to test")
    parser.add_argument("--strategy", type=str, default="inverse",
                       choices=["inverse", "random", "extreme", "safety_inverse"])
    
    args = parser.parse_args()
    
    results = run_contamination_experiment(args.data, args.rates, args.strategy)
    
    # Save results
    output_file = f"contamination_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
