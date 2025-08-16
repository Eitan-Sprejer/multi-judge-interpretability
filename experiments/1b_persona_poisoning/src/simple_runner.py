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

# Import judge rubrics to get correct judge count and IDs
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS

# Judge IDs
JUDGE_IDS = list(JUDGE_RUBRICS.keys())

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

# Constants based on current judge configuration
N_JUDGES = len(JUDGE_IDS)  # Should be 10

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
        # Handle different possible column names for human feedback
        if 'human_feedback_score' in df_contaminated.columns:
            original_score = float(df_contaminated.iloc[idx]['human_feedback_score'])
        elif 'score' in df_contaminated.columns:
            original_score = float(df_contaminated.iloc[idx]['score'])
        else:
            # Try to extract from human_feedback if it's a dict
            hf = df_contaminated.iloc[idx].get('human_feedback', {})
            if isinstance(hf, dict) and 'score' in hf:
                original_score = float(hf['score'])
            else:
                logger.warning(f"Could not find human feedback score for row {idx}, using default 5.0")
                original_score = 5.0
        
        judge_scores = list(df_contaminated.iloc[idx]['judge_scores'])  # Ensure it's a list
        
        # Generate troll rating
        troll_rating = troll.generate_rating(original_score, judge_scores)
        
        # Update the dataframe - only modify the score, not the judge scores
        if 'human_feedback_score' in df_contaminated.columns:
            df_contaminated.at[idx, 'human_feedback_score'] = float(troll_rating)
        elif 'score' in df_contaminated.columns:
            df_contaminated.at[idx, 'score'] = float(troll_rating)
        else:
            # Update within human_feedback dict
            hf = df_contaminated.iloc[idx].get('human_feedback', {})
            if isinstance(hf, dict):
                hf['score'] = float(troll_rating)
                df_contaminated.at[idx, 'human_feedback'] = hf
        
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
    
    # Ensure we have the required columns - handle various naming conventions
    if 'judge_scores' not in df.columns:
        if 'scores' in df.columns:
            df['judge_scores'] = df['scores']
        else:
            raise ValueError("Dataset must have 'judge_scores' or 'scores' column")
    
    # Handle different human feedback column formats
    if 'human_feedback_score' not in df.columns:
        if 'score' in df.columns:
            df['human_feedback_score'] = df['score']
        elif 'human_feedback' in df.columns:
            # Extract score from human_feedback dict/object
            def extract_score(hf):
                if isinstance(hf, dict) and 'score' in hf:
                    return hf['score']
                elif isinstance(hf, dict) and 'average_score' in hf:
                    return hf['average_score']
                else:
                    return 5.0  # Default fallback
            df['human_feedback_score'] = df['human_feedback'].apply(extract_score)
        else:
            raise ValueError("Dataset must have 'human_feedback_score', 'score', or 'human_feedback' column")
    
    # Remove rows with missing values
    df = df.dropna(subset=['human_feedback_score'])
    
    # Filter out rows where judge_scores don't have the right length
    expected_judge_count = N_JUDGES
    df = df[df['judge_scores'].apply(lambda x: len(x) == expected_judge_count if isinstance(x, list) else False)]
    logger.info(f"Loaded {len(df)} samples (after removing missing values and invalid scores)")
    logger.info(f"Expected {expected_judge_count} judge scores per sample")
    
    # Prepare clean test set (20% of data)
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[:test_size].copy().reset_index(drop=True)
    df_train_base = df.iloc[test_size:].copy().reset_index(drop=True)
    
    # Prepare test data
    X_test = np.array(df_test['judge_scores'].tolist())
    y_test = np.array(df_test['human_feedback_score'].values, dtype=np.float32)
    
    results = {}
    
    for rate in rates:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing contamination rate: {rate*100:.0f}%")
        logger.info(f"{'='*50}")
        
        # Contaminate training data
        df_train = contaminate_dataset_simple(df_train_base, rate, strategy)
        
        # Debug check
        logger.info(f"Checking df_train after contamination...")
        bad_rows = []
        for i, scores in enumerate(df_train['judge_scores']):
            if not isinstance(scores, list) or len(scores) != 10:
                bad_rows.append(i)
        if bad_rows:
            logger.warning(f"Found {len(bad_rows)} bad rows: {bad_rows[:5]}")
            # Filter out bad rows
            df_train = df_train[df_train['judge_scores'].apply(lambda x: isinstance(x, list) and len(x) == 10)]
            logger.info(f"Filtered to {len(df_train)} good rows")
        
        # Prepare training data
        X_train = np.array(df_train['judge_scores'].tolist())
        y_train = np.array(df_train['human_feedback_score'].values, dtype=np.float32)
        
        # Train model
        model, metrics = train_simple_model(X_train, y_train, X_test, y_test)
        
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