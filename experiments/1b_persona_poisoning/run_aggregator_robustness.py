#!/usr/bin/env python3
"""
Aggregator Robustness Study

Studies how robust the learned aggregator method is to realistic 
human feedback contamination scenarios, without comparing to other architectures.

Research Question: At what contamination rate does the aggregator break down, 
and which real-world issues are most problematic?
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import contamination function
from pipeline.core.baseline_models import contaminate_human_feedback, BaselineEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple MLP aggregator model
class MLPAggregator(nn.Module):
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

def train_aggregator(X_train, y_train, X_test, y_test, epochs=100):
    """Train MLP aggregator model."""
    n_features = X_train.shape[1]
    model = MLPAggregator(n_judges=n_features, hidden_dim=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().numpy()
    
    return compute_metrics(y_test, y_pred)

def run_robustness_study(data_path: str, 
                        contamination_rates: List[float],
                        strategies: List[str] = ['systematic_bias', 'random_noise', 'scaled_down']) -> Dict:
    """Run aggregator robustness study across contamination scenarios."""
    
    logger.info(f"🔬 AGGREGATOR ROBUSTNESS STUDY")
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    # Initialize evaluator for data preparation
    evaluator = BaselineEvaluator(random_seed=42, test_size=0.2)
    
    # Prepare data with uniform persona sampling
    X, y = evaluator.prepare_data_uniform_sampling(df)
    logger.info(f"Prepared {len(X)} samples with uniform persona sampling")
    logger.info(f"Target distribution: mean={y.mean():.3f}, std={y.std():.3f}")
    
    # Split data consistently
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply normalization (fit on training data only - NO DATA LEAKAGE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\\n{'='*60}")
        logger.info(f"TESTING CONTAMINATION: {strategy.upper().replace('_', ' ')}")
        logger.info(f"{'='*60}")
        
        strategy_results = {}
        
        for rate in contamination_rates:
            logger.info(f"\\n--- Contamination Rate: {rate*100:.0f}% ---")
            
            # Contaminate human feedback
            y_train_contaminated = contaminate_human_feedback(
                y_train, rate, strategy, seed=42
            )
            
            logger.info(f"Original train mean: {y_train.mean():.3f}")
            logger.info(f"Contaminated train mean: {y_train_contaminated.mean():.3f}")
            
            # Train aggregator on contaminated data
            metrics = train_aggregator(
                X_train_scaled, y_train_contaminated, X_test_scaled, y_test
            )
            
            logger.info(f"Aggregator R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.3f}")
            
            strategy_results[str(rate)] = {
                'metrics': metrics,
                'contamination_info': {
                    'rate': rate,
                    'strategy': strategy,
                    'original_mean': float(y_train.mean()),
                    'contaminated_mean': float(y_train_contaminated.mean()),
                    'mean_shift': float(y_train_contaminated.mean() - y_train.mean())
                }
            }
        
        results[strategy] = strategy_results
    
    return results

def analyze_and_visualize_results(results: Dict, save_path: str = None):
    """Analyze results and create focused visualization."""
    
    print("\\n" + "="*80)
    print("📊 AGGREGATOR ROBUSTNESS ANALYSIS")
    print("="*80)
    
    # Create single clean plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    strategies = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    markers = ['o', 's', '^']
    
    for i, strategy in enumerate(strategies):
        strategy_results = results[strategy]
        rates = sorted([float(k) for k in strategy_results.keys()])
        r2_scores = [strategy_results[str(r)]['metrics']['r2'] for r in rates]
        
        # Convert to percentages for plotting
        rates_pct = [r * 100 for r in rates]
        
        # Plot with nice formatting
        ax.plot(rates_pct, r2_scores, 'o-', 
               color=colors[i], linewidth=2.5, markersize=7,
               label=f'{strategy.replace("_", " ").title()}',
               marker=markers[i])
        
        # Print summary stats
        clean_r2 = strategy_results[str(0.0)]['metrics']['r2']
        max_contam_r2 = strategy_results[str(max(rates))]['metrics']['r2']
        degradation = (clean_r2 - max_contam_r2) / clean_r2 * 100
        
        print(f"\\n{strategy.upper().replace('_', ' ')}:")
        print(f"  Clean performance (0%):     R² = {clean_r2:.3f}")
        print(f"  High contamination (50%):   R² = {max_contam_r2:.3f}")  
        print(f"  Performance degradation:    {degradation:.1f}%")
        
        # Find breaking point (R² < 0.3)
        breaking_point = None
        for rate in rates:
            r2 = strategy_results[str(rate)]['metrics']['r2']
            if r2 < 0.3:
                breaking_point = rate * 100
                break
        
        if breaking_point:
            print(f"  Breaking point (R² < 0.3):  {breaking_point:.0f}% contamination")
        else:
            print(f"  Breaking point (R² < 0.3):  Not reached (≤50%)")
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=1, 
               label='Acceptable Threshold (R² = 0.3)')
    
    ax.set_xlabel('Contamination Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Aggregator Robustness to Human Feedback Contamination', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(-2, 52)
    
    # Improve y-axis formatting
    y_min = min([min([strategy_results[k]['metrics']['r2'] 
                     for k in strategy_results.keys()]) 
                for strategy_results in results.values()])
    y_max = max([max([strategy_results[k]['metrics']['r2'] 
                     for k in strategy_results.keys()]) 
                for strategy_results in results.values()])
    
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\\n📊 Visualization saved to: {save_path}")
    
    plt.show()
    
    # Summary insights
    print(f"\\n🔍 KEY INSIGHTS:")
    print("=" * 50)
    
    # Find most robust and most vulnerable scenarios
    all_degradations = {}
    for strategy in strategies:
        strategy_rates = sorted([float(k) for k in results[strategy].keys()])
        clean = results[strategy][str(0.0)]['metrics']['r2']
        max_contam = results[strategy][str(max(strategy_rates))]['metrics']['r2']
        degradation = (clean - max_contam) / clean * 100
        all_degradations[strategy] = degradation
    
    most_robust = min(all_degradations.keys(), key=lambda k: all_degradations[k])
    most_vulnerable = max(all_degradations.keys(), key=lambda k: all_degradations[k])
    
    print(f"Most robust scenario:     {most_robust.replace('_', ' ').title()} ({all_degradations[most_robust]:.1f}% degradation)")
    print(f"Most vulnerable scenario: {most_vulnerable.replace('_', ' ').title()} ({all_degradations[most_vulnerable]:.1f}% degradation)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregator Robustness Study")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--rates", type=float, nargs="+", 
                       default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                       help="Contamination rates to test")
    parser.add_argument("--strategies", type=str, nargs="+", 
                       default=['systematic_bias', 'random_noise', 'scaled_down'],
                       choices=['systematic_bias', 'random_noise', 'scaled_down'],
                       help="Contamination strategies to test")
    parser.add_argument("--output", type=str, default="aggregator_robustness_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Run robustness study
    results = run_robustness_study(args.data, args.rates, args.strategies)
    
    # Analyze and visualize results
    plot_path = 'results/aggregator_robustness_analysis.png'
    analyze_and_visualize_results(results, plot_path)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"results/aggregator_robustness_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy(obj)
    
    results_serializable = recursive_convert(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\\nResults saved to {output_file}")

if __name__ == "__main__":
    main()