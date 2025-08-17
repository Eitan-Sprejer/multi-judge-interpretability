#!/usr/bin/env python3
"""
Rapid Hyperparameter Testing

Quick tests of the most promising configurations with reduced epochs.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.core.aggregator_training import MLPTrainer, compute_metrics, plot_training_curves
from pipeline.core.persona_simulation import PERSONAS

def load_experiment_data(experiment_path: str) -> tuple:
    """Load and prepare data from experiment."""
    data_path = Path(experiment_path) / "data" / "data_with_judge_scores.pkl"
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    # Prepare training data
    X_list = []
    y_list = []
    
    available_personas = list(PERSONAS.keys())
    samples_per_persona = len(data) // len(available_personas)
    remaining_samples = len(data) % len(available_personas)
    
    persona_assignment = []
    for persona in available_personas:
        persona_assignment.extend([persona] * samples_per_persona)
    for _ in range(remaining_samples):
        persona_assignment.append(np.random.choice(available_personas))
    np.random.shuffle(persona_assignment)
    
    for idx, (row, assigned_persona) in enumerate(zip(data.iterrows(), persona_assignment)):
        row = row[1]
        
        if ('human_feedback' not in row or 'personas' not in row['human_feedback'] or
            'judge_scores' not in row or not isinstance(row['judge_scores'], list)):
            continue
        
        personas_feedback = row['human_feedback']['personas']
        if assigned_persona not in personas_feedback or 'score' not in personas_feedback[assigned_persona]:
            continue
        
        selected_score = personas_feedback[assigned_persona]['score']
        judge_scores = row['judge_scores']
        
        if selected_score is None or len(judge_scores) != 10:
            continue
        
        X_list.append(judge_scores)
        y_list.append(selected_score)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y

def test_config(X_train, X_test, y_train, y_test, config, config_idx=None):
    """Test a single configuration with early stopping and loss plotting."""
    try:
        # Use early stopping for better generalization
        fast_config = config.copy()
        fast_config['early_stopping_patience'] = 10  # Faster early stopping for rapid testing
        
        trainer = MLPTrainer(**fast_config)
        train_losses, val_losses = trainer.fit(X_train, y_train, X_test, y_test)
        
        # Plot training curves for visualization
        if config_idx is not None:
            plot_path = f"rapid_tune_curves_config_{config_idx+1}.png"
            plot_training_curves(train_losses, val_losses, save_path=plot_path, show=False)
        
        # Get both training and test metrics
        train_pred = trainer.predict(X_train)
        test_pred = trainer.predict(X_test)
        
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)
        
        # Add training info
        final_epoch = len(train_losses)
        best_epoch = np.argmin(val_losses) + 1 if val_losses else final_epoch
        
        return {
            'train_r2': train_metrics['r2'],
            'train_mae': train_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'test_mae': test_metrics['mae'],
            'final_epoch': final_epoch,
            'best_epoch': best_epoch,
            'early_stopped': final_epoch < fast_config['n_epochs']
        }
    except Exception as e:
        print(f"Failed: {e}")
        return {
            'train_r2': -1,
            'train_mae': 999,
            'test_r2': -1,
            'test_mae': 999,
            'final_epoch': 0,
            'best_epoch': 0,
            'early_stopped': False
        }

def main():
    experiment_path = "full_experiment_runs/baseline_ultrafeedback_2000samples_20250816_213023"
    
    print("⚡ Rapid Hyperparameter Testing (Fast Mode)")
    print(f"📂 Loading data from {experiment_path}")
    
    # Load data
    X, y = load_experiment_data(experiment_path)
    print(f"✅ Loaded {len(X)} samples")
    
    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"🎯 Baseline R²: 0.539")
    print(f"🎯 Previous best: 0.566")
    print(f"⚡ Using reduced epochs for speed")
    print()
    
    # Most promising configurations (reduced epochs for speed)
    configs = [
        # Current best from initial search
        {'hidden_dim': 128, 'learning_rate': 0.0005, 'batch_size': 16, 'n_epochs': 100, 'dropout': 0.2, 'l2_reg': 0.001},
        
        # Promising variations
        {'hidden_dim': 192, 'learning_rate': 0.0005, 'batch_size': 16, 'n_epochs': 100, 'dropout': 0.2, 'l2_reg': 0.001},
        {'hidden_dim': 256, 'learning_rate': 0.0003, 'batch_size': 16, 'n_epochs': 100, 'dropout': 0.15, 'l2_reg': 0.001},
        {'hidden_dim': 160, 'learning_rate': 0.0007, 'batch_size': 16, 'n_epochs': 100, 'dropout': 0.25, 'l2_reg': 0.0005},
        {'hidden_dim': 128, 'learning_rate': 0.0003, 'batch_size': 8, 'n_epochs': 100, 'dropout': 0.2, 'l2_reg': 0.002},
        
        # Different architectures
        {'hidden_dim': 96, 'learning_rate': 0.001, 'batch_size': 32, 'n_epochs': 100, 'dropout': 0.1, 'l2_reg': 0.001},
        {'hidden_dim': 320, 'learning_rate': 0.0002, 'batch_size': 8, 'n_epochs': 100, 'dropout': 0.3, 'l2_reg': 0.001},
        
        # Aggressive regularization
        {'hidden_dim': 256, 'learning_rate': 0.0005, 'batch_size': 16, 'n_epochs': 100, 'dropout': 0.4, 'l2_reg': 0.005},
        
        # Low regularization
        {'hidden_dim': 128, 'learning_rate': 0.0005, 'batch_size': 16, 'n_epochs': 100, 'dropout': 0.0, 'l2_reg': 0.0},
        
        # High learning rate
        {'hidden_dim': 64, 'learning_rate': 0.002, 'batch_size': 32, 'n_epochs': 100, 'dropout': 0.2, 'l2_reg': 0.001},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"Testing {i+1}/{len(configs)}: Hidden={config['hidden_dim']}, LR={config['learning_rate']}, Dropout={config['dropout']}")
        metrics = test_config(X_train, X_test, y_train, y_test, config, config_idx=i)
        
        results.append({
            'config': config,
            'train_r2': metrics['train_r2'],
            'train_mae': metrics['train_mae'],
            'test_r2': metrics['test_r2'],
            'test_mae': metrics['test_mae'],
            'improvement': metrics['test_r2'] - 0.539,
            'final_epoch': metrics['final_epoch'],
            'best_epoch': metrics['best_epoch'],
            'early_stopped': metrics['early_stopped']
        })
        
        print(f"  → Train R² = {metrics['train_r2']:.4f}, Test R² = {metrics['test_r2']:.4f}")
        print(f"  → Train MAE = {metrics['train_mae']:.4f}, Test MAE = {metrics['test_mae']:.4f}")
        print(f"  → Improvement = {metrics['test_r2']-0.539:+.4f}")
        print(f"  → Training: {metrics['final_epoch']} epochs (best at {metrics['best_epoch']})")
        if metrics['early_stopped']:
            print(f"  ⏹️  Early stopped!")
        
        if metrics['test_r2'] > 0.566:
            print(f"  🎉 NEW BEST!")
        print()
    
    # Sort by Test R²
    results.sort(key=lambda x: x['test_r2'], reverse=True)
    
    print("="*60)
    print("🏆 RAPID TESTING RESULTS")
    print("="*60)
    
    best = results[0]
    print(f"🥇 Best Test R²: {best['test_r2']:.4f}")
    print(f"🏃 Best Train R²: {best['train_r2']:.4f}")
    print(f"📈 Improvement: {best['improvement']:+.4f}")
    print(f"📉 Test MAE: {best['test_mae']:.4f}")
    print(f"🔧 Config: {best['config']}")
    
    print(f"\n📊 All Results (sorted by Test R²):")
    for i, result in enumerate(results):
        print(f"{i+1:2d}. Test R²={result['test_r2']:.4f} Train R²={result['train_r2']:.4f} (+{result['improvement']:+.4f}) | "
              f"H={result['config']['hidden_dim']:3d} LR={result['config']['learning_rate']:.4f} "
              f"D={result['config']['dropout']:.1f} L2={result['config']['l2_reg']:.4f}")
    
    # Save results to organized location
    results_dir = Path("results/quick_tests")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'rapid_tune_results.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    
    # Recommend best config for full training
    if best['test_r2'] > 0.566:
        print(f"\n🎯 RECOMMENDED FOR FULL TRAINING:")
        print(f"   Use this config with n_epochs=300-500 for best results:")
        full_config = best['config'].copy()
        full_config['n_epochs'] = 300
        print(f"   {full_config}")

if __name__ == "__main__":
    main()