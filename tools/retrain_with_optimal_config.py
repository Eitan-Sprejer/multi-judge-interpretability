#!/usr/bin/env python3
"""
Retrain Model with Optimal Hyperparameters

Retrains the baseline experiment model using the second best configuration
from hyperparameter search (64 hidden dim for efficiency).
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.core.aggregator_training import MLPTrainer, compute_metrics, plot_training_curves
from pipeline.core.persona_simulation import PERSONAS

def main():
    # Configuration from second best hyperparameter result
    # R² = 0.5809 (64 hidden dim vs 512 hidden dim with R² = 0.5813)
    optimal_config = {
        "hidden_dim": 64,
        "learning_rate": 0.001,
        "batch_size": 32,
        "n_epochs": 400,
        "dropout": 0.1,
        "l2_reg": 0.1,
        "early_stopping_patience": 20,
        "min_delta": 0.0001
    }
    
    print("🎯 Retraining Model with Optimal Hyperparameters")
    print("=" * 60)
    print(f"🔧 Configuration (2nd best - efficient 64-dim):")
    for key, value in optimal_config.items():
        print(f"   {key}: {value}")
    print(f"📊 Expected R²: ~0.581 (vs baseline 0.539)")
    print()
    
    # Paths
    baseline_experiment = "results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023"
    data_path = Path(baseline_experiment) / "data" / "data_with_judge_scores.pkl"
    
    # Load existing data
    print(f"📂 Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    print(f"✅ Loaded {len(data)} samples")
    
    # Prepare training data (same method as original experiment)
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
    
    print(f"✅ Prepared {len(X)} training samples")
    
    # Split data (same random seed as original)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"🎯 Training set: {len(X_train)} samples")
    print(f"🎯 Test set: {len(X_test)} samples")
    print()
    
    # Train model with optimal configuration
    print("🚀 Training model with optimal hyperparameters...")
    trainer = MLPTrainer(**optimal_config)
    
    # Train and capture training history
    train_losses, val_losses = trainer.fit(X_train, y_train, X_test, y_test)
    
    # Generate predictions
    train_pred = trainer.predict(X_train)
    test_pred = trainer.predict(X_test)
    
    # Compute metrics
    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    
    # Print results
    print("\n" + "=" * 60)
    print("🏆 OPTIMAL MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Training Results:")
    print(f"   Train R²: {train_metrics['r2']:.4f}")
    print(f"   Train MAE: {train_metrics['mae']:.4f}")
    print(f"   Test R²: {test_metrics['r2']:.4f}")
    print(f"   Test MAE: {test_metrics['mae']:.4f}")
    
    # Compare with baseline
    baseline_r2 = 0.539  # From original experiment
    improvement = test_metrics['r2'] - baseline_r2
    print(f"\n🎯 Performance vs Baseline:")
    print(f"   Baseline R²: {baseline_r2:.4f}")
    print(f"   New R²: {test_metrics['r2']:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement/baseline_r2*100:+.1f}%)")
    
    # Training info
    final_epoch = len(train_losses)
    best_epoch = np.argmin(val_losses) + 1 if val_losses else final_epoch
    early_stopped = final_epoch < optimal_config['n_epochs']
    
    print(f"\n⚡ Training Details:")
    print(f"   Total epochs: {final_epoch}")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Early stopped: {'Yes' if early_stopped else 'No'}")
    
    # Save updated results to baseline experiment directory
    results_dir = Path(baseline_experiment)
    
    # Save the trained model
    model_path = results_dir / "optimal_model.pt"
    trainer.save_model(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save scaler for future inference
    scaler_path = results_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # Create training curves plot
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    curve_path = plots_dir / "optimal_training_curves.png"
    plot_training_curves(train_losses, val_losses, save_path=curve_path, show=False)
    print(f"📊 Training curves saved to: {curve_path}")
    
    # Update experiment results
    updated_results = {
        "experiment_info": {
            "original_r2": baseline_r2,
            "optimal_r2": test_metrics['r2'],
            "improvement": improvement,
            "improvement_percent": improvement/baseline_r2*100,
            "retrained_timestamp": datetime.now().isoformat()
        },
        "optimal_config": optimal_config,
        "training_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_details": {
            "total_epochs": final_epoch,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None
        }
    }
    
    # Save updated results
    updated_results_path = results_dir / "optimal_model_results.json"
    with open(updated_results_path, 'w') as f:
        json.dump(updated_results, f, indent=2, default=str)
    print(f"📋 Updated results saved to: {updated_results_path}")
    
    # Update experiment summary
    summary_path = results_dir / "experiment_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Add optimal model info to summary
        summary["optimal_model"] = {
            "r2_score": test_metrics['r2'],
            "mae_score": test_metrics['mae'],
            "improvement_vs_baseline": improvement,
            "config": optimal_config,
            "model_path": str(model_path),
            "retrained_timestamp": datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📝 Updated experiment summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ RETRAIN COMPLETE! Baseline experiment updated with optimal model.")
    print("=" * 60)
    
    return {
        "test_r2": test_metrics['r2'],
        "improvement": improvement,
        "model_path": model_path,
        "config": optimal_config
    }

if __name__ == "__main__":
    results = main()