#!/usr/bin/env python3
"""
Train MLP for Existing Experiment Analysis

This script trains an MLP model on existing experiment data and integrates
the results into the analysis pipeline.
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.core.aggregator_training import MLPTrainer, compute_metrics, FEATURE_LABELS


class MLPAnalyzer:
    """Train and analyze MLP models on existing experiment data."""
    
    def __init__(self, experiment_dir: str, random_seed: int = 42):
        self.experiment_dir = Path(experiment_dir)
        self.random_seed = random_seed
        self.results = {}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """Load experiment data and prepare for MLP training."""
        data_path = self.experiment_dir / "data" / "data_with_judge_scores.pkl"
        
        print(f"📂 Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        print(f"📊 Processing {len(data)} samples")
        
        # Extract judge scores and human feedback
        judge_scores = []
        human_scores = []
        
        for _, row in data.iterrows():
            if 'judge_scores' in row and 'human_feedback' in row:
                # Extract judge scores (should be list of 10 scores)
                js = row['judge_scores']
                if isinstance(js, list) and len(js) == 10:
                    judge_scores.append(js)
                    
                    # Extract human feedback average score
                    hf = row['human_feedback']
                    if isinstance(hf, dict) and 'average_score' in hf:
                        human_scores.append(hf['average_score'])
                    else:
                        # Skip this sample if no valid human feedback
                        judge_scores.pop()
        
        X = np.array(judge_scores)
        y = np.array(human_scores)
        
        print(f"✅ Prepared {len(X)} samples for training")
        print(f"📈 Judge scores shape: {X.shape}, Human scores shape: {y.shape}")
        
        return X, y, FEATURE_LABELS
    
    def train_mlp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Train MLP model with the same train/test split as other analyses."""
        
        print(f"\n🧠 Training MLP model...")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Epochs: {n_epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Use same random seed as baseline analysis for consistency
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )
        
        # Initialize trainer
        trainer = MLPTrainer(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            early_stopping_patience=10
        )
        
        # Train model
        train_losses, val_losses = trainer.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate
        train_pred = trainer.predict(X_train)
        test_pred = trainer.predict(X_test)
        
        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)
        
        print(f"\n📊 MLP Results:")
        print(f"   Train - MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"   Test  - MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")
        
        # Save model
        model_dir = self.experiment_dir / "mlp_analysis"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "mlp_model.pt"
        trainer.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Create training curves plot
        if val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss', linewidth=2)
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title(f'MLP Training Curves (Hidden={hidden_dim}, LR={learning_rate})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            curves_path = model_dir / "mlp_training_curves.png"
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📈 Training curves saved to {curves_path}")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_path': str(model_path),
            'hyperparameters': {
                'hidden_dim': hidden_dim,
                'learning_rate': learning_rate,
                'n_epochs': n_epochs,
                'batch_size': batch_size
            }
        }
    
    def update_experiment_summary(self, mlp_results: Dict[str, Any]):
        """Update experiment summary with MLP results."""
        summary_path = self.experiment_dir / "experiment_summary.json"
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Add MLP analysis section
        summary['mlp_analysis'] = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_train_r2': mlp_results['train_metrics']['r2'],
            'best_test_r2': mlp_results['test_metrics']['r2'],
            'best_train_mse': mlp_results['train_metrics']['mse'],
            'best_test_mse': mlp_results['test_metrics']['mse'],
            'best_train_mae': mlp_results['train_metrics']['mae'],
            'best_test_mae': mlp_results['test_metrics']['mae'],
            'hyperparameters': mlp_results['hyperparameters'],
            'model_path': mlp_results['model_path']
        }
        
        # Update model comparison to include MLP
        if 'model_comparison' in summary:
            summary['model_comparison']['all_r2_scores']['mlp'] = mlp_results['test_metrics']['r2']
            
            # Update best model if MLP is better
            current_best_r2 = summary['model_comparison']['best_r2']
            if mlp_results['test_metrics']['r2'] > current_best_r2:
                summary['model_comparison']['best_model'] = 'mlp'
                summary['model_comparison']['best_r2'] = mlp_results['test_metrics']['r2']
        
        # Save updated summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Updated experiment summary: {summary_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MLP for existing experiment")
    parser.add_argument('--experiment-dir', required=True,
                        help='Path to experiment directory')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting MLP analysis for: {args.experiment_dir}")
    
    # Initialize analyzer
    analyzer = MLPAnalyzer(args.experiment_dir, args.random_seed)
    
    # Load data
    X, y, feature_labels = analyzer.load_data()
    
    # Train MLP
    results = analyzer.train_mlp(
        X, y,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Update experiment summary
    analyzer.update_experiment_summary(results)
    
    print(f"\n🎉 MLP analysis complete!")
    print(f"📊 Test R² score: {results['test_metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()