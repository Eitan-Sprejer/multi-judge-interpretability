#!/usr/bin/env python3
"""
Hyperparameter Tuning for MLP Models

Uses existing experiment data to test different MLP configurations
and find optimal hyperparameters for improving R² scores.
"""

import json
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import itertools

# Import project modules
from pipeline.core.aggregator_training import MLPTrainer, compute_metrics
from pipeline.core.persona_simulation import PERSONAS

class HyperparameterTuner:
    """
    Hyperparameter tuning for MLP aggregation models.
    """
    
    def __init__(
        self,
        experiment_data_path: str,
        output_dir: str = "hyperparameter_tuning_results",
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        self.experiment_data_path = experiment_data_path
        self.output_dir = Path(output_dir) if not str(output_dir).startswith("results/") else Path(output_dir)
        # If default output dir, use organized structure
        if output_dir == "hyperparameter_tuning_results":
            self.output_dir = Path("results/hyperparameter_search")
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"tuning_run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Hyperparameter tuning output: {self.run_dir}")
    
    def load_experiment_data(self) -> pd.DataFrame:
        """Load data from completed experiment."""
        data_path = Path(self.experiment_data_path) / "data" / "data_with_judge_scores.pkl"
        
        print(f"📂 Loading experiment data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(data)} samples with judge scores and persona feedback")
        return data
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare X and y for training."""
        X_list = []
        y_list = []
        
        # Uniform persona sampling for consistency
        available_personas = list(PERSONAS.keys())
        samples_per_persona = len(data) // len(available_personas)
        remaining_samples = len(data) % len(available_personas)
        
        persona_assignment = []
        for persona in available_personas:
            persona_assignment.extend([persona] * samples_per_persona)
        for _ in range(remaining_samples):
            persona_assignment.append(random.choice(available_personas))
        random.shuffle(persona_assignment)
        
        # Extract features and targets
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
        return X, y
    
    def define_hyperparameter_grid(self) -> Dict[str, List]:
        """Define hyperparameter search grid (with early stopping, epochs matter less)."""
        return {
            'hidden_dim': [32, 64, 128, 256, 512],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [16, 32, 64, 128],
            'n_epochs': [200, 300, 400],  # Reduced since early stopping will find optimal point
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'l2_reg': [0.0, 0.001, 0.01, 0.1]
        }
    
    def random_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_trials: int = 50,
        normalize: bool = True
    ) -> List[Dict]:
        """
        Perform random hyperparameter search.
        """
        print(f"🔍 Starting random search with {n_trials} trials")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        hyperparams = self.define_hyperparameter_grid()
        results = []
        
        for trial in range(n_trials):
            # Sample random hyperparameters
            config = {
                'hidden_dim': random.choice(hyperparams['hidden_dim']),
                'learning_rate': random.choice(hyperparams['learning_rate']),
                'batch_size': min(random.choice(hyperparams['batch_size']), len(X_train) // 2),
                'n_epochs': random.choice(hyperparams['n_epochs']),
                'dropout': random.choice(hyperparams['dropout']),
                'l2_reg': random.choice(hyperparams['l2_reg']),
                'early_stopping_patience': 20,  # Appropriate for hyperparameter search
                'min_delta': 1e-4
            }
            
            print(f"Trial {trial + 1}/{n_trials}: {config}")
            
            try:
                # Train model
                trainer = MLPTrainer(**config)
                train_losses, val_losses = trainer.fit(X_train, y_train, X_test, y_test)
                
                # Evaluate
                train_pred = trainer.predict(X_train)
                test_pred = trainer.predict(X_test)
                
                train_metrics = compute_metrics(y_train, train_pred)
                test_metrics = compute_metrics(y_test, test_pred)
                
                result = {
                    'trial': trial + 1,
                    'config': config,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'normalize': normalize,
                    'final_train_loss': train_losses[-1] if train_losses else None,
                    'final_val_loss': val_losses[-1] if val_losses else None
                }
                
                results.append(result)
                
                print(f"  ✅ R² = {test_metrics['r2']:.4f}, MAE = {test_metrics['mae']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Trial failed: {e}")
                continue
        
        # Sort by test R²
        results.sort(key=lambda x: x['test_metrics']['r2'], reverse=True)
        
        return results
    
    def grid_search_top_configs(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Fine-tune around the best configurations from random search.
        """
        print(f"🎯 Fine-tuning top {top_n} configurations")
        
        # First do random search to find promising regions
        random_results = self.random_search(X, y, n_trials=30, normalize=True)
        
        if len(random_results) < top_n:
            return random_results
        
        # Get top configurations
        top_configs = random_results[:top_n]
        
        # Define fine-tuning grids around best configs
        fine_tune_results = []
        
        for i, result in enumerate(top_configs):
            base_config = result['config']
            print(f"\n🔍 Fine-tuning config {i+1}: {base_config}")
            
            # Create variation grids around best config
            variations = self._create_config_variations(base_config)
            
            for variation in variations:
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=self.test_size, random_state=self.random_seed + i
                    )
                    
                    # Normalize
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Add early stopping to variation
                    variation['early_stopping_patience'] = 20
                    variation['min_delta'] = 1e-4
                    
                    # Train
                    trainer = MLPTrainer(**variation)
                    train_losses, val_losses = trainer.fit(X_train, y_train, X_test, y_test)
                    
                    # Evaluate
                    train_pred = trainer.predict(X_train)
                    test_pred = trainer.predict(X_test)
                    
                    train_metrics = compute_metrics(y_train, train_pred)
                    test_metrics = compute_metrics(y_test, test_pred)
                    
                    fine_result = {
                        'base_trial': i + 1,
                        'config': variation,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'normalize': True
                    }
                    
                    fine_tune_results.append(fine_result)
                    
                except Exception as e:
                    continue
        
        # Combine and sort all results
        all_results = random_results + fine_tune_results
        all_results.sort(key=lambda x: x['test_metrics']['r2'], reverse=True)
        
        return all_results
    
    def _create_config_variations(self, base_config: Dict) -> List[Dict]:
        """Create variations around a base configuration."""
        variations = []
        
        # Learning rate variations
        for lr_factor in [0.5, 0.75, 1.25, 1.5]:
            new_lr = base_config['learning_rate'] * lr_factor
            if 0.0001 <= new_lr <= 0.01:
                var = base_config.copy()
                var['learning_rate'] = new_lr
                var['early_stopping_patience'] = 20
                var['min_delta'] = 1e-4
                variations.append(var)
        
        # Hidden dimension variations
        for hidden_factor in [0.5, 0.75, 1.25, 1.5]:
            new_hidden = int(base_config['hidden_dim'] * hidden_factor)
            if 16 <= new_hidden <= 1024:
                var = base_config.copy()
                var['hidden_dim'] = new_hidden
                var['early_stopping_patience'] = 20
                var['min_delta'] = 1e-4
                variations.append(var)
        
        # Epoch variations (less important with early stopping)
        for epoch_factor in [1.5, 2.0]:  # Only increase epochs since early stopping will handle it
            new_epochs = int(base_config['n_epochs'] * epoch_factor)
            if 25 <= new_epochs <= 500:
                var = base_config.copy()
                var['n_epochs'] = new_epochs
                var['early_stopping_patience'] = 25  # More patience for longer training
                var['min_delta'] = 1e-4
                variations.append(var)
        
        return variations[:10]  # Limit variations
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze hyperparameter tuning results."""
        if not results:
            return {}
        
        # Best overall result
        best_result = results[0]
        
        # Extract all test R² scores
        r2_scores = [r['test_metrics']['r2'] for r in results]
        
        analysis = {
            'best_config': best_result['config'],
            'best_r2': best_result['test_metrics']['r2'],
            'best_mae': best_result['test_metrics']['mae'],
            'improvement_vs_baseline': best_result['test_metrics']['r2'] - 0.539,  # Your baseline
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores),
            'top_5_configs': [r['config'] for r in results[:5]],
            'top_5_r2': [r['test_metrics']['r2'] for r in results[:5]]
        }
        
        return analysis
    
    def create_visualizations(self, results: List[Dict], analysis: Dict):
        """Create comprehensive visualization plots including heatmap."""
        if not results:
            return
        
        # Create main analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. R² distribution
        r2_scores = [r['test_metrics']['r2'] for r in results]
        axes[0, 0].hist(r2_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0.539, color='red', linestyle='--', label='Baseline R²')
        axes[0, 0].axvline(analysis['best_r2'], color='green', linestyle='--', label='Best R²')
        axes[0, 0].set_xlabel('Test R² Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of R² Scores')
        axes[0, 0].legend()
        
        # 2. Learning rate vs R²
        lr_vs_r2 = [(r['config']['learning_rate'], r['test_metrics']['r2']) for r in results]
        lrs, r2s = zip(*lr_vs_r2)
        axes[0, 1].scatter(lrs, r2s, alpha=0.6)
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Test R² Score')
        axes[0, 1].set_title('Learning Rate vs R²')
        
        # 3. Hidden dimension vs R²
        hidden_vs_r2 = [(r['config']['hidden_dim'], r['test_metrics']['r2']) for r in results]
        hiddens, r2s = zip(*hidden_vs_r2)
        axes[1, 0].scatter(hiddens, r2s, alpha=0.6)
        axes[1, 0].set_xlabel('Hidden Dimension')
        axes[1, 0].set_ylabel('Test R² Score')
        axes[1, 0].set_title('Hidden Dimension vs R²')
        
        # 4. Top 10 configurations
        top_10 = results[:10]
        config_names = [f"Config {i+1}" for i in range(len(top_10))]
        top_r2s = [r['test_metrics']['r2'] for r in top_10]
        
        bars = axes[1, 1].bar(config_names, top_r2s)
        axes[1, 1].axhline(y=0.539, color='red', linestyle='--', label='Baseline')
        axes[1, 1].set_xlabel('Configuration')
        axes[1, 1].set_ylabel('Test R² Score')
        axes[1, 1].set_title('Top 10 Configurations')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        # Color top bars
        for i, bar in enumerate(bars):
            if i < 3:
                bar.set_color('green')
            elif i < 5:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        
        # Save main analysis plot
        analysis_path = self.run_dir / 'hyperparameter_analysis.png'
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create comprehensive heatmap
        self.create_hyperparameter_heatmap(results, analysis)
        
        print(f"📊 Analysis plots saved to {analysis_path}")
        print(f"🔥 Heatmap saved to {self.run_dir / 'hyperparameter_heatmap.png'}")
    
    def create_hyperparameter_heatmap(self, results: List[Dict], analysis: Dict):
        """Create best validation R² heatmap for hyperparameter combinations."""
        if not results:
            return
        
        # Extract data for heatmap
        data_rows = []
        for result in results:
            config = result['config']
            data_rows.append({
                'hidden_dim': config['hidden_dim'],
                'learning_rate': config['learning_rate'],
                'dropout': config['dropout'],
                'l2_reg': config['l2_reg'],
                'test_r2': result['test_metrics']['r2'],
                'train_r2': result['train_metrics']['r2'],
                'overfitting_gap': result['train_metrics']['r2'] - result['test_metrics']['r2']
            })
        
        df = pd.DataFrame(data_rows)
        
        # Create comprehensive heatmap visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heatmap 1: Best R² by Hidden Dim vs Learning Rate
        pivot_r2 = df.pivot_table(
            values='test_r2', 
            index='learning_rate', 
            columns='hidden_dim', 
            aggfunc='max'  # Take best R² for each combination
        )
        
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Best Test R²'}, ax=ax1)
        ax1.set_title('Best Test R² by Configuration\n(Hidden Dimension vs Learning Rate)')
        ax1.set_xlabel('Hidden Dimension')
        ax1.set_ylabel('Learning Rate')
        
        # Heatmap 2: Best R² by Dropout vs L2 Regularization
        pivot_reg = df.pivot_table(
            values='test_r2', 
            index='dropout', 
            columns='l2_reg', 
            aggfunc='max'
        )
        
        sns.heatmap(pivot_reg, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Best Test R²'}, ax=ax2)
        ax2.set_title('Best Test R² by Regularization\n(Dropout vs L2 Regularization)')
        ax2.set_xlabel('L2 Regularization')
        ax2.set_ylabel('Dropout Rate')
        
        # Heatmap 3: Overfitting Gap by Hidden Dim vs Learning Rate
        pivot_gap = df.pivot_table(
            values='overfitting_gap', 
            index='learning_rate', 
            columns='hidden_dim', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_gap, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Overfitting Gap (Train R² - Test R²)'}, ax=ax3)
        ax3.set_title('Overfitting Analysis\n(Train R² - Test R²)')
        ax3.set_xlabel('Hidden Dimension')
        ax3.set_ylabel('Learning Rate')
        
        # Heatmap 4: Configuration frequency
        freq_table = df.groupby(['learning_rate', 'hidden_dim']).size().unstack(fill_value=0)
        sns.heatmap(freq_table, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Number of Trials'}, ax=ax4)
        ax4.set_title('Configuration Trial Frequency\n(Hidden Dimension vs Learning Rate)')
        ax4.set_xlabel('Hidden Dimension')
        ax4.set_ylabel('Learning Rate')
        
        plt.tight_layout()
        
        # Add overall title with best config info
        best_config = results[0]['config']
        best_r2 = analysis['best_r2']
        fig.suptitle(f'Hyperparameter Analysis - Best: R²={best_r2:.3f} '
                    f'(H={best_config["hidden_dim"]}, LR={best_config["learning_rate"]:.4f}, '
                    f'D={best_config["dropout"]:.2f}, L2={best_config["l2_reg"]:.4f})', 
                    fontsize=14, y=0.98)
        
        # Save heatmap
        heatmap_path = self.run_dir / "hyperparameter_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional parameter interaction plots
        self.create_parameter_interaction_plots(df)
    
    def create_parameter_interaction_plots(self, df: pd.DataFrame):
        """Create detailed parameter interaction analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Hidden Dimension vs R² with trend
        hidden_dims = sorted(df['hidden_dim'].unique())
        hidden_r2_means = [df[df['hidden_dim'] == h]['test_r2'].mean() for h in hidden_dims]
        hidden_r2_maxs = [df[df['hidden_dim'] == h]['test_r2'].max() for h in hidden_dims]
        
        ax1.scatter(df['hidden_dim'], df['test_r2'], alpha=0.5, color='lightblue', label='Individual trials')
        ax1.plot(hidden_dims, hidden_r2_means, 'bo-', label='Mean R²', linewidth=2)
        ax1.plot(hidden_dims, hidden_r2_maxs, 'ro-', label='Max R²', linewidth=2)
        ax1.set_xlabel('Hidden Dimension')
        ax1.set_ylabel('Test R²')
        ax1.set_title('Hidden Dimension Impact Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning Rate vs R² (log scale)
        ax2.scatter(df['learning_rate'], df['test_r2'], alpha=0.6, c='green')
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Learning Rate vs Performance')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Dropout vs Overfitting
        ax3.scatter(df['dropout'], df['overfitting_gap'], alpha=0.6, c='orange')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No overfitting')
        ax3.set_xlabel('Dropout Rate')
        ax3.set_ylabel('Overfitting Gap (Train R² - Test R²)')
        ax3.set_title('Dropout Effect on Overfitting')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: L2 Regularization vs Performance
        ax4.scatter(df['l2_reg'], df['test_r2'], alpha=0.6, c='purple')
        ax4.set_xlabel('L2 Regularization')
        ax4.set_ylabel('Test R²')
        ax4.set_title('L2 Regularization vs Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save interaction plots
        interaction_path = self.run_dir / "parameter_interactions.png"
        plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: List[Dict], analysis: Dict):
        """Save all results to files."""
        # Save detailed results
        results_path = self.run_dir / 'detailed_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        serializable_result[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
                    else:
                        serializable_result[key] = float(value) if isinstance(value, np.number) else value
                serializable_results.append(serializable_result)
            json.dump(serializable_results, f, indent=2)
        
        # Save analysis summary
        analysis_path = self.run_dir / 'analysis_summary.json'
        with open(analysis_path, 'w') as f:
            serializable_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, (np.ndarray, list)):
                    serializable_analysis[key] = [float(x) if isinstance(x, np.number) else x for x in value]
                else:
                    serializable_analysis[key] = float(value) if isinstance(value, np.number) else value
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"💾 Results saved to {self.run_dir}")
    
    def run_tuning(self, search_type: str = "random", n_trials: int = 50) -> Dict:
        """Run complete hyperparameter tuning."""
        print(f"🚀 Starting {search_type} hyperparameter search")
        
        # Load data
        data = self.load_experiment_data()
        X, y = self.prepare_training_data(data)
        
        # Run search
        if search_type == "random":
            results = self.random_search(X, y, n_trials=n_trials)
        elif search_type == "grid_fine":
            results = self.grid_search_top_configs(X, y)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Create visualizations
        self.create_visualizations(results, analysis)
        
        # Save results
        self.save_results(results, analysis)
        
        return analysis


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MLP models")
    parser.add_argument('--experiment-path', required=True,
                        help='Path to completed experiment directory')
    parser.add_argument('--search-type', choices=['random', 'grid_fine'], default='random',
                        help='Type of search to perform')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for random search')
    parser.add_argument('--output-dir', default='hyperparameter_tuning_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run tuning
    tuner = HyperparameterTuner(
        experiment_data_path=args.experiment_path,
        output_dir=args.output_dir
    )
    
    analysis = tuner.run_tuning(
        search_type=args.search_type,
        n_trials=args.n_trials
    )
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    print(f"🏆 Best R² Score: {analysis['best_r2']:.4f}")
    print(f"📈 Improvement vs Baseline: +{analysis['improvement_vs_baseline']:.4f}")
    print(f"🔧 Best Configuration:")
    for key, value in analysis['best_config'].items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Mean R²: {analysis['mean_r2']:.4f}")
    print(f"   Std R²: {analysis['std_r2']:.4f}")
    print(f"   Top 5 R² scores: {[f'{r:.4f}' for r in analysis['top_5_r2']]}")
    
    print("="*80)


if __name__ == "__main__":
    main()