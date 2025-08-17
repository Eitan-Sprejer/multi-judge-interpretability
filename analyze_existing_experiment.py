#!/usr/bin/env python3
"""
Analyze Existing Experiment with GAM and Baseline Comparisons

This script loads an existing experiment's data and runs:
1. GAM hyperparameter tuning (75 trials with optimized parameter ranges)
2. Baseline model comparisons (naive mean, best single judge, correlation-weighted mean)
3. Updates experiment_summary.json with all metrics
4. Creates comprehensive visualizations and model comparison plots

Usage:
    python analyze_existing_experiment.py --experiment-dir path/to/experiment

Key Features:
- Non-destructive: Doesn't re-run judge inference or persona simulation
- Comprehensive: Includes GAM heatmaps, partial dependence plots, and baseline analysis
- Complete: Updates experiment_summary.json with all new metrics for research papers
- Organized: Saves all results in structured subdirectories within the experiment folder

Example:
    python analyze_existing_experiment.py --experiment-dir results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023
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

# Import GAM dependencies
try:
    from pygam import LinearGAM, s, f, te
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("❌ PyGAM not installed. Install with: pip install pygam")
    exit(1)

# Import project modules
from pipeline.core.aggregator_training import GAMAggregator, compute_metrics, FEATURE_LABELS
from pipeline.core.persona_simulation import PERSONAS
from gam_hyperparameter_tuning import GAMHyperparameterTuner


class ExistingExperimentAnalyzer:
    """
    Analyzes existing experiment data with GAM tuning and baseline comparisons.
    """
    
    def __init__(
        self,
        experiment_dir: str,
        random_seed: int = 42
    ):
        self.experiment_dir = Path(experiment_dir)
        self.random_seed = random_seed
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Verify experiment directory exists
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")
        
        print(f"🔍 Analyzing existing experiment: {self.experiment_dir}")
    
    def load_experiment_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load experiment data and existing summary."""
        # Try different possible data file locations
        possible_paths = [
            self.experiment_dir / "data_with_judge_scores.pkl",
            self.experiment_dir / "data" / "data_with_judge_scores.pkl",
            self.experiment_dir / "experiment_results.pkl"
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(f"Judge scores data not found in: {possible_paths}")
        
        print(f"📂 Loading experiment data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(data)} samples with judge scores and persona feedback")
        
        # Load existing experiment summary if it exists
        summary_path = self.experiment_dir / "experiment_summary.json"
        existing_summary = {}
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
            print(f"📊 Loaded existing experiment summary")
        
        return data, existing_summary
    
    def compute_baseline_comparisons(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute baseline model comparisons: best single judge and naive mean."""
        print("🔬 Computing baseline model comparisons...")
        
        # Prepare data for baseline analysis
        X_list = []
        y_list = []
        
        # Uniform persona sampling for consistency (with fixed seed for reproducibility)
        random.seed(self.random_seed)  # Ensure consistent persona assignment
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
        
        print(f"✅ Prepared {len(X)} samples for baseline analysis")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        
        baselines = {}
        
        # 1. Naive Mean: Simple average of all judge scores
        print("📊 Computing Naive Mean baseline...")
        naive_mean_train = np.mean(X_train, axis=1)
        naive_mean_test = np.mean(X_test, axis=1)
        
        # Scale to match target range [0, 10] (same as human scores)
        target_min, target_max = 0, 10
        naive_min, naive_max = np.min(naive_mean_train), np.max(naive_mean_train)
        
        if naive_max > naive_min:
            naive_mean_train_scaled = ((naive_mean_train - naive_min) / (naive_max - naive_min)) * (target_max - target_min) + target_min
            naive_mean_test_scaled = ((naive_mean_test - naive_min) / (naive_max - naive_min)) * (target_max - target_min) + target_min
        else:
            naive_mean_train_scaled = np.full_like(naive_mean_train, (target_min + target_max) / 2)
            naive_mean_test_scaled = np.full_like(naive_mean_test, (target_min + target_max) / 2)
        
        naive_metrics = compute_metrics(y_test, naive_mean_test_scaled)
        baselines['naive_mean'] = {
            'method': 'Simple average of all judge scores, scaled to [0,10]',
            'train_metrics': compute_metrics(y_train, naive_mean_train_scaled),
            'test_metrics': naive_metrics
        }
        
        # 2. Best Single Judge: Find the judge with highest correlation to human feedback
        print("🏆 Finding best single judge...")
        judge_correlations = []
        judge_metrics = []
        
        for j in range(10):
            judge_scores_train = X_train[:, j]
            judge_scores_test = X_test[:, j]
            
            # Scale judge scores to [0, 10] range
            judge_min, judge_max = np.min(judge_scores_train), np.max(judge_scores_train)
            if judge_max > judge_min:
                judge_train_scaled = ((judge_scores_train - judge_min) / (judge_max - judge_min)) * (target_max - target_min) + target_min
                judge_test_scaled = ((judge_scores_test - judge_min) / (judge_max - judge_min)) * (target_max - target_min) + target_min
            else:
                judge_train_scaled = np.full_like(judge_scores_train, (target_min + target_max) / 2)
                judge_test_scaled = np.full_like(judge_scores_test, (target_min + target_max) / 2)
            
            correlation = np.corrcoef(y_test, judge_test_scaled)[0, 1] if len(np.unique(judge_test_scaled)) > 1 else 0
            metrics = compute_metrics(y_test, judge_test_scaled)
            
            judge_correlations.append(correlation)
            judge_metrics.append(metrics)
        
        best_judge_idx = np.argmax(judge_correlations)
        best_judge_name = FEATURE_LABELS[best_judge_idx]
        
        baselines['best_single_judge'] = {
            'method': f'Best performing single judge: {best_judge_name}',
            'judge_index': int(best_judge_idx),
            'judge_name': best_judge_name,
            'correlation': float(judge_correlations[best_judge_idx]),
            'test_metrics': judge_metrics[best_judge_idx],
            'all_judge_correlations': {
                FEATURE_LABELS[i]: float(corr) for i, corr in enumerate(judge_correlations)
            }
        }
        
        # 3. Scaled Mean: Weighted average using correlation-based weights
        print("⚖️ Computing correlation-weighted baseline...")
        weights = np.array([max(0, corr) for corr in judge_correlations])  # Only positive correlations
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)  # Normalize
        else:
            weights = np.ones(10) / 10  # Equal weights if all correlations are negative
        
        scaled_mean_train = np.average(X_train, axis=1, weights=weights)
        scaled_mean_test = np.average(X_test, axis=1, weights=weights)
        
        # Scale to [0, 10] range
        scaled_min, scaled_max = np.min(scaled_mean_train), np.max(scaled_mean_train)
        if scaled_max > scaled_min:
            scaled_mean_train_scaled = ((scaled_mean_train - scaled_min) / (scaled_max - scaled_min)) * (target_max - target_min) + target_min
            scaled_mean_test_scaled = ((scaled_mean_test - scaled_min) / (scaled_max - scaled_min)) * (target_max - target_min) + target_min
        else:
            scaled_mean_train_scaled = np.full_like(scaled_mean_train, (target_min + target_max) / 2)
            scaled_mean_test_scaled = np.full_like(scaled_mean_test, (target_min + target_max) / 2)
        
        scaled_metrics = compute_metrics(y_test, scaled_mean_test_scaled)
        baselines['correlation_weighted_mean'] = {
            'method': 'Correlation-weighted average of judge scores',
            'weights': {FEATURE_LABELS[i]: float(w) for i, w in enumerate(weights)},
            'test_metrics': scaled_metrics
        }
        
        # Summary comparison
        print("\n📊 Baseline Comparison Results:")
        print(f"   Naive Mean R²: {naive_metrics['r2']:.4f}")
        print(f"   Best Judge R² ({best_judge_name}): {judge_metrics[best_judge_idx]['r2']:.4f}")
        print(f"   Weighted Mean R²: {scaled_metrics['r2']:.4f}")
        
        return baselines
    
    def run_gam_analysis(self) -> Dict[str, Any]:
        """Run GAM hyperparameter tuning on existing experiment data."""
        print("🧠 Running GAM hyperparameter analysis...")
        
        # Create GAM tuner with experiment data
        gam_tuner = GAMHyperparameterTuner(
            experiment_data_path=str(self.experiment_dir),
            output_dir=str(self.experiment_dir / "gam_analysis"),
            random_seed=self.random_seed
        )
        
        # Run GAM tuning with more trials for thorough analysis
        gam_analysis = gam_tuner.run_tuning(n_trials=75, normalize=True)
        
        if gam_analysis:
            print(f"✅ GAM analysis complete - Best R²: {gam_analysis['best_r2']:.4f}")
            return gam_analysis
        else:
            print("❌ GAM analysis failed")
            return {}
    
    def update_experiment_summary(
        self, 
        existing_summary: Dict, 
        baseline_results: Dict, 
        gam_results: Dict
    ):
        """Update experiment_summary.json with new analysis results."""
        print("📝 Updating experiment summary with new results...")
        
        # Add baseline analysis
        existing_summary['baseline_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'baselines': baseline_results,
            'baseline_comparison': {
                'naive_mean_r2': baseline_results['naive_mean']['test_metrics']['r2'],
                'best_judge_r2': baseline_results['best_single_judge']['test_metrics']['r2'],
                'weighted_mean_r2': baseline_results['correlation_weighted_mean']['test_metrics']['r2'],
                'best_baseline_method': max(baseline_results.keys(), 
                                          key=lambda k: baseline_results[k]['test_metrics']['r2'])
            }
        }
        
        # Add GAM analysis if successful
        if gam_results:
            existing_summary['gam_analysis'] = {
                'timestamp': datetime.now().isoformat(),
                'best_r2': gam_results['best_r2'],
                'best_mae': gam_results['best_mae'],
                'best_aic': gam_results['best_aic'],
                'best_gcv': gam_results['best_gcv'],
                'best_edof': gam_results['best_edof'],
                'best_config': gam_results['best_config'],
                'successful_trials': gam_results['successful_trials'],
                'mean_r2': gam_results['mean_r2'],
                'std_r2': gam_results['std_r2'],
                'top_5_r2': gam_results['top_5_r2'],
                'feature_importance': gam_results['feature_importance_best']
            }
        
        # Add overall comparison
        all_r2_scores = {
            'naive_mean': baseline_results['naive_mean']['test_metrics']['r2'],
            'best_judge': baseline_results['best_single_judge']['test_metrics']['r2'],
            'weighted_mean': baseline_results['correlation_weighted_mean']['test_metrics']['r2']
        }
        
        if gam_results:
            all_r2_scores['gam'] = gam_results['best_r2']
        
        # Check if MLP results exist (could be in 'mlp_analysis' or 'optimal_model')
        if 'mlp_analysis' in existing_summary:
            all_r2_scores['mlp'] = existing_summary['mlp_analysis'].get('best_test_r2', 0)
        elif 'optimal_model' in existing_summary:
            all_r2_scores['mlp'] = existing_summary['optimal_model'].get('r2_score', 0)
        
        existing_summary['model_comparison'] = {
            'all_r2_scores': all_r2_scores,
            'best_model': max(all_r2_scores.keys(), key=lambda k: all_r2_scores[k]),
            'best_r2': max(all_r2_scores.values()),
            'improvement_over_baseline': {
                model: all_r2_scores[model] - all_r2_scores['naive_mean'] 
                for model in all_r2_scores.keys()
            }
        }
        
        # Save updated summary
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(existing_summary, f, indent=2)
        
        print(f"✅ Updated experiment summary: {summary_path}")
        
        return existing_summary
    
    def create_comparison_visualization(
        self, 
        baseline_results: Dict, 
        gam_results: Dict, 
        existing_summary: Dict
    ):
        """Create visualization comparing all model results."""
        print("📊 Creating model comparison visualization...")
        
        # Prepare data for visualization
        models = ['Naive Mean', 'Best Judge', 'Weighted Mean']
        r2_scores = [
            baseline_results['naive_mean']['test_metrics']['r2'],
            baseline_results['best_single_judge']['test_metrics']['r2'],
            baseline_results['correlation_weighted_mean']['test_metrics']['r2']
        ]
        mae_scores = [
            baseline_results['naive_mean']['test_metrics']['mae'],
            baseline_results['best_single_judge']['test_metrics']['mae'],
            baseline_results['correlation_weighted_mean']['test_metrics']['mae']
        ]
        
        if gam_results:
            models.append('GAM')
            r2_scores.append(gam_results['best_r2'])
            mae_scores.append(gam_results['best_mae'])
        
        # Check if MLP results exist (could be in 'mlp_analysis' or 'optimal_model')
        if 'mlp_analysis' in existing_summary:
            models.append('MLP')
            r2_scores.append(existing_summary['mlp_analysis'].get('best_test_r2', 0))
            mae_scores.append(existing_summary['mlp_analysis'].get('best_test_mae', 0))
        elif 'optimal_model' in existing_summary:
            models.append('MLP')
            r2_scores.append(existing_summary['optimal_model'].get('r2_score', 0))
            mae_scores.append(existing_summary['optimal_model'].get('mae_score', 0))
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        bars1 = ax1.bar(models, r2_scores, color=['lightcoral', 'orange', 'gold', 'lightgreen', 'lightblue'][:len(models)])
        ax1.set_title('Model Performance Comparison - R² Score', fontsize=14)
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, max(r2_scores) * 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Color best performing model
        best_idx = np.argmax(r2_scores)
        bars1[best_idx].set_color('gold')
        bars1[best_idx].set_edgecolor('darkgoldenrod')
        bars1[best_idx].set_linewidth(2)
        
        # MAE comparison (lower is better)
        bars2 = ax2.bar(models, mae_scores, color=['lightcoral', 'orange', 'gold', 'lightgreen', 'lightblue'][:len(models)])
        ax2.set_title('Model Performance Comparison - MAE', fontsize=14)
        ax2.set_ylabel('Mean Absolute Error (Lower is Better)')
        ax2.set_ylim(0, max(mae_scores) * 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Color best performing model (lowest MAE)
        best_mae_idx = np.argmin(mae_scores)
        bars2[best_mae_idx].set_color('gold')
        bars2[best_mae_idx].set_edgecolor('darkgoldenrod')
        bars2[best_mae_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = self.experiment_dir / "model_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Model comparison plot saved: {comparison_path}")
    
    def run_analysis(self):
        """Run complete analysis of existing experiment."""
        print(f"🚀 Starting analysis of existing experiment...")
        
        # Load experiment data
        data, existing_summary = self.load_experiment_data()
        
        # Run baseline comparisons
        baseline_results = self.compute_baseline_comparisons(data)
        
        # Run GAM analysis
        gam_results = self.run_gam_analysis()
        
        # Update experiment summary
        updated_summary = self.update_experiment_summary(
            existing_summary, baseline_results, gam_results
        )
        
        # Create comparison visualization
        self.create_comparison_visualization(
            baseline_results, gam_results, updated_summary
        )
        
        # Print final summary
        self.print_analysis_summary(baseline_results, gam_results, updated_summary)
    
    def print_analysis_summary(
        self, 
        baseline_results: Dict, 
        gam_results: Dict, 
        updated_summary: Dict
    ):
        """Print comprehensive analysis summary."""
        print("\n" + "="*80)
        print("🎯 EXISTING EXPERIMENT ANALYSIS COMPLETE!")
        print("="*80)
        
        print("\n📊 BASELINE MODEL RESULTS:")
        print("-" * 40)
        for name, result in baseline_results.items():
            r2 = result['test_metrics']['r2']
            mae = result['test_metrics']['mae']
            print(f"   {name.replace('_', ' ').title()}: R²={r2:.4f}, MAE={mae:.3f}")
            if name == 'best_single_judge':
                print(f"      Best Judge: {result['judge_name']}")
        
        if gam_results:
            print(f"\n🧠 GAM MODEL RESULTS:")
            print("-" * 40)
            print(f"   Best R² Score: {gam_results['best_r2']:.4f}")
            print(f"   Best MAE: {gam_results['best_mae']:.3f}")
            print(f"   Best AIC: {gam_results['best_aic']:.2f}")
            print(f"   Effective DOF: {gam_results['best_edof']:.1f}")
            print(f"   Successful Trials: {gam_results['successful_trials']}")
            
            print(f"\n   Best GAM Configuration:")
            for key, value in gam_results['best_config'].items():
                print(f"      {key}: {value}")
        
        if 'model_comparison' in updated_summary:
            comparison = updated_summary['model_comparison']
            print(f"\n🏆 OVERALL MODEL COMPARISON:")
            print("-" * 40)
            for model, r2 in comparison['all_r2_scores'].items():
                improvement = comparison['improvement_over_baseline'][model]
                status = "👑" if model == comparison['best_model'] else "  "
                print(f"   {status} {model.upper()}: R²={r2:.4f} (+{improvement:.4f} vs baseline)")
            
            print(f"\n🎖️  Best Model: {comparison['best_model'].upper()}")
            print(f"🎯 Best R² Score: {comparison['best_r2']:.4f}")
        
        print(f"\n📁 Results saved to: {self.experiment_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze existing experiment with GAM and baseline comparisons")
    parser.add_argument('--experiment-dir', required=True,
                        help='Path to existing experiment directory')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ExistingExperimentAnalyzer(
        experiment_dir=args.experiment_dir,
        random_seed=args.random_seed
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()