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
from pipeline.core.baseline_models import BaselineEvaluator
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
        """Compute comprehensive baseline model comparisons using unified system."""
        print("🔬 Computing comprehensive baseline comparisons...")
        
        try:
            # Use unified baseline evaluation system
            evaluator = BaselineEvaluator(
                random_seed=self.random_seed,
                test_size=0.2
            )
            
            # Get comprehensive baseline results
            baseline_results = evaluator.evaluate_all_baselines(data)
            
            # Extract key metrics for summary
            baselines = baseline_results['baselines']
            summary = baseline_results['summary']
            
            print(f"✅ Computed {len(baselines)} baseline approaches")
            print(f"   Best baseline: {summary['best_baseline']} (R² = {summary['best_r2']:.4f})")
            print(f"   Samples used: {summary['data_info']['total_samples']}")
            
            # Debug: Print all R² scores
            print(f"\n🔍 Debug: All baseline R² scores:")
            for name, result in baselines.items():
                r2 = result['metrics']['r2']
                print(f"   {name}: {r2:.4f}")
            
            return baseline_results
            
        except Exception as e:
            print(f"❌ Unified baseline system failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to legacy method
            return self._legacy_baseline_comparison(data)
    
    def _legacy_baseline_comparison(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Legacy baseline comparison for fallback."""
        print("⚠️  Using legacy baseline comparison as fallback...")
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
        
        # 1. Naive Mean: Simple average of all judge scores (NO SCALING - truly naive)
        print("📊 Computing Naive Mean baseline...")
        naive_mean_test = np.mean(X_test, axis=1)
        
        # NO SCALING - this is the naive baseline (judges [0-4] vs personas [0-10])
        naive_metrics = compute_metrics(y_test, naive_mean_test)
        baselines['naive_mean'] = {
            'method': 'Simple average of all judge scores (no scaling)',
            'test_metrics': naive_metrics
        }
        
        # 2. Best Single Judge: Find the judge with best proportional scaling performance
        print("🏆 Finding best single judge...")
        judge_metrics = []
        
        for j in range(10):
            judge_scores_test = X_test[:, j]
            
            # Proportional scaling: judges [0-4] to personas [0-10]
            judge_scaled = judge_scores_test * 2.5
            
            metrics = compute_metrics(y_test, judge_scaled)
            judge_metrics.append(metrics)
        
        best_judge_idx = np.argmax([m['r2'] for m in judge_metrics])
        best_judge_name = FEATURE_LABELS[best_judge_idx]
        
        baselines['best_single_judge'] = {
            'method': f'Best single judge with proportional scaling: {best_judge_name}',
            'judge_index': int(best_judge_idx),
            'judge_name': best_judge_name,
            'scaling_factor': 2.5,
            'test_metrics': judge_metrics[best_judge_idx],
            'all_judge_r2': {
                FEATURE_LABELS[i]: float(m['r2']) for i, m in enumerate(judge_metrics)
            }
        }
        
        # 3. Linear Scaling Mean: Proportional scaling from [0-4] to [0-10]
        print("⚖️ Computing linear scaling baseline...")
        judge_mean_test = np.mean(X_test, axis=1)
        
        # Proportional scaling: judges [0-4] to personas [0-10]
        scaled_mean_test = judge_mean_test * 2.5
        
        scaled_metrics = compute_metrics(y_test, scaled_mean_test)
        baselines['linear_scaling_mean'] = {
            'method': 'Mean of judges scaled proportionally from [0-4] to [0-10]',
            'scaling_factor': 2.5,
            'test_metrics': scaled_metrics
        }
        
        # Summary comparison
        print("\n📊 Baseline Comparison Results:")
        print(f"   Naive Mean R²: {naive_metrics['r2']:.4f}")
        print(f"   Best Judge R² ({best_judge_name}): {judge_metrics[best_judge_idx]['r2']:.4f}")
        print(f"   Linear Scaling Mean R²: {scaled_metrics['r2']:.4f}")
        
        return {
            'baselines': baselines,
            'summary': {
                'best_baseline': 'legacy_fallback',
                'methodology': 'legacy_fallback',
                'best_r2': max([b['test_metrics']['r2'] for b in baselines.values()]),
                'data_info': {'total_samples': len(X)}
            }
        }
    
    def run_gam_analysis(self) -> Dict[str, Any]:
        """Run GAM hyperparameter tuning on existing experiment data."""
        print("🧠 Running GAM hyperparameter analysis...")
        
        # Create GAM tuner with experiment data
        # IMPORTANT: Use the same random seed as baseline evaluation to ensure consistency
        gam_tuner = GAMHyperparameterTuner(
            experiment_data_path=str(self.experiment_dir),
            output_dir=str(self.experiment_dir / "gam_analysis"),
            random_seed=self.random_seed
        )
        
        # Run GAM tuning with more trials for thorough analysis
        # Use normalize=True to match the standards used in aggregator training
        gam_analysis = gam_tuner.run_tuning(n_trials=75, normalize=True)
        
        if gam_analysis:
            print(f"✅ GAM analysis complete - Best R²: {gam_analysis['best_r2']:.4f}")
            print(f"   GAM used same random seed ({self.random_seed}) as baselines for consistency")
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
        
        # Extract R2 scores safely from either structure
        def safe_extract_r2(model_dict):
            if isinstance(model_dict, dict):
                # Try 'metrics' first (unified system), then 'test_metrics' (legacy)
                return (model_dict.get('metrics', {}).get('r2', 0) or 
                       model_dict.get('test_metrics', {}).get('r2', 0))
            return 0
        
        # Add baseline analysis (handle unified baseline structure)
        baselines = baseline_results.get('baselines', baseline_results)
        
        existing_summary['baseline_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'baselines': baseline_results,
            'methodology': baseline_results.get('summary', {}).get('methodology', 'unknown'),
            'baseline_comparison': {
                'naive_mean_r2': safe_extract_r2(baselines.get('naive_mean', {})),
                'best_judge_r2': (safe_extract_r2(baselines.get('best_judge_linear_scaling', {})) or
                                safe_extract_r2(baselines.get('best_single_judge', {}))),
                'linear_scaling_mean_r2': safe_extract_r2(baselines.get('linear_scaling_mean', {})),
                'standardscaler_lr_mean_r2': safe_extract_r2(baselines.get('standardscaler_lr_mean', {})),
                'best_baseline_method': baseline_results.get('summary', {}).get('best_baseline', 'unknown')
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
        
        # Add overall comparison (handle both unified and legacy baseline structures)
        all_r2_scores = {}
        
        # Add heuristic baselines
        if baselines.get('linear_scaling_mean'):
            all_r2_scores['10_judge_mean'] = safe_extract_r2(baselines['linear_scaling_mean'])
        if baselines.get('best_judge_linear_scaling'):
            all_r2_scores['best_judge'] = safe_extract_r2(baselines['best_judge_linear_scaling'])
        if baselines.get('ultrafeedback_4judge'):
            all_r2_scores['ultrafeedback_4judge'] = safe_extract_r2(baselines['ultrafeedback_4judge'])
        
        # Add learned baselines
        if baselines.get('standardscaler_lr_mean'):
            all_r2_scores['standardscaler_lr_mean'] = safe_extract_r2(baselines['standardscaler_lr_mean'])
        if baselines.get('best_judge_standardscaler_lr'):
            all_r2_scores['best_judge_lr'] = safe_extract_r2(baselines['best_judge_standardscaler_lr'])
        
        # Remove zero scores to clean up comparison
        all_r2_scores = {k: v for k, v in all_r2_scores.items() if v > 0}
        
        if gam_results:
            all_r2_scores['gam'] = gam_results['best_r2']
        
        # Check if MLP results exist (could be in 'mlp_analysis' or 'optimal_model')
        if 'mlp_analysis' in existing_summary:
            all_r2_scores['mlp'] = existing_summary['mlp_analysis'].get('best_test_r2', 0)
        elif 'optimal_model' in existing_summary:
            all_r2_scores['mlp'] = existing_summary['optimal_model'].get('r2_score', 0)
        
        # Calculate improvement over best heuristic baseline
        heuristic_keys = ['10_judge_mean', 'best_judge', 'ultrafeedback_4judge']
        heuristic_scores = {k: v for k, v in all_r2_scores.items() if k in heuristic_keys}
        best_heuristic_score = max(heuristic_scores.values()) if heuristic_scores else 0
        
        existing_summary['model_comparison'] = {
            'all_r2_scores': all_r2_scores,
            'best_model': max(all_r2_scores.keys(), key=lambda k: all_r2_scores[k]) if all_r2_scores else 'none',
            'best_r2': max(all_r2_scores.values()) if all_r2_scores else 0,
            'best_heuristic_baseline': best_heuristic_score,
            'improvement_over_heuristic': {
                model: all_r2_scores[model] - best_heuristic_score 
                for model in all_r2_scores.keys()
            } if all_r2_scores else {}
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.integer)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # Convert the entire summary
        existing_summary_clean = convert_numpy_types(existing_summary)
        
        # Save updated summary
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(existing_summary_clean, f, indent=2)
        
        print(f"✅ Updated experiment summary: {summary_path}")
        
        return existing_summary
    
    def create_comparison_visualization(
        self, 
        baseline_results: Dict, 
        gam_results: Dict, 
        existing_summary: Dict
    ):
        """Create single comprehensive comparison visualization with heuristic vs learned distinction."""
        print("📊 Creating comprehensive model comparison visualization...")
        
        # Handle both unified and legacy baseline structures
        baselines = baseline_results.get('baselines', baseline_results)
        
        # Create single comprehensive plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Extract models and organize by type
        def safe_extract_metrics(model_dict, metrics_key='metrics'):
            # Try 'metrics' first (unified system), then 'test_metrics' (legacy)
            if 'metrics' in model_dict:
                return model_dict['metrics'].get('r2', 0), model_dict['metrics'].get('mae', 0)
            elif 'test_metrics' in model_dict:
                return model_dict['test_metrics'].get('r2', 0), model_dict['test_metrics'].get('mae', 0)
            return 0, 0
        
        # Organize models by type
        heuristic_models = {}
        learned_models = {}
        
        # Heuristic models (no training)
        heuristic_keys = ['linear_scaling_mean', 'best_judge_linear_scaling', 'ultrafeedback_4judge']
        heuristic_labels = ['10-Judge Mean', 'Best Judge', 'UltraFeedback 4-Judge']
        
        for key, label in zip(heuristic_keys, heuristic_labels):
            if baselines.get(key):
                r2, mae = safe_extract_metrics(baselines[key])
                if r2 != 0:  # Only include valid results
                    heuristic_models[label] = r2
        
        # Learned models (trained parameters)
        if baselines.get('standardscaler_lr_mean'):
            r2, mae = safe_extract_metrics(baselines['standardscaler_lr_mean'])
            learned_models['StandardScaler + LR'] = r2
            
        if baselines.get('best_judge_standardscaler_lr'):
            r2, mae = safe_extract_metrics(baselines['best_judge_standardscaler_lr'])
            learned_models['Best Judge + LR'] = r2
            
        if gam_results and gam_results.get('best_r2', -1) > 0:
            learned_models['GAM'] = gam_results['best_r2']
            
        if 'mlp_analysis' in existing_summary:
            learned_models['MLP'] = existing_summary['mlp_analysis'].get('best_test_r2', 0)
        elif 'optimal_model' in existing_summary:
            learned_models['MLP'] = existing_summary['optimal_model'].get('r2_score', 0)
        
        # Combine all models for plotting
        all_models = list(heuristic_models.keys()) + list(learned_models.keys())
        all_r2s = list(heuristic_models.values()) + list(learned_models.values())
        
        # Create colors: coral for heuristic, steelblue for learned
        colors = ['lightcoral'] * len(heuristic_models) + ['steelblue'] * len(learned_models)
        
        # Create the plot
        bars = ax.bar(all_models, all_r2s, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Styling
        ax.set_title('Model Performance Comparison - R² Score\n(Heuristic vs Learned Approaches)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(all_r2s) * 1.15 if all_r2s else 1)
        
        # Add value labels
        for bar, score in zip(bars, all_r2s):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Highlight best model
        if all_r2s:
            best_idx = np.argmax(all_r2s)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkgoldenrod')
            bars[best_idx].set_linewidth(3)
            
            # Add crown emoji to best model
            best_bar = bars[best_idx]
            ax.text(best_bar.get_x() + best_bar.get_width()/2., best_bar.get_height() + 0.05,
                    '👑', ha='center', va='bottom', fontsize=20)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', alpha=0.8, label='Heuristic Approaches'),
            Patch(facecolor='steelblue', alpha=0.8, label='Learned Models')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, shadow=True)
        
        # Add dividing line between heuristic and learned
        if len(heuristic_models) > 0:
            divider_x = len(heuristic_models) - 0.5
            ax.axvline(x=divider_x, color='gray', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add text labels for sections
            ax.text(len(heuristic_models)/2 - 0.5, max(all_r2s) * 1.05, 'HEURISTIC', 
                   ha='center', va='center', fontweight='bold', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
            ax.text(len(heuristic_models) + len(learned_models)/2 - 0.5, max(all_r2s) * 1.05, 'LEARNED', 
                   ha='center', va='center', fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='steelblue', alpha=0.3))
        
        plt.tight_layout()
        
        # Save comprehensive comparison
        comparison_path = self.experiment_dir / "model_comparison_comprehensive.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comprehensive model comparison plot saved: {comparison_path}")
        
        # Also save as the standard model_comparison.png for compatibility
        simple_path = self.experiment_dir / "model_comparison.png" 
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Recreate the same plot
        bars = ax.bar(all_models, all_r2s, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title('Model Performance Comparison - R² Score\n(Heuristic vs Learned Approaches)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(all_r2s) * 1.15 if all_r2s else 1)
        
        for bar, score in zip(bars, all_r2s):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        if all_r2s:
            best_idx = np.argmax(all_r2s)
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkgoldenrod')
            bars[best_idx].set_linewidth(3)
            best_bar = bars[best_idx]
            ax.text(best_bar.get_x() + best_bar.get_width()/2., best_bar.get_height() + 0.05,
                    '👑', ha='center', va='bottom', fontsize=20)
        
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', alpha=0.8, label='Heuristic Approaches'),
            Patch(facecolor='steelblue', alpha=0.8, label='Learned Models')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, shadow=True)
        
        if len(heuristic_models) > 0:
            divider_x = len(heuristic_models) - 0.5
            ax.axvline(x=divider_x, color='gray', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(len(heuristic_models)/2 - 0.5, max(all_r2s) * 1.05, 'HEURISTIC', 
                   ha='center', va='center', fontweight='bold', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
            ax.text(len(heuristic_models) + len(learned_models)/2 - 0.5, max(all_r2s) * 1.05, 'LEARNED', 
                   ha='center', va='center', fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='steelblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(simple_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Standard model comparison plot saved: {simple_path}")
    
    
    def run_analysis(self):
        """Run complete analysis of existing experiment."""
        print(f"🚀 Starting analysis of existing experiment...")
        print(f"🎯 Using random seed {self.random_seed} for all components to ensure consistency")
        
        # Load experiment data
        data, existing_summary = self.load_experiment_data()
        
        # Run baseline comparisons with unified system
        print(f"\n📊 Phase 1: Computing baselines with unified evaluation system...")
        baseline_results = self.compute_baseline_comparisons(data)
        
        # Run GAM analysis (uses same sampling methodology)
        print(f"\n🧠 Phase 2: Running GAM analysis with consistent sampling...")
        gam_results = self.run_gam_analysis()
        
        # Update experiment summary
        print(f"\n📝 Phase 3: Updating experiment summary...")
        updated_summary = self.update_experiment_summary(
            existing_summary, baseline_results, gam_results
        )
        
        # Create comparison visualization
        print(f"\n📊 Phase 4: Creating visualizations...")
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
        
        # Handle both unified and legacy baseline structures
        baselines = baseline_results.get('baselines', baseline_results)
        
        for name, result in baselines.items():
            # Try to get metrics from either metrics (unified) or test_metrics (legacy)
            metrics = result.get('metrics', result.get('test_metrics', {}))
            r2 = metrics.get('r2', 0)
            mae = metrics.get('mae', 0)
            print(f"   {name.replace('_', ' ').title()}: R²={r2:.4f}, MAE={mae:.3f}")
            if name in ['best_single_judge', 'best_judge_linear_scaling'] and 'judge_name' in result:
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
            
            # Separate heuristic and learned models
            heuristic_keys = ['10_judge_mean', 'best_judge', 'ultrafeedback_4judge']
            learned_keys = [k for k in comparison['all_r2_scores'].keys() if k not in heuristic_keys]
            
            print("   HEURISTIC APPROACHES:")
            for model in heuristic_keys:
                if model in comparison['all_r2_scores']:
                    r2 = comparison['all_r2_scores'][model]
                    status = "👑" if model == comparison['best_model'] else "  "
                    print(f"   {status} {model.replace('_', ' ').title()}: R²={r2:.4f}")
            
            print("\n   LEARNED MODELS:")
            for model in learned_keys:
                if model in comparison['all_r2_scores']:
                    r2 = comparison['all_r2_scores'][model]
                    improvement = comparison['improvement_over_heuristic'][model]
                    status = "👑" if model == comparison['best_model'] else "  "
                    print(f"   {status} {model.replace('_', ' ').upper()}: R²={r2:.4f} (+{improvement:.4f} vs best heuristic)")
            
            print(f"\n🎖️  Best Model: {comparison['best_model'].replace('_', ' ').upper()}")
            print(f"🎯 Best R² Score: {comparison['best_r2']:.4f}")
            print(f"📈 Best Heuristic Baseline: {comparison['best_heuristic_baseline']:.4f}")
        
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