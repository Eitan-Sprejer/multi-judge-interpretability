#!/usr/bin/env python3
"""
GAM Stability Analysis

Analyzes the stability of GAM feature importance and partial dependence by:
1. Training multiple GAM models with slightly different configurations
2. Computing mean and standard deviation of feature importance scores
3. Analyzing stability of partial dependence curves
4. Identifying which features have stable vs. unstable relationships

This helps determine which interpretability insights are robust vs. potentially spurious.
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
from scipy import stats

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


class GAMStabilityAnalyzer:
    """
    Analyzes stability of GAM interpretability features across multiple training runs.
    """
    
    def __init__(
        self,
        experiment_dir: str,
        n_stability_runs: int = 20,
        random_seed: int = 42
    ):
        self.experiment_dir = Path(experiment_dir)
        self.n_stability_runs = n_stability_runs
        self.random_seed = random_seed
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.experiment_dir / f"gam_stability_analysis_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 GAM stability analysis output: {self.output_dir}")
    
    def load_best_gam_config(self) -> Dict:
        """Load the best GAM configuration from previous analysis."""
        gam_dirs = list(self.experiment_dir.glob("gam_analysis/gam_tuning_run_*"))
        if not gam_dirs:
            raise FileNotFoundError("No GAM analysis found. Run analyze_existing_experiment.py first.")
        
        # Use most recent GAM analysis
        latest_gam_dir = max(gam_dirs, key=lambda p: p.stat().st_mtime)
        summary_path = latest_gam_dir / "gam_analysis_summary.json"
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        best_config = summary['best_config']
        print(f"📊 Using best GAM config: splines={best_config['n_splines']}, λ={best_config['lam']}")
        return best_config
    
    def load_experiment_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load experiment data with consistent persona assignment."""
        data_path = self.experiment_dir / "data" / "data_with_judge_scores.pkl"
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Prepare training data with FIXED seed for consistent persona assignment
        X_list = []
        y_list = []
        
        # Use fixed seed for persona assignment to ensure same data across all stability runs
        random.seed(self.random_seed)
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
        
        return np.array(X_list), np.array(y_list)
    
    def create_gam_variants(self, base_config: Dict) -> List[Dict]:
        """Create slight variations of the best GAM config for stability testing."""
        variants = []
        
        # Base configuration (exact best)
        variants.append(base_config.copy())
        
        # Lambda variations (±20% around best)
        base_lam = base_config['lam']
        for lam_factor in [0.8, 0.9, 1.1, 1.2]:
            variant = base_config.copy()
            variant['lam'] = base_lam * lam_factor
            variants.append(variant)
        
        # Spline variations (±1 around best)
        base_splines = base_config['n_splines']
        for spline_delta in [-2, -1, 1, 2]:
            new_splines = max(5, base_splines + spline_delta)  # Minimum 5 splines
            if new_splines <= 20:  # Maximum reasonable
                variant = base_config.copy()
                variant['n_splines'] = new_splines
                variants.append(variant)
        
        # Tolerance variations
        base_tol = base_config['tol']
        for tol_factor in [0.1, 10.0]:  # More/less strict convergence
            variant = base_config.copy()
            variant['tol'] = base_tol * tol_factor
            variants.append(variant)
        
        # Different interaction patterns (if the best had interactions)
        if base_config['interaction_features']:
            # Version without interactions
            variant = base_config.copy()
            variant['interaction_features'] = []
            variants.append(variant)
            
            # Alternative interaction patterns
            alternative_interactions = [
                [(0, 2)],  # Truthfulness & Helpfulness
                [(1, 2)],  # Harmlessness & Helpfulness  
                [(6, 7)],  # Clarity & Conciseness
                [(8, 9)],  # Logic & Creativity
            ]
            for interaction in alternative_interactions:
                if interaction != base_config['interaction_features']:
                    variant = base_config.copy()
                    variant['interaction_features'] = interaction
                    variants.append(variant)
        
        # Limit to requested number of runs
        if len(variants) > self.n_stability_runs:
            # Keep base config and randomly sample others
            selected = [variants[0]]  # Always include base
            selected.extend(random.sample(variants[1:], self.n_stability_runs - 1))
            variants = selected
        
        print(f"🔧 Created {len(variants)} GAM variants for stability analysis")
        return variants
    
    def create_gam_model(self, config: Dict) -> LinearGAM:
        """Create GAM model with specified configuration."""
        # Build terms for each feature
        terms = []
        
        # Add individual spline terms for each judge
        for i in range(10):  # 10 judges
            terms.append(s(i, n_splines=config['n_splines'], lam=config['lam']))
        
        # Add interaction terms if specified
        for interaction in config['interaction_features']:
            if len(interaction) == 2:
                terms.append(te(interaction[0], interaction[1], 
                               n_splines=max(5, config['n_splines']//2), 
                               lam=config['lam']))
        
        # Combine all terms
        if len(terms) == 1:
            gam_terms = terms[0]
        else:
            gam_terms = terms[0]
            for term in terms[1:]:
                gam_terms = gam_terms + term
        
        # Create GAM model
        gam = LinearGAM(
            gam_terms,
            fit_intercept=True,
            max_iter=config['max_iter'],
            tol=config['tol']
        )
        
        return gam
    
    def run_stability_analysis(self) -> Dict[str, Any]:
        """Run stability analysis across multiple GAM configurations."""
        print(f"🚀 Starting GAM stability analysis with {self.n_stability_runs} runs")
        
        # Load data and best config
        X, y = self.load_experiment_data()
        base_config = self.load_best_gam_config()
        variants = self.create_gam_variants(base_config)
        
        # Split data once (consistent across all runs) - same as baseline analysis
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        
        # Normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Run multiple GAM fits
        results = []
        feature_importance_runs = []
        partial_dependence_runs = []
        
        for run_idx, config in enumerate(variants):
            print(f"Run {run_idx + 1}/{len(variants)}: splines={config['n_splines']}, λ={config['lam']:.2f}")
            
            try:
                # Fit GAM
                gam = self.create_gam_model(config)
                gam.fit(X_train_scaled, y_train)
                
                # Compute metrics
                test_pred = gam.predict(X_test_scaled)
                r2 = r2_score(y_test, test_pred)
                mae = mean_absolute_error(y_test, test_pred)
                
                # Feature importance
                try:
                    p_values = gam.statistics_['p_values']
                    importance = {}
                    for i, label in enumerate(FEATURE_LABELS):
                        if i < len(p_values):
                            importance[label] = max(0, 1.0 - p_values[i])
                        else:
                            importance[label] = 0.0
                    feature_importance_runs.append(importance)
                except:
                    feature_importance_runs.append({label: 0.0 for label in FEATURE_LABELS})
                
                # Partial dependence curves
                pd_curves = {}
                for i in range(10):
                    try:
                        XX = gam.generate_X_grid(term=i, meshgrid=False)
                        x_values = XX[:, i]
                        y_values = gam.partial_dependence(term=i, X=XX)
                        
                        # Standardize x-values to [-2, 2] range for comparison
                        x_std = (x_values - np.mean(x_values)) / (np.std(x_values) + 1e-8)
                        pd_curves[FEATURE_LABELS[i]] = {
                            'x': x_std,
                            'y': y_values
                        }
                    except:
                        pd_curves[FEATURE_LABELS[i]] = {'x': np.array([]), 'y': np.array([])}
                
                partial_dependence_runs.append(pd_curves)
                
                results.append({
                    'config': config,
                    'r2': r2,
                    'mae': mae,
                    'success': True
                })
                
            except Exception as e:
                print(f"  ❌ Run failed: {e}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'success': False
                })
        
        print(f"✅ Completed {sum(1 for r in results if r['success'])}/{len(results)} successful runs")
        
        # Analyze stability
        stability_analysis = self.analyze_stability(
            feature_importance_runs, partial_dependence_runs, results
        )
        
        return stability_analysis
    
    def analyze_stability(
        self, 
        importance_runs: List[Dict], 
        pd_runs: List[Dict], 
        results: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze stability of feature importance and partial dependence."""
        
        # Feature importance stability
        importance_stats = {}
        for feature in FEATURE_LABELS:
            values = [run[feature] for run in importance_runs if feature in run]
            if values:
                importance_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / (np.mean(values) + 1e-8),  # Coefficient of variation
                    'min': np.min(values),
                    'max': np.max(values),
                    'runs': len(values)
                }
        
        # Partial dependence stability (simplified: analyze correlation between curves)
        pd_stability = {}
        for feature in FEATURE_LABELS:
            correlations = []
            
            # Extract all curves for this feature
            curves = []
            for run in pd_runs:
                if feature in run and len(run[feature]['y']) > 0:
                    curves.append(run[feature]['y'])
            
            if len(curves) >= 2:
                # Compute pairwise correlations between curves
                for i in range(len(curves)):
                    for j in range(i+1, len(curves)):
                        if len(curves[i]) == len(curves[j]):
                            corr = np.corrcoef(curves[i], curves[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                if correlations:
                    pd_stability[feature] = {
                        'mean_correlation': np.mean(correlations),
                        'std_correlation': np.std(correlations),
                        'min_correlation': np.min(correlations),
                        'n_comparisons': len(correlations),
                        'stability_score': np.mean(correlations)  # Higher = more stable
                    }
        
        # Performance stability
        successful_results = [r for r in results if r['success']]
        r2_values = [r['r2'] for r in successful_results]
        mae_values = [r['mae'] for r in successful_results]
        
        performance_stats = {
            'r2': {
                'mean': np.mean(r2_values),
                'std': np.std(r2_values),
                'cv': np.std(r2_values) / np.mean(r2_values),
                'range': np.max(r2_values) - np.min(r2_values)
            },
            'mae': {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values),
                'cv': np.std(mae_values) / np.mean(mae_values),
                'range': np.max(mae_values) - np.min(mae_values)
            }
        }
        
        # Reliability scores: combine high importance with low variability
        reliability_scores = {}
        for feature in FEATURE_LABELS:
            if feature in importance_stats:
                stats = importance_stats[feature]
                mean_importance = stats['mean']
                cv = stats['cv']
                
                # Reliability = high importance AND low variability
                # Penalize both low importance and high variability
                reliability = mean_importance * (1 - min(cv, 1.0))  # CV capped at 1.0
                reliability_scores[feature] = {
                    'reliability_score': reliability,
                    'mean_importance': mean_importance,
                    'cv': cv,
                    'interpretation': 'reliable' if cv < 0.1 and mean_importance > 0.7 else 
                                   'moderate' if cv < 0.2 and mean_importance > 0.5 else 'unreliable'
                }
        
        # Rankings
        importance_stability_ranking = sorted(
            importance_stats.items(),
            key=lambda x: x[1]['cv']  # Lower CV = more stable
        )
        
        reliability_ranking = sorted(
            reliability_scores.items(),
            key=lambda x: x[1]['reliability_score'],
            reverse=True  # Higher reliability = better
        )
        
        pd_stability_ranking = sorted(
            pd_stability.items(),
            key=lambda x: x[1]['stability_score'],
            reverse=True  # Higher correlation = more stable
        )
        
        analysis = {
            'feature_importance_stability': importance_stats,
            'partial_dependence_stability': pd_stability,
            'performance_stability': performance_stats,
            'reliability_scores': reliability_scores,
            'rankings': {
                'most_stable_importance': [item[0] for item in importance_stability_ranking[:5]],
                'least_stable_importance': [item[0] for item in importance_stability_ranking[-5:]],
                'most_reliable_features': [item[0] for item in reliability_ranking[:5]],
                'least_reliable_features': [item[0] for item in reliability_ranking[-5:]],
                'most_stable_partial_dependence': [item[0] for item in pd_stability_ranking[:5]],
                'least_stable_partial_dependence': [item[0] for item in pd_stability_ranking[-5:]]
            },
            'summary': {
                'n_successful_runs': len(successful_results),
                'performance_is_stable': performance_stats['r2']['cv'] < 0.05,  # CV < 5%
                'reliable_features': [
                    f for f, scores in reliability_scores.items() 
                    if scores['interpretation'] == 'reliable'
                ],
                'moderate_features': [
                    f for f, scores in reliability_scores.items() 
                    if scores['interpretation'] == 'moderate'
                ],
                'unreliable_features': [
                    f for f, scores in reliability_scores.items() 
                    if scores['interpretation'] == 'unreliable'
                ]
            }
        }
        
        return analysis
    
    def create_stability_visualizations(self, analysis: Dict):
        """Create visualizations for stability analysis."""
        
        # 1. Feature importance stability plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Feature importance means and error bars
        features = list(analysis['feature_importance_stability'].keys())
        means = [analysis['feature_importance_stability'][f]['mean'] for f in features]
        stds = [analysis['feature_importance_stability'][f]['std'] for f in features]
        
        bars = ax1.bar(range(len(features)), means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_title('Feature Importance Stability\n(Mean ± Std across runs)', fontsize=12)
        ax1.set_ylabel('Feature Importance')
        ax1.set_xticks(range(len(features)))
        ax1.set_xticklabels([f.replace(' / ', '/\n') for f in features], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by stability (CV)
        cvs = [analysis['feature_importance_stability'][f]['cv'] for f in features]
        for bar, cv in zip(bars, cvs):
            if cv < 0.1:
                bar.set_color('green')  # Very stable
            elif cv < 0.2:
                bar.set_color('orange')  # Moderately stable
            else:
                bar.set_color('red')  # Unstable
        
        # Coefficient of variation plot
        ax2.bar(range(len(features)), cvs, alpha=0.7, color='lightblue')
        ax2.set_title('Feature Importance Stability\n(Coefficient of Variation)', fontsize=12)
        ax2.set_ylabel('CV (Std/Mean)')
        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels([f.replace(' / ', '/\n') for f in features], rotation=45, ha='right')
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Partial dependence stability
        if analysis['partial_dependence_stability']:
            pd_features = list(analysis['partial_dependence_stability'].keys())
            stability_scores = [analysis['partial_dependence_stability'][f]['stability_score'] for f in pd_features]
            
            bars3 = ax3.bar(range(len(pd_features)), stability_scores, alpha=0.7, color='lightgreen')
            ax3.set_title('Partial Dependence Stability\n(Mean correlation between curves)', fontsize=12)
            ax3.set_ylabel('Stability Score (Correlation)')
            ax3.set_xticks(range(len(pd_features)))
            ax3.set_xticklabels([f.replace(' / ', '/\n') for f in pd_features], rotation=45, ha='right')
            ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High stability')
            ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate stability')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Performance stability
        perf_metrics = ['R²', 'MAE']
        perf_cvs = [
            analysis['performance_stability']['r2']['cv'],
            analysis['performance_stability']['mae']['cv']
        ]
        
        bars4 = ax4.bar(perf_metrics, perf_cvs, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax4.set_title('Performance Stability\n(Coefficient of Variation)', fontsize=12)
        ax4.set_ylabel('CV (Std/Mean)')
        ax4.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='High stability (<5%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "gam_stability_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Stability visualization saved: {plot_path}")
    
    def save_results(self, analysis: Dict):
        """Save stability analysis results."""
        # Save detailed results
        results_path = self.output_dir / "stability_analysis.json"
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, np.number):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                else:
                    return obj
            
            json.dump(convert_numpy(analysis), f, indent=2)
        
        print(f"💾 Stability analysis saved: {results_path}")
    
    def print_summary(self, analysis: Dict):
        """Print comprehensive stability analysis summary focused on feature reliability."""
        print("\n" + "="*80)
        print("🔬 GAM FEATURE IMPORTANCE RELIABILITY ANALYSIS")
        print("="*80)
        
        # Performance stability
        perf = analysis['performance_stability']
        print(f"\n📊 PERFORMANCE STABILITY:")
        print(f"   R² Mean: {perf['r2']['mean']:.4f} ± {perf['r2']['std']:.4f} (CV: {perf['r2']['cv']:.3f})")
        
        stability_status = "🟢 STABLE" if perf['r2']['cv'] < 0.05 else "🟡 MODERATE" if perf['r2']['cv'] < 0.1 else "🔴 UNSTABLE"
        print(f"   Overall Performance: {stability_status}")
        
        # Feature reliability ranking
        print(f"\n🏆 MOST RELIABLE FEATURES (High importance + Low variability):")
        for i, feature in enumerate(analysis['rankings']['most_reliable_features'][:5]):
            reliability = analysis['reliability_scores'][feature]
            importance = reliability['mean_importance']
            cv = reliability['cv']
            score = reliability['reliability_score']
            interpretation = reliability['interpretation']
            
            emoji = "🟢" if interpretation == 'reliable' else "🟡" if interpretation == 'moderate' else "🔴"
            print(f"   {i+1}. {emoji} {feature}:")
            print(f"      Reliability Score: {score:.3f}")
            print(f"      Mean Importance: {importance:.3f} (CV: {cv:.3f})")
        
        print(f"\n⚠️  LEAST RELIABLE FEATURES:")
        for i, feature in enumerate(analysis['rankings']['least_reliable_features'][-3:]):
            reliability = analysis['reliability_scores'][feature]
            importance = reliability['mean_importance']
            cv = reliability['cv']
            score = reliability['reliability_score']
            interpretation = reliability['interpretation']
            
            emoji = "🟢" if interpretation == 'reliable' else "🟡" if interpretation == 'moderate' else "🔴"
            print(f"   {i+1}. {emoji} {feature}:")
            print(f"      Reliability Score: {score:.3f}")
            print(f"      Mean Importance: {importance:.3f} (CV: {cv:.3f})")
        
        # Categorized features
        reliable = analysis['summary']['reliable_features']
        moderate = analysis['summary']['moderate_features'] 
        unreliable = analysis['summary']['unreliable_features']
        
        print(f"\n📊 FEATURE RELIABILITY SUMMARY:")
        print(f"   🟢 RELIABLE ({len(reliable)} features): Can trust for interpretation")
        for feature in reliable:
            score = analysis['reliability_scores'][feature]['reliability_score']
            print(f"      • {feature} (Score: {score:.3f})")
        
        print(f"\n   🟡 MODERATE ({len(moderate)} features): Use with caution")
        for feature in moderate:
            score = analysis['reliability_scores'][feature]['reliability_score']
            print(f"      • {feature} (Score: {score:.3f})")
        
        print(f"\n   🔴 UNRELIABLE ({len(unreliable)} features): High uncertainty")
        for feature in unreliable:
            score = analysis['reliability_scores'][feature]['reliability_score']
            print(f"      • {feature} (Score: {score:.3f})")
        
        print(f"\n💡 INTERPRETATION GUIDANCE:")
        print(f"   • Reliable features: Consistent high importance across model variants")
        print(f"   • Moderate features: Reasonably stable but interpret with confidence intervals") 
        print(f"   • Unreliable features: Importance varies significantly - avoid strong claims")
        
        print(f"\n📋 SUMMARY:")
        print(f"   • Successful runs: {analysis['summary']['n_successful_runs']}")
        print(f"   • Performance stable: {analysis['summary']['performance_is_stable']}")
        print(f"   • Can trust: {len(reliable)}/10 features for interpretation")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="GAM stability analysis")
    parser.add_argument('--experiment-dir', required=True,
                        help='Path to experiment directory')
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Number of stability runs')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run stability analysis
    analyzer = GAMStabilityAnalyzer(
        experiment_dir=args.experiment_dir,
        n_stability_runs=args.n_runs,
        random_seed=args.random_seed
    )
    
    analysis = analyzer.run_stability_analysis()
    analyzer.create_stability_visualizations(analysis)
    analyzer.save_results(analysis)
    analyzer.print_summary(analysis)


if __name__ == "__main__":
    main()