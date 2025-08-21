#!/usr/bin/env python3
"""
Experiment Runner for Experiment 2b: Aggregator Validation with Less Varied Data

Runs all 11 experiments (mixed personas, ultrafeedback, 8 individual personas, persona mean)
and collects comprehensive results for analysis.
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pipeline.core.persona_simulation import PERSONAS
from data_preparation import load_experiment_data, prepare_all_targets, validate_data_alignment
from training_functions import train_both_models, validate_results_consistency


class ExperimentRunner:
    """
    Runs all experiments for the aggregator validation study.
    """
    
    def __init__(
        self,
        data_path: str,
        output_dir: str = None,
        test_size: float = 0.2,
        random_seed: int = 42,
        normalize: bool = True
    ):
        """
        Initialize experiment runner.
        
        Args:
            data_path: Path to the experiment data with judge scores and ultrafeedback
            output_dir: Output directory for results (default: auto-generated)
            test_size: Test set fraction
            random_seed: Random seed for reproducibility
            normalize: Whether to normalize features
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_seed = random_seed
        self.normalize = normalize
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/experiment_2b_{timestamp}"
        
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Experiment 2b: Aggregator Validation")
        print(f"   Data: {data_path}")
        print(f"   Output: {self.output_dir}")
        print(f"   Settings: test_size={test_size}, seed={random_seed}, normalize={normalize}")
        
        # Store configuration
        self.config = {
            'data_path': data_path,
            'test_size': test_size,
            'random_seed': random_seed,
            'normalize': normalize,
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'aggregator_validation_varied_data'
        }
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_and_prepare_data(self) -> Dict[str, np.ndarray]:
        """Load and prepare all data targets."""
        print("\n📂 Loading and preparing data...")
        
        # Load raw data
        data = load_experiment_data(self.data_path)
        
        # Prepare all targets
        targets = prepare_all_targets(data, self.random_seed)
        
        # Validate data alignment
        validate_data_alignment(targets)
        
        # Save prepared data
        data_path = self.output_dir / "prepared_targets.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(targets, f)
        
        print(f"✅ Data prepared and saved to {data_path}")
        return targets
    
    def run_mixed_personas_experiment(self, X: np.ndarray, y_mixed: np.ndarray) -> Dict[str, Any]:
        """Run experiment with mixed personas (baseline replication)."""
        print("\n🎭 Running Mixed Personas Experiment (Baseline)...")
        
        results = train_both_models(
            X, y_mixed, 
            target_name="Mixed Personas",
            test_size=self.test_size,
            random_seed=self.random_seed,
            normalize=self.normalize
        )
        
        # Save results
        results_path = self.output_dir / "mixed_personas_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def run_ultrafeedback_experiment(self, X: np.ndarray, y_ultrafeedback: np.ndarray) -> Dict[str, Any]:
        """Run experiment with UltraFeedback scores."""
        print("\n🌟 Running UltraFeedback Experiment...")
        
        results = train_both_models(
            X, y_ultrafeedback,
            target_name="UltraFeedback",
            test_size=self.test_size,
            random_seed=self.random_seed,
            normalize=self.normalize
        )
        
        # Save results
        results_path = self.output_dir / "ultrafeedback_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def run_individual_persona_experiments(self, X: np.ndarray, y_personas: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run experiments for each individual persona."""
        print(f"\n👤 Running Individual Persona Experiments ({len(y_personas)} personas)...")
        
        individual_results = {}
        
        for i, (persona_name, y_persona) in enumerate(y_personas.items(), 1):
            print(f"\n  [{i}/{len(y_personas)}] {persona_name}...")
            
            # Run experiment for this persona
            results = train_both_models(
                X, y_persona,
                target_name=f"{persona_name} Persona",
                test_size=self.test_size,
                random_seed=self.random_seed,
                normalize=self.normalize
            )
            
            individual_results[persona_name] = results
            
            # Save individual results
            results_path = self.output_dir / f"persona_{persona_name.lower().replace(' ', '_')}_results.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
        
        # Calculate mean metrics across all personas
        gam_r2_scores = []
        mlp_r2_scores = []
        variances = []
        
        for persona_name, results in individual_results.items():
            if 'summary' in results:
                gam_r2_scores.append(results['summary']['gam_r2'])
                mlp_r2_scores.append(results['summary']['mlp_r2'])
                variances.append(results['data_stats']['target_variance'])
        
        mean_metrics = {
            'gam_r2_mean': float(np.mean(gam_r2_scores)),
            'gam_r2_std': float(np.std(gam_r2_scores)),
            'mlp_r2_mean': float(np.mean(mlp_r2_scores)),
            'mlp_r2_std': float(np.std(mlp_r2_scores)),
            'variance_mean': float(np.mean(variances)),
            'variance_std': float(np.std(variances)),
            'n_personas': len(individual_results),
            'best_persona_gam': max(individual_results.items(), key=lambda x: x[1]['summary']['gam_r2'])[0],
            'worst_persona_gam': min(individual_results.items(), key=lambda x: x[1]['summary']['gam_r2'])[0],
            'best_persona_mlp': max(individual_results.items(), key=lambda x: x[1]['summary']['mlp_r2'])[0],
            'worst_persona_mlp': min(individual_results.items(), key=lambda x: x[1]['summary']['mlp_r2'])[0]
        }
        
        # Combine individual results with summary
        combined_results = {
            'individual_results': individual_results,
            'mean_metrics': mean_metrics
        }
        
        # Save combined results
        results_path = self.output_dir / "individual_personas_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(combined_results, f)
        
        print(f"\n✅ Individual persona experiments complete!")
        print(f"   Mean GAM R²: {mean_metrics['gam_r2_mean']:.4f} ± {mean_metrics['gam_r2_std']:.4f}")
        print(f"   Mean MLP R²: {mean_metrics['mlp_r2_mean']:.4f} ± {mean_metrics['mlp_r2_std']:.4f}")
        print(f"   Best GAM persona: {mean_metrics['best_persona_gam']}")
        print(f"   Best MLP persona: {mean_metrics['best_persona_mlp']}")
        
        return combined_results
    
    def run_persona_mean_experiment(self, X: np.ndarray, y_persona_mean: np.ndarray) -> Dict[str, Any]:
        """Run experiment with mean of all persona scores (bonus experiment)."""
        print("\n📊 Running Persona Mean Experiment (Bonus)...")
        
        results = train_both_models(
            X, y_persona_mean,
            target_name="Persona Mean",
            test_size=self.test_size,
            random_seed=self.random_seed,
            normalize=self.normalize
        )
        
        # Save results
        results_path = self.output_dir / "persona_mean_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all 11 experiments and collect comprehensive results."""
        print("\n🎯 Running All Experiments...")
        
        # Load and prepare data
        targets = self.load_and_prepare_data()
        
        X = targets['X']
        
        # Run all experiments
        results = {}
        
        # 1. Mixed personas (baseline)
        results['mixed_personas'] = self.run_mixed_personas_experiment(X, targets['y_mixed'])
        
        # 2. UltraFeedback
        results['ultrafeedback'] = self.run_ultrafeedback_experiment(X, targets['y_ultrafeedback'])
        
        # 3. Individual personas (8 experiments)
        results['individual_personas'] = self.run_individual_persona_experiments(X, targets['y_personas'])
        
        # 4. Persona mean (bonus)
        results['persona_mean'] = self.run_persona_mean_experiment(X, targets['y_persona_mean'])
        
        # Create summary comparison
        summary = self.create_experiment_summary(results)
        results['summary'] = summary
        
        # Save all results
        results_path = self.output_dir / "all_experiments_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as JSON
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 All experiments complete!")
        print(f"   Results saved to: {self.output_dir}")
        
        return results
    
    def create_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive summary of all experiments."""
        print("\n📋 Creating experiment summary...")
        
        summary = {
            'experiment_info': self.config,
            'main_comparison': {},
            'individual_persona_analysis': {},
            'variance_analysis': {},
            'hypothesis_validation': {}
        }
        
        # Main comparison (for the primary plot)
        main_comparison = {}
        
        # Mixed personas
        if 'summary' in results['mixed_personas']:
            main_comparison['mixed_personas'] = {
                'gam_r2': results['mixed_personas']['summary']['gam_r2'],
                'mlp_r2': results['mixed_personas']['summary']['mlp_r2'],
                'variance': results['mixed_personas']['data_stats']['target_variance']
            }
        
        # UltraFeedback
        if 'summary' in results['ultrafeedback']:
            main_comparison['ultrafeedback'] = {
                'gam_r2': results['ultrafeedback']['summary']['gam_r2'],
                'mlp_r2': results['ultrafeedback']['summary']['mlp_r2'],
                'variance': results['ultrafeedback']['data_stats']['target_variance']
            }
        
        # Individual personas (mean)
        if 'mean_metrics' in results['individual_personas']:
            mean_metrics = results['individual_personas']['mean_metrics']
            main_comparison['individual_personas_mean'] = {
                'gam_r2': mean_metrics['gam_r2_mean'],
                'mlp_r2': mean_metrics['mlp_r2_mean'],
                'variance': mean_metrics['variance_mean']
            }
        
        # Persona mean (bonus)
        if 'summary' in results['persona_mean']:
            main_comparison['persona_mean'] = {
                'gam_r2': results['persona_mean']['summary']['gam_r2'],
                'mlp_r2': results['persona_mean']['summary']['mlp_r2'],
                'variance': results['persona_mean']['data_stats']['target_variance']
            }
        
        summary['main_comparison'] = main_comparison
        
        # Individual persona analysis
        if 'individual_results' in results['individual_personas']:
            individual_analysis = {}
            for persona_name, persona_results in results['individual_personas']['individual_results'].items():
                if 'summary' in persona_results:
                    individual_analysis[persona_name] = {
                        'gam_r2': persona_results['summary']['gam_r2'],
                        'mlp_r2': persona_results['summary']['mlp_r2'],
                        'variance': persona_results['data_stats']['target_variance'],
                        'best_model': persona_results['summary']['best_model']
                    }
            
            summary['individual_persona_analysis'] = individual_analysis
        
        # Variance analysis (for hypothesis testing)
        variance_data = []
        r2_data = []
        names = []
        
        for exp_name, exp_data in main_comparison.items():
            variance_data.append(exp_data['variance'])
            r2_data.append(max(exp_data['gam_r2'], exp_data['mlp_r2']))  # Best R²
            names.append(exp_name)
        
        # Calculate correlation between variance and R²
        if len(variance_data) > 1:
            correlation = float(np.corrcoef(variance_data, r2_data)[0, 1])
        else:
            correlation = 0.0
        
        summary['variance_analysis'] = {
            'variance_vs_r2_correlation': correlation,
            'variance_range': [float(min(variance_data)), float(max(variance_data))],
            'r2_range': [float(min(r2_data)), float(max(r2_data))],
            'experiments': names
        }
        
        # Hypothesis validation
        hypothesis_results = {}
        
        # H1: UltraFeedback R² > 0.70
        uf_r2 = max(main_comparison['ultrafeedback']['gam_r2'], main_comparison['ultrafeedback']['mlp_r2'])
        hypothesis_results['h1_ultrafeedback_high_r2'] = {
            'hypothesis': 'UltraFeedback R² > 0.70',
            'result': float(uf_r2),
            'passed': uf_r2 > 0.70
        }
        
        # H2: Individual personas mean R² > 0.65
        ind_r2 = max(main_comparison['individual_personas_mean']['gam_r2'], 
                    main_comparison['individual_personas_mean']['mlp_r2'])
        hypothesis_results['h2_individual_personas_high_r2'] = {
            'hypothesis': 'Individual personas mean R² > 0.65',
            'result': float(ind_r2),
            'passed': ind_r2 > 0.65
        }
        
        # H3: Variance ordering: Individual < UltraFeedback < Mixed
        individual_var = main_comparison['individual_personas_mean']['variance']
        uf_var = main_comparison['ultrafeedback']['variance']
        mixed_var = main_comparison['mixed_personas']['variance']
        
        hypothesis_results['h3_variance_ordering'] = {
            'hypothesis': 'Variance order: Individual < UltraFeedback < Mixed',
            'individual_variance': float(individual_var),
            'ultrafeedback_variance': float(uf_var),
            'mixed_variance': float(mixed_var),
            'passed': individual_var < uf_var < mixed_var
        }
        
        summary['hypothesis_validation'] = hypothesis_results
        
        # Print summary
        print(f"📊 Experiment Summary:")
        print(f"   Mixed Personas R²: {main_comparison['mixed_personas']['gam_r2']:.4f} / {main_comparison['mixed_personas']['mlp_r2']:.4f}")
        print(f"   UltraFeedback R²: {main_comparison['ultrafeedback']['gam_r2']:.4f} / {main_comparison['ultrafeedback']['mlp_r2']:.4f}")
        print(f"   Individual Mean R²: {main_comparison['individual_personas_mean']['gam_r2']:.4f} / {main_comparison['individual_personas_mean']['mlp_r2']:.4f}")
        print(f"   Variance correlation: {correlation:.3f}")
        print(f"   H1 (UF > 0.70): {'✅' if hypothesis_results['h1_ultrafeedback_high_r2']['passed'] else '❌'}")
        print(f"   H2 (Ind > 0.65): {'✅' if hypothesis_results['h2_individual_personas_high_r2']['passed'] else '❌'}")
        print(f"   H3 (Var order): {'✅' if hypothesis_results['h3_variance_ordering']['passed'] else '❌'}")
        
        return summary


if __name__ == "__main__":
    # Example usage
    data_path = "/Users/eitu/Documents/Eitu/AI Safety/AIS_hackathons/model_routing/multi-judge-interpretability/results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023/data/data_with_judge_scores_and_ultrafeedback.pkl"
    
    # Create experiment runner
    runner = ExperimentRunner(
        data_path=data_path,
        output_dir="test_run",
        test_size=0.2,
        random_seed=42,
        normalize=True
    )
    
    # Run all experiments
    results = runner.run_all_experiments()