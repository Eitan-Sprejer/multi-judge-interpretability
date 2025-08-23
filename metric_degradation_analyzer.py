#!/usr/bin/env python3
"""
Metric Degradation Analyzer

Measures performance degradation when using contaminated judge mixtures.
Uses existing baseline models and clean performance metrics for comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MetricDegradationAnalyzer:
    """Analyze performance degradation with contaminated judge mixtures."""
    
    def __init__(self, baseline_results_path: str = None):
        """
        Initialize with baseline (clean) performance results.
        
        Args:
            baseline_results_path: Path to clean baseline experiment results
        """
        self.baseline_results_path = baseline_results_path or "results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023/experiment_summary.json"
        self.baseline_metrics = None
        self.baseline_methods = [
            'linear_scaling_mean',
            'best_judge_linear_scaling', 
            'ultrafeedback_4judge',
            'standardscaler_lr_mean',
            'best_judge_standardscaler_lr'
        ]
    
    def load_baseline_metrics(self) -> Dict[str, Any]:
        """Load clean baseline performance metrics."""
        if self.baseline_metrics is not None:
            return self.baseline_metrics
            
        logger.info(f"Loading baseline metrics from {self.baseline_results_path}")
        
        with open(self.baseline_results_path, 'r') as f:
            baseline_data = json.load(f)
        
        # Extract baseline R² scores
        if 'baseline_analysis' in baseline_data:
            baseline_r2_scores = baseline_data['baseline_analysis']['baselines']['summary']['r2_scores']
        elif 'model_comparison' in baseline_data:
            baseline_r2_scores = baseline_data['model_comparison']['all_r2_scores']
        else:
            raise ValueError("Could not find baseline R² scores in results file")
        
        # Extract overall performance metrics
        clean_performance = {
            'best_model_r2': baseline_data.get('best_model_r2', baseline_data.get('optimal_model', {}).get('r2_score', 0.578)),
            'gam_r2': baseline_data.get('gam_analysis', {}).get('best_r2', 0.575),
            'baseline_r2_scores': baseline_r2_scores
        }
        
        self.baseline_metrics = clean_performance
        logger.info(f"Loaded baseline metrics: {list(baseline_r2_scores.keys())}")
        
        return self.baseline_metrics
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard evaluation metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def apply_baseline_method(
        self,
        method_name: str,
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Apply a specific baseline method to contaminated data."""
        
        if method_name == 'linear_scaling_mean':
            # Scale mean of judges to [0,10] range
            judge_mean_train = np.mean(X_train, axis=1)
            judge_mean_test = np.mean(X_test, axis=1)
            
            target_min, target_max = 0, 10
            judge_min, judge_max = np.min(judge_mean_train), np.max(judge_mean_train)
            
            if judge_max > judge_min:
                y_pred = ((judge_mean_test - judge_min) / (judge_max - judge_min)) * (target_max - target_min) + target_min
            else:
                y_pred = np.full_like(judge_mean_test, (target_min + target_max) / 2)
            
            return {
                'method': 'linear_scaling_mean',
                'metrics': self.compute_metrics(y_test, y_pred),
                'predictions': y_pred
            }
        
        elif method_name == 'best_judge_linear_scaling':
            # Find best judge and scale
            best_r2 = -1
            best_pred = None
            
            target_min, target_max = 0, 10
            
            for judge_idx in range(X_train.shape[1]):
                judge_scores_train = X_train[:, judge_idx]
                judge_scores_test = X_test[:, judge_idx]
                
                judge_min, judge_max = np.min(judge_scores_train), np.max(judge_scores_train)
                
                if judge_max > judge_min:
                    y_pred_candidate = ((judge_scores_test - judge_min) / (judge_max - judge_min)) * (target_max - target_min) + target_min
                else:
                    y_pred_candidate = np.full_like(judge_scores_test, (target_min + target_max) / 2)
                
                r2_candidate = r2_score(y_test, y_pred_candidate)
                
                if r2_candidate > best_r2:
                    best_r2 = r2_candidate
                    best_pred = y_pred_candidate
            
            return {
                'method': 'best_judge_linear_scaling',
                'metrics': self.compute_metrics(y_test, best_pred),
                'predictions': best_pred
            }
        
        elif method_name == 'ultrafeedback_4judge':
            # Use only 4 UltraFeedback judges (indices 0, 2, 3, 5)
            ultrafeedback_indices = [0, 2, 3, 5]
            if X_train.shape[1] >= 6:  # Ensure we have enough judges
                X_train_uf = X_train[:, ultrafeedback_indices]
                X_test_uf = X_test[:, ultrafeedback_indices]
            else:
                X_train_uf = X_train[:, :4]  # Use first 4 judges
                X_test_uf = X_test[:, :4]
            
            # Scale mean of 4 judges
            judge_mean_train = np.mean(X_train_uf, axis=1)
            judge_mean_test = np.mean(X_test_uf, axis=1)
            
            target_min, target_max = 0, 10
            judge_min, judge_max = np.min(judge_mean_train), np.max(judge_mean_train)
            
            if judge_max > judge_min:
                y_pred = ((judge_mean_test - judge_min) / (judge_max - judge_min)) * (target_max - target_min) + target_min
            else:
                y_pred = np.full_like(judge_mean_test, (target_min + target_max) / 2)
            
            return {
                'method': 'ultrafeedback_4judge',
                'metrics': self.compute_metrics(y_test, y_pred),
                'predictions': y_pred
            }
        
        elif method_name == 'standardscaler_lr_mean':
            # StandardScaler + LinearRegression on mean
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_mean = X_train_scaled.mean(axis=1).reshape(-1, 1)
            X_test_mean = X_test_scaled.mean(axis=1).reshape(-1, 1)
            
            lr = LinearRegression()
            lr.fit(X_train_mean, y_train)
            y_pred = lr.predict(X_test_mean)
            
            return {
                'method': 'standardscaler_lr_mean',
                'metrics': self.compute_metrics(y_test, y_pred),
                'predictions': y_pred
            }
        
        elif method_name == 'best_judge_standardscaler_lr':
            # Find best judge with StandardScaler + LinearRegression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            best_r2 = -1
            best_pred = None
            
            for judge_idx in range(X_train_scaled.shape[1]):
                X_train_single = X_train_scaled[:, judge_idx].reshape(-1, 1)
                X_test_single = X_test_scaled[:, judge_idx].reshape(-1, 1)
                
                lr = LinearRegression()
                lr.fit(X_train_single, y_train)
                y_pred_candidate = lr.predict(X_test_single)
                
                r2_candidate = r2_score(y_test, y_pred_candidate)
                
                if r2_candidate > best_r2:
                    best_r2 = r2_candidate
                    best_pred = y_pred_candidate
            
            return {
                'method': 'best_judge_standardscaler_lr',
                'metrics': self.compute_metrics(y_test, best_pred),
                'predictions': best_pred
            }
        
        else:
            raise ValueError(f"Unknown baseline method: {method_name}")
    
    def extract_human_feedback(self, data: pd.DataFrame) -> np.ndarray:
        """Extract human feedback scores from dataset."""
        human_scores = []
        
        for _, row in data.iterrows():
            if 'human_feedback' in row and row['human_feedback'] is not None:
                # Handle different human feedback formats
                if isinstance(row['human_feedback'], dict):
                    if 'personas' in row['human_feedback']:
                        # Get random persona score
                        personas = row['human_feedback']['personas']
                        persona_scores = [p['score'] for p in personas.values() if p.get('score') is not None]
                        if persona_scores:
                            human_scores.append(np.random.choice(persona_scores))
                        else:
                            human_scores.append(5.0)  # Default
                    elif 'score' in row['human_feedback']:
                        human_scores.append(row['human_feedback']['score'])
                    else:
                        human_scores.append(5.0)  # Default
                elif isinstance(row['human_feedback'], (int, float)):
                    human_scores.append(float(row['human_feedback']))
                else:
                    human_scores.append(5.0)  # Default
            else:
                human_scores.append(5.0)  # Default
        
        return np.array(human_scores)
    
    def analyze_mixture_degradation(
        self,
        mixtures: Dict[str, Dict],
        test_size: float = 0.2,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Analyze performance degradation for each contaminated mixture.
        
        Args:
            mixtures: Dictionary of contaminated judge mixtures from ContaminatedJudgeMixtureGenerator
            test_size: Fraction of data to use for testing
            random_seed: Random seed for train/test split
            
        Returns:
            Dictionary with degradation analysis results
        """
        # Load baseline metrics
        baseline_metrics = self.load_baseline_metrics()
        
        degradation_results = {
            'baseline_metrics': baseline_metrics,
            'mixture_results': {},
            'degradation_summary': {}
        }
        
        logger.info(f"Analyzing degradation for {len(mixtures)} contaminated mixtures...")
        
        for mixture_key, mixture_data in mixtures.items():
            logger.info(f"Processing mixture: {mixture_key}")
            
            try:
                # Extract contaminated judge scores and human feedback
                contaminated_matrix = mixture_data['contaminated_judge_matrix']
                
                # Load original data to get human feedback
                if hasattr(self, 'original_data') and self.original_data is not None:
                    human_scores = self.extract_human_feedback(self.original_data)
                else:
                    # Try to load from clean data path
                    try:
                        from contamination_mixture_generator import ContaminatedJudgeMixtureGenerator
                        generator = ContaminatedJudgeMixtureGenerator()
                        original_data = generator.load_clean_data()
                        human_scores = self.extract_human_feedback(original_data)
                        self.original_data = original_data
                    except:
                        # Generate synthetic human scores as fallback
                        human_scores = np.random.uniform(0, 10, len(contaminated_matrix))
                        logger.warning(f"Using synthetic human scores for {mixture_key}")
                
                # Ensure same length
                min_length = min(len(contaminated_matrix), len(human_scores))
                X = contaminated_matrix[:min_length]
                y = human_scores[:min_length]
                
                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed
                )
                
                # Apply each baseline method
                mixture_results = {}
                for method_name in self.baseline_methods:
                    try:
                        method_results = self.apply_baseline_method(
                            method_name, X_train, y_train, X_test, y_test
                        )
                        mixture_results[method_name] = method_results
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply {method_name} to {mixture_key}: {e}")
                        mixture_results[method_name] = {
                            'method': method_name,
                            'metrics': {'r2': 0.0, 'mse': 999.0, 'mae': 999.0},
                            'error': str(e)
                        }
                
                # Store results for this mixture
                degradation_results['mixture_results'][mixture_key] = {
                    'mixture_info': {
                        'strategy': mixture_data['strategy'],
                        'judge_mixture_rate': mixture_data['judge_mixture_rate'],
                        'contamination_rate': mixture_data['contamination_rate'],
                        'contaminated_judges': len(mixture_data['contaminated_judges_indices']),
                        'n_samples': mixture_data['n_samples']
                    },
                    'method_results': mixture_results
                }
                
            except Exception as e:
                logger.error(f"Failed to process mixture {mixture_key}: {e}")
                degradation_results['mixture_results'][mixture_key] = {
                    'error': str(e)
                }
        
        # Compute degradation summary
        degradation_results['degradation_summary'] = self._compute_degradation_summary(
            degradation_results, baseline_metrics
        )
        
        logger.info(f"Completed degradation analysis for {len(mixtures)} mixtures")
        return degradation_results
    
    def _compute_degradation_summary(
        self,
        results: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute summary statistics about performance degradation."""
        
        summary = {
            'method_degradation': {},
            'strategy_analysis': {},
            'contamination_trends': {}
        }
        
        baseline_r2_scores = baseline_metrics['baseline_r2_scores']
        
        # Analyze degradation by method
        for method_name in self.baseline_methods:
            if method_name in baseline_r2_scores:
                clean_r2 = baseline_r2_scores[method_name]
                
                method_degradations = []
                for mixture_key, mixture_result in results['mixture_results'].items():
                    if 'method_results' in mixture_result and method_name in mixture_result['method_results']:
                        contaminated_r2 = mixture_result['method_results'][method_name]['metrics']['r2']
                        degradation = clean_r2 - contaminated_r2
                        relative_degradation = degradation / clean_r2 if clean_r2 != 0 else 0
                        method_degradations.append({
                            'mixture': mixture_key,
                            'clean_r2': clean_r2,
                            'contaminated_r2': contaminated_r2,
                            'absolute_degradation': degradation,
                            'relative_degradation': relative_degradation
                        })
                
                if method_degradations:
                    summary['method_degradation'][method_name] = {
                        'clean_r2': clean_r2,
                        'mean_degradation': np.mean([d['absolute_degradation'] for d in method_degradations]),
                        'max_degradation': max([d['absolute_degradation'] for d in method_degradations]),
                        'mean_relative_degradation': np.mean([d['relative_degradation'] for d in method_degradations]),
                        'degradations': method_degradations
                    }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str = "metric_degradation_analysis.pkl"):
        """Save degradation analysis results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved degradation analysis to {output_path}")
        
        # Save summary as JSON
        json_path = output_path.with_suffix('.summary.json')
        summary_data = {
            'baseline_metrics': results['baseline_metrics'],
            'degradation_summary': results['degradation_summary'],
            'total_mixtures_analyzed': len(results['mixture_results'])
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Saved summary to {json_path}")


def main():
    """Run metric degradation analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze metric degradation with contaminated judges")
    parser.add_argument('--mixtures', default='contaminated_judge_mixtures.pkl',
                        help='Path to contaminated judge mixtures')
    parser.add_argument('--baseline-results', 
                        help='Path to baseline experiment results')
    parser.add_argument('--output', default='metric_degradation_analysis.pkl',
                        help='Output path for analysis results')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load contaminated mixtures
    logger.info(f"Loading contaminated mixtures from {args.mixtures}")
    with open(args.mixtures, 'rb') as f:
        mixtures = pickle.load(f)
    
    # Initialize analyzer
    analyzer = MetricDegradationAnalyzer(baseline_results_path=args.baseline_results)
    
    # Run degradation analysis
    results = analyzer.analyze_mixture_degradation(
        mixtures,
        test_size=args.test_size,
        random_seed=args.random_seed
    )
    
    # Save results
    analyzer.save_results(results, args.output)
    
    # Print summary
    print(f"\n✅ Degradation analysis complete!")
    print(f"📁 Results saved to: {args.output}")
    print(f"📋 Summary: {Path(args.output).with_suffix('.summary.json')}")
    
    summary = results['degradation_summary']
    if 'method_degradation' in summary:
        print(f"\n📊 Method Degradation Summary:")
        for method, data in summary['method_degradation'].items():
            print(f"  {method}:")
            print(f"    Clean R²: {data['clean_r2']:.3f}")
            print(f"    Mean degradation: {data['mean_degradation']:.3f}")
            print(f"    Max degradation: {data['max_degradation']:.3f}")
            print(f"    Mean relative degradation: {data['mean_relative_degradation']:.1%}")


if __name__ == "__main__":
    main()