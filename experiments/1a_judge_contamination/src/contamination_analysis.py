#!/usr/bin/env python3
"""
Advanced Contamination Analysis Module

Provides sophisticated statistical analysis and robustness metrics for judge contamination studies.
Implements comprehensive detection algorithms, impact assessment, and baseline comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kstest, mannwhitneyu
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ContaminationAnalyzer:
    """
    Comprehensive analyzer for judge contamination effects.
    
    Provides statistical testing, robustness metrics, baseline comparisons,
    and contamination detection algorithms.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize the contamination analyzer.
        
        Args:
            significance_level: Statistical significance threshold for tests
        """
        self.significance_level = significance_level
        self.analysis_results = {}
        
    def analyze_judge_inversion(self, 
                               baseline_scores: pd.DataFrame, 
                               contaminated_scores: pd.DataFrame,
                               judge_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze judge inversion effects with comprehensive statistical testing.
        
        Args:
            baseline_scores: Clean judge scores (samples x judges)
            contaminated_scores: Contaminated judge scores (samples x judges)
            judge_mapping: Mapping from baseline judge ID to contaminated judge ID
            
        Returns:
            Comprehensive analysis results dictionary
        """
        results = {
            'individual_judges': {},
            'aggregate_metrics': {},
            'statistical_tests': {},
            'inversion_detection': {}
        }
        
        # Analyze each judge pair individually
        for baseline_judge, contaminated_judge in judge_mapping.items():
            if baseline_judge in baseline_scores.columns and contaminated_judge in contaminated_scores.columns:
                baseline_vec = baseline_scores[baseline_judge].dropna()
                contaminated_vec = contaminated_scores[contaminated_judge].dropna()
                
                # Ensure same samples for comparison
                min_len = min(len(baseline_vec), len(contaminated_vec))
                baseline_vec = baseline_vec.iloc[:min_len]
                contaminated_vec = contaminated_vec.iloc[:min_len]
                
                judge_analysis = self._analyze_judge_pair(baseline_vec, contaminated_vec, baseline_judge)
                results['individual_judges'][baseline_judge] = judge_analysis
        
        # Aggregate-level analysis
        results['aggregate_metrics'] = self._compute_aggregate_metrics(
            baseline_scores, contaminated_scores, judge_mapping
        )
        
        # Statistical significance tests
        results['statistical_tests'] = self._perform_statistical_tests(
            baseline_scores, contaminated_scores, judge_mapping
        )
        
        # Inversion detection
        results['inversion_detection'] = self._detect_inversion_patterns(
            baseline_scores, contaminated_scores, judge_mapping
        )
        
        self.analysis_results['judge_inversion'] = results
        return results
    
    def _analyze_judge_pair(self, baseline: pd.Series, contaminated: pd.Series, judge_id: str) -> Dict[str, Any]:
        """Analyze individual judge pair for contamination effects."""
        
        # Basic statistics
        baseline_stats = {
            'mean': float(baseline.mean()),
            'std': float(baseline.std()),
            'min': float(baseline.min()),
            'max': float(baseline.max()),
            'median': float(baseline.median())
        }
        
        contaminated_stats = {
            'mean': float(contaminated.mean()),
            'std': float(contaminated.std()),
            'min': float(contaminated.min()),
            'max': float(contaminated.max()),
            'median': float(contaminated.median())
        }
        
        # Score shift analysis
        mean_shift = contaminated_stats['mean'] - baseline_stats['mean']
        std_shift = contaminated_stats['std'] - baseline_stats['std']
        
        # Correlation analysis
        pearson_corr, pearson_p = pearsonr(baseline, contaminated)
        spearman_corr, spearman_p = spearmanr(baseline, contaminated)
        
        # Distribution comparison
        ks_stat, ks_p = kstest(contaminated, baseline)
        mw_stat, mw_p = mannwhitneyu(baseline, contaminated, alternative='two-sided')
        
        # Inversion indicators
        is_inverted = pearson_corr < -0.5
        inversion_strength = abs(pearson_corr) if pearson_corr < 0 else 0
        
        return {
            'baseline_stats': baseline_stats,
            'contaminated_stats': contaminated_stats,
            'shifts': {
                'mean_shift': float(mean_shift),
                'std_shift': float(std_shift),
                'relative_mean_shift': float(mean_shift / baseline_stats['mean']) if baseline_stats['mean'] != 0 else 0
            },
            'correlations': {
                'pearson': {'correlation': float(pearson_corr), 'p_value': float(pearson_p)},
                'spearman': {'correlation': float(spearman_corr), 'p_value': float(spearman_p)}
            },
            'distribution_tests': {
                'kolmogorov_smirnov': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
                'mann_whitney_u': {'statistic': float(mw_stat), 'p_value': float(mw_p)}
            },
            'inversion_analysis': {
                'is_inverted': is_inverted,
                'inversion_strength': float(inversion_strength),
                'inversion_quality': self._assess_inversion_quality(pearson_corr)
            }
        }
    
    def _assess_inversion_quality(self, correlation: float) -> str:
        """Assess the quality of judge inversion based on correlation."""
        if correlation >= 0:
            return "no_inversion"
        elif correlation >= -0.3:
            return "weak_inversion"
        elif correlation >= -0.7:
            return "moderate_inversion"
        else:
            return "strong_inversion"
    
    def _compute_aggregate_metrics(self, 
                                  baseline_scores: pd.DataFrame, 
                                  contaminated_scores: pd.DataFrame,
                                  judge_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Compute aggregate-level contamination metrics."""
        
        # Overall correlation matrix
        baseline_means = baseline_scores.mean()
        contaminated_means = contaminated_scores.mean()
        
        # Cross-correlation analysis
        correlations = []
        for baseline_judge, contaminated_judge in judge_mapping.items():
            if baseline_judge in baseline_scores.columns and contaminated_judge in contaminated_scores.columns:
                corr, _ = pearsonr(
                    baseline_scores[baseline_judge].dropna(),
                    contaminated_scores[contaminated_judge].dropna()
                )
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        # System-wide metrics
        contamination_rate = sum(1 for corr in correlations if corr < -0.5) / len(correlations)
        
        return {
            'average_correlation': float(avg_correlation),
            'correlation_std': float(np.std(correlations)),
            'contamination_rate': float(contamination_rate),
            'correlations_list': [float(c) for c in correlations],
            'system_inversion_detected': avg_correlation < -0.3,
            'severe_contamination': contamination_rate > 0.5
        }
    
    def _perform_statistical_tests(self, 
                                  baseline_scores: pd.DataFrame, 
                                  contaminated_scores: pd.DataFrame,
                                  judge_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Perform comprehensive statistical significance testing."""
        
        results = {}
        
        # System-level distribution test
        baseline_flat = baseline_scores.values.flatten()
        contaminated_flat = contaminated_scores.values.flatten()
        
        # Remove NaN values
        baseline_flat = baseline_flat[~np.isnan(baseline_flat)]
        contaminated_flat = contaminated_flat[~np.isnan(contaminated_flat)]
        
        # Kolmogorov-Smirnov test for distribution shift
        ks_stat, ks_p = kstest(contaminated_flat, baseline_flat)
        
        # Mann-Whitney U test for median shift
        mw_stat, mw_p = mannwhitneyu(baseline_flat, contaminated_flat, alternative='two-sided')
        
        results['system_level'] = {
            'distribution_shift': {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'significant': ks_p < self.significance_level
            },
            'median_shift': {
                'mw_statistic': float(mw_stat),
                'mw_p_value': float(mw_p),
                'significant': mw_p < self.significance_level
            }
        }
        
        # Multiple comparisons correction (Bonferroni)
        n_comparisons = len(judge_mapping)
        corrected_alpha = self.significance_level / n_comparisons
        
        results['multiple_comparison_correction'] = {
            'original_alpha': self.significance_level,
            'corrected_alpha': corrected_alpha,
            'n_comparisons': n_comparisons
        }
        
        return results
    
    def _detect_inversion_patterns(self, 
                                  baseline_scores: pd.DataFrame, 
                                  contaminated_scores: pd.DataFrame,
                                  judge_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Detect and classify inversion patterns across judges."""
        
        inversion_patterns = {
            'complete_inversion': [],
            'partial_inversion': [],
            'no_inversion': [],
            'amplification': []
        }
        
        for baseline_judge, contaminated_judge in judge_mapping.items():
            if baseline_judge in baseline_scores.columns and contaminated_judge in contaminated_scores.columns:
                baseline_vec = baseline_scores[baseline_judge].dropna()
                contaminated_vec = contaminated_scores[contaminated_judge].dropna()
                
                min_len = min(len(baseline_vec), len(contaminated_vec))
                baseline_vec = baseline_vec.iloc[:min_len]
                contaminated_vec = contaminated_vec.iloc[:min_len]
                
                corr, _ = pearsonr(baseline_vec, contaminated_vec)
                
                if corr < -0.8:
                    inversion_patterns['complete_inversion'].append(baseline_judge)
                elif corr < -0.3:
                    inversion_patterns['partial_inversion'].append(baseline_judge)
                elif corr > 0.8:
                    inversion_patterns['amplification'].append(baseline_judge)
                else:
                    inversion_patterns['no_inversion'].append(baseline_judge)
        
        return {
            'patterns': inversion_patterns,
            'pattern_summary': {
                'complete_inversion_count': len(inversion_patterns['complete_inversion']),
                'partial_inversion_count': len(inversion_patterns['partial_inversion']),
                'no_inversion_count': len(inversion_patterns['no_inversion']),
                'amplification_count': len(inversion_patterns['amplification'])
            },
            'contamination_success_rate': (
                len(inversion_patterns['complete_inversion']) + 
                len(inversion_patterns['partial_inversion'])
            ) / len(judge_mapping)
        }
    
    def analyze_aggregator_robustness(self, 
                                    clean_data: pd.DataFrame,
                                    contaminated_data: pd.DataFrame,
                                    human_feedback: pd.Series,
                                    contamination_rates: List[float] = None) -> Dict[str, Any]:
        """
        Analyze aggregator robustness to contamination at different levels.
        
        Args:
            clean_data: Clean judge scores
            contaminated_data: Contaminated judge scores  
            human_feedback: Ground truth human feedback
            contamination_rates: List of contamination rates to test
            
        Returns:
            Robustness analysis results
        """
        if contamination_rates is None:
            contamination_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            
        results = {
            'contamination_curves': {},
            'breakdown_analysis': {},
            'robustness_metrics': {}
        }
        
        # Test different contamination mixing ratios
        for rate in contamination_rates:
            mixed_data = self._mix_clean_contaminated_data(clean_data, contaminated_data, rate)
            
            # Split data for training/testing
            X_train, X_test, y_train, y_test = train_test_split(
                mixed_data, human_feedback, test_size=0.2, random_state=42
            )
            
            # Test different aggregation methods
            methods_performance = {}
            
            # Naive averaging
            naive_pred = X_test.mean(axis=1)
            methods_performance['naive_average'] = {
                'r2': r2_score(y_test, naive_pred),
                'mse': mean_squared_error(y_test, naive_pred),
                'mae': mean_absolute_error(y_test, naive_pred)
            }
            
            # Best single judge
            best_judge_scores = []
            for col in X_test.columns:
                score = r2_score(y_test, X_test[col])
                best_judge_scores.append(score)
            
            best_r2 = max(best_judge_scores)
            methods_performance['best_single_judge'] = {
                'r2': best_r2,
                'mse': None,  # Would need to compute for specific judge
                'mae': None
            }
            
            results['contamination_curves'][f'rate_{rate:.1f}'] = methods_performance
        
        # Compute robustness metrics
        results['robustness_metrics'] = self._compute_robustness_metrics(
            results['contamination_curves']
        )
        
        # Breakdown analysis
        results['breakdown_analysis'] = self._analyze_performance_breakdown(
            results['contamination_curves'], contamination_rates
        )
        
        return results
    
    def _mix_clean_contaminated_data(self, 
                                   clean_data: pd.DataFrame, 
                                   contaminated_data: pd.DataFrame, 
                                   contamination_rate: float) -> pd.DataFrame:
        """Mix clean and contaminated data at specified rate."""
        n_samples = len(clean_data)
        n_contaminated = int(n_samples * contamination_rate)
        
        # Randomly select samples to contaminate
        contaminate_indices = np.random.choice(n_samples, n_contaminated, replace=False)
        
        mixed_data = clean_data.copy()
        for idx in contaminate_indices:
            if idx < len(contaminated_data):
                mixed_data.iloc[idx] = contaminated_data.iloc[idx]
        
        return mixed_data
    
    def _compute_robustness_metrics(self, contamination_curves: Dict) -> Dict[str, Any]:
        """Compute robustness metrics from contamination curves."""
        
        # Extract R² scores for naive averaging across contamination rates
        naive_r2_scores = []
        rates = []
        
        for rate_key, methods in contamination_curves.items():
            rate = float(rate_key.split('_')[1])
            rates.append(rate)
            naive_r2_scores.append(methods['naive_average']['r2'])
        
        # Compute robustness metrics
        clean_performance = naive_r2_scores[0]  # Rate 0.0
        final_performance = naive_r2_scores[-1]  # Highest rate
        
        relative_degradation = (clean_performance - final_performance) / clean_performance
        
        # Find breakdown point (where performance drops below 50% of clean)
        breakdown_threshold = clean_performance * 0.5
        breakdown_rate = None
        
        for i, (rate, score) in enumerate(zip(rates, naive_r2_scores)):
            if score < breakdown_threshold:
                breakdown_rate = rate
                break
        
        return {
            'clean_performance': float(clean_performance),
            'final_performance': float(final_performance),
            'relative_degradation': float(relative_degradation),
            'breakdown_rate': breakdown_rate,
            'performance_curve': {
                'rates': rates,
                'r2_scores': naive_r2_scores
            }
        }
    
    def _analyze_performance_breakdown(self, 
                                     contamination_curves: Dict,
                                     contamination_rates: List[float]) -> Dict[str, Any]:
        """Analyze where and how performance breaks down."""
        
        # Performance stability analysis
        naive_scores = []
        single_judge_scores = []
        
        for rate in contamination_rates:
            rate_key = f'rate_{rate:.1f}'
            if rate_key in contamination_curves:
                naive_scores.append(contamination_curves[rate_key]['naive_average']['r2'])
                single_judge_scores.append(contamination_curves[rate_key]['best_single_judge']['r2'])
        
        # Compute derivatives to find steepest decline
        naive_derivatives = np.diff(naive_scores)
        steepest_decline_idx = np.argmin(naive_derivatives)
        steepest_decline_rate = contamination_rates[steepest_decline_idx]
        
        return {
            'steepest_decline': {
                'rate': float(steepest_decline_rate),
                'performance_drop': float(abs(naive_derivatives[steepest_decline_idx]))
            },
            'comparative_robustness': {
                'naive_vs_single_judge': float(np.mean(naive_scores) - np.mean(single_judge_scores))
            }
        }
    
    def generate_contamination_report(self, output_path: Path) -> Dict[str, Any]:
        """Generate comprehensive contamination analysis report."""
        
        report = {
            'analysis_summary': {},
            'key_findings': [],
            'recommendations': [],
            'detailed_results': self.analysis_results
        }
        
        # Extract key findings from analysis results
        if 'judge_inversion' in self.analysis_results:
            inversion_results = self.analysis_results['judge_inversion']
            
            # Summary statistics
            avg_correlation = inversion_results['aggregate_metrics']['average_correlation']
            contamination_rate = inversion_results['aggregate_metrics']['contamination_rate']
            
            report['analysis_summary'] = {
                'overall_contamination_detected': avg_correlation < -0.3,
                'average_judge_correlation': avg_correlation,
                'contamination_success_rate': contamination_rate,
                'statistical_significance': inversion_results['statistical_tests']['system_level']['distribution_shift']['significant']
            }
            
            # Key findings
            if avg_correlation < -0.5:
                report['key_findings'].append("Strong evidence of systematic judge inversion detected")
            elif avg_correlation < -0.3:
                report['key_findings'].append("Moderate judge contamination effects observed")
            else:
                report['key_findings'].append("Limited contamination effects detected")
            
            # Recommendations
            if contamination_rate > 0.7:
                report['recommendations'].append("High contamination rate suggests need for enhanced detection systems")
            if inversion_results['statistical_tests']['system_level']['distribution_shift']['significant']:
                report['recommendations'].append("Significant distribution shift indicates systematic bias introduction")
        
        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report