"""
Bias Analysis Module for Experiment 4C

Implements comprehensive framing effects and frequency bias analysis
following the methodology from Christian et al. (2024).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasAnalyzer:
    """Comprehensive bias analysis for judge aggregation systems"""
    
    def __init__(self, normalize_scores: bool = False):
        """
        Initialize the bias analyzer
        
        Args:
            normalize_scores: Whether to normalize scores to [0,1] range
        """
        self.results = {}
        self.comparisons = {}
        self.normalize_scores = normalize_scores
    
    def _normalize_scores(self, scores_df: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
        """
        Normalize model scores to [0,1] range if enabled
        
        Args:
            scores_df: DataFrame with scores
            model_columns: List of score columns to normalize
            
        Returns:
            DataFrame with normalized scores (if enabled)
        """
        if not self.normalize_scores:
            return scores_df
        
        df_normalized = scores_df.copy()
        
        for col in model_columns:
            if col in df_normalized.columns:
                min_score = df_normalized[col].min()
                max_score = df_normalized[col].max()
                
                if max_score > min_score:  # Avoid division by zero
                    df_normalized[col] = (df_normalized[col] - min_score) / (max_score - min_score)
                else:
                    logger.warning(f"No score variation in {col}, normalization skipped")
        
        return df_normalized
        
    def analyze_framing_effects(self, 
                               scores_df: pd.DataFrame,
                               model_columns: List[str]) -> Dict[str, Dict]:
        """
        Comprehensive framing effects analysis
        
        Args:
            scores_df: DataFrame with scores for different prompts
            model_columns: List of column names for different models/judges
            
        Returns:
            Dictionary with framing analysis results for each model
        """
        logger.info("Starting framing effects analysis...")
        
        # Normalize scores if enabled
        scores_df_norm = self._normalize_scores(scores_df, model_columns)
        if self.normalize_scores:
            logger.info("Scores normalized to [0,1] range")
        
        framing_results = {}
        
        for model_col in model_columns:
            logger.info(f"Analyzing framing effects for {model_col}")
            
            model_results = self._analyze_single_model_framing(scores_df_norm, model_col)
            framing_results[model_col] = model_results
        
        self.results['framing_effects'] = framing_results
        return framing_results
    
    def _analyze_single_model_framing(self, 
                                     scores_df: pd.DataFrame,
                                     model_column: str) -> Dict[str, float]:
        """
        Analyze framing effects for a single model/judge
        
        Args:
            scores_df: DataFrame with scores
            model_column: Column name for the model to analyze
            
        Returns:
            Dictionary with framing metrics
        """
        results = {}
        
        # Separate by prompt type
        positive_prompt = scores_df[scores_df['prompt_type'] == 'positive']
        negative_prompt = scores_df[scores_df['prompt_type'] == 'negative']
        
        # Analyze each prompt type
        for prompt_type, prompt_df in [('positive', positive_prompt), ('negative', negative_prompt)]:
            
            # Separate tokens by sentiment
            pos_tokens = prompt_df[prompt_df['sentiment_score'] > 0]
            neg_tokens = prompt_df[prompt_df['sentiment_score'] < 0]
            
            # Calculate regression slopes for positive and negative tokens
            if len(pos_tokens) > 3:
                pos_slope, pos_intercept, pos_r, pos_p, pos_se = stats.linregress(
                    pos_tokens['sentiment_score'], 
                    pos_tokens[model_column]
                )
                pos_variance = pos_tokens[model_column].var()
            else:
                pos_slope = pos_r = pos_variance = 0
                logger.warning(f"Insufficient positive tokens for {prompt_type} prompt")
            
            if len(neg_tokens) > 3:
                neg_slope, neg_intercept, neg_r, neg_p, neg_se = stats.linregress(
                    neg_tokens['sentiment_score'], 
                    neg_tokens[model_column]
                )
                neg_variance = neg_tokens[model_column].var()
            else:
                neg_slope = neg_r = neg_variance = 0
                logger.warning(f"Insufficient negative tokens for {prompt_type} prompt")
            
            # Store detailed results
            results[f'{prompt_type}_pos_slope'] = pos_slope
            results[f'{prompt_type}_pos_r2'] = pos_r ** 2
            results[f'{prompt_type}_pos_variance'] = pos_variance
            results[f'{prompt_type}_neg_slope'] = neg_slope
            results[f'{prompt_type}_neg_r2'] = neg_r ** 2
            results[f'{prompt_type}_neg_variance'] = neg_variance
            
            # Calculate asymmetries
            slope_asymmetry = abs(pos_slope) - abs(neg_slope)
            variance_asymmetry = abs(pos_variance - neg_variance)
            
            results[f'{prompt_type}_slope_asymmetry'] = slope_asymmetry
            results[f'{prompt_type}_variance_asymmetry'] = variance_asymmetry
        
        # Calculate overall framing flip metric
        # Following Christian et al.: dominance measures how much model favors
        # positive vs negative tokens under each framing
        best_dominance = (results['positive_pos_slope'] - abs(results['positive_neg_slope']))
        worst_dominance = (abs(results['negative_neg_slope']) - results['negative_pos_slope'])
        
        framing_flip = best_dominance + worst_dominance
        results['framing_flip'] = framing_flip
        
        # Additional asymmetry measures
        results['overall_slope_asymmetry'] = (
            (results['positive_slope_asymmetry'] + results['negative_slope_asymmetry']) / 2
        )
        results['overall_variance_asymmetry'] = (
            (results['positive_variance_asymmetry'] + results['negative_variance_asymmetry']) / 2
        )
        
        return results
    
    def analyze_frequency_bias(self, 
                              scores_df: pd.DataFrame,
                              model_columns: List[str]) -> Dict[str, Dict]:
        """
        Analyze frequency bias using partial correlation
        
        Args:
            scores_df: DataFrame with scores and frequency data
            model_columns: List of column names for models to analyze
            
        Returns:
            Dictionary with frequency bias results for each model
        """
        logger.info("Starting frequency bias analysis...")
        
        # Normalize scores if enabled
        scores_df_norm = self._normalize_scores(scores_df, model_columns)
        
        frequency_results = {}
        
        for model_col in model_columns:
            logger.info(f"Analyzing frequency bias for {model_col}")
            
            # Calculate bias for each prompt type separately
            model_results = {}
            
            for prompt_type in ['positive', 'negative']:
                prompt_df = scores_df_norm[scores_df_norm['prompt_type'] == prompt_type]
                bias_result = self._calculate_frequency_bias(prompt_df, model_col)
                model_results[f'{prompt_type}_frequency_bias'] = bias_result
            
            # Overall frequency bias (averaged across prompts)
            overall_bias = np.mean([
                model_results['positive_frequency_bias']['correlation'],
                model_results['negative_frequency_bias']['correlation']
            ])
            model_results['overall_frequency_bias'] = overall_bias
            
            frequency_results[model_col] = model_results
        
        self.results['frequency_bias'] = frequency_results
        return frequency_results
    
    def _calculate_frequency_bias(self, 
                                 prompt_df: pd.DataFrame,
                                 model_column: str) -> Dict[str, float]:
        """
        Calculate frequency bias for a specific model and prompt
        
        Args:
            prompt_df: DataFrame filtered to specific prompt type
            model_column: Column name for the model
            
        Returns:
            Dictionary with bias metrics
        """
        # Clean data - remove NaN values
        clean_df = prompt_df[[model_column, 'log_frequency', 'sentiment_score']].dropna()
        
        if len(clean_df) < 10:
            logger.warning(f"Insufficient data for frequency bias: {len(clean_df)} samples")
            return {'correlation': np.nan, 'p_value': np.nan, 'partial_correlation': np.nan}
        
        # Raw correlation between frequency and scores
        raw_corr, raw_p = pearsonr(clean_df['log_frequency'], clean_df[model_column])
        
        # Partial correlation controlling for sentiment
        # Method: regress out sentiment from both variables, then correlate residuals
        
        # Regress sentiment out of log_frequency
        freq_slope, freq_intercept = np.polyfit(
            clean_df['sentiment_score'], clean_df['log_frequency'], 1
        )
        freq_residuals = (clean_df['log_frequency'] - 
                         (freq_slope * clean_df['sentiment_score'] + freq_intercept))
        
        # Regress sentiment out of model scores
        score_slope, score_intercept = np.polyfit(
            clean_df['sentiment_score'], clean_df[model_column], 1
        )
        score_residuals = (clean_df[model_column] - 
                          (score_slope * clean_df['sentiment_score'] + score_intercept))
        
        # Correlation between residuals = partial correlation
        partial_corr, partial_p = pearsonr(freq_residuals, score_residuals)
        
        return {
            'correlation': raw_corr,
            'p_value': raw_p,
            'partial_correlation': partial_corr,
            'partial_p_value': partial_p,
            'n_samples': len(clean_df)
        }
    
    def compare_bias_reduction(self,
                              individual_results: Dict,
                              aggregator_results: Dict) -> Dict[str, float]:
        """
        Compare bias levels between individual judges and aggregators
        
        Args:
            individual_results: Results from individual judges
            aggregator_results: Results from learned aggregators
            
        Returns:
            Dictionary with bias reduction percentages
        """
        logger.info("Calculating bias reduction percentages...")
        
        # Calculate average bias across individual judges
        individual_framing_flips = []
        individual_freq_biases = []
        
        for judge, results in individual_results.items():
            if 'framing_flip' in results:
                individual_framing_flips.append(abs(results['framing_flip']))
            
            if 'overall_frequency_bias' in results:
                individual_freq_biases.append(abs(results['overall_frequency_bias']))
        
        avg_individual_framing = np.mean(individual_framing_flips) if individual_framing_flips else np.nan
        avg_individual_frequency = np.mean(individual_freq_biases) if individual_freq_biases else np.nan
        
        # Get aggregator biases
        aggregator_framing = {}
        aggregator_frequency = {}
        
        for agg_name, results in aggregator_results.items():
            if 'framing_flip' in results:
                aggregator_framing[agg_name] = abs(results['framing_flip'])
            
            if 'overall_frequency_bias' in results:
                aggregator_frequency[agg_name] = abs(results['overall_frequency_bias'])
        
        # Calculate reduction percentages
        comparisons = {}
        
        if not np.isnan(avg_individual_framing):
            for agg_name, agg_framing in aggregator_framing.items():
                reduction = ((avg_individual_framing - agg_framing) / avg_individual_framing) * 100
                comparisons[f'{agg_name}_framing_reduction'] = reduction
        
        if not np.isnan(avg_individual_frequency):
            for agg_name, agg_frequency in aggregator_frequency.items():
                reduction = ((avg_individual_frequency - agg_frequency) / avg_individual_frequency) * 100
                comparisons[f'{agg_name}_frequency_reduction'] = reduction
        
        # Add baseline comparisons
        comparisons['naive_average_available'] = 'naive_average' in aggregator_results
        comparisons['avg_individual_framing_flip'] = avg_individual_framing
        comparisons['avg_individual_frequency_bias'] = avg_individual_frequency
        
        self.comparisons = comparisons
        return comparisons
    
    def run_significance_tests(self, 
                              scores_df: pd.DataFrame,
                              model_columns: List[str]) -> Dict[str, Dict]:
        """
        Run statistical significance tests for bias differences
        
        Args:
            scores_df: DataFrame with all scores
            model_columns: List of models to test
            
        Returns:
            Dictionary with significance test results
        """
        logger.info("Running statistical significance tests...")
        
        significance_results = {}
        
        for model_col in model_columns:
            model_tests = {}
            
            # Test for significant framing effects
            positive_df = scores_df[scores_df['prompt_type'] == 'positive']
            negative_df = scores_df[scores_df['prompt_type'] == 'negative']
            
            # Paired t-test for same tokens under different framings
            common_tokens = set(positive_df['token']) & set(negative_df['token'])
            
            if len(common_tokens) > 5:
                pos_scores = []
                neg_scores = []
                
                for token in common_tokens:
                    pos_score = positive_df[positive_df['token'] == token][model_col].iloc[0]
                    neg_score = negative_df[negative_df['token'] == token][model_col].iloc[0]
                    pos_scores.append(pos_score)
                    neg_scores.append(neg_score)
                
                framing_t_stat, framing_p_val = stats.ttest_rel(pos_scores, neg_scores)
                model_tests['framing_effect_significant'] = framing_p_val < 0.05
                model_tests['framing_t_statistic'] = framing_t_stat
                model_tests['framing_p_value'] = framing_p_val
            
            # Test frequency bias significance
            clean_df = scores_df[[model_col, 'log_frequency', 'sentiment_score']].dropna()
            if len(clean_df) > 20:
                # Test if partial correlation is significantly different from zero
                # Using Fisher z-transform for correlation significance
                partial_corr = self._calculate_frequency_bias(clean_df, model_col)['partial_correlation']
                
                if not np.isnan(partial_corr):
                    n = len(clean_df)
                    z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))  # Fisher z-transform
                    se = 1 / np.sqrt(n - 3)  # Standard error
                    z_stat = z / se
                    freq_p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
                    
                    model_tests['frequency_bias_significant'] = freq_p_val < 0.05
                    model_tests['frequency_z_statistic'] = z_stat
                    model_tests['frequency_p_value'] = freq_p_val
            
            significance_results[model_col] = model_tests
        
        self.results['significance_tests'] = significance_results
        return significance_results
    
    def generate_summary_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive summary of all bias analyses
        
        Returns:
            Dictionary with summary statistics and key findings
        """
        logger.info("Generating summary report...")
        
        summary = {
            'experiment_name': 'Experiment 4C: Framing Effects and Bias Transfer',
            'analysis_date': pd.Timestamp.now().isoformat(),
            'methods': [
                'framing_effects_analysis',
                'frequency_bias_analysis',
                'bias_reduction_comparison',
                'significance_testing'
            ]
        }
        
        # Add key findings
        if 'framing_effects' in self.results:
            framing_summary = self._summarize_framing_effects()
            summary['framing_effects_summary'] = framing_summary
        
        if 'frequency_bias' in self.results:
            frequency_summary = self._summarize_frequency_bias()
            summary['frequency_bias_summary'] = frequency_summary
        
        if self.comparisons:
            summary['bias_reduction_summary'] = self.comparisons
        
        # Overall conclusions
        summary['conclusions'] = self._generate_conclusions()
        
        return summary
    
    def _summarize_framing_effects(self) -> Dict[str, any]:
        """Summarize framing effects findings"""
        framing_results = self.results['framing_effects']
        
        all_framing_flips = []
        model_names = []
        
        for model, results in framing_results.items():
            if 'framing_flip' in results and not np.isnan(results['framing_flip']):
                all_framing_flips.append(abs(results['framing_flip']))
                model_names.append(model)
        
        if not all_framing_flips:
            return {'error': 'No valid framing flip data available'}
        
        return {
            'n_models_analyzed': len(all_framing_flips),
            'mean_framing_flip': np.mean(all_framing_flips),
            'std_framing_flip': np.std(all_framing_flips),
            'min_framing_flip': np.min(all_framing_flips),
            'max_framing_flip': np.max(all_framing_flips),
            'models_with_strong_bias': sum(1 for x in all_framing_flips if x > 1.0)
        }
    
    def _summarize_frequency_bias(self) -> Dict[str, any]:
        """Summarize frequency bias findings"""
        frequency_results = self.results['frequency_bias']
        
        all_freq_biases = []
        model_names = []
        
        for model, results in frequency_results.items():
            if 'overall_frequency_bias' in results and not np.isnan(results['overall_frequency_bias']):
                all_freq_biases.append(abs(results['overall_frequency_bias']))
                model_names.append(model)
        
        if not all_freq_biases:
            return {'error': 'No valid frequency bias data available'}
        
        return {
            'n_models_analyzed': len(all_freq_biases),
            'mean_frequency_bias': np.mean(all_freq_biases),
            'std_frequency_bias': np.std(all_freq_biases),
            'min_frequency_bias': np.min(all_freq_biases),
            'max_frequency_bias': np.max(all_freq_biases),
            'models_with_strong_bias': sum(1 for x in all_freq_biases if x > 0.3)
        }
    
    def _generate_conclusions(self) -> List[str]:
        """Generate high-level conclusions from the analysis"""
        conclusions = []
        
        # Analyze bias reduction effectiveness
        if self.comparisons:
            framing_reductions = [v for k, v in self.comparisons.items() if 'framing_reduction' in k]
            frequency_reductions = [v for k, v in self.comparisons.items() if 'frequency_reduction' in k]
            
            if framing_reductions:
                avg_framing_reduction = np.mean(framing_reductions)
                if avg_framing_reduction > 30:
                    conclusions.append(f"Strong framing bias reduction: {avg_framing_reduction:.1f}% average reduction")
                elif avg_framing_reduction > 10:
                    conclusions.append(f"Moderate framing bias reduction: {avg_framing_reduction:.1f}% average reduction")
                else:
                    conclusions.append(f"Limited framing bias reduction: {avg_framing_reduction:.1f}% average reduction")
            
            if frequency_reductions:
                avg_frequency_reduction = np.mean(frequency_reductions)
                if avg_frequency_reduction > 25:
                    conclusions.append(f"Strong frequency bias reduction: {avg_frequency_reduction:.1f}% average reduction")
                elif avg_frequency_reduction > 10:
                    conclusions.append(f"Moderate frequency bias reduction: {avg_frequency_reduction:.1f}% average reduction")
                else:
                    conclusions.append(f"Limited frequency bias reduction: {avg_frequency_reduction:.1f}% average reduction")
        
        # Success criteria evaluation
        if hasattr(self, 'comparisons') and self.comparisons:
            success_criteria = []
            
            # Check if learned aggregator shows >30% reduction in framing flip
            framing_reductions = [v for k, v in self.comparisons.items() if 'framing_reduction' in k]
            if framing_reductions and max(framing_reductions) > 30:
                success_criteria.append("✓ >30% framing bias reduction achieved")
            else:
                success_criteria.append("✗ <30% framing bias reduction")
            
            # Check if frequency bias reduced by >25%
            frequency_reductions = [v for k, v in self.comparisons.items() if 'frequency_reduction' in k]
            if frequency_reductions and max(frequency_reductions) > 25:
                success_criteria.append("✓ >25% frequency bias reduction achieved")
            else:
                success_criteria.append("✗ <25% frequency bias reduction")
            
            conclusions.extend(success_criteria)
        
        if not conclusions:
            conclusions.append("Analysis completed but insufficient data for conclusions")
        
        return conclusions
    
    def save_results(self, filepath: str):
        """Save all analysis results to file"""
        results_to_save = {
            'analysis_results': self.results,
            'comparisons': self.comparisons,
            'summary': self.generate_summary_report()
        }
        
        logger.info(f"Saving analysis results to {filepath}")
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(results_to_save, f)


def main():
    """Example usage of BiasAnalyzer"""
    
    # Create sample data for testing
    np.random.seed(42)
    n_tokens = 100
    
    sample_data = []
    tokens = [f"token_{i}" for i in range(n_tokens)]
    
    for token in tokens:
        for prompt_type in ['positive', 'negative']:
            sentiment = np.random.normal(0, 2)  # Random sentiment
            log_freq = np.random.normal(-7, 1)  # Random log frequency
            
            # Simulate judge scores with some bias
            base_score = 2.5 + 0.2 * sentiment
            if prompt_type == 'positive':
                base_score += 0.1 * max(0, sentiment)  # Framing effect
            else:
                base_score -= 0.1 * max(0, -sentiment)  # Framing effect
            
            # Add frequency bias
            base_score += 0.05 * log_freq
            
            # Individual judges with noise
            judge_scores = {f'judge_{i+1}': base_score + np.random.normal(0, 0.3) 
                          for i in range(5)}
            
            # Naive average
            naive_avg = np.mean(list(judge_scores.values()))
            
            # Mock aggregator (slightly less biased)
            mock_agg = base_score * 0.8 + np.random.normal(0, 0.2)
            
            sample_data.append({
                'token': token,
                'prompt_type': prompt_type,
                'sentiment_score': sentiment,
                'log_frequency': log_freq,
                'naive_average': naive_avg,
                'mock_aggregator': mock_agg,
                **judge_scores
            })
    
    scores_df = pd.DataFrame(sample_data)
    
    # Run analysis
    analyzer = BiasAnalyzer()
    
    model_columns = ['naive_average', 'mock_aggregator'] + [f'judge_{i+1}' for i in range(5)]
    
    # Run all analyses
    framing_results = analyzer.analyze_framing_effects(scores_df, model_columns)
    frequency_results = analyzer.analyze_frequency_bias(scores_df, model_columns)
    significance_results = analyzer.run_significance_tests(scores_df, model_columns)
    
    # Compare bias reduction
    individual_results = {col: {**framing_results[col], **frequency_results[col]} 
                         for col in model_columns if col.startswith('judge_')}
    aggregator_results = {col: {**framing_results[col], **frequency_results[col]} 
                         for col in model_columns if not col.startswith('judge_')}
    
    comparisons = analyzer.compare_bias_reduction(individual_results, aggregator_results)
    
    # Generate summary
    summary = analyzer.generate_summary_report()
    
    print("Analysis Complete!")
    print(f"Models analyzed: {len(model_columns)}")
    print(f"Bias reduction findings: {len(comparisons)} metrics")
    
    return analyzer, summary


if __name__ == "__main__":
    analyzer, summary = main()