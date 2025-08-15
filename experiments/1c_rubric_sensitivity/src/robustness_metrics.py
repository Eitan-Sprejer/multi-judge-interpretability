"""
Robustness Metrics Analyzer

Calculates robustness metrics for aggregation models across rubric variations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Analyzes robustness of aggregation models to rubric variations."""
    
    def __init__(
        self,
        scores_df: pd.DataFrame,
        model_path: Optional[str] = None
    ):
        """
        Initialize the robustness analyzer.
        
        Args:
            scores_df: DataFrame with scores from all judge variants
            model_path: Path to trained aggregation model
        """
        self.scores_df = scores_df.copy()
        self.model_path = model_path
        
        # Parse judge variants
        self.variant_groups = self._parse_variant_groups()
        
        # Load model if provided
        self.model = None
        if model_path:
            self.model = self._load_model(model_path)
    
    def _parse_variant_groups(self) -> Dict[str, Dict[str, str]]:
        """
        Parse judge columns into base judges and their variants.
        
        Returns:
            Dictionary mapping base judge names to variant columns
        """
        variant_groups = {}
        
        for col in self.scores_df.columns:
            if '-' in col and col not in ['example_idx']:
                parts = col.rsplit('-', 1)
                if len(parts) == 2:
                    base_judge, variant_type = parts
                    
                    if base_judge not in variant_groups:
                        variant_groups[base_judge] = {}
                    
                    variant_groups[base_judge][variant_type] = col
        
        logger.info(f"Found {len(variant_groups)} base judges with variants")
        return variant_groups
    
    def _load_model(self, model_path: str):
        """Load trained aggregation model."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def calculate_score_variance(self) -> Dict[str, Dict]:
        """
        Calculate variance in judge scores across rubric variations.
        
        Returns:
            Dictionary with variance metrics for each judge
        """
        variance_metrics = {}
        
        for base_judge, variants in self.variant_groups.items():
            if len(variants) < 2:
                continue
            
            # Get scores for all variants
            variant_cols = list(variants.values())
            scores_matrix = self.scores_df[variant_cols].values
            
            # Calculate variance metrics
            row_variances = np.nanvar(scores_matrix, axis=1)
            row_std = np.nanstd(scores_matrix, axis=1)
            
            variance_metrics[base_judge] = {
                'mean_variance': np.nanmean(row_variances),
                'std_variance': np.nanstd(row_variances),
                'max_variance': np.nanmax(row_variances),
                'mean_std': np.nanmean(row_std),
                'variance_distribution': row_variances,
                'n_variants': len(variants)
            }
        
        return variance_metrics
    
    def calculate_cross_rubric_correlation(self) -> Dict[str, Dict]:
        """
        Calculate correlation between original and variant rubric scores.
        
        Returns:
            Dictionary with correlation metrics for each judge
        """
        correlation_metrics = {}
        
        for base_judge, variants in self.variant_groups.items():
            if 'original' not in variants:
                continue
            
            original_col = variants['original']
            original_scores = self.scores_df[original_col].values
            
            correlations = {}
            for variant_type, variant_col in variants.items():
                if variant_type == 'original':
                    continue
                
                variant_scores = self.scores_df[variant_col].values
                
                # Calculate correlations (handle NaNs)
                mask = ~(np.isnan(original_scores) | np.isnan(variant_scores))
                if mask.sum() > 1:
                    pearson_r, pearson_p = stats.pearsonr(
                        original_scores[mask], 
                        variant_scores[mask]
                    )
                    spearman_r, spearman_p = stats.spearmanr(
                        original_scores[mask], 
                        variant_scores[mask]
                    )
                else:
                    pearson_r = pearson_p = spearman_r = spearman_p = np.nan
                
                correlations[variant_type] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_valid': mask.sum()
                }
            
            # Calculate average correlations
            if correlations:
                pearson_values = [c['pearson_r'] for c in correlations.values() if not np.isnan(c['pearson_r'])]
                spearman_values = [c['spearman_r'] for c in correlations.values() if not np.isnan(c['spearman_r'])]
                
                correlation_metrics[base_judge] = {
                    'individual_correlations': correlations,
                    'mean_pearson': np.mean(pearson_values) if pearson_values else np.nan,
                    'min_pearson': np.min(pearson_values) if pearson_values else np.nan,
                    'mean_spearman': np.mean(spearman_values) if spearman_values else np.nan,
                    'min_spearman': np.min(spearman_values) if spearman_values else np.nan
                }
        
        return correlation_metrics
    
    def calculate_rank_order_consistency(self) -> Dict[str, Dict]:
        """
        Calculate rank-order consistency across rubric variations.
        
        Returns:
            Dictionary with rank consistency metrics for each judge
        """
        rank_metrics = {}
        
        for base_judge, variants in self.variant_groups.items():
            if len(variants) < 2:
                continue
            
            variant_cols = list(variants.values())
            
            # Calculate ranks for each variant
            ranks = {}
            for variant_type, col in variants.items():
                scores = self.scores_df[col].values
                # Handle NaNs by giving them the lowest rank
                ranks[variant_type] = stats.rankdata(
                    np.where(np.isnan(scores), -np.inf, scores),
                    method='average'
                )
            
            # Calculate pairwise rank correlations
            rank_correlations = {}
            for i, (var1, ranks1) in enumerate(ranks.items()):
                for var2, ranks2 in list(ranks.items())[i+1:]:
                    # Calculate Kendall's tau
                    mask = ~(np.isnan(ranks1) | np.isnan(ranks2))
                    if mask.sum() > 1:
                        tau, p_value = stats.kendalltau(ranks1[mask], ranks2[mask])
                    else:
                        tau = p_value = np.nan
                    
                    rank_correlations[f"{var1}_vs_{var2}"] = {
                        'kendall_tau': tau,
                        'p_value': p_value,
                        'n_valid': mask.sum()
                    }
            
            # Calculate average rank consistency
            tau_values = [rc['kendall_tau'] for rc in rank_correlations.values() 
                         if not np.isnan(rc['kendall_tau'])]
            
            rank_metrics[base_judge] = {
                'pairwise_correlations': rank_correlations,
                'mean_kendall_tau': np.mean(tau_values) if tau_values else np.nan,
                'min_kendall_tau': np.min(tau_values) if tau_values else np.nan,
                'n_pairs': len(rank_correlations)
            }
        
        return rank_metrics
    
    def calculate_aggregator_robustness(
        self,
        aggregation_methods: List[str] = ['learned', 'mean', 'single_best']
    ) -> Dict[str, Dict]:
        """
        Calculate robustness of different aggregation methods.
        
        Args:
            aggregation_methods: List of aggregation methods to compare
            
        Returns:
            Dictionary with robustness metrics for each method
        """
        robustness_metrics = {}
        
        for method in aggregation_methods:
            logger.info(f"Calculating robustness for {method} aggregation")
            
            # Calculate aggregated scores for each variant combination
            variant_combinations = self._get_variant_combinations()
            aggregated_scores = {}
            
            for combo_name, combo_cols in variant_combinations.items():
                if method == 'learned' and self.model:
                    aggregated_scores[combo_name] = self._aggregate_learned(combo_cols)
                elif method == 'mean':
                    aggregated_scores[combo_name] = self._aggregate_mean(combo_cols)
                elif method == 'single_best':
                    aggregated_scores[combo_name] = self._aggregate_single_best(combo_cols)
            
            if not aggregated_scores:
                continue
            
            # Calculate robustness metrics across combinations
            scores_matrix = np.column_stack(list(aggregated_scores.values()))
            
            # Variance across variant combinations
            row_variances = np.nanvar(scores_matrix, axis=1)
            
            # Correlation with "original" combination if available
            correlations = {}
            if 'original' in aggregated_scores:
                original_scores = aggregated_scores['original']
                for combo_name, combo_scores in aggregated_scores.items():
                    if combo_name == 'original':
                        continue
                    
                    mask = ~(np.isnan(original_scores) | np.isnan(combo_scores))
                    if mask.sum() > 1:
                        r, p = stats.pearsonr(original_scores[mask], combo_scores[mask])
                        correlations[combo_name] = {'r': r, 'p': p}
            
            robustness_metrics[method] = {
                'mean_variance': np.nanmean(row_variances),
                'max_variance': np.nanmax(row_variances),
                'variance_distribution': row_variances,
                'correlations_with_original': correlations,
                'n_combinations': len(aggregated_scores)
            }
        
        return robustness_metrics
    
    def _get_variant_combinations(self) -> Dict[str, List[str]]:
        """
        Get different combinations of judge variants for testing robustness.
        
        Returns:
            Dictionary mapping combination names to lists of column names
        """
        combinations = {}
        
        # All original judges
        original_cols = []
        for base_judge, variants in self.variant_groups.items():
            if 'original' in variants:
                original_cols.append(variants['original'])
        if original_cols:
            combinations['original'] = original_cols
        
        # All formal variants
        formal_cols = []
        for base_judge, variants in self.variant_groups.items():
            if 'formal' in variants:
                formal_cols.append(variants['formal'])
        if formal_cols:
            combinations['formal'] = formal_cols
        
        # All casual variants
        casual_cols = []
        for base_judge, variants in self.variant_groups.items():
            if 'casual' in variants:
                casual_cols.append(variants['casual'])
        if casual_cols:
            combinations['casual'] = casual_cols
        
        # All restructured variants
        restructured_cols = []
        for base_judge, variants in self.variant_groups.items():
            if 'restructured' in variants:
                restructured_cols.append(variants['restructured'])
        if restructured_cols:
            combinations['restructured'] = restructured_cols
        
        # Mixed combinations (e.g., some original, some formal)
        if len(original_cols) > 1 and len(formal_cols) > 1:
            mixed_cols = original_cols[:len(original_cols)//2] + formal_cols[len(formal_cols)//2:]
            combinations['mixed_original_formal'] = mixed_cols
        
        return combinations
    
    def _aggregate_learned(self, judge_cols: List[str]) -> np.ndarray:
        """Aggregate using learned model."""
        if self.model is None:
            return np.full(len(self.scores_df), np.nan)
        
        # Get scores for the specified judges
        scores_matrix = self.scores_df[judge_cols].values
        
        # Handle missing values (fill with median for each judge)
        for i in range(scores_matrix.shape[1]):
            col_median = np.nanmedian(scores_matrix[:, i])
            scores_matrix[np.isnan(scores_matrix[:, i]), i] = col_median
        
        try:
            # Predict using the model
            predictions = self.model.predict(scores_matrix)
            return predictions
        except Exception as e:
            logger.error(f"Failed to use learned model: {e}")
            return np.full(len(self.scores_df), np.nan)
    
    def _aggregate_mean(self, judge_cols: List[str]) -> np.ndarray:
        """Aggregate using simple mean."""
        scores_matrix = self.scores_df[judge_cols].values
        return np.nanmean(scores_matrix, axis=1)
    
    def _aggregate_single_best(self, judge_cols: List[str]) -> np.ndarray:
        """Aggregate using single best judge (highest correlation with ground truth)."""
        if 'ground_truth' not in self.scores_df.columns:
            # Fallback: use mean of first judge
            return self.scores_df[judge_cols[0]].values
        
        ground_truth = self.scores_df['ground_truth'].values
        best_judge = None
        best_correlation = -np.inf
        
        for col in judge_cols:
            scores = self.scores_df[col].values
            mask = ~(np.isnan(scores) | np.isnan(ground_truth))
            
            if mask.sum() > 1:
                r, _ = stats.pearsonr(scores[mask], ground_truth[mask])
                if r > best_correlation:
                    best_correlation = r
                    best_judge = col
        
        if best_judge:
            return self.scores_df[best_judge].values
        else:
            return self.scores_df[judge_cols[0]].values
    
    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report.
        
        Returns:
            Dictionary with all robustness metrics
        """
        logger.info("Generating comprehensive robustness report...")
        
        report = {
            'score_variance': self.calculate_score_variance(),
            'cross_rubric_correlation': self.calculate_cross_rubric_correlation(),
            'rank_order_consistency': self.calculate_rank_order_consistency(),
            'aggregator_robustness': self.calculate_aggregator_robustness()
        }
        
        # Summary statistics
        variance_metrics = report['score_variance']
        correlation_metrics = report['cross_rubric_correlation']
        aggregator_metrics = report['aggregator_robustness']
        
        # Overall variance across all judges
        all_variances = []
        for judge_metrics in variance_metrics.values():
            all_variances.extend(judge_metrics['variance_distribution'])
        
        # Overall correlations
        all_correlations = []
        for judge_metrics in correlation_metrics.values():
            pearson_vals = [c['pearson_r'] for c in judge_metrics['individual_correlations'].values()]
            all_correlations.extend([c for c in pearson_vals if not np.isnan(c)])
        
        report['summary'] = {
            'overall_mean_variance': np.mean(all_variances) if all_variances else np.nan,
            'overall_max_variance': np.max(all_variances) if all_variances else np.nan,
            'variance_below_5_percent': np.mean(np.array(all_variances) < 0.05) if all_variances else 0,
            'overall_mean_correlation': np.mean(all_correlations) if all_correlations else np.nan,
            'correlation_above_95_percent': np.mean(np.array(all_correlations) > 0.95) if all_correlations else 0,
            'n_judges_tested': len(variance_metrics),
            'n_examples': len(self.scores_df)
        }
        
        return report
    
    def save_report(self, output_path: str, report: Dict):
        """Save robustness report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(report, f)
        
        logger.info(f"Saved robustness report to {output_path}")


def main():
    """Main entry point for testing robustness analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze robustness of judge variants")
    parser.add_argument('--scores', required=True, help='Path to scores DataFrame')
    parser.add_argument('--model', help='Path to trained aggregation model')
    parser.add_argument('--output', default='robustness_report.pkl',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Load scores
    if args.scores.endswith('.csv'):
        scores_df = pd.read_csv(args.scores)
    else:
        with open(args.scores, 'rb') as f:
            scores_df = pickle.load(f)
    
    # Initialize analyzer
    analyzer = RobustnessAnalyzer(scores_df, args.model)
    
    # Generate report
    report = analyzer.generate_summary_report()
    
    # Print summary
    print("\nRobustness Analysis Summary:")
    print(f"Mean variance: {report['summary']['overall_mean_variance']:.4f}")
    print(f"% with <5% variance: {report['summary']['variance_below_5_percent']*100:.1f}%")
    print(f"Mean correlation: {report['summary']['overall_mean_correlation']:.4f}")
    print(f"% with >95% correlation: {report['summary']['correlation_above_95_percent']*100:.1f}%")
    
    # Save report
    analyzer.save_report(args.output, report)


if __name__ == "__main__":
    main()