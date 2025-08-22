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
from sklearn.model_selection import train_test_split
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import training components
from pipeline.core.aggregator_training import MLPTrainer, GAMAggregator, load_training_config, determine_training_scale

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Analyzes robustness of aggregation models to rubric variations."""
    
    def __init__(
        self,
        scores_df: pd.DataFrame,
        ground_truth_df: Optional[pd.DataFrame] = None,
        model_type: str = 'mlp',
        use_restructured_data: bool = True
    ):
        """
        Initialize the robustness analyzer.
        
        Args:
            scores_df: DataFrame with scores from all judge variants (original format)
            ground_truth_df: DataFrame with ground truth scores for training
            model_type: Type of model to train ('mlp' or 'gam')
            use_restructured_data: Whether to use properly restructured data for training
        """
        self.model_type = model_type.lower()
        self.ground_truth_df = ground_truth_df
        self.use_restructured_data = use_restructured_data
        
        # CRITICAL FIX: Load proper data structure for meaningful aggregator training
        if use_restructured_data:
            # Use the properly structured data (1000 examples × all judge-variant combinations)
            output_dir = Path(__file__).parent.parent.parent / 'results_full_20250818_215910'
            restructured_path = output_dir / 'restructured_scores.pkl'
            
            if restructured_path.exists():
                logger.info(f"Using restructured data for proper aggregator training: {restructured_path}")
                with open(restructured_path, 'rb') as f:
                    self.training_data = pickle.load(f)
                logger.info(f"Loaded dense training data: {self.training_data.shape} (vs sparse original: {scores_df.shape})")
            else:
                logger.warning(f"Restructured data not found, falling back to original (may give misleading results)")
                self.training_data = scores_df.copy()
                self.use_restructured_data = False
        else:
            self.training_data = scores_df.copy()
            
        # Keep original scores_df for backward compatibility
        self.scores_df = scores_df.copy()
        
        # Parse judge variants
        self.variant_groups = self._parse_variant_groups()
        
        # Load training configuration
        self.training_config = load_training_config()
        
        # Storage for trained models per variant
        self.trained_models = {}
    
    def _parse_variant_groups(self) -> Dict[str, Dict[str, str]]:
        """
        Parse judge columns into base judges and their variants.
        
        Returns:
            Dictionary mapping base judge names to variant columns
        """
        variant_groups = {}
        
        for col in self.scores_df.columns:
            if col != 'example_idx' and '_' in col:
                # Split on the last underscore to separate judge name from variant
                parts = col.rsplit('_', 1)
                if len(parts) == 2:
                    base_judge, variant_type = parts
                    
                    if base_judge not in variant_groups:
                        variant_groups[base_judge] = {}
                    
                    variant_groups[base_judge][variant_type] = col
        
        logger.info(f"Found {len(variant_groups)} base judges with variants")
        return variant_groups
    
    def _train_variant_model(self, variant_name: str, judge_cols: List[str]) -> Optional[object]:
        """
        Train a new aggregator model for a specific rubric variant using restructured data.
        
        CRITICAL CHANGE: Now trains on the same 1000 examples with different judge variants
        to properly test robustness across rubric changes.
        
        Args:
            variant_name: Name of the variant (e.g., 'original', 'strict', 'lenient')
            judge_cols: List of judge column names to use (from restructured data)
            
        Returns:
            Trained model or None if training fails
        """
        if self.ground_truth_df is None:
            logger.warning(f"No ground truth data available for training {variant_name} model")
            return None
            
        try:
            if self.use_restructured_data:
                # Use restructured data: same 1000 examples, different judge variants
                judge_scores = self.training_data[judge_cols].values
                logger.info(f"Training {variant_name} on restructured data: {judge_scores.shape}")
            else:
                # Fallback to original method (less meaningful)
                n_examples = min(1000, len(self.ground_truth_df))
                judge_scores = self.scores_df[judge_cols].iloc[:n_examples].values
                logger.warning(f"Using original method for {variant_name} (results may be misleading)")}
            
            # Get corresponding ground truth scores
            # Check for ground truth score column (try multiple possible names)
            score_column = None
            for col_name in ['human_score', 'human_feedback', 'score', 'feedback']:
                if col_name in self.ground_truth_df.columns:
                    score_column = col_name
                    break
            
            if score_column is None:
                logger.error(f"Ground truth data must have a score column. Available columns: {list(self.ground_truth_df.columns)}")
                return None
            
            # Extract numeric scores from the ground truth data
            raw_scores = self.ground_truth_df[score_column].values
            
            # Handle different data formats
            human_scores = []
            for score_data in raw_scores:
                if isinstance(score_data, dict):
                    # Extract average score from dictionary structure
                    if 'score' in score_data:
                        human_scores.append(float(score_data['score']))
                    elif 'average_score' in score_data:
                        human_scores.append(float(score_data['average_score']))
                    else:
                        # If no direct score, try to compute from personas
                        logger.warning(f"Complex score structure found, using fallback extraction")
                        human_scores.append(5.0)  # Fallback to neutral score
                elif isinstance(score_data, (int, float)):
                    human_scores.append(float(score_data))
                else:
                    # Try to convert to numeric, fallback to median
                    try:
                        human_scores.append(float(score_data))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert score data to numeric: {type(score_data)}")
                        human_scores.append(5.0)  # Fallback to neutral score
            
            human_scores = np.array(human_scores)
            
            # Ensure human scores match the number of examples (always 1000 for restructured data)
            if self.use_restructured_data:
                # Restructured data always has 1000 examples
                human_scores = human_scores[:1000]
            else:
                # Fallback: truncate to match n_examples
                human_scores = human_scores[:n_examples]
            
            # Final check for matching lengths
            if len(judge_scores) != len(human_scores):
                logger.warning(f"Mismatch after truncation: {len(judge_scores)} judge vs {len(human_scores)} human scores")
                min_len = min(len(judge_scores), len(human_scores))
                judge_scores = judge_scores[:min_len]
                human_scores = human_scores[:min_len]
            
            # Check for NaN values and handle them
            if np.any(np.isnan(judge_scores)):
                logger.warning(f"Found NaN values in judge scores for {variant_name}, filling with median")
                for i in range(judge_scores.shape[1]):
                    col_median = np.nanmedian(judge_scores[:, i])
                    if np.isnan(col_median):
                        col_median = 2.0  # Default mid-range value
                    judge_scores[np.isnan(judge_scores[:, i]), i] = col_median
            
            if np.any(np.isnan(human_scores)):
                logger.warning(f"Found NaN values in human scores for {variant_name}, filling with median")
                human_median = np.nanmedian(human_scores)
                if np.isnan(human_median):
                    human_median = 5.0  # Default mid-range value
                human_scores = np.where(np.isnan(human_scores), human_median, human_scores)
            
            # Split data for training
            X_train, X_val, y_train, y_val = train_test_split(
                judge_scores, human_scores, test_size=0.2, random_state=42
            )
            
            logger.info(f"Training {self.model_type} model for variant '{variant_name}' "
                       f"with {len(X_train)} training and {len(X_val)} validation samples")
            
            if self.model_type == 'mlp':
                # Determine training scale and config
                scale = determine_training_scale(len(X_train))
                mlp_config = self.training_config["mlp_training"].get(
                    scale, self.training_config["mlp_training"]["medium_scale"]
                )
                
                # Create and train MLP
                trainer = MLPTrainer(
                    hidden_dim=mlp_config["hidden_dim"],
                    learning_rate=mlp_config["learning_rate"],
                    batch_size=min(mlp_config["batch_size"], max(2, len(X_train) // 2)),
                    n_epochs=mlp_config["n_epochs"]
                )
                
                train_losses, val_losses = trainer.fit(X_train, y_train, X_val, y_val)
                logger.info(f"MLP training completed for {variant_name}, final val loss: {val_losses[-1]:.4f}")
                return trainer
                
            elif self.model_type == 'gam':
                # Create and train GAM
                gam_config = self.training_config.get("gam_training", {
                    "n_splines": 10,
                    "lam": 0.6
                })
                
                model = GAMAggregator(
                    n_splines=gam_config["n_splines"],
                    lam=gam_config["lam"]
                )
                
                model.fit(X_train, y_train)
                logger.info(f"GAM training completed for {variant_name}")
                return model
            
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to train {self.model_type} model for variant {variant_name}: {e}")
            import traceback
            traceback.print_exc()
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
                if method == 'learned':
                    aggregated_scores[combo_name] = self._aggregate_learned(combo_name, combo_cols)
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
        CRITICAL FIX: Use restructured data format for meaningful comparisons.
        
        Returns:
            Dictionary mapping combination names to lists of column names
        """
        combinations = {}
        
        if self.use_restructured_data:
            # Use restructured data with proper judge-variant columns
            judge_names = ['truthfulness-judge', 'harmlessness-judge', 'helpfulness-judge', 
                          'honesty-judge', 'explanatory-depth-judge', 'instruction-following-judge',
                          'clarity-judge', 'conciseness-judge', 'logical-consistency-judge', 'creativity-judge']
            variant_types = ['original', 'strict', 'lenient', 'bottom_heavy', 'top_heavy']
            
            logger.info(f"Using restructured data format with {len(judge_names)} judges and {len(variant_types)} variants")
            
            # Create pure variant combinations (all judges use same variant)
            for variant_type in variant_types:
                variant_cols = [f"{judge}_{variant_type}" for judge in judge_names]
                # Check which columns actually exist in training data
                existing_cols = [col for col in variant_cols if col in self.training_data.columns]
                
                if len(existing_cols) == 10:  # We want complete sets
                    combinations[variant_type] = existing_cols
                    logger.info(f"Created combination '{variant_type}' with 10 judges (all using {variant_type} variant)")
                else:
                    logger.warning(f"Incomplete variant '{variant_type}': only {len(existing_cols)}/10 judges available")
            
            logger.info(f"Created {len(combinations)} pure variant combinations: {list(combinations.keys())}")
            return combinations
        
        else:
            # Fallback to original method for backward compatibility
            logger.warning("Using original data format - results may be misleading")
            available_cols = [col for col in self.scores_df.columns if col != 'example_idx']
        
        # Parse judge names and variant types from column names
        judge_variants = {}  # judge_name -> [variant_types]
        all_judges = set()  # All unique judge names
        
        for col in available_cols:
            if '_' in col:
                # Handle compound variants like "bottom_heavy" and "top_heavy"
                if '_bottom_heavy' in col:
                    judge_name = col.replace('_bottom_heavy', '')
                    variant_type = 'bottom_heavy'
                elif '_top_heavy' in col:
                    judge_name = col.replace('_top_heavy', '')
                    variant_type = 'top_heavy'
                elif col.endswith('_original'):
                    judge_name = col.replace('_original', '')
                    variant_type = 'original'
                elif col.endswith('_strict'):
                    judge_name = col.replace('_strict', '')
                    variant_type = 'strict'
                elif col.endswith('_lenient'):
                    judge_name = col.replace('_lenient', '')
                    variant_type = 'lenient'
                else:
                    # Fallback: split on last underscore
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2:
                        judge_name, variant_type = parts
                    else:
                        continue
                
                all_judges.add(judge_name)
                if judge_name not in judge_variants:
                    judge_variants[judge_name] = {}
                judge_variants[judge_name][variant_type] = col
        
        logger.info(f"Found {len(all_judges)} unique judges: {sorted(all_judges)}")
        
        # Get all available variant types
        all_variant_types = set()
        for variants_dict in judge_variants.values():
            all_variant_types.update(variants_dict.keys())
        
        logger.info(f"Found variant types: {sorted(all_variant_types)}")
        
        # Now create actual variant combinations
        # Since we have partial variants, we need to create mixed combinations
        
        # Strategy: For each variant type that exists, create a combination that uses
        # that variant where available, and falls back to 'original' for other judges
        
        for variant_type in sorted(all_variant_types):
            if variant_type == 'original':
                continue  # Handle original separately
                
            combination_cols = []
            variant_judges_used = 0
            
            # For each judge, try to use the requested variant, fallback to original
            for judge_name in sorted(all_judges):
                if judge_name in judge_variants:
                    if variant_type in judge_variants[judge_name]:
                        # Use the requested variant for this judge
                        combination_cols.append(judge_variants[judge_name][variant_type])
                        variant_judges_used += 1
                    elif 'original' in judge_variants[judge_name]:
                        # Fallback to original for this judge
                        combination_cols.append(judge_variants[judge_name]['original'])
                    else:
                        # Use the first available variant for this judge
                        first_variant = list(judge_variants[judge_name].keys())[0]
                        combination_cols.append(judge_variants[judge_name][first_variant])
            
            # Only create combination if we have 10 judges and at least some use the variant
            if len(combination_cols) == 10 and variant_judges_used > 0:
                combinations[variant_type] = combination_cols
                logger.info(f"Created combination '{variant_type}' with 10 judges ({variant_judges_used} using {variant_type} variant)")
            else:
                logger.info(f"Variant '{variant_type}': {len(combination_cols)} judges, {variant_judges_used} using variant")
        
        # Create original combination (baseline)
        original_cols = []
        for judge_name in sorted(all_judges):
            if judge_name in judge_variants and 'original' in judge_variants[judge_name]:
                original_cols.append(judge_variants[judge_name]['original'])
            elif judge_name in judge_variants:
                # If no original, use first available
                first_variant = list(judge_variants[judge_name].keys())[0]
                original_cols.append(judge_variants[judge_name][first_variant])
        
        if len(original_cols) == 10:
            combinations['original'] = original_cols
            logger.info(f"Created 'original' combination with 10 judges")
        
        # If still no combinations, create fallback
        if not combinations and len(available_cols) >= 10:
            combinations['fallback'] = available_cols[:10]
            logger.info(f"Created 'fallback' combination with first 10 columns")
        
        logger.info(f"Created {len(combinations)} variant combinations: {list(combinations.keys())}")
        return combinations
    
    def _aggregate_learned(self, combo_name: str, judge_cols: List[str]) -> np.ndarray:
        """
        Aggregate using a model trained specifically for this variant combination.
        
        Args:
            combo_name: Name of the variant combination
            judge_cols: List of judge column names to use
            
        Returns:
            Predicted human preference scores
        """
        # Check if we already have a trained model for this combination
        if combo_name not in self.trained_models:
            logger.info(f"Training new {self.model_type} model for variant '{combo_name}'")
            model = self._train_variant_model(combo_name, judge_cols)
            if model is None:
                logger.warning(f"Failed to train model for {combo_name}, returning NaN values")
                return np.full(len(self.scores_df), np.nan)
            self.trained_models[combo_name] = model
        
        model = self.trained_models[combo_name]
        
        # Get scores for the specified judges - use appropriate data source
        if self.use_restructured_data:
            # Use training data (restructured) for predictions to match training
            scores_matrix = self.training_data[judge_cols].values
            logger.info(f"Using trained {self.model_type} model for '{combo_name}' on restructured data, input shape: {scores_matrix.shape}")
        else:
            # Fallback to original method
            scores_matrix = self.scores_df[judge_cols].values
            logger.info(f"Using trained {self.model_type} model for '{combo_name}' on original data, input shape: {scores_matrix.shape}")
        
        # Handle missing values (fill with median for each judge)
        for i in range(scores_matrix.shape[1]):
            col_median = np.nanmedian(scores_matrix[:, i])
            scores_matrix[np.isnan(scores_matrix[:, i]), i] = col_median
        
        try:
            if self.model_type == 'mlp':
                # Use MLPTrainer's predict method
                predictions = model.predict(scores_matrix)
                logger.info(f"MLP predictions for '{combo_name}': shape {predictions.shape}, sample values: {predictions[:5]}")
                return predictions
            
            elif self.model_type == 'gam':
                # Use GAM's predict method
                predictions = model.predict(scores_matrix)
                logger.info(f"GAM predictions for '{combo_name}': shape {predictions.shape}, sample values: {predictions[:5]}")
                return predictions
            
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                return np.full(len(self.scores_df), np.nan)
                
        except Exception as e:
            logger.error(f"Failed to predict with {self.model_type} model for {combo_name}: {e}")
            import traceback
            traceback.print_exc()
            return np.full(len(self.scores_df), np.nan)
    
    def _aggregate_mean(self, judge_cols: List[str]) -> np.ndarray:
        """Aggregate using simple mean."""
        if self.use_restructured_data:
            scores_matrix = self.training_data[judge_cols].values
        else:
            scores_matrix = self.scores_df[judge_cols].values
        return np.nanmean(scores_matrix, axis=1)
    
    def _aggregate_single_best(self, judge_cols: List[str]) -> np.ndarray:
        """Aggregate using single best judge (highest correlation with ground truth)."""
        data_source = self.training_data if self.use_restructured_data else self.scores_df
        
        if 'ground_truth' not in data_source.columns:
            # Fallback: use first judge
            return data_source[judge_cols[0]].values
        
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
    
    def create_robustness_plots(self, output_dir: str, report: Dict):
        """
        Create comprehensive robustness visualization plots.
        
        Args:
            output_dir: Directory to save plots
            report: Robustness report from generate_summary_report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        plot_functions = [
            ("Variance Comparison", self._plot_variance_comparison),
            ("Correlation Heatmaps", self._plot_correlation_heatmaps),
            ("Variance Distribution", self._plot_variance_distribution),
            ("Judge Robustness Heatmap", self._plot_judge_robustness_heatmap),
        ]
        
        for plot_name, plot_func in plot_functions:
            try:
                logger.info(f"Generating {plot_name}...")
                plot_func(report, output_dir)
            except Exception as e:
                logger.error(f"Failed to generate {plot_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # 5. Aggregator Performance Comparison (conditional)
        if 'aggregator_robustness' in report:
            try:
                logger.info("Generating Aggregator Performance Comparison...")
                self._plot_aggregator_comparison(report, output_dir)
            except Exception as e:
                logger.error(f"Failed to generate Aggregator Performance Comparison: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"Saved robustness plots to {output_dir}")
    
    def _plot_variance_comparison(self, report: Dict, output_dir: Path):
        """Create variance comparison bar chart."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Prepare data for plotting
        judges = []
        variances = []
        
        for judge_name, metrics in report['score_variance'].items():
            judges.append(judge_name.replace('-', '\n'))  # Line break for readability
            variances.append(metrics['mean_variance'])
        
        # Create bar plot
        bars = ax.bar(judges, variances, color=plt.cm.Set3(np.linspace(0, 1, len(judges))))
        
        # Add 5% threshold line
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                   label='5% Robustness Threshold')
        
        # Formatting
        ax.set_title('Judge Score Variance Across Rubric Variations', fontsize=16, fontweight='bold')
        ax.set_xlabel('Judge', fontsize=12)
        ax.set_ylabel('Mean Variance', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, variance in zip(bars, variances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{variance:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'variance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, report: Dict, output_dir: Path):
        """Create correlation heatmaps for each judge."""
        correlation_data = report['cross_rubric_correlation']
        
        if len(correlation_data) == 0:
            return
        
        # Create subplot grid
        n_judges = len(correlation_data)
        cols = min(3, n_judges)
        rows = (n_judges + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_judges == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (judge_name, metrics) in enumerate(correlation_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Extract correlation matrix
            correlations = metrics['individual_correlations']
            if not correlations:
                ax.text(0.5, 0.5, 'No correlations', ha='center', va='center')
                ax.set_title(judge_name.replace('-', '\n'))
                continue
            
            # Build correlation matrix - handle missing '_vs_' separator
            variant_names = set()
            for key in correlations.keys():
                if '_vs_' in key:
                    parts = key.split('_vs_')
                    if len(parts) >= 2:
                        variant_names.add(parts[0])
                        variant_names.add(parts[1])
            variants = sorted(variant_names)
            
            corr_matrix = np.eye(len(variants))
            variant_to_idx = {v: i for i, v in enumerate(variants)}
            
            for key, corr_data in correlations.items():
                if '_vs_' in key:
                    parts = key.split('_vs_')
                    if len(parts) >= 2:
                        v1, v2 = parts[0], parts[1]
                        if v1 in variant_to_idx and v2 in variant_to_idx:
                            i, j = variant_to_idx[v1], variant_to_idx[v2]
                            r = corr_data['pearson_r']
                            if not np.isnan(r):
                                corr_matrix[i, j] = corr_matrix[j, i] = r
            
            # Skip if no variants found
            if len(variants) == 0:
                ax.text(0.5, 0.5, 'No variants found', ha='center', va='center')
                ax.set_title(judge_name.replace('-', '\n'))
                continue
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(variants)))
            ax.set_yticks(range(len(variants)))
            ax.set_xticklabels(variants, rotation=45, ha='right')
            ax.set_yticklabels(variants)
            
            # Add correlation values
            for i in range(len(variants)):
                for j in range(len(variants)):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=9)
            
            ax.set_title(judge_name.replace('-', '\n'), fontsize=12)
        
        # Remove unused subplots
        for idx in range(n_judges, len(axes)):
            axes[idx].remove()
        
        plt.suptitle('Cross-Rubric Correlation Heatmaps by Judge', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_variance_distribution(self, report: Dict, output_dir: Path):
        """Create variance distribution box plots."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Prepare data
        variance_data = []
        judge_names = []
        
        for judge_name, metrics in report['score_variance'].items():
            variance_dist = metrics['variance_distribution']
            variance_data.extend(variance_dist)
            judge_names.extend([judge_name] * len(variance_dist))
        
        # Create DataFrame for seaborn
        plot_df = pd.DataFrame({
            'Judge': judge_names,
            'Variance': variance_data
        })
        
        # Create box plot
        sns.boxplot(data=plot_df, x='Judge', y='Variance', ax=ax)
        
        # Add 5% threshold line
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                   label='5% Robustness Threshold')
        
        # Formatting
        ax.set_title('Score Variance Distribution by Judge', fontsize=16, fontweight='bold')
        ax.set_xlabel('Judge', fontsize=12)
        ax.set_ylabel('Variance', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'variance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_judge_robustness_heatmap(self, report: Dict, output_dir: Path):
        """Create heatmap showing robustness metrics across judges."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data
        judges = list(report['score_variance'].keys())
        variance_metrics = [report['score_variance'][j]['mean_variance'] for j in judges]
        correlation_metrics = [report['cross_rubric_correlation'][j]['mean_pearson'] 
                              for j in judges if j in report['cross_rubric_correlation']]
        
        # Pad correlation metrics if needed
        while len(correlation_metrics) < len(judges):
            correlation_metrics.append(np.nan)
        
        # Create heatmap data
        heatmap_data = np.array([variance_metrics, correlation_metrics]).T
        
        # Variance heatmap
        im1 = ax1.imshow(heatmap_data[:, [0]], cmap='Reds', aspect='auto')
        ax1.set_yticks(range(len(judges)))
        ax1.set_yticklabels([j.replace('-', '\n') for j in judges])
        ax1.set_xticks([0])
        ax1.set_xticklabels(['Variance'])
        ax1.set_title('Mean Variance by Judge', fontweight='bold')
        
        # Add values
        for i, variance in enumerate(variance_metrics):
            ax1.text(0, i, f'{variance:.3f}', ha='center', va='center', 
                    color='white' if variance > 0.025 else 'black', fontsize=10)
        
        # Correlation heatmap
        im2 = ax2.imshow(heatmap_data[:, [1]], cmap='Blues', aspect='auto')
        ax2.set_yticks(range(len(judges)))
        ax2.set_yticklabels([j.replace('-', '\n') for j in judges])
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Correlation'])
        ax2.set_title('Mean Correlation by Judge', fontweight='bold')
        
        # Add values
        for i, corr in enumerate(correlation_metrics):
            if not np.isnan(corr):
                ax2.text(0, i, f'{corr:.3f}', ha='center', va='center', 
                        color='white' if corr < 0.5 else 'black', fontsize=10)
        
        plt.suptitle('Judge Robustness Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'judge_robustness_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_aggregator_comparison(self, report: Dict, output_dir: Path):
        """Create aggregator performance comparison plot."""
        aggregator_data = report['aggregator_robustness']
        if not aggregator_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        methods = list(aggregator_data.keys())
        mean_variances = [aggregator_data[m]['mean_variance'] for m in methods]
        max_variances = [aggregator_data[m]['max_variance'] for m in methods]
        
        # Variance comparison
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mean_variances, width, label='Mean Variance', alpha=0.8)
        bars2 = ax1.bar(x + width/2, max_variances, width, label='Max Variance', alpha=0.8)
        
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                   label='5% Threshold')
        
        ax1.set_xlabel('Aggregation Method')
        ax1.set_ylabel('Variance')
        ax1.set_title('Aggregator Robustness Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, mean_variances):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Improvement factors
        if 'mean' in aggregator_data and 'learned' in aggregator_data:
            baseline_var = aggregator_data['mean']['mean_variance']
            improvement_factors = []
            method_labels = []
            
            for method in methods:
                if method != 'mean' and not np.isnan(aggregator_data[method]['mean_variance']):
                    method_var = aggregator_data[method]['mean_variance']
                    if method_var > 0:  # Avoid division by zero
                        factor = baseline_var / method_var
                        improvement_factors.append(factor)
                        method_labels.append(method)
                    else:
                        # Perfect robustness case
                        improvement_factors.append(float('inf'))
                        method_labels.append(method)
            
            if improvement_factors:
                bars3 = ax2.bar(method_labels, improvement_factors, color='green', alpha=0.7)
                ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                           label='No Improvement')
                ax2.set_xlabel('Aggregation Method')
                ax2.set_ylabel('Improvement Factor vs Mean Baseline')
                ax2.set_title('Robustness Improvement Factors')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, factor in zip(bars3, improvement_factors):
                    if np.isfinite(factor):
                        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                                f'{factor:.1f}x', ha='center', va='bottom', fontsize=10)
                    else:
                        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 0.5,
                                'Perfect', ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'aggregator_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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