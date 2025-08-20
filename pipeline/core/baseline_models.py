#!/usr/bin/env python3
"""
Unified Baseline Models for Multi-Judge Aggregation

This module provides standardized baseline implementations that can be used
across all experiments for consistent comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .persona_simulation import PERSONAS
import random

def compute_metrics(y_true, y_pred):
    """Compute standard evaluation metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def contaminate_human_feedback(y_clean: np.ndarray, contamination_rate: float, strategy: str = 'inversion', seed: int = 42) -> np.ndarray:
    """
    Contaminate human feedback scores using different strategies.
    
    Args:
        y_clean: Clean human feedback scores
        contamination_rate: Fraction of samples to contaminate (0.0 to 1.0)
        strategy: Contamination strategy ('inversion', 'random_bias', 'scaled_down')
        seed: Random seed for reproducibility
    
    Returns:
        Contaminated human feedback scores
    """
    if contamination_rate == 0:
        return y_clean.copy()
    
    np.random.seed(seed)
    y_contaminated = y_clean.copy()
    n_contaminate = int(len(y_contaminated) * contamination_rate)
    contaminate_indices = np.random.choice(len(y_contaminated), n_contaminate, replace=False)
    
    for idx in contaminate_indices:
        original = float(y_contaminated[idx])
        
        if strategy == 'inversion':
            # Invert score: 10 - original
            contaminated = 10 - original
            
        elif strategy == 'systematic_bias':
            # Systematic bias: consistent +2 or -2 offset per contaminated annotator
            # Use index to ensure same annotator always gets same bias
            np.random.seed(seed + idx)  # Deterministic bias per sample
            bias = np.random.choice([-2, 2])
            contaminated = np.clip(original + bias, 0, 10)
            
        elif strategy == 'random_noise':
            # Random noise: ±1-3 random error per rating
            noise = np.random.uniform(-3, 3)
            contaminated = np.clip(original + noise, 0, 10)
            
        elif strategy == 'scaled_down':
            # Scale compression: [0,10] to [3,7] range (avoiding extremes)
            # Map 0->3, 10->7, linear interpolation between
            contaminated = 3 + (original / 10) * (7 - 3)
            
        else:
            raise ValueError(f"Unknown contamination strategy: {strategy}")
        
        y_contaminated[idx] = float(contaminated)
    
    return y_contaminated

class BaselineEvaluator:
    """
    Unified baseline evaluation system with multiple baseline approaches.
    
    Implements consistent methodology across experiments:
    - Uniform persona sampling
    - Multiple baseline strategies
    - Consistent train/test splits
    """
    
    def __init__(self, random_seed: int = 42, test_size: float = 0.2):
        self.random_seed = random_seed
        self.test_size = test_size
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def prepare_data_uniform_sampling(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data using uniform persona sampling (same as main experiment).
        
        Returns:
            X: Judge scores array (n_samples, n_judges)
            y: Human feedback scores array (n_samples,)
        """
        # Set up uniform persona assignment
        available_personas = list(PERSONAS.keys())
        samples_per_persona = len(df) // len(available_personas)
        remaining_samples = len(df) % len(available_personas)
        
        persona_assignment = []
        for persona in available_personas:
            persona_assignment.extend([persona] * samples_per_persona)
        for _ in range(remaining_samples):
            persona_assignment.append(random.choice(available_personas))
        random.shuffle(persona_assignment)
        
        # Extract features and targets
        X_list = []
        y_list = []
        
        for idx, (row, assigned_persona) in enumerate(zip(df.iterrows(), persona_assignment)):
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
    
    def naive_mean_baseline(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Naive Mean Baseline: Simple average of all judge scores.
        No scaling or training involved. Judges are [0-4], personas are [0-10].
        """
        # Simple mean of judges - NO SCALING (truly naive)
        y_pred = np.mean(X_test, axis=1)
        
        return {
            'method': 'Simple average of all judge scores (no scaling)',
            'approach': 'Direct averaging, judges [0-4] vs personas [0-10]',
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }
    
    def linear_scaling_mean_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Linear Scaling Mean Baseline: Scale judge mean proportionally from [0-4] to [0-10].
        This is a proper heuristic scaling (no training involved).
        """
        # Compute means
        judge_mean_test = np.mean(X_test, axis=1)
        
        # Proportional scaling: judges [0-4] to personas [0-10]
        # y = (x / 4) * 10 = x * 2.5
        y_pred = judge_mean_test * 2.5
        
        return {
            'method': 'Mean of judges scaled proportionally from [0-4] to [0-10]',
            'approach': 'Proportional scaling (multiply by 2.5)',
            'scaling_params': {'scaling_factor': 2.5, 'judge_range': [0, 4], 'target_range': [0, 10]},
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }
    
    def standardscaler_lr_mean_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        StandardScaler + LinearRegression Mean Baseline: Learn optimal mapping.
        This is what the persona poisoning experiment originally used.
        """
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train on mean of normalized judges
        X_train_mean = X_train_scaled.mean(axis=1).reshape(-1, 1)
        X_test_mean = X_test_scaled.mean(axis=1).reshape(-1, 1)
        
        lr = LinearRegression()
        lr.fit(X_train_mean, y_train)
        y_pred = lr.predict(X_test_mean)
        
        return {
            'method': 'Mean of judges with StandardScaler + LinearRegression',
            'approach': 'Learned optimal mapping (trained baseline)',
            'model_params': {'slope': lr.coef_[0], 'intercept': lr.intercept_},
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }
    
    def best_single_judge_naive(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               judge_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Best Single Judge Baseline: Use raw scores from best performing judge.
        No scaling or training involved.
        """
        if judge_names is None:
            judge_names = [f'Judge_{i}' for i in range(X_train.shape[1])]
        
        # Find best judge based on correlation with training targets
        best_r2 = -1
        best_judge_idx = 0
        best_pred = None
        
        for judge_idx in range(X_train.shape[1]):
            y_pred = X_test[:, judge_idx]
            r2 = r2_score(y_test, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_judge_idx = judge_idx
                best_pred = y_pred
        
        return {
            'method': f'Best single judge: {judge_names[best_judge_idx]}',
            'approach': 'Direct judge scores (no scaling)',
            'judge_index': best_judge_idx,
            'judge_name': judge_names[best_judge_idx],
            'metrics': compute_metrics(y_test, best_pred),
            'predictions': best_pred
        }
    
    def best_single_judge_linear_scaling(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: np.ndarray, y_test: np.ndarray,
                                        judge_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Best Single Judge with Proportional Scaling: Scale best judge from [0-4] to [0-10].
        This is a proper heuristic scaling (no training involved).
        """
        if judge_names is None:
            judge_names = [f'Judge_{i}' for i in range(X_train.shape[1])]
        
        # Find best judge with proportional scaling
        best_r2 = -1
        best_judge_idx = 0
        best_pred = None
        
        for judge_idx in range(X_train.shape[1]):
            judge_scores_test = X_test[:, judge_idx]
            
            # Proportional scaling: judges [0-4] to personas [0-10]
            y_pred = judge_scores_test * 2.5
            
            r2 = r2_score(y_test, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_judge_idx = judge_idx
                best_pred = y_pred
        
        return {
            'method': f'Best single judge with proportional scaling: {judge_names[best_judge_idx]}',
            'approach': 'Proportional scaling from [0-4] to [0-10] (multiply by 2.5)',
            'judge_index': best_judge_idx,
            'judge_name': judge_names[best_judge_idx],
            'scaling_params': {'scaling_factor': 2.5, 'judge_range': [0, 4], 'target_range': [0, 10]},
            'metrics': compute_metrics(y_test, best_pred),
            'predictions': best_pred
        }
    
    def best_single_judge_standardscaler_lr(self, X_train: np.ndarray, y_train: np.ndarray,
                                           X_test: np.ndarray, y_test: np.ndarray,
                                           judge_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Best Single Judge with StandardScaler + LinearRegression: Learn optimal mapping.
        """
        if judge_names is None:
            judge_names = [f'Judge_{i}' for i in range(X_train.shape[1])]
        
        # Normalize all features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Find best judge with learned mapping
        best_r2 = -1
        best_judge_idx = 0
        best_pred = None
        best_model_params = None
        
        for judge_idx in range(X_train_scaled.shape[1]):
            X_train_single = X_train_scaled[:, judge_idx].reshape(-1, 1)
            X_test_single = X_test_scaled[:, judge_idx].reshape(-1, 1)
            
            lr = LinearRegression()
            lr.fit(X_train_single, y_train)
            y_pred = lr.predict(X_test_single)
            
            r2 = r2_score(y_test, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_judge_idx = judge_idx
                best_pred = y_pred
                best_model_params = {'slope': lr.coef_[0], 'intercept': lr.intercept_}
        
        return {
            'method': f'Best single judge with StandardScaler + LinearRegression: {judge_names[best_judge_idx]}',
            'approach': 'Learned optimal mapping (trained baseline)',
            'judge_index': best_judge_idx,
            'judge_name': judge_names[best_judge_idx],
            'model_params': best_model_params,
            'metrics': compute_metrics(y_test, best_pred),
            'predictions': best_pred
        }
    
    # =================================================================================
    # CONTAMINATION-AWARE BASELINES (Tier 1)
    # These baselines train on contaminated human feedback and should be compared 
    # against the aggregator under the same contamination conditions.
    # =================================================================================
    
    def linear_regression_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 normalize: bool = True) -> Dict[str, Any]:
        """
        Linear Regression Baseline: Train LinearRegression(judges -> human_feedback).
        This is what most practitioners would try first with contaminated data.
        """
        if normalize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train linear regression on (potentially contaminated) human feedback
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)
        
        return {
            'method': 'Linear Regression (judges -> human feedback)',
            'approach': f'Naive approach (normalization: {normalize})',
            'model_type': 'linear_regression',
            'normalized': normalize,
            'model_params': {
                'coef': lr.coef_.tolist() if hasattr(lr.coef_, 'tolist') else lr.coef_,
                'intercept': float(lr.intercept_)
            },
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }
    
    def ridge_regression_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                alpha: float = 1.0, normalize: bool = True) -> Dict[str, Any]:
        """
        Ridge Regression Baseline: Regularized LinearRegression(judges -> human_feedback).
        Slightly more sophisticated approach that practitioners might use.
        """
        if normalize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train ridge regression on (potentially contaminated) human feedback
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        
        return {
            'method': f'Ridge Regression (alpha={alpha}, judges -> human feedback)',
            'approach': f'Regularized approach (normalization: {normalize})',
            'model_type': 'ridge_regression',
            'normalized': normalize,
            'alpha': alpha,
            'model_params': {
                'coef': ridge.coef_.tolist() if hasattr(ridge.coef_, 'tolist') else ridge.coef_,
                'intercept': float(ridge.intercept_)
            },
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }
    
    def evaluate_contamination_baselines(self, X_train: np.ndarray, y_train_contaminated: np.ndarray,
                                       X_test: np.ndarray, y_test_clean: np.ndarray,
                                       contamination_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate contamination-aware baselines that train on contaminated human feedback.
        
        Args:
            X_train: Judge scores for training
            y_train_contaminated: Contaminated human feedback for training  
            X_test: Judge scores for testing
            y_test_clean: Clean human feedback for testing
            contamination_info: Dict with 'rate', 'strategy', etc.
            
        Returns:
            Dict with baseline results trained on contaminated data
        """
        baselines = {}
        
        # Tier 1: What practitioners would actually do with contaminated data
        baselines['linear_regression_norm'] = self.linear_regression_baseline(
            X_train, y_train_contaminated, X_test, y_test_clean, normalize=True)
        
        baselines['linear_regression_raw'] = self.linear_regression_baseline(
            X_train, y_train_contaminated, X_test, y_test_clean, normalize=False)
            
        baselines['ridge_regression_norm'] = self.ridge_regression_baseline(
            X_train, y_train_contaminated, X_test, y_test_clean, alpha=1.0, normalize=True)
            
        baselines['ridge_regression_raw'] = self.ridge_regression_baseline(
            X_train, y_train_contaminated, X_test, y_test_clean, alpha=1.0, normalize=False)
        
        # Summary statistics
        r2_scores = {name: result['metrics']['r2'] for name, result in baselines.items()}
        best_baseline = max(r2_scores.keys(), key=lambda k: r2_scores[k])
        
        return {
            'baselines': baselines,
            'contamination_info': contamination_info,
            'summary': {
                'best_contamination_baseline': best_baseline,
                'best_r2': r2_scores[best_baseline],
                'r2_scores': r2_scores,
                'contamination_rate': contamination_info.get('rate', 0),
                'contamination_strategy': contamination_info.get('strategy', 'unknown')
            }
        }
    
    def ultrafeedback_4judge_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        UltraFeedback 4-Judge Baseline: Use only the 4 judges from original UltraFeedback.
        Indices: Truthfulness(0), Helpfulness(2), Honesty(3), Instruction Following(5)
        """
        # Select UltraFeedback judges: Truthfulness, Helpfulness, Honesty, Instruction Following
        ultrafeedback_indices = [0, 2, 3, 5]  
        
        X_test_uf = X_test[:, ultrafeedback_indices]
        
        # Compute mean of 4 judges and scale proportionally
        judge_mean_test = np.mean(X_test_uf, axis=1)
        
        # Proportional scaling: judges [0-4] to personas [0-10]
        y_pred = judge_mean_test * 2.5
        
        return {
            'method': 'UltraFeedback 4-judge subset (Truthfulness, Helpfulness, Honesty, Instruction Following)',
            'approach': 'Proportional scaling of 4 UltraFeedback judges from [0-4] to [0-10]',
            'judge_indices': ultrafeedback_indices,
            'judge_names': ['Truthfulness', 'Helpfulness', 'Honesty', 'Instruction Following'],
            'scaling_params': {'scaling_factor': 2.5, 'judge_range': [0, 4], 'target_range': [0, 10]},
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }
    
    def ultrafeedback_overall_baseline(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray,
                                     df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        UltraFeedback Overall Score Baseline: Use the original overall scores from UltraFeedback.
        This extracts the overall_score from the original UltraFeedback completions.
        """
        # Extract overall scores from the dataframe
        y_pred = []
        missing_count = 0
        
        for idx, row in df_test.iterrows():
            if 'ultrafeedback_overall' in row and row['ultrafeedback_overall'] is not None:
                # Scale from [1-5] to [0-10] (UltraFeedback overall scores are typically 1-5)
                score = float(row['ultrafeedback_overall'])
                scaled_score = (score - 1) / 4 * 10  # Map [1,5] to [0,10]
                y_pred.append(scaled_score)
            else:
                # Fallback: use mean of available scores
                y_pred.append(5.0)  # Middle value
                missing_count += 1
        
        y_pred = np.array(y_pred)
        
        return {
            'method': 'UltraFeedback original overall scores',
            'approach': 'Original dataset overall scores scaled from [1-5] to [0-10]',
            'scaling_params': {'source_range': [1, 5], 'target_range': [0, 10], 'missing_scores': missing_count},
            'metrics': compute_metrics(y_test, y_pred),
            'predictions': y_pred
        }

    def evaluate_all_baselines(self, df: pd.DataFrame, 
                             judge_names: Optional[List[str]] = None,
                             include_contamination_baselines: bool = False,
                             contamination_rate: float = 0.0,
                             contamination_strategy: str = 'inversion') -> Dict[str, Any]:
        """
        Evaluate all baseline approaches on the given dataset.
        
        Args:
            df: Dataset with judge scores and human feedback
            judge_names: Names of judges (for display purposes)
            include_contamination_baselines: Whether to include contamination-aware baselines
            contamination_rate: Rate of contamination for contamination baselines
            contamination_strategy: Strategy for contamination
        
        Returns:
            Comprehensive baseline comparison results
        """
        # Prepare data with uniform persona sampling
        X, y = self.prepare_data_uniform_sampling(df)
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data: {len(X)} samples. Need at least 10.")
        
        # Split data consistently
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )
        
        # Default judge names
        if judge_names is None:
            judge_names = [
                'Truthfulness / Factual Accuracy',
                'Harmlessness / Safety', 
                'Helpfulness / Utility',
                'Honesty / Transparency',
                'Explanatory Depth / Detail',
                'Instruction Following / Compliance',
                'Clarity / Understandability',
                'Conciseness / Efficiency',
                'Logical Consistency / Reasoning',
                'Creativity / Originality'
            ]
        
        # Evaluate all baselines
        baselines = {}
        
        # Heuristic baselines (no training, just scaling/selection heuristics)
        baselines['naive_mean'] = self.naive_mean_baseline(X_train, y_train, X_test, y_test)
        baselines['linear_scaling_mean'] = self.linear_scaling_mean_baseline(X_train, y_train, X_test, y_test)
        baselines['best_judge_linear_scaling'] = self.best_single_judge_linear_scaling(X_train, y_train, X_test, y_test, judge_names)
        baselines['ultrafeedback_4judge'] = self.ultrafeedback_4judge_baseline(X_train, y_train, X_test, y_test)
        
        # UltraFeedback overall baseline (if available in the dataframe)
        if 'ultrafeedback_overall' in df.columns:
            # Split the dataframe to match the train/test split
            df_test = df.iloc[len(X_train):].reset_index(drop=True)
            baselines['ultrafeedback_overall'] = self.ultrafeedback_overall_baseline(X_train, y_train, X_test, y_test, df_test)
        
        # Learned baselines (train parameters on data)
        baselines['standardscaler_lr_mean'] = self.standardscaler_lr_mean_baseline(X_train, y_train, X_test, y_test)
        baselines['best_judge_standardscaler_lr'] = self.best_single_judge_standardscaler_lr(X_train, y_train, X_test, y_test, judge_names)
        
        # Contamination-aware baselines (train on contaminated human feedback)
        if include_contamination_baselines:
            y_train_contaminated = contaminate_human_feedback(
                y_train, contamination_rate, contamination_strategy, seed=self.random_seed
            )
            
            contamination_info = {
                'rate': contamination_rate,
                'strategy': contamination_strategy,
                'n_contaminated': int(len(y_train) * contamination_rate),
                'original_mean': float(y_train.mean()),
                'contaminated_mean': float(y_train_contaminated.mean())
            }
            
            contamination_results = self.evaluate_contamination_baselines(
                X_train, y_train_contaminated, X_test, y_test, contamination_info
            )
            
            # Add contamination baselines to main results
            baselines.update(contamination_results['baselines'])
        
        # Summary statistics  
        r2_scores = {name: result['metrics']['r2'] for name, result in baselines.items()}
        best_baseline = max(r2_scores.keys(), key=lambda k: r2_scores[k])
        
        result = {
            'baselines': baselines,
            'summary': {
                'best_baseline': best_baseline,
                'best_r2': r2_scores[best_baseline],
                'r2_scores': r2_scores,
                'data_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'n_judges': X.shape[1],
                    'persona_sampling': 'uniform',
                    'random_seed': self.random_seed
                }
            }
        }
        
        if include_contamination_baselines:
            result['contamination_info'] = contamination_info
        
        return result

# Convenience functions for backward compatibility
def compute_baseline_comparisons(df: pd.DataFrame, 
                               random_seed: int = 42,
                               test_size: float = 0.2,
                               judge_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute baseline comparisons using the unified system.
    
    This function maintains compatibility with existing code while providing
    the new unified baseline system.
    """
    evaluator = BaselineEvaluator(random_seed=random_seed, test_size=test_size)
    return evaluator.evaluate_all_baselines(df, judge_names)

def get_main_experiment_baselines(df: pd.DataFrame,
                                random_seed: int = 42, 
                                test_size: float = 0.2,
                                judge_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get baselines matching the main experiment methodology.
    
    Returns only the baselines that match what the main experiment uses:
    - Naive mean (for reference)
    - Linear scaling mean (main experiment method)
    - Best judge with linear scaling (main experiment method)
    """
    evaluator = BaselineEvaluator(random_seed=random_seed, test_size=test_size)
    all_results = evaluator.evaluate_all_baselines(df, judge_names)
    
    # Return only the main experiment compatible baselines
    main_baselines = {
        'naive_mean': all_results['baselines']['naive_mean'],
        'linear_scaling_mean': all_results['baselines']['linear_scaling_mean'], 
        'best_judge_linear_scaling': all_results['baselines']['best_judge_linear_scaling']
    }
    
    # Update summary
    r2_scores = {name: result['metrics']['r2'] for name, result in main_baselines.items()}
    best_baseline = max(r2_scores.keys(), key=lambda k: r2_scores[k])
    
    return {
        'baselines': main_baselines,
        'summary': {
            'best_baseline': best_baseline,
            'best_r2': r2_scores[best_baseline],
            'r2_scores': r2_scores,
            'methodology': 'main_experiment_compatible',
            'data_info': all_results['summary']['data_info']
        }
    }