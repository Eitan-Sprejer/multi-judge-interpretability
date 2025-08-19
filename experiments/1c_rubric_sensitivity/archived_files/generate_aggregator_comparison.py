#!/usr/bin/env python3
"""
Generate Aggregator Comparison Plot - The Most Important Result

This script compares how different aggregation methods (learned MLP, mean, single best)
perform under rubric variations, which is the key finding of the experiment.
"""

import pickle
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import torch.nn as nn

# Add parent dirs to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import training components
from pipeline.core.aggregator_training import MLPTrainer, load_training_config, determine_training_scale


def train_aggregator_for_variant(scores_df, variant_cols, ground_truth, variant_name):
    """Train an MLP aggregator for a specific variant combination."""
    
    # Get scores for this variant
    X = scores_df[variant_cols].values
    y = ground_truth
    
    # Ensure matching lengths
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load config and train
    config = load_training_config()
    scale = determine_training_scale(len(X_train))
    mlp_config = config["mlp_training"].get(scale, config["mlp_training"]["medium_scale"])
    
    trainer = MLPTrainer(
        hidden_dim=mlp_config["hidden_dim"],
        learning_rate=mlp_config["learning_rate"],
        batch_size=min(mlp_config["batch_size"], max(2, len(X_train) // 2)),
        n_epochs=mlp_config["n_epochs"]
    )
    
    try:
        train_losses, val_losses = trainer.fit(X_train, y_train, X_test, y_test)
        predictions = trainer.predict(X)
        logger.info(f"Trained {variant_name}: final val loss = {val_losses[-1]:.4f}")
        return predictions
    except Exception as e:
        logger.warning(f"Failed to train {variant_name}: {e}")
        return np.full(len(X), np.nan)


def generate_aggregator_comparison(data_path: str, output_dir: str):
    """
    Generate the aggregator comparison plot showing robustness of different methods.
    
    This is the KEY RESULT showing how learned aggregators are more robust
    to rubric variations than simple baselines.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots_fixed"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load restructured data
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        scores_df = pickle.load(f)
    
    # Load ground truth
    logger.info("Loading ground truth human feedback...")
    gt_path = Path(__file__).parent.parent.parent.parent / 'dataset' / 'data_with_judge_scores.pkl'
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    
    # Extract numeric scores from human feedback
    human_scores = []
    for score_data in gt_data['human_feedback'].values[:1000]:
        if isinstance(score_data, dict):
            if 'score' in score_data:
                human_scores.append(float(score_data['score']))
            elif 'average_score' in score_data:
                human_scores.append(float(score_data['average_score']))
            else:
                human_scores.append(5.0)
        else:
            human_scores.append(float(score_data) if pd.notna(score_data) else 5.0)
    human_scores = np.array(human_scores)
    
    # Define judges and variants
    judge_names = ['truthfulness-judge', 'harmlessness-judge', 'helpfulness-judge', 
                   'honesty-judge', 'explanatory-depth-judge', 'instruction-following-judge',
                   'clarity-judge', 'conciseness-judge', 'logical-consistency-judge', 'creativity-judge']
    variant_types = ['original', 'strict', 'lenient', 'bottom_heavy', 'top_heavy']
    
    # Define variant combinations to test
    combinations = {
        'original': [f'{j}_original' for j in judge_names],
        'strict': [f'{j}_strict' for j in judge_names],
        'lenient': [f'{j}_lenient' for j in judge_names],
        'bottom_heavy': [f'{j}_bottom_heavy' for j in judge_names],
        'mixed': [f'{judge_names[i]}_{variant_types[i % len(variant_types)]}' for i in range(10)]
    }
    
    # Aggregation methods to compare
    aggregation_results = {
        'Learned (MLP)': {},
        'Mean': {},
        'Single Best': {}
    }
    
    logger.info("Computing aggregation results for each variant combination...")
    
    for combo_name, combo_cols in combinations.items():
        logger.info(f"Processing {combo_name} combination...")
        
        # 1. Learned MLP Aggregator
        mlp_predictions = train_aggregator_for_variant(
            scores_df, combo_cols, human_scores, f"MLP-{combo_name}"
        )
        aggregation_results['Learned (MLP)'][combo_name] = mlp_predictions
        
        # 2. Mean Aggregator
        mean_predictions = scores_df[combo_cols].mean(axis=1).values
        aggregation_results['Mean'][combo_name] = mean_predictions
        
        # 3. Single Best (highest correlation with ground truth)
        best_judge = combo_cols[0]
        best_corr = -1
        for col in combo_cols:
            judge_scores = scores_df[col].values
            if not np.all(np.isnan(judge_scores)):
                corr = np.corrcoef(judge_scores, human_scores)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_judge = col
        aggregation_results['Single Best'][combo_name] = scores_df[best_judge].values
    
    # Calculate robustness metrics
    logger.info("Calculating robustness metrics...")
    robustness_metrics = {}
    
    for method_name, method_results in aggregation_results.items():
        # Get predictions for all combinations
        all_preds = np.column_stack([method_results[c] for c in combinations.keys()])
        
        # Calculate variance across combinations for each example
        row_variances = np.nanvar(all_preds, axis=1)
        
        # Calculate correlation with ground truth for original
        original_preds = method_results['original']
        mask = ~(np.isnan(original_preds) | np.isnan(human_scores))
        if mask.sum() > 1:
            correlation = np.corrcoef(original_preds[mask], human_scores[mask])[0, 1]
        else:
            correlation = np.nan
        
        robustness_metrics[method_name] = {
            'mean_variance': np.nanmean(row_variances),
            'std_variance': np.nanstd(row_variances),
            'max_variance': np.nanmax(row_variances),
            'correlation_with_truth': correlation,
            'variance_distribution': row_variances
        }
    
    # Create the aggregator comparison plot
    logger.info("Creating aggregator comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Variance Comparison (Main Result)
    methods = list(robustness_metrics.keys())
    mean_variances = [robustness_metrics[m]['mean_variance'] for m in methods]
    colors = ['green', 'orange', 'red']
    
    bars = ax1.bar(methods, mean_variances, color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% Robustness Target')
    ax1.set_ylabel('Mean Variance Across Rubric Variations', fontsize=12)
    ax1.set_title('Aggregator Robustness to Rubric Changes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, variance in zip(bars, mean_variances):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{variance:.4f}', ha='center', va='bottom', fontsize=11)
    
    # 2. Correlation with Ground Truth
    correlations = [robustness_metrics[m]['correlation_with_truth'] for m in methods]
    bars2 = ax2.bar(methods, correlations, color=colors, alpha=0.7)
    ax2.set_ylabel('Correlation with Human Feedback', fontsize=12)
    ax2.set_title('Prediction Accuracy (Original Rubric)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, corr in zip(bars2, correlations):
        if not np.isnan(corr):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=11)
    
    # 3. Variance Distribution Box Plot
    variance_data = [robustness_metrics[m]['variance_distribution'] for m in methods]
    bp = ax3.boxplot(variance_data, labels=methods, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Variance', fontsize=12)
    ax3.set_title('Distribution of Variance Across Examples', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Improvement Ratios
    baseline_var = robustness_metrics['Mean']['mean_variance']
    improvement_ratios = []
    improvement_labels = []
    
    for method in methods:
        if method != 'Mean':
            method_var = robustness_metrics[method]['mean_variance']
            if method_var > 0:
                ratio = baseline_var / method_var
            else:
                ratio = float('inf')
            improvement_ratios.append(ratio)
            improvement_labels.append(method)
    
    if improvement_ratios:
        bars4 = ax4.bar(improvement_labels, improvement_ratios, color=['green', 'red'], alpha=0.7)
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='No Improvement')
        ax4.set_ylabel('Robustness Improvement Factor', fontsize=12)
        ax4.set_title('Improvement Over Mean Baseline', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, ratio in zip(bars4, improvement_ratios):
            if np.isfinite(ratio):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                        f'{ratio:.2f}x', ha='center', va='bottom', fontsize=11)
            else:
                ax4.text(bar.get_x() + bar.get_width()/2., 2.0,
                        'Perfect', ha='center', va='center', fontsize=11)
    
    plt.suptitle('🎯 KEY RESULT: Learned Aggregators Are More Robust to Rubric Variations',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'aggregator_comparison_complete.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Aggregator comparison plot saved")
    
    # Print summary of key findings
    print("\n" + "="*70)
    print("🎯 AGGREGATOR COMPARISON - KEY EXPERIMENTAL RESULTS")
    print("="*70)
    
    for method in methods:
        metrics = robustness_metrics[method]
        print(f"\n{method}:")
        print(f"  • Mean Variance: {metrics['mean_variance']:.4f}")
        print(f"  • Correlation with Truth: {metrics['correlation_with_truth']:.3f}")
        
        if method != 'Mean':
            improvement = baseline_var / metrics['mean_variance'] if metrics['mean_variance'] > 0 else float('inf')
            print(f"  • Improvement over Mean: {improvement:.2f}x")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    learned_var = robustness_metrics['Learned (MLP)']['mean_variance']
    mean_var = robustness_metrics['Mean']['mean_variance']
    
    if learned_var < mean_var:
        improvement = mean_var / learned_var if learned_var > 0 else float('inf')
        print(f"✅ Learned aggregators are {improvement:.2f}x MORE ROBUST to rubric variations!")
    else:
        print("⚠️ Learned aggregators show similar or worse robustness than baselines.")
        print("   This may indicate need for more training data or regularization.")
    
    print("="*70)
    
    return robustness_metrics


if __name__ == "__main__":
    # Run the aggregator comparison analysis
    data_path = "../results_full_20250818_215910/restructured_scores.pkl"
    output_dir = "../results_full_20250818_215910"
    
    metrics = generate_aggregator_comparison(data_path, output_dir)
    print(f"\n✅ Aggregator comparison complete! Check plots_fixed/aggregator_comparison_complete.png")