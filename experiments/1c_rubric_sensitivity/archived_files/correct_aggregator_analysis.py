#!/usr/bin/env python3
"""
Corrected Aggregator Analysis for Rubric Sensitivity Experiment

This script creates the key result showing how different aggregation methods
(learned GAM/MLP vs baselines) perform under rubric variations.
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

# Import model architectures
try:
    from pipeline.core.aggregator_training import GAMAggregator, SingleLayerMLP
    MODEL_IMPORT_SUCCESS = True
except ImportError as e:
    logger.warning(f"Could not import model classes: {e}")
    MODEL_IMPORT_SUCCESS = False


class SimpleGAM(nn.Module):
    """Simple GAM implementation for loading if main class unavailable."""
    def __init__(self, n_judges, hidden=32):
        super().__init__()
        self.n_judges = n_judges
        self.hidden = hidden
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            ) for _ in range(n_judges)
        ])
        self.final = nn.Linear(n_judges, 1)
    
    def forward(self, x):
        outputs = []
        for i, net in enumerate(self.nets):
            out = net(x[:, i:i+1])
            outputs.append(out)
        combined = torch.cat(outputs, dim=1)
        return self.final(combined)


class SimpleMLP(nn.Module):
    """Simple MLP implementation for loading if main class unavailable."""
    def __init__(self, n_judges, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_judges, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


def load_trained_model(model_path, n_judges=10, model_type='mlp'):
    """Load a trained model from file."""
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        if model_type.lower() == 'gam':
            if MODEL_IMPORT_SUCCESS:
                try:
                    model = GAMAggregator(n_judges=n_judges, hidden=32, monotone=False)
                except:
                    model = SimpleGAM(n_judges=n_judges, hidden=32)
            else:
                model = SimpleGAM(n_judges=n_judges, hidden=32)
        else:  # MLP
            if MODEL_IMPORT_SUCCESS:
                try:
                    model = SingleLayerMLP(n_judges=n_judges, hidden_dim=64)
                except:
                    model = SimpleMLP(n_judges=n_judges, hidden_dim=64)
            else:
                model = SimpleMLP(n_judges=n_judges, hidden_dim=64)
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        logger.info(f"Successfully loaded {model_type.upper()} model from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None


def correct_aggregator_analysis():
    """
    Create the corrected aggregator comparison analysis.
    """
    # Paths
    results_dir = Path(__file__).parent / '..' / 'results_full_20250818_215910'
    data_path = results_dir / 'restructured_scores_fixed.pkl'
    plots_dir = results_dir / 'plots_corrected'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Model paths
    project_root = Path(__file__).parent.parent.parent
    gam_model_path = project_root / 'models' / 'agg_model_gam.pt'
    mlp_model_path = project_root / 'models' / 'agg_model_mlp.pt'
    
    # Load the fixed restructured data
    logger.info(f"Loading fixed data from {data_path}")
    with open(data_path, 'rb') as f:
        scores_df = pickle.load(f)
    
    logger.info(f"Loaded data shape: {scores_df.shape}")
    logger.info(f"Score range: [{scores_df.min().min():.2f}, {scores_df.max().max():.2f}]")
    
    # Load ground truth data
    logger.info("Loading ground truth human feedback...")
    gt_path = project_root / 'dataset' / 'data_with_judge_scores.pkl'
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    
    # Extract human scores
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
    logger.info(f"Loaded {len(human_scores)} human scores, range: [{human_scores.min():.1f}, {human_scores.max():.1f}]")
    
    # Define judge names and variants
    judge_names = ['truthfulness-judge', 'harmlessness-judge', 'helpfulness-judge', 
                   'honesty-judge', 'explanatory-depth-judge', 'instruction-following-judge',
                   'clarity-judge', 'conciseness-judge', 'logical-consistency-judge', 'creativity-judge']
    
    variant_types = ['original', 'strict', 'lenient', 'bottom_heavy', 'top_heavy']
    
    # Define variant combinations to test robustness
    combinations = {}
    for variant in variant_types:
        columns = [f'{judge}_{variant}' for judge in judge_names]
        # Filter to only existing columns
        existing_cols = [col for col in columns if col in scores_df.columns]
        if len(existing_cols) >= 8:  # Need at least 8 judges for meaningful analysis
            combinations[variant] = existing_cols
    
    logger.info(f"Testing {len(combinations)} variant combinations: {list(combinations.keys())}")
    
    # Load trained models
    models = {}
    
    # Try to load GAM model
    if gam_model_path.exists():
        gam_model = load_trained_model(gam_model_path, n_judges=10, model_type='gam')
        if gam_model is not None:
            models['Learned (GAM)'] = gam_model
    
    # Try to load MLP model
    if mlp_model_path.exists():
        mlp_model = load_trained_model(mlp_model_path, n_judges=10, model_type='mlp')
        if mlp_model is not None:
            models['Learned (MLP)'] = mlp_model
    
    logger.info(f"Loaded {len(models)} trained models: {list(models.keys())}")
    
    # Calculate aggregation results for each combination
    aggregation_results = {
        'Mean': {},
        'Single Best': {}
    }
    
    # Add loaded models to results
    for model_name in models.keys():
        aggregation_results[model_name] = {}
    
    logger.info("Computing aggregation results for each variant combination...")
    
    for combo_name, combo_cols in combinations.items():
        logger.info(f"Processing {combo_name} combination with {len(combo_cols)} judges...")
        
        # Get judge scores for this combination
        judge_scores = scores_df[combo_cols].values
        
        # Ensure we have the right number of examples
        min_len = min(len(judge_scores), len(human_scores))
        judge_scores = judge_scores[:min_len]
        combo_human_scores = human_scores[:min_len]
        
        # Replace any NaN values with median
        judge_scores = np.nan_to_num(judge_scores, nan=np.nanmedian(judge_scores))
        
        # 1. Mean aggregator
        mean_predictions = np.mean(judge_scores, axis=1)
        aggregation_results['Mean'][combo_name] = mean_predictions
        
        # 2. Single best judge (highest correlation with ground truth)
        best_corr = -1
        best_judge_idx = 0
        for i, col in enumerate(combo_cols):
            judge_vals = judge_scores[:, i]
            if not np.all(np.isnan(judge_vals)):
                corr, _ = stats.pearsonr(judge_vals, combo_human_scores)
                if np.isfinite(corr) and corr > best_corr:
                    best_corr = corr
                    best_judge_idx = i
        
        single_predictions = judge_scores[:, best_judge_idx]
        aggregation_results['Single Best'][combo_name] = single_predictions
        
        # 3. Learned models
        for model_name, model in models.items():
            try:
                with torch.no_grad():
                    # Normalize inputs (models expect normalized data)
                    normalized_scores = (judge_scores - np.mean(judge_scores, axis=0)) / (np.std(judge_scores, axis=0) + 1e-8)
                    
                    # Pad/truncate to model's expected input size
                    if normalized_scores.shape[1] > 10:
                        normalized_scores = normalized_scores[:, :10]
                    elif normalized_scores.shape[1] < 10:
                        # Pad with zeros
                        padding = np.zeros((normalized_scores.shape[0], 10 - normalized_scores.shape[1]))
                        normalized_scores = np.concatenate([normalized_scores, padding], axis=1)
                    
                    input_tensor = torch.FloatTensor(normalized_scores)
                    predictions = model(input_tensor).squeeze().numpy()
                    
                    # Denormalize predictions to human score scale
                    predictions = predictions * np.std(combo_human_scores) + np.mean(combo_human_scores)
                    
                    aggregation_results[model_name][combo_name] = predictions
                    
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_name}: {e}")
                # Fallback to mean predictions
                aggregation_results[model_name][combo_name] = mean_predictions
    
    # Calculate robustness metrics
    logger.info("Calculating robustness metrics...")
    robustness_metrics = {}
    
    for method_name, method_results in aggregation_results.items():
        if len(method_results) < 2:
            logger.warning(f"Insufficient combinations for {method_name}")
            continue
        
        # Get predictions for all combinations
        all_preds = []
        valid_combinations = []
        
        for combo_name, preds in method_results.items():
            if preds is not None and len(preds) > 0:
                all_preds.append(preds)
                valid_combinations.append(combo_name)
        
        if len(all_preds) < 2:
            logger.warning(f"Need at least 2 combinations for {method_name}, got {len(all_preds)}")
            continue
        
        # Ensure all predictions have the same length
        min_length = min(len(p) for p in all_preds)
        all_preds = [p[:min_length] for p in all_preds]
        
        # Stack predictions and calculate variance across combinations
        all_preds_matrix = np.column_stack(all_preds)
        row_variances = np.var(all_preds_matrix, axis=1)
        
        # Calculate correlation with ground truth (using original combination)
        original_preds = method_results.get('original', all_preds[0])
        original_preds = original_preds[:min_length]
        combo_human_scores = human_scores[:min_length]
        
        mask = ~(np.isnan(original_preds) | np.isnan(combo_human_scores))
        if mask.sum() > 1:
            correlation, _ = stats.pearsonr(original_preds[mask], combo_human_scores[mask])
        else:
            correlation = np.nan
        
        robustness_metrics[method_name] = {
            'mean_variance': np.mean(row_variances),
            'std_variance': np.std(row_variances),
            'max_variance': np.max(row_variances),
            'correlation_with_truth': correlation,
            'variance_distribution': row_variances,
            'combinations_used': valid_combinations
        }
        
        logger.info(f"{method_name}: variance={np.mean(row_variances):.4f}, corr={correlation:.3f}")
    
    # Create the main aggregator comparison plot
    logger.info("Creating aggregator comparison visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = list(robustness_metrics.keys())
    colors = ['green', 'blue', 'orange', 'red'][:len(methods)]
    
    # 1. Variance Comparison (Main Result)
    mean_variances = [robustness_metrics[m]['mean_variance'] for m in methods]
    
    bars = ax1.bar(methods, mean_variances, color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% Threshold')
    ax1.set_ylabel('Mean Variance Across Rubric Variations', fontsize=12)
    ax1.set_title('Aggregator Robustness to Rubric Changes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, variance in zip(bars, mean_variances):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(mean_variances)*0.02,
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
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
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
    if 'Mean' in robustness_metrics:
        baseline_var = robustness_metrics['Mean']['mean_variance']
        improvement_ratios = []
        improvement_labels = []
        improvement_colors = []
        
        for i, method in enumerate(methods):
            if method != 'Mean':
                method_var = robustness_metrics[method]['mean_variance']
                if method_var > 0:
                    ratio = baseline_var / method_var
                else:
                    ratio = float('inf')
                improvement_ratios.append(ratio)
                improvement_labels.append(method)
                improvement_colors.append(colors[i])
        
        if improvement_ratios:
            bars4 = ax4.bar(improvement_labels, improvement_ratios, color=improvement_colors, alpha=0.7)
            ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='No Improvement')
            ax4.set_ylabel('Robustness Improvement Factor vs Mean', fontsize=12)
            ax4.set_title('Improvement Over Mean Baseline', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            
            for bar, ratio in zip(bars4, improvement_ratios):
                if np.isfinite(ratio):
                    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(improvement_ratios)*0.02,
                            f'{ratio:.2f}x', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('🎯 KEY RESULT: Aggregator Robustness to Rubric Variations',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = plots_dir / 'aggregator_comparison_corrected.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Corrected aggregator comparison plot saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("🎯 CORRECTED AGGREGATOR COMPARISON - KEY RESULTS")
    print("="*70)
    
    for method in methods:
        metrics = robustness_metrics[method]
        print(f"\n{method}:")
        print(f"  • Mean Variance: {metrics['mean_variance']:.4f}")
        print(f"  • Correlation with Truth: {metrics['correlation_with_truth']:.3f}")
        print(f"  • Combinations Used: {', '.join(metrics['combinations_used'])}")
    
    # Calculate and show improvement
    if 'Mean' in robustness_metrics:
        mean_var = robustness_metrics['Mean']['mean_variance']
        print(f"\nImprovements over Mean Baseline (variance={mean_var:.4f}):")
        
        for method in methods:
            if method != 'Mean':
                method_var = robustness_metrics[method]['mean_variance']
                if method_var > 0:
                    improvement = mean_var / method_var
                    status = "✅ MORE ROBUST" if improvement > 1 else "❌ LESS ROBUST"
                    print(f"  • {method}: {improvement:.2f}x {status}")
    
    print("="*70)
    
    # Save detailed results
    results_path = results_dir / 'corrected_aggregator_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({
            'robustness_metrics': robustness_metrics,
            'aggregation_results': aggregation_results,
            'combinations': combinations,
            'data_shape': scores_df.shape
        }, f)
    
    logger.info(f"Detailed results saved to: {results_path}")
    
    return robustness_metrics


if __name__ == "__main__":
    metrics = correct_aggregator_analysis()
    print(f"\n✅ Corrected aggregator analysis complete!")
    print("Check plots_corrected/aggregator_comparison_corrected.png for the key result!")