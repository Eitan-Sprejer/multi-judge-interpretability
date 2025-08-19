#!/usr/bin/env python3
"""
Fixed Analysis Script for Rubric Sensitivity Experiment

This script properly analyzes the rubric sensitivity data with correct
data structure handling and visualization generation.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_rubric_sensitivity(data_path: str, output_dir: str):
    """
    Analyze rubric sensitivity with proper data handling.
    
    Args:
        data_path: Path to restructured scores pickle file
        output_dir: Directory to save results and plots
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots_fixed"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load restructured data
    logger.info(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        scores_df = pickle.load(f)
    
    logger.info(f"Data shape: {scores_df.shape}")
    
    # Define judges and variants
    judge_names = ['truthfulness-judge', 'harmlessness-judge', 'helpfulness-judge', 
                   'honesty-judge', 'explanatory-depth-judge', 'instruction-following-judge',
                   'clarity-judge', 'conciseness-judge', 'logical-consistency-judge', 'creativity-judge']
    variant_types = ['original', 'strict', 'lenient', 'bottom_heavy', 'top_heavy']
    
    # Calculate variance metrics for each judge
    variance_results = {}
    correlation_results = {}
    
    for judge in judge_names:
        # Get all variant columns for this judge
        variant_cols = [f'{judge}_{v}' for v in variant_types]
        valid_cols = [c for c in variant_cols if c in scores_df.columns]
        
        if len(valid_cols) < 2:
            logger.warning(f"Insufficient variants for {judge}")
            continue
        
        # Get scores matrix
        scores_matrix = scores_df[valid_cols].values
        
        # Calculate variance across variants for each example
        row_variances = np.var(scores_matrix, axis=1)
        
        variance_results[judge] = {
            'mean_variance': np.mean(row_variances),
            'std_variance': np.std(row_variances),
            'max_variance': np.max(row_variances),
            'variance_distribution': row_variances
        }
        
        # Calculate correlations with original
        original_col = f'{judge}_original'
        if original_col in scores_df.columns:
            original_scores = scores_df[original_col].values
            correlations = {}
            
            for variant in ['strict', 'lenient', 'bottom_heavy', 'top_heavy']:
                variant_col = f'{judge}_{variant}'
                if variant_col in scores_df.columns:
                    variant_scores = scores_df[variant_col].values
                    r, p = stats.pearsonr(original_scores, variant_scores)
                    correlations[variant] = {'r': r, 'p': p}
            
            correlation_results[judge] = correlations
    
    # Create visualizations
    logger.info("Generating visualizations...")
    
    # 1. Variance Comparison Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    judges_clean = [j.replace('-judge', '').replace('-', '\n') for j in variance_results.keys()]
    variances = [variance_results[j]['mean_variance'] for j in variance_results.keys()]
    
    bars = ax.bar(judges_clean, variances, color=plt.cm.Set3(np.linspace(0, 1, len(judges_clean))))
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% Threshold')
    
    ax.set_title('Judge Score Variance Across Rubric Variations', fontsize=16, fontweight='bold')
    ax.set_xlabel('Judge', fontsize=12)
    ax.set_ylabel('Mean Variance', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, variance in zip(bars, variances):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{variance:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'variance_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Variance comparison plot saved")
    
    # 2. Correlation Matrix Heatmap
    if correlation_results:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Build correlation matrix
        judges_with_corr = list(correlation_results.keys())
        variants_to_show = ['strict', 'lenient', 'bottom_heavy', 'top_heavy']
        
        corr_matrix = []
        for judge in judges_with_corr:
            row = []
            for variant in variants_to_show:
                if variant in correlation_results[judge]:
                    row.append(correlation_results[judge][variant]['r'])
                else:
                    row.append(np.nan)
            corr_matrix.append(row)
        
        corr_matrix = np.array(corr_matrix)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0.8, vmax=1.0, aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(variants_to_show)))
        ax.set_xticklabels([v.replace('_', ' ').title() for v in variants_to_show])
        ax.set_yticks(range(len(judges_with_corr)))
        ax.set_yticklabels([j.replace('-judge', '').replace('-', ' ').title() for j in judges_with_corr])
        
        # Add correlation values
        for i in range(len(judges_with_corr)):
            for j in range(len(variants_to_show)):
                if not np.isnan(corr_matrix[i, j]):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                                 ha="center", va="center", 
                                 color="white" if corr_matrix[i, j] < 0.9 else "black",
                                 fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation with Original', rotation=270, labelpad=20)
        
        ax.set_title('Correlation Between Original and Variant Rubrics', fontsize=16, fontweight='bold')
        ax.set_xlabel('Rubric Variant', fontsize=12)
        ax.set_ylabel('Judge', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_matrix_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Correlation matrix plot saved")
    
    # 3. Variance Distribution Box Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for box plot
    plot_data = []
    plot_labels = []
    
    for judge in variance_results.keys():
        dist = variance_results[judge]['variance_distribution']
        plot_data.append(dist)
        plot_labels.append(judge.replace('-judge', '').replace('-', ' ').title())
    
    # Create box plot
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% Threshold')
    ax.set_title('Distribution of Score Variance Across Examples', fontsize=16, fontweight='bold')
    ax.set_xlabel('Judge', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'variance_distribution_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Variance distribution plot saved")
    
    # 4. Summary Statistics Table
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for judge in variance_results.keys():
        judge_clean = judge.replace('-judge', '').replace('-', ' ').title()
        mean_var = variance_results[judge]['mean_variance']
        std_var = variance_results[judge]['std_variance']
        max_var = variance_results[judge]['max_variance']
        
        # Get average correlation if available
        avg_corr = 'N/A'
        if judge in correlation_results:
            corrs = [c['r'] for c in correlation_results[judge].values()]
            if corrs:
                avg_corr = f'{np.mean(corrs):.3f}'
        
        table_data.append([judge_clean, f'{mean_var:.4f}', f'{std_var:.4f}', 
                          f'{max_var:.4f}', avg_corr])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Judge', 'Mean Var', 'Std Var', 'Max Var', 'Avg Corr'],
                    cellLoc='center',
                    loc='center',
                    colColours=['lightgray'] * 5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on variance (red for high variance)
    for i in range(len(table_data)):
        mean_var = float(table_data[i][1])
        if mean_var > 0.1:
            table[(i+1, 1)].set_facecolor('#ffcccc')
        elif mean_var > 0.05:
            table[(i+1, 1)].set_facecolor('#ffffcc')
        else:
            table[(i+1, 1)].set_facecolor('#ccffcc')
    
    ax.set_title('Rubric Sensitivity Summary Statistics', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'summary_table_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Summary table saved")
    
    # 5. SIMPLE AGGREGATOR COMPARISON
    logger.info("Generating simple aggregator comparison...")
    
    # Load human feedback for training
    gt_path = "/Users/eitu/Documents/Eitu/AI Safety/AIS_hackathons/model_routing/multi-judge-interpretability/dataset/data_with_judge_scores.pkl"
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    
    # Extract human scores (same logic as before)
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
    
    # Test aggregation methods on different variant combinations
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    
    variants_to_test = ['original', 'strict', 'lenient']
    aggregator_variances = {'Learned (MLP)': [], 'Mean': [], 'Single Best': []}
    
    for variant in variants_to_test:
        variant_cols = [f"{judge}_{variant}" for judge in judge_names]
        existing_cols = [col for col in variant_cols if col in scores_df.columns]
        
        if len(existing_cols) >= 5:  # Need at least 5 judges
            X = scores_df[existing_cols].values
            y = human_scores
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=2.0)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Learned aggregator (MLP)
            mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
            mlp.fit(X_train, y_train)
            learned_preds = mlp.predict(X)
            
            # Mean aggregator
            mean_preds = np.mean(X, axis=1)
            
            # Single best (first judge)
            single_preds = X[:, 0]
            
            # Store predictions for this variant
            aggregator_variances['Learned (MLP)'].append(learned_preds)
            aggregator_variances['Mean'].append(mean_preds)
            aggregator_variances['Single Best'].append(single_preds)
    
    # Calculate robustness (variance across variants)
    final_aggregator_results = {}
    for method, predictions_list in aggregator_variances.items():
        if len(predictions_list) >= 2:
            # Stack predictions and calculate variance across variants
            pred_matrix = np.column_stack(predictions_list)
            variances = np.var(pred_matrix, axis=1)
            final_aggregator_results[method] = {
                'mean_variance': np.mean(variances),
                'predictions': predictions_list
            }
    
    # Create aggregator comparison plot
    if final_aggregator_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        methods = list(final_aggregator_results.keys())
        variances = [final_aggregator_results[m]['mean_variance'] for m in methods]
        colors = ['green', 'orange', 'red']
        
        bars = ax.bar(methods, variances, color=colors[:len(methods)], alpha=0.7)
        ax.set_ylabel('Mean Variance Across Rubric Variants', fontsize=12)
        ax.set_title('🎯 KEY RESULT: Aggregator Robustness to Rubric Changes', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, variance in zip(bars, variances):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{variance:.4f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'aggregator_comparison_simple.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Simple aggregator comparison plot saved")
    
    # Save analysis results
    results = {
        'variance_results': variance_results,
        'correlation_results': correlation_results,
        'aggregator_results': final_aggregator_results,
        'summary': {
            'total_judges': len(variance_results),
            'avg_variance': np.mean([v['mean_variance'] for v in variance_results.values()]),
            'judges_below_5pct': sum(1 for v in variance_results.values() if v['mean_variance'] < 0.05),
            'avg_correlation': np.mean([c['r'] for j in correlation_results.values() for c in j.values()])
        }
    }
    
    with open(output_dir / 'fixed_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n" + "="*60)
    print("RUBRIC SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Judges Analyzed: {results['summary']['total_judges']}")
    print(f"Average Variance: {results['summary']['avg_variance']:.4f}")
    print(f"Judges Below 5% Variance: {results['summary']['judges_below_5pct']}/{results['summary']['total_judges']}")
    print(f"Average Correlation with Original: {results['summary']['avg_correlation']:.4f}")
    print("\nJudge-Specific Results:")
    for judge in sorted(variance_results.keys(), key=lambda x: variance_results[x]['mean_variance']):
        print(f"  {judge.replace('-judge', ''):20s}: var={variance_results[judge]['mean_variance']:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Run the fixed analysis
    data_path = "../results_full_20250818_215910/restructured_scores.pkl"
    output_dir = "../results_full_20250818_215910"
    
    results = analyze_rubric_sensitivity(data_path, output_dir)
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/plots_fixed/")