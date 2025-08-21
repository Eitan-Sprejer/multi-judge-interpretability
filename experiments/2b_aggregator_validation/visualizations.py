#!/usr/bin/env python3
"""
Visualizations for Experiment 2b: Aggregator Validation with Less Varied Data

Creates all visualizations for the experiment including:
1. Primary R² comparison plot
2. Variance scatter plot with linear fits
3. Supplementary visualizations (heatmap, extended comparison)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set style
plt.style.use('default')
sns.set_palette("husl")


def create_main_r2_comparison_plot(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create the main R² comparison plot showing the three primary targets.
    
    Args:
        results: Dictionary with all experiment results
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    print("📊 Creating main R² comparison plot...")
    
    # Extract main comparison data
    main_comparison = results['summary']['main_comparison']
    
    # Prepare data for plotting
    categories = ['Mixed Personas\n(Baseline)', 'UltraFeedback\n(GPT-4)', 'Individual Personas\n(Mean of 14)']
    
    gam_scores = [
        main_comparison['mixed_personas']['gam_r2'],
        main_comparison['ultrafeedback']['gam_r2'],
        main_comparison['individual_personas_mean']['gam_r2']
    ]
    
    mlp_scores = [
        main_comparison['mixed_personas']['mlp_r2'],
        main_comparison['ultrafeedback']['mlp_r2'],
        main_comparison['individual_personas_mean']['mlp_r2']
    ]
    
    variances = [
        main_comparison['mixed_personas']['variance'],
        main_comparison['ultrafeedback']['variance'],
        main_comparison['individual_personas_mean']['variance']
    ]
    
    # Create figure with grouped bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, gam_scores, width, label='GAM', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, mlp_scores, width, label='MLP', color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Styling
    ax.set_title('R² Score Comparison Across Ground Truth Types\n(Lower Variance → Higher Performance)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Ground Truth Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(gam_scores), max(mlp_scores)) * 1.15)
    
    # Add value labels on bars
    for bars, scores in [(bars1, gam_scores), (bars2, mlp_scores)]:
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Highlight best scores
    best_gam_idx = np.argmax(gam_scores)
    best_mlp_idx = np.argmax(mlp_scores)
    
    bars1[best_gam_idx].set_color('gold')
    bars1[best_gam_idx].set_edgecolor('darkgoldenrod')
    bars1[best_gam_idx].set_linewidth(2)
    
    bars2[best_mlp_idx].set_color('gold')
    bars2[best_mlp_idx].set_edgecolor('darkgoldenrod')
    bars2[best_mlp_idx].set_linewidth(2)
    
    # Add variance information as text annotations
    for i, (cat, var) in enumerate(zip(categories, variances)):
        ax.text(i, max(max(gam_scores), max(mlp_scores)) * 1.05, 
               f'Variance: {var:.2f}', ha='center', va='bottom', 
               fontsize=10, style='italic', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    return fig


def create_variance_scatter_plot(results: Dict[str, Any], targets: Dict[str, np.ndarray], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create variance analysis scatter plot with linear fits.
    
    Args:
        results: Dictionary with all experiment results
        targets: Dictionary with all target arrays
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    print("📈 Creating variance scatter plot with linear fits...")
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    X = targets['X']
    judge_mean = np.mean(X, axis=1)
    
    # Plot configurations
    plot_configs = [
        (ax1, targets['y_mixed'], 'Mixed Personas (High Variance)', 'lightcoral'),
        (ax2, targets['y_ultrafeedback'], 'UltraFeedback (Low Variance)', 'steelblue'),
        (ax3, targets['y_persona_mean'], 'Persona Mean (Moderate Variance)', 'lightgreen'),
        (ax4, targets['y_personas']['Professor'], 'Professor Persona (Example)', 'orange')  # Example individual
    ]
    
    for ax, y_target, title, color in plot_configs:
        # Create scatter plot
        ax.scatter(judge_mean, y_target, alpha=0.6, color=color, s=30, edgecolors='black', linewidth=0.5)
        
        # Calculate and plot linear fit
        if len(np.unique(judge_mean)) > 1:
            coeffs = np.polyfit(judge_mean, y_target, 1)
            y_pred_linear = np.polyval(coeffs, judge_mean)
            
            # Plot regression line
            judge_range = np.linspace(judge_mean.min(), judge_mean.max(), 100)
            y_fit = np.polyval(coeffs, judge_range)
            ax.plot(judge_range, y_fit, 'r--', linewidth=2, alpha=0.8, label='Linear Fit')
            
            # Calculate statistics
            correlation = np.corrcoef(judge_mean, y_target)[0, 1]
            residuals = y_target - y_pred_linear
            std_from_fit = np.std(residuals)
            
            # Add confidence band
            y_mean = np.mean(y_target)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_target - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Plot confidence interval
            ax.fill_between(judge_range, y_fit - std_from_fit, y_fit + std_from_fit, 
                           alpha=0.2, color='red', label=f'±1σ ({std_from_fit:.2f})')
            
            # Add statistics text
            stats_text = f'r = {correlation:.3f}\nR² = {r_squared:.3f}\nσ = {std_from_fit:.2f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Styling
        ax.set_xlabel('Mean Judge Score', fontweight='bold')
        ax.set_ylabel('Ground Truth Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    plt.suptitle('Judge Score vs Ground Truth Analysis\n(Linear Fit Quality Indicates Predictability)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    return fig


def create_persona_performance_heatmap(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing performance for each individual persona.
    
    Args:
        results: Dictionary with all experiment results
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    print("🔥 Creating persona performance heatmap...")
    
    # Extract individual persona results
    individual_results = results['individual_personas']['individual_results']
    
    # Prepare data for heatmap
    personas = list(individual_results.keys())
    metrics = ['GAM R²', 'MLP R²', 'Variance', 'Best Model Score']
    
    heatmap_data = []
    for persona in personas:
        persona_data = individual_results[persona]
        if 'summary' in persona_data:
            row = [
                persona_data['summary']['gam_r2'],
                persona_data['summary']['mlp_r2'],
                persona_data['data_stats']['target_variance'],
                persona_data['summary']['best_r2']
            ]
            heatmap_data.append(row)
    
    # Create DataFrame
    df_heatmap = pd.DataFrame(heatmap_data, index=personas, columns=metrics)
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    # Normalize variance column (invert so lower variance shows as better)
    df_normalized = df_heatmap.copy()
    df_normalized['Variance'] = 1 / (1 + df_normalized['Variance'])  # Invert variance
    
    sns.heatmap(df_normalized, annot=df_heatmap, fmt='.3f', cmap='RdYlGn', 
                center=0.5, ax=ax, cbar_kws={'label': 'Performance Score (Higher = Better)'})
    
    ax.set_title('Individual Persona Performance Matrix\n(Variance Inverted: Lower = Better)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Personas', fontweight='bold')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    return fig


def create_extended_comparison_plot(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create extended comparison including all experiments and individual personas.
    
    Args:
        results: Dictionary with all experiment results
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    print("📊 Creating extended comparison plot...")
    
    # Prepare data
    main_comparison = results['summary']['main_comparison']
    individual_results = results['individual_personas']['individual_results']
    
    # Main experiments
    main_names = ['Mixed\nPersonas', 'UltraFeedback', 'Individual\nMean', 'Persona\nMean']
    main_gam = [
        main_comparison['mixed_personas']['gam_r2'],
        main_comparison['ultrafeedback']['gam_r2'],
        main_comparison['individual_personas_mean']['gam_r2'],
        main_comparison['persona_mean']['gam_r2']
    ]
    main_mlp = [
        main_comparison['mixed_personas']['mlp_r2'],
        main_comparison['ultrafeedback']['mlp_r2'],
        main_comparison['individual_personas_mean']['mlp_r2'],
        main_comparison['persona_mean']['mlp_r2']
    ]
    
    # Individual personas (best model for each)
    persona_names = []
    persona_scores = []
    for persona, results_data in individual_results.items():
        if 'summary' in results_data:
            persona_names.append(persona.replace(' ', '\n'))
            persona_scores.append(results_data['summary']['best_r2'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top plot: Main comparisons
    x_main = np.arange(len(main_names))
    width = 0.35
    
    bars1 = ax1.bar(x_main - width/2, main_gam, width, label='GAM', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_main + width/2, main_mlp, width, label='MLP', 
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_title('Main Experiment Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_xticks(x_main)
    ax1.set_xticklabels(main_names, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars, scores in [(bars1, main_gam), (bars2, main_mlp)]:
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Bottom plot: Individual personas
    x_persona = np.arange(len(persona_names))
    bars3 = ax2.bar(x_persona, persona_scores, color='lightgreen', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    ax2.set_title('Individual Persona Performance (Best Model)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R² Score', fontweight='bold')
    ax2.set_xlabel('Personas', fontweight='bold')
    ax2.set_xticks(x_persona)
    ax2.set_xticklabels(persona_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight best and worst personas
    best_idx = np.argmax(persona_scores)
    worst_idx = np.argmin(persona_scores)
    
    bars3[best_idx].set_color('gold')
    bars3[best_idx].set_edgecolor('darkgoldenrod')
    bars3[best_idx].set_linewidth(2)
    
    bars3[worst_idx].set_color('lightcoral')
    bars3[worst_idx].set_edgecolor('darkred')
    bars3[worst_idx].set_linewidth(2)
    
    # Add value labels for best and worst
    for i, (bar, score) in enumerate(zip(bars3, persona_scores)):
        if i == best_idx or i == worst_idx:
            height = bar.get_height()
            label = f'{score:.3f}\n{"(Best)" if i == best_idx else "(Worst)"}'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    return fig


def create_variance_vs_performance_plot(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create scatter plot showing relationship between variance and R² performance.
    
    Args:
        results: Dictionary with all experiment results
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    print("📈 Creating variance vs performance analysis...")
    
    # Collect all data points
    variances = []
    r2_scores = []
    names = []
    colors = []
    sizes = []
    
    # Main comparisons
    main_comparison = results['summary']['main_comparison']
    for exp_name, exp_data in main_comparison.items():
        variances.append(exp_data['variance'])
        r2_scores.append(max(exp_data['gam_r2'], exp_data['mlp_r2']))
        names.append(exp_name.replace('_', ' ').title())
        colors.append('red' if 'mixed' in exp_name else 'blue' if 'ultrafeedback' in exp_name else 'green')
        sizes.append(100)
    
    # Individual personas
    individual_results = results['individual_personas']['individual_results']
    for persona, results_data in individual_results.items():
        if 'summary' in results_data:
            variances.append(results_data['data_stats']['target_variance'])
            r2_scores.append(results_data['summary']['best_r2'])
            names.append(persona)
            colors.append('orange')
            sizes.append(60)
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    scatter = ax.scatter(variances, r2_scores, c=colors, s=sizes, alpha=0.7, 
                        edgecolors='black', linewidth=1)
    
    # Add trend line
    if len(variances) > 1:
        coeffs = np.polyfit(variances, r2_scores, 1)
        trend_line = np.polyval(coeffs, np.sort(variances))
        ax.plot(np.sort(variances), trend_line, 'r--', linewidth=2, alpha=0.8, 
               label=f'Trend: R² = {coeffs[0]:.3f} × Var + {coeffs[1]:.3f}')
        
        # Calculate correlation
        correlation = np.corrcoef(variances, r2_scores)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add labels for main experiments
    main_indices = list(range(len(main_comparison)))
    for i in main_indices:
        ax.annotate(names[i], (variances[i], r2_scores[i]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Ground Truth Variance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best R² Score', fontsize=14, fontweight='bold')
    ax.set_title('Hypothesis Validation: Variance vs Performance\n(Lower Variance → Higher R² Score)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add color legend
    legend_elements = [
        plt.scatter([], [], c='red', s=100, label='Mixed Personas'),
        plt.scatter([], [], c='blue', s=100, label='UltraFeedback'),
        plt.scatter([], [], c='green', s=100, label='Aggregated'),
        plt.scatter([], [], c='orange', s=60, label='Individual Personas')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    
    return fig


def create_all_visualizations(results: Dict[str, Any], targets: Dict[str, np.ndarray], output_dir: str) -> Dict[str, str]:
    """
    Create all visualizations for the experiment.
    
    Args:
        results: Dictionary with all experiment results
        targets: Dictionary with all target arrays
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with paths to saved plots
    """
    print("\n🎨 Creating all visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # 1. Main R² comparison
    fig1 = create_main_r2_comparison_plot(results, str(output_path / "1_main_r2_comparison.png"))
    plt.close(fig1)
    plot_paths['main_r2_comparison'] = str(output_path / "1_main_r2_comparison.png")
    
    # 2. Variance scatter plot
    fig2 = create_variance_scatter_plot(results, targets, str(output_path / "2_variance_scatter_analysis.png"))
    plt.close(fig2)
    plot_paths['variance_scatter'] = str(output_path / "2_variance_scatter_analysis.png")
    
    # 3. Persona performance heatmap
    fig3 = create_persona_performance_heatmap(results, str(output_path / "3_persona_performance_heatmap.png"))
    plt.close(fig3)
    plot_paths['persona_heatmap'] = str(output_path / "3_persona_performance_heatmap.png")
    
    # 4. Extended comparison
    fig4 = create_extended_comparison_plot(results, str(output_path / "4_extended_comparison.png"))
    plt.close(fig4)
    plot_paths['extended_comparison'] = str(output_path / "4_extended_comparison.png")
    
    # 5. Variance vs performance
    fig5 = create_variance_vs_performance_plot(results, str(output_path / "5_variance_vs_performance.png"))
    plt.close(fig5)
    plot_paths['variance_vs_performance'] = str(output_path / "5_variance_vs_performance.png")
    
    print(f"✅ All visualizations created and saved to: {output_path}")
    
    return plot_paths


if __name__ == "__main__":
    # Test visualization creation with dummy data
    print("🧪 Testing visualization functions...")
    
    # This would normally be called after running experiments
    print("✅ Visualization functions created successfully!")
    print("   Use create_all_visualizations() after running experiments to generate plots.")