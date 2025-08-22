#!/usr/bin/env python3
"""
Comprehensive Visualization System for Judge Contamination Analysis

Creates publication-quality visualizations including distribution plots, correlation heatmaps,
robustness curves, and comparative analysis charts for judge contamination experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

class ContaminationVisualizer:
    """
    Comprehensive visualization suite for judge contamination analysis.
    
    Provides methods to create various types of analysis plots including
    distribution comparisons, correlation analysis, robustness curves,
    and publication-ready summary figures.
    """
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save generated plots
            dpi: Resolution for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
    def plot_score_distributions(self, 
                                baseline_scores: pd.DataFrame,
                                contaminated_scores: pd.DataFrame,
                                judge_mapping: Dict[str, str],
                                save_name: str = "score_distributions") -> plt.Figure:
        """
        Create comprehensive score distribution comparison plots.
        
        Args:
            baseline_scores: Clean judge scores
            contaminated_scores: Contaminated judge scores
            judge_mapping: Mapping from baseline to contaminated judge IDs
            save_name: Name for saved figure
            
        Returns:
            matplotlib Figure object
        """
        n_judges = len(judge_mapping)
        n_cols = min(3, n_judges)
        n_rows = (n_judges + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_judges == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Score Distribution Comparison: Baseline vs Contaminated Judges', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plot_idx = 0
        for baseline_judge, contaminated_judge in judge_mapping.items():
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if baseline_judge in baseline_scores.columns and contaminated_judge in contaminated_scores.columns:
                baseline_data = baseline_scores[baseline_judge].dropna()
                contaminated_data = contaminated_scores[contaminated_judge].dropna()
                
                # Create overlapping histograms
                ax.hist(baseline_data, bins=20, alpha=0.6, label='Baseline', 
                       color='steelblue', density=True, edgecolor='black', linewidth=0.5)
                ax.hist(contaminated_data, bins=20, alpha=0.6, label='Contaminated', 
                       color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
                
                # Add statistical annotations
                baseline_mean = baseline_data.mean()
                contaminated_mean = contaminated_data.mean()
                
                ax.axvline(baseline_mean, color='darkblue', linestyle='--', linewidth=2, 
                          label=f'Baseline μ={baseline_mean:.2f}')
                ax.axvline(contaminated_mean, color='darkred', linestyle='--', linewidth=2,
                          label=f'Contaminated μ={contaminated_mean:.2f}')
                
                # Compute correlation for title
                from scipy.stats import pearsonr
                corr, p_value = pearsonr(baseline_data[:min(len(baseline_data), len(contaminated_data))], 
                                       contaminated_data[:min(len(baseline_data), len(contaminated_data))])
                
                ax.set_title(f'{baseline_judge}\nCorrelation: {corr:.3f} (p={p_value:.3f})', 
                           fontweight='bold')
                ax.set_xlabel('Score')
                ax.set_ylabel('Density')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                if idx < len(axes):
                    axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, 
                                analysis_results: Dict[str, Any],
                                save_name: str = "correlation_heatmap") -> plt.Figure:
        """
        Create correlation heatmap showing judge-to-judge relationships.
        
        Args:
            analysis_results: Results from contamination analysis
            save_name: Name for saved figure
            
        Returns:
            matplotlib Figure object
        """
        # Extract correlation data
        if 'judge_inversion' not in analysis_results:
            raise ValueError("Analysis results must contain judge_inversion data")
            
        individual_judges = analysis_results['judge_inversion']['individual_judges']
        
        # Prepare correlation matrix data
        judge_names = list(individual_judges.keys())
        correlations = []
        p_values = []
        
        for judge in judge_names:
            judge_data = individual_judges[judge]
            correlations.append(judge_data['correlations']['pearson']['correlation'])
            p_values.append(judge_data['correlations']['pearson']['p_value'])
        
        # Create figure with multiple panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Panel 1: Correlation values
        corr_data = pd.DataFrame({
            'Judge': judge_names,
            'Correlation': correlations,
            'P-Value': p_values
        })
        
        # Create heatmap-style bar plot for correlations
        colors = ['darkred' if c < -0.5 else 'orange' if c < 0 else 'lightgreen' for c in correlations]
        bars = ax1.barh(range(len(judge_names)), correlations, color=colors, 
                       edgecolor='black', linewidth=1)
        
        ax1.set_yticks(range(len(judge_names)))
        ax1.set_yticklabels(judge_names, fontsize=10)
        ax1.set_xlabel('Pearson Correlation (Baseline vs Contaminated)')
        ax1.set_title('Judge Contamination Correlations', fontweight='bold')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.axvline(-0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Inversion Threshold')
        ax1.legend()
        
        # Add value labels on bars
        for i, (bar, corr, p_val) in enumerate(zip(bars, correlations, p_values)):
            width = bar.get_width()
            label_x = width + 0.05 if width >= 0 else width - 0.05
            ha = 'left' if width >= 0 else 'right'
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}{significance}', ha=ha, va='center', fontweight='bold')
        
        # Panel 2: Inversion pattern summary
        inversion_patterns = analysis_results['judge_inversion']['inversion_detection']['patterns']
        pattern_counts = [
            len(inversion_patterns['complete_inversion']),
            len(inversion_patterns['partial_inversion']),
            len(inversion_patterns['no_inversion']),
            len(inversion_patterns['amplification'])
        ]
        pattern_labels = ['Complete\nInversion', 'Partial\nInversion', 'No\nInversion', 'Amplification']
        pattern_colors = ['darkred', 'orange', 'lightgreen', 'blue']
        
        wedges, texts, autotexts = ax2.pie(pattern_counts, labels=pattern_labels, colors=pattern_colors, 
                                          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Contamination Pattern Distribution', fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_robustness_curves(self, 
                              robustness_results: Dict[str, Any],
                              save_name: str = "robustness_curves") -> plt.Figure:
        """
        Create robustness curves showing performance degradation vs contamination rate.
        
        Args:
            robustness_results: Results from robustness analysis
            save_name: Name for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data from robustness results
        contamination_curves = robustness_results['contamination_curves']
        rates = []
        naive_r2 = []
        naive_mse = []
        naive_mae = []
        single_judge_r2 = []
        
        for rate_key in sorted(contamination_curves.keys()):
            rate = float(rate_key.split('_')[1])
            rates.append(rate)
            
            naive_metrics = contamination_curves[rate_key]['naive_average']
            naive_r2.append(naive_metrics['r2'])
            naive_mse.append(naive_metrics['mse'])
            naive_mae.append(naive_metrics['mae'])
            
            single_judge_r2.append(contamination_curves[rate_key]['best_single_judge']['r2'])
        
        # Plot 1: R² Score vs Contamination Rate
        ax1.plot(rates, naive_r2, 'o-', linewidth=3, markersize=8, label='Naive Average', color='steelblue')
        ax1.plot(rates, single_judge_r2, 's--', linewidth=3, markersize=8, label='Best Single Judge', color='lightcoral')
        ax1.set_xlabel('Contamination Rate')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Performance vs Contamination Rate', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add breakdown threshold line if available
        if 'breakdown_rate' in robustness_results['robustness_metrics'] and robustness_results['robustness_metrics']['breakdown_rate']:
            breakdown_rate = robustness_results['robustness_metrics']['breakdown_rate']
            ax1.axvline(breakdown_rate, color='red', linestyle=':', linewidth=2, 
                       label=f'Breakdown at {breakdown_rate:.1f}')
            ax1.legend()
        
        # Plot 2: MSE vs Contamination Rate
        ax2.plot(rates, naive_mse, 'o-', linewidth=3, markersize=8, color='darkred')
        ax2.set_xlabel('Contamination Rate')
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('MSE vs Contamination Rate', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MAE vs Contamination Rate
        ax3.plot(rates, naive_mae, 'o-', linewidth=3, markersize=8, color='darkgreen')
        ax3.set_xlabel('Contamination Rate')
        ax3.set_ylabel('Mean Absolute Error')
        ax3.set_title('MAE vs Contamination Rate', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Degradation
        clean_r2 = naive_r2[0]
        degradation = [(clean_r2 - r2) / clean_r2 * 100 for r2 in naive_r2]
        
        ax4.plot(rates, degradation, 'o-', linewidth=3, markersize=8, color='purple')
        ax4.set_xlabel('Contamination Rate')
        ax4.set_ylabel('Performance Degradation (%)')
        ax4.set_title('Relative Performance Degradation', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% Degradation')
        ax4.legend()
        
        plt.suptitle('Aggregator Robustness Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_individual_judge_analysis(self, 
                                     analysis_results: Dict[str, Any],
                                     baseline_scores: pd.DataFrame,
                                     contaminated_scores: pd.DataFrame,
                                     judge_mapping: Dict[str, str],
                                     save_name: str = "individual_judge_analysis") -> plt.Figure:
        """
        Create detailed individual judge analysis plots.
        
        Args:
            analysis_results: Results from contamination analysis
            baseline_scores: Clean judge scores
            contaminated_scores: Contaminated judge scores
            judge_mapping: Mapping from baseline to contaminated judge IDs
            save_name: Name for saved figure
            
        Returns:
            matplotlib Figure object
        """
        individual_judges = analysis_results['judge_inversion']['individual_judges']
        n_judges = len(individual_judges)
        
        # Create figure with subplots for each judge
        fig, axes = plt.subplots(n_judges, 3, figsize=(18, 4*n_judges))
        if n_judges == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Individual Judge Contamination Analysis', fontsize=16, fontweight='bold')
        
        for idx, (judge_id, judge_data) in enumerate(individual_judges.items()):
            baseline_judge = judge_id
            contaminated_judge = judge_mapping[judge_id]
            
            # Get score data
            baseline_data = baseline_scores[baseline_judge].dropna()
            contaminated_data = contaminated_scores[contaminated_judge].dropna()
            
            min_len = min(len(baseline_data), len(contaminated_data))
            baseline_data = baseline_data.iloc[:min_len]
            contaminated_data = contaminated_data.iloc[:min_len]
            
            # Plot 1: Scatter plot with correlation
            ax1 = axes[idx, 0]
            ax1.scatter(baseline_data, contaminated_data, alpha=0.6, s=30)
            
            # Add regression line
            z = np.polyfit(baseline_data, contaminated_data, 1)
            p = np.poly1d(z)
            ax1.plot(baseline_data, p(baseline_data), "r--", alpha=0.8, linewidth=2)
            
            corr = judge_data['correlations']['pearson']['correlation']
            ax1.set_xlabel('Baseline Score')
            ax1.set_ylabel('Contaminated Score')
            ax1.set_title(f'{judge_id}\nCorrelation: {corr:.3f}')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Score shift analysis
            ax2 = axes[idx, 1]
            shifts = contaminated_data - baseline_data
            ax2.hist(shifts, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax2.axvline(shifts.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean Shift: {shifts.mean():.3f}')
            ax2.set_xlabel('Score Shift (Contaminated - Baseline)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Shift Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Performance metrics
            ax3 = axes[idx, 2]
            metrics = ['Mean Shift', 'Std Shift', 'Correlation', 'Inversion Strength']
            values = [
                judge_data['shifts']['mean_shift'],
                judge_data['shifts']['std_shift'],
                judge_data['correlations']['pearson']['correlation'],
                judge_data['inversion_analysis']['inversion_strength']
            ]
            
            colors = ['blue', 'green', 'red' if values[2] < 0 else 'lightgreen', 'purple']
            bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_title('Judge Performance Metrics')
            ax3.set_ylabel('Metric Value')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_summary_dashboard(self, 
                                analysis_results: Dict[str, Any],
                                robustness_results: Optional[Dict[str, Any]] = None,
                                save_name: str = "contamination_dashboard") -> plt.Figure:
        """
        Create a comprehensive dashboard summarizing all contamination analysis.
        
        Args:
            analysis_results: Results from contamination analysis
            robustness_results: Optional robustness analysis results
            save_name: Name for saved figure
            
        Returns:
            matplotlib Figure object
        """
        # Create figure with grid layout
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Judge Contamination Analysis Dashboard', fontsize=20, fontweight='bold', y=0.97)
        
        # Panel 1: Overall contamination summary (top-left, spans 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Extract key metrics
        agg_metrics = analysis_results['judge_inversion']['aggregate_metrics']
        avg_corr = agg_metrics['average_correlation']
        contam_rate = agg_metrics['contamination_rate']
        
        # Create summary text with metrics
        summary_text = f"""
CONTAMINATION ANALYSIS SUMMARY

Average Judge Correlation: {avg_corr:.3f}
Contamination Success Rate: {contam_rate:.1%}
System Inversion Detected: {agg_metrics['system_inversion_detected']}
Severe Contamination: {agg_metrics['severe_contamination']}

Statistical Significance:
• Distribution Shift: {analysis_results['judge_inversion']['statistical_tests']['system_level']['distribution_shift']['significant']}
• Median Shift: {analysis_results['judge_inversion']['statistical_tests']['system_level']['median_shift']['significant']}
        """
        
        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=14,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        
        # Panel 2: Correlation distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        correlations = agg_metrics['correlations_list']
        ax2.hist(correlations, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(avg_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_corr:.3f}')
        ax2.axvline(-0.5, color='orange', linestyle=':', linewidth=2, label='Inversion Threshold')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Number of Judges')
        ax2.set_title('Judge Correlation Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Inversion pattern pie chart (middle-right)
        ax3 = fig.add_subplot(gs[1, 2:4])
        patterns = analysis_results['judge_inversion']['inversion_detection']['patterns']
        pattern_counts = [
            len(patterns['complete_inversion']),
            len(patterns['partial_inversion']),
            len(patterns['no_inversion']),
            len(patterns['amplification'])
        ]
        pattern_labels = ['Complete\nInversion', 'Partial\nInversion', 'No\nInversion', 'Amplification']
        colors = ['darkred', 'orange', 'lightgreen', 'blue']
        
        wedges, texts, autotexts = ax3.pie(pattern_counts, labels=pattern_labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Contamination Patterns', fontweight='bold')
        
        # Panel 4: Judge performance comparison (bottom-left, spans 2x2)
        ax4 = fig.add_subplot(gs[2:4, 0:2])
        
        individual_judges = analysis_results['judge_inversion']['individual_judges']
        judge_names = list(individual_judges.keys())
        correlations = [individual_judges[j]['correlations']['pearson']['correlation'] for j in judge_names]
        inversion_strengths = [individual_judges[j]['inversion_analysis']['inversion_strength'] for j in judge_names]
        
        scatter = ax4.scatter(correlations, inversion_strengths, s=100, alpha=0.7, 
                             c=correlations, cmap='RdYlBu_r', edgecolor='black')
        
        # Add judge labels
        for i, judge in enumerate(judge_names):
            ax4.annotate(judge.replace('-judge', ''), (correlations[i], inversion_strengths[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Correlation Coefficient')
        ax4.set_ylabel('Inversion Strength')
        ax4.set_title('Judge Performance Map', fontweight='bold')
        ax4.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(-0.5, color='red', linestyle='--', alpha=0.5, label='Inversion Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        # Panel 5: Robustness curve (bottom-right) - if robustness data available
        ax5 = fig.add_subplot(gs[2:4, 2:4])
        if robustness_results:
            contamination_curves = robustness_results['contamination_curves']
            rates = []
            naive_r2 = []
            
            for rate_key in sorted(contamination_curves.keys()):
                rate = float(rate_key.split('_')[1])
                rates.append(rate)
                naive_r2.append(contamination_curves[rate_key]['naive_average']['r2'])
            
            ax5.plot(rates, naive_r2, 'o-', linewidth=3, markersize=8, color='steelblue')
            ax5.set_xlabel('Contamination Rate')
            ax5.set_ylabel('R² Score')
            ax5.set_title('Aggregator Robustness Curve', fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Add breakdown threshold if available
            if 'breakdown_rate' in robustness_results['robustness_metrics'] and robustness_results['robustness_metrics']['breakdown_rate']:
                breakdown_rate = robustness_results['robustness_metrics']['breakdown_rate']
                ax5.axvline(breakdown_rate, color='red', linestyle=':', linewidth=2,
                           label=f'Breakdown at {breakdown_rate:.1f}')
                ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Robustness Analysis\nNot Available', 
                    ha='center', va='center', transform=ax5.transAxes,
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            ax5.set_title('Robustness Analysis', fontweight='bold')
            ax5.axis('off')
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_publication_figure(self, 
                                 analysis_results: Dict[str, Any],
                                 baseline_scores: pd.DataFrame,
                                 contaminated_scores: pd.DataFrame,
                                 judge_mapping: Dict[str, str],
                                 save_name: str = "publication_figure") -> plt.Figure:
        """
        Create a publication-ready figure for papers and presentations.
        
        Args:
            analysis_results: Results from contamination analysis
            baseline_scores: Clean judge scores
            contaminated_scores: Contaminated judge scores
            judge_mapping: Mapping from baseline to contaminated judge IDs
            save_name: Name for saved figure
            
        Returns:
            matplotlib Figure object
        """
        # Create figure with specific layout for publication
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel A: Representative judge correlation
        representative_judge = list(judge_mapping.keys())[0]
        baseline_data = baseline_scores[representative_judge].dropna()
        contaminated_data = contaminated_scores[judge_mapping[representative_judge]].dropna()
        
        min_len = min(len(baseline_data), len(contaminated_data))
        baseline_data = baseline_data.iloc[:min_len]
        contaminated_data = contaminated_data.iloc[:min_len]
        
        ax1.scatter(baseline_data, contaminated_data, alpha=0.6, s=20, color='steelblue')
        z = np.polyfit(baseline_data, contaminated_data, 1)
        p = np.poly1d(z)
        ax1.plot(baseline_data, p(baseline_data), "r-", linewidth=2, alpha=0.8)
        
        from scipy.stats import pearsonr
        corr, _ = pearsonr(baseline_data, contaminated_data)
        ax1.set_xlabel('Baseline Judge Score')
        ax1.set_ylabel('Contaminated Judge Score')
        ax1.set_title(f'A. Judge Score Correlation\n(r = {corr:.3f})', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Correlation summary
        individual_judges = analysis_results['judge_inversion']['individual_judges']
        correlations = [individual_judges[j]['correlations']['pearson']['correlation'] 
                       for j in individual_judges.keys()]
        
        ax2.hist(correlations, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(correlations):.3f}')
        ax2.axvline(-0.5, color='orange', linestyle=':', linewidth=2, label='Inversion Threshold')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Number of Judges')
        ax2.set_title('B. Correlation Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Inversion success rates
        patterns = analysis_results['judge_inversion']['inversion_detection']['patterns']
        pattern_counts = [
            len(patterns['complete_inversion']),
            len(patterns['partial_inversion']),
            len(patterns['no_inversion'])
        ]
        pattern_labels = ['Complete\nInversion', 'Partial\nInversion', 'No Inversion']
        colors = ['darkred', 'orange', 'lightgreen']
        
        bars = ax3.bar(pattern_labels, pattern_counts, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Number of Judges')
        ax3.set_title('C. Contamination Success', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, pattern_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Panel D: Performance impact summary
        agg_metrics = analysis_results['judge_inversion']['aggregate_metrics']
        metrics = ['Avg Correlation', 'Contamination\nRate', 'Inversion\nDetected']
        values = [
            agg_metrics['average_correlation'],
            agg_metrics['contamination_rate'],
            1.0 if agg_metrics['system_inversion_detected'] else 0.0
        ]
        colors_d = ['steelblue', 'lightcoral', 'green' if values[2] > 0 else 'red']
        
        bars_d = ax4.bar(metrics, values, color=colors_d, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Metric Value')
        ax4.set_title('D. System Impact Summary', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars_d, values):
            height = bar.get_height()
            label = f'{value:.3f}' if isinstance(value, float) and value != 1.0 and value != 0.0 else ('Yes' if value == 1.0 else 'No')
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontweight='bold')
        
        # Overall styling
        plt.suptitle('Judge Contamination Analysis Results', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def generate_all_visualizations(self, 
                                   analysis_results: Dict[str, Any],
                                   baseline_scores: pd.DataFrame,
                                   contaminated_scores: pd.DataFrame,
                                   judge_mapping: Dict[str, str],
                                   robustness_results: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
        """
        Generate all visualization types in one call.
        
        Args:
            analysis_results: Results from contamination analysis
            baseline_scores: Clean judge scores
            contaminated_scores: Contaminated judge scores
            judge_mapping: Mapping from baseline to contaminated judge IDs
            robustness_results: Optional robustness analysis results
            
        Returns:
            Dictionary mapping visualization names to Figure objects
        """
        figures = {}
        
        # Generate all visualization types
        figures['score_distributions'] = self.plot_score_distributions(
            baseline_scores, contaminated_scores, judge_mapping
        )
        
        figures['correlation_heatmap'] = self.plot_correlation_heatmap(analysis_results)
        
        if robustness_results:
            figures['robustness_curves'] = self.plot_robustness_curves(robustness_results)
        
        figures['individual_analysis'] = self.plot_individual_judge_analysis(
            analysis_results, baseline_scores, contaminated_scores, judge_mapping
        )
        
        figures['dashboard'] = self.create_summary_dashboard(
            analysis_results, robustness_results
        )
        
        figures['publication_figure'] = self.create_publication_figure(
            analysis_results, baseline_scores, contaminated_scores, judge_mapping
        )
        
        return figures