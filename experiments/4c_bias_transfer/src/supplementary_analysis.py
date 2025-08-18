"""
Supplementary Analysis for Experiment 4C

Creates detailed individual judge analyses and additional visualizations
that complement the main clean report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SupplementaryAnalyzer:
    """Creates detailed supplementary analysis and visualizations"""
    
    def __init__(self, timestamp: str):
        """
        Initialize supplementary analyzer
        
        Args:
            timestamp: Timestamp for file naming
        """
        self.timestamp = timestamp
        
    def create_individual_judge_analysis(self,
                                       bias_reduction_results: Dict,
                                       save_dir: str) -> List[str]:
        """
        Create detailed individual judge bias analysis
        
        Returns:
            List of saved file paths
        """
        logger.info("Creating individual judge analysis...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Extract individual judges data from the bias reduction structure
        individual_judges_data = bias_reduction_results.get('individual_judges', {})
        
        # Convert to format expected by our plotting functions
        individual_results = {}
        if 'framing_biases' in individual_judges_data and 'frequency_biases' in individual_judges_data:
            framing_biases = individual_judges_data['framing_biases']
            frequency_biases = individual_judges_data['frequency_biases']
            
            # Create individual results structure for plotting
            for i in range(individual_judges_data.get('count', 0)):
                judge_name = f'judge_{i+1}'
                individual_results[judge_name] = {
                    'framing_flip': framing_biases[i] if i < len(framing_biases) else np.nan,
                    'overall_frequency_bias': frequency_biases[i] if i < len(frequency_biases) else np.nan
                }
        
        saved_files = []
        
        if individual_results:
            # 1. Individual Judge Bias Heatmap
            heatmap_path = self._create_judge_bias_heatmap(individual_results, save_dir)
            saved_files.append(heatmap_path)
            
            # 2. Individual Judge Distribution Plot
            distribution_path = self._create_judge_distribution_plot(individual_results, save_dir)
            saved_files.append(distribution_path)
            
            # 3. Judge Performance Ranking
            ranking_path = self._create_judge_ranking_plot(individual_results, save_dir)
            saved_files.append(ranking_path)
        else:
            logger.warning("No individual judge results found for analysis")
        
        return saved_files
    
    def _create_judge_bias_heatmap(self, individual_results: Dict, save_dir: Path) -> str:
        """Create heatmap of individual judge biases"""
        logger.info("Creating judge bias heatmap...")
        
        # Prepare data for heatmap
        judges = []
        framing_biases = []
        frequency_biases = []
        
        for judge_name, results in individual_results.items():
            judges.append(judge_name.replace('judge_', 'Judge '))
            framing_biases.append(results.get('framing_flip', np.nan))
            frequency_biases.append(results.get('overall_frequency_bias', np.nan))
        
        # Create heatmap data
        heatmap_data = np.array([
            [abs(x) if not np.isnan(x) else 0 for x in framing_biases],
            [abs(x) if not np.isnan(x) else 0 for x in frequency_biases]
        ])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        im = sns.heatmap(heatmap_data, 
                        xticklabels=judges,
                        yticklabels=['Framing Bias', 'Frequency Bias'],
                        annot=True, 
                        fmt='.3f',
                        cmap='YlOrRd',
                        ax=ax)
        
        ax.set_title('Individual Judge Bias Levels', fontsize=14, fontweight='bold')
        ax.set_xlabel('Individual Judges')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = save_dir / f"{self.timestamp}_individual_judge_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Judge bias heatmap saved: {save_path}")
        return str(save_path)
    
    def _create_judge_distribution_plot(self, individual_results: Dict, save_dir: Path) -> str:
        """Create distribution plots for individual judges"""
        logger.info("Creating judge distribution plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract bias values
        framing_biases = []
        frequency_biases = []
        judge_names = []
        
        for judge_name, results in individual_results.items():
            if not np.isnan(results.get('framing_flip', np.nan)):
                framing_biases.append(abs(results['framing_flip']))
                judge_names.append(judge_name.replace('judge_', 'J'))
        
        for judge_name, results in individual_results.items():
            if not np.isnan(results.get('overall_frequency_bias', np.nan)):
                frequency_biases.append(abs(results['overall_frequency_bias']))
        
        # Plot 1: Framing Bias Distribution
        if framing_biases:
            ax1.bar(range(len(framing_biases)), framing_biases, 
                   color='lightcoral', alpha=0.7)
            ax1.set_title('Framing Bias by Individual Judge')
            ax1.set_xlabel('Judge')
            ax1.set_ylabel('Absolute Framing Bias')
            ax1.set_xticks(range(len(framing_biases)))
            ax1.set_xticklabels(judge_names[:len(framing_biases)], rotation=45)
            
            # Add mean line
            mean_framing = np.mean(framing_biases)
            ax1.axhline(y=mean_framing, color='red', linestyle='--', 
                       label=f'Mean: {mean_framing:.3f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No framing bias data', ha='center', va='center', 
                    transform=ax1.transAxes)
            ax1.set_title('Framing Bias by Individual Judge')
        
        # Plot 2: Frequency Bias Distribution
        if frequency_biases:
            ax2.bar(range(len(frequency_biases)), frequency_biases, 
                   color='lightblue', alpha=0.7)
            ax2.set_title('Frequency Bias by Individual Judge')
            ax2.set_xlabel('Judge')
            ax2.set_ylabel('Absolute Frequency Bias')
            ax2.set_xticks(range(len(frequency_biases)))
            ax2.set_xticklabels([f'J{i+1}' for i in range(len(frequency_biases))], rotation=45)
            
            # Add mean line
            mean_frequency = np.mean(frequency_biases)
            ax2.axhline(y=mean_frequency, color='blue', linestyle='--', 
                       label=f'Mean: {mean_frequency:.3f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No frequency bias data', ha='center', va='center', 
                    transform=ax2.transAxes)
            ax2.set_title('Frequency Bias by Individual Judge')
        
        plt.tight_layout()
        
        save_path = save_dir / f"{self.timestamp}_individual_judge_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Judge distribution plots saved: {save_path}")
        return str(save_path)
    
    def _create_judge_ranking_plot(self, individual_results: Dict, save_dir: Path) -> str:
        """Create ranking plot of judges by bias levels"""
        logger.info("Creating judge ranking plot...")
        
        # Prepare data
        judge_data = []
        for judge_name, results in individual_results.items():
            framing_bias = abs(results.get('framing_flip', np.nan))
            frequency_bias = abs(results.get('overall_frequency_bias', np.nan))
            
            # Combined bias score (average of available biases)
            biases = [x for x in [framing_bias, frequency_bias] if not np.isnan(x)]
            combined_bias = np.mean(biases) if biases else np.nan
            
            judge_data.append({
                'judge': judge_name.replace('judge_', 'Judge '),
                'framing_bias': framing_bias,
                'frequency_bias': frequency_bias,
                'combined_bias': combined_bias
            })
        
        # Sort by combined bias
        judge_data.sort(key=lambda x: x['combined_bias'] if not np.isnan(x['combined_bias']) else 999)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        judges = [d['judge'] for d in judge_data if not np.isnan(d['combined_bias'])]
        combined_biases = [d['combined_bias'] for d in judge_data if not np.isnan(d['combined_bias'])]
        
        if judges and combined_biases:
            # Create horizontal bar chart
            y_pos = np.arange(len(judges))
            bars = ax.barh(y_pos, combined_biases, color='lightgreen', alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(judges)
            ax.set_xlabel('Combined Bias Score (Average of Framing + Frequency)')
            ax.set_title('Judge Ranking by Overall Bias Level\n(Lower = Better)', 
                        fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            # Color code: green for low bias, red for high bias
            if len(combined_biases) > 1:
                max_bias = max(combined_biases)
                min_bias = min(combined_biases)
                for i, bar in enumerate(bars):
                    bias_level = combined_biases[i]
                    normalized_bias = (bias_level - min_bias) / (max_bias - min_bias) if max_bias > min_bias else 0
                    color = plt.cm.RdYlGn_r(normalized_bias)  # Red for high, green for low
                    bar.set_color(color)
                    bar.set_alpha(0.8)
        else:
            ax.text(0.5, 0.5, 'No bias data for ranking', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Judge Ranking by Overall Bias Level')
        
        plt.tight_layout()
        
        save_path = save_dir / f"{self.timestamp}_judge_ranking.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Judge ranking plot saved: {save_path}")
        return str(save_path)
    
    def create_method_comparison_details(self,
                                       bias_reduction_results: Dict,
                                       save_dir: str) -> List[str]:
        """
        Create detailed method comparison visualizations
        
        Returns:
            List of saved file paths
        """
        logger.info("Creating detailed method comparison...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        saved_files = []
        
        # 1. Detailed Performance Matrix
        matrix_path = self._create_performance_matrix(bias_reduction_results, save_dir)
        saved_files.append(matrix_path)
        
        # 2. Bias Reduction Effectiveness Chart
        effectiveness_path = self._create_effectiveness_chart(bias_reduction_results, save_dir)
        saved_files.append(effectiveness_path)
        
        return saved_files
    
    def _create_performance_matrix(self, bias_reduction_results: Dict, save_dir: Path) -> str:
        """Create detailed performance comparison matrix"""
        logger.info("Creating performance matrix...")
        
        individual = bias_reduction_results['individual_judges']
        naive = bias_reduction_results['naive_average']
        mlp = bias_reduction_results['mlp_aggregator']
        
        # Create performance matrix
        methods = ['Individual\nJudges', 'Naive\nAverage', 'MLP\nAggregator']
        framing_values = [
            individual['mean_framing_bias'],
            naive['framing_bias'],
            mlp['framing_bias']
        ]
        frequency_values = [
            individual['mean_frequency_bias'],
            naive['frequency_bias'],
            mlp['frequency_bias']
        ]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Method Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Absolute Bias Levels
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, framing_values, width, label='Framing Bias', 
                       color='lightcoral', alpha=0.7)
        bars2 = ax1.bar(x + width/2, frequency_values, width, label='Frequency Bias', 
                       color='lightblue', alpha=0.7)
        
        ax1.set_ylabel('Absolute Bias Level')
        ax1.set_title('Absolute Bias Levels by Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Reduction Percentages
        reduction_methods = ['Naive Average', 'MLP Aggregator']
        framing_reductions = [naive['framing_reduction_percent'], mlp['framing_reduction_percent']]
        frequency_reductions = [naive['frequency_reduction_percent'], mlp['frequency_reduction_percent']]
        
        x = np.arange(len(reduction_methods))
        bars1 = ax2.bar(x - width/2, framing_reductions, width, label='Framing Reduction', 
                       color='lightcoral', alpha=0.7)
        bars2 = ax2.bar(x + width/2, frequency_reductions, width, label='Frequency Reduction', 
                       color='lightblue', alpha=0.7)
        
        ax2.set_ylabel('Reduction Percentage (%)')
        ax2.set_title('Bias Reduction Percentages')
        ax2.set_xticks(x)
        ax2.set_xticklabels(reduction_methods)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax2.text(bar.get_x() + bar.get_width()/2., 
                           height + (2 if height >= 0 else -5),
                           f'{height:.1f}%', ha='center', 
                           va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Plot 3: Effectiveness Score
        effectiveness_scores = []
        for method in ['naive', 'mlp']:
            if method == 'naive':
                framing_eff = max(0, naive['framing_reduction_percent']) / 100 if not np.isnan(naive['framing_reduction_percent']) else 0
                frequency_eff = max(0, naive['frequency_reduction_percent']) / 100 if not np.isnan(naive['frequency_reduction_percent']) else 0
            else:
                framing_eff = max(0, mlp['framing_reduction_percent']) / 100 if not np.isnan(mlp['framing_reduction_percent']) else 0
                frequency_eff = max(0, mlp['frequency_reduction_percent']) / 100 if not np.isnan(mlp['frequency_reduction_percent']) else 0
            
            overall_eff = (framing_eff + frequency_eff) / 2
            effectiveness_scores.append(overall_eff * 100)
        
        bars = ax3.bar(['Naive Average', 'MLP Aggregator'], effectiveness_scores, 
                      color=['orange', 'lightgreen'], alpha=0.7)
        ax3.set_ylabel('Overall Effectiveness Score (%)')
        ax3.set_title('Overall Bias Reduction Effectiveness')
        ax3.set_ylim(0, 100)
        
        for bar, score in zip(bars, effectiveness_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 4: Winner Analysis
        categories = ['Framing Bias\nReduction', 'Frequency Bias\nReduction', 'Overall\nPerformance']
        naive_wins = [
            naive['framing_reduction_percent'] > mlp['framing_reduction_percent'] if not np.isnan(naive['framing_reduction_percent']) and not np.isnan(mlp['framing_reduction_percent']) else False,
            naive['frequency_reduction_percent'] > mlp['frequency_reduction_percent'] if not np.isnan(naive['frequency_reduction_percent']) and not np.isnan(mlp['frequency_reduction_percent']) else False,
            effectiveness_scores[0] > effectiveness_scores[1]
        ]
        
        colors = ['green' if win else 'red' for win in naive_wins]
        bars = ax4.bar(categories, [100 if win else 0 for win in naive_wins], color=colors, alpha=0.7)
        ax4.set_ylabel('Winner')
        ax4.set_title('Method Performance Winners')
        ax4.set_ylim(0, 100)
        ax4.set_yticks([0, 100])
        ax4.set_yticklabels(['MLP Wins', 'Naive Wins'])
        
        # Add winner labels
        for i, (bar, win) in enumerate(zip(bars, naive_wins)):
            winner = 'Naive\nAverage' if win else 'MLP\nAggregator'
            ax4.text(bar.get_x() + bar.get_width()/2., 50,
                   winner, ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white')
        
        plt.tight_layout()
        
        save_path = save_dir / f"{self.timestamp}_performance_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance matrix saved: {save_path}")
        return str(save_path)
    
    def _create_effectiveness_chart(self, bias_reduction_results: Dict, save_dir: Path) -> str:
        """Create bias reduction effectiveness radar/spider chart"""
        logger.info("Creating effectiveness chart...")
        
        naive = bias_reduction_results['naive_average']
        mlp = bias_reduction_results['mlp_aggregator']
        
        # Prepare metrics (normalize to 0-100 scale)
        metrics = ['Framing\nReduction', 'Frequency\nReduction', 'Consistency', 'Overall\nScore']
        
        # Calculate scores (0-100 scale)
        naive_scores = [
            max(0, naive['framing_reduction_percent']) if not np.isnan(naive['framing_reduction_percent']) else 0,
            max(0, naive['frequency_reduction_percent']) if not np.isnan(naive['frequency_reduction_percent']) else 0,
            100 - abs(naive['framing_reduction_percent'] - naive['frequency_reduction_percent']) if not np.isnan(naive['framing_reduction_percent']) and not np.isnan(naive['frequency_reduction_percent']) else 0,
            0  # Will calculate
        ]
        
        mlp_scores = [
            max(0, mlp['framing_reduction_percent']) if not np.isnan(mlp['framing_reduction_percent']) else 0,
            max(0, mlp['frequency_reduction_percent']) if not np.isnan(mlp['frequency_reduction_percent']) else 0,
            100 - abs(mlp['framing_reduction_percent'] - mlp['frequency_reduction_percent']) if not np.isnan(mlp['framing_reduction_percent']) and not np.isnan(mlp['frequency_reduction_percent']) else 0,
            0  # Will calculate
        ]
        
        # Calculate overall scores
        naive_scores[3] = np.mean(naive_scores[:3])
        mlp_scores[3] = np.mean(mlp_scores[:3])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, naive_scores, width, label='Naive Average', 
                      color='orange', alpha=0.7)
        bars2 = ax.bar(x + width/2, mlp_scores, width, label='MLP Aggregator', 
                      color='lightgreen', alpha=0.7)
        
        ax.set_ylabel('Effectiveness Score (0-100)')
        ax.set_title('Bias Reduction Effectiveness Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Add reference lines
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Threshold')
        ax.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='75% Good Performance')
        
        plt.tight_layout()
        
        save_path = save_dir / f"{self.timestamp}_effectiveness_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Effectiveness chart saved: {save_path}")
        return str(save_path)
    
    def generate_supplementary_readme(self, saved_files: List[str], save_dir: str) -> str:
        """Generate README for supplementary analysis folder"""
        save_dir = Path(save_dir)
        readme_path = save_dir / "README.md"
        
        readme_content = f"""# Supplementary Analysis - Experiment 4C

This folder contains detailed supplementary visualizations that complement the main clean report.

## Generated on: {self.timestamp}

## Individual Judge Analysis

### Judge Bias Heatmap
- **File**: `{self.timestamp}_individual_judge_heatmap.png`
- **Description**: Heatmap showing framing and frequency bias levels for each individual judge
- **Purpose**: Identify which judges have the highest/lowest bias levels

### Judge Distribution Plots  
- **File**: `{self.timestamp}_individual_judge_distributions.png`
- **Description**: Bar charts showing bias distribution across individual judges
- **Purpose**: Compare individual judge performance and identify outliers

### Judge Ranking
- **File**: `{self.timestamp}_judge_ranking.png`
- **Description**: Horizontal bar chart ranking judges by combined bias score
- **Purpose**: Identify best and worst performing individual judges

## Method Comparison Details

### Performance Matrix
- **File**: `{self.timestamp}_performance_matrix.png`
- **Description**: Comprehensive 4-panel comparison of naive vs MLP performance
- **Purpose**: Detailed analysis of method effectiveness across multiple metrics

### Effectiveness Chart
- **File**: `{self.timestamp}_effectiveness_chart.png`
- **Description**: Bar chart comparing methods across effectiveness dimensions
- **Purpose**: Overall performance comparison with scoring system

## How to Use

These supplementary figures provide additional insights beyond the main clean report:

1. **Individual Judge Analysis** - Use to understand variability in judge performance and identify potential problematic judges
2. **Method Comparison Details** - Use for deeper analysis of why one method outperforms another

## Main Report Files

For the primary results, see the main results folder:
- `{self.timestamp}_COMPLETE_REPORT.json` - Full detailed report
- `{self.timestamp}_CLEAN_SUMMARY.md` - Clean summary report  
- `{self.timestamp}_bias_reduction_plots.png` - Main visualization

## Interpretation Notes

- **Lower bias values** = Better performance (less cognitive bias)
- **Higher reduction percentages** = Better bias mitigation
- **Negative reduction percentages** = Bias amplification (concerning)
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Supplementary README saved: {readme_path}")
        return str(readme_path)


def main():
    """Example usage"""
    analyzer = SupplementaryAnalyzer("20250814_example")
    print("Supplementary Analyzer initialized")


if __name__ == "__main__":
    main()