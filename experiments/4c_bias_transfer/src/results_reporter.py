"""
Clean Results Reporter for Experiment 4C

Generates three types of outputs:
1. Complete detailed report (JSON)
2. Clean bias reduction summary (Markdown)
3. Bias reduction visualizations (PNG)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CleanResultsReporter:
    """Clean and organized results reporting for bias transfer analysis"""
    
    def __init__(self, timestamp: str = None):
        """
        Initialize reporter
        
        Args:
            timestamp: Optional timestamp for file naming
        """
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def calculate_bias_reduction(self, 
                               individual_results: Dict,
                               naive_results: Dict,
                               mlp_results: Dict) -> Dict[str, float]:
        """
        Calculate clean bias reduction percentages
        
        Args:
            individual_results: Results from individual judges  
            naive_results: Results from naive averaging
            mlp_results: Results from MLP aggregator
            
        Returns:
            Clean bias reduction metrics
        """
        logger.info("Calculating bias reduction metrics...")
        
        # Extract individual judge biases
        individual_framing_biases = []
        individual_frequency_biases = []
        
        for judge_name, results in individual_results.items():
            if 'framing_flip' in results and not np.isnan(results['framing_flip']):
                individual_framing_biases.append(abs(results['framing_flip']))
            
            if 'overall_frequency_bias' in results and not np.isnan(results['overall_frequency_bias']):
                individual_frequency_biases.append(abs(results['overall_frequency_bias']))
        
        # Extract naive mean biases
        naive_framing_bias = np.nan
        naive_frequency_bias = np.nan
        
        if naive_results and 'framing_flip' in naive_results and not np.isnan(naive_results['framing_flip']):
            naive_framing_bias = abs(naive_results['framing_flip'])
        
        if naive_results and 'overall_frequency_bias' in naive_results and not np.isnan(naive_results['overall_frequency_bias']):
            naive_frequency_bias = abs(naive_results['overall_frequency_bias'])
        
        # Extract MLP aggregator biases
        mlp_framing_bias = np.nan
        mlp_frequency_bias = np.nan
        
        if mlp_results and 'framing_flip' in mlp_results and not np.isnan(mlp_results['framing_flip']):
            mlp_framing_bias = abs(mlp_results['framing_flip'])
        
        if mlp_results and 'overall_frequency_bias' in mlp_results and not np.isnan(mlp_results['overall_frequency_bias']):
            mlp_frequency_bias = abs(mlp_results['overall_frequency_bias'])
        
        # Calculate mean individual biases
        mean_individual_framing = np.mean(individual_framing_biases) if individual_framing_biases else np.nan
        mean_individual_frequency = np.mean(individual_frequency_biases) if individual_frequency_biases else np.nan
        
        # Calculate reduction percentages vs naive
        naive_framing_reduction = np.nan
        naive_frequency_reduction = np.nan
        
        if not np.isnan(mean_individual_framing) and not np.isnan(naive_framing_bias) and mean_individual_framing > 0:
            naive_framing_reduction = ((mean_individual_framing - naive_framing_bias) / mean_individual_framing) * 100
        
        if not np.isnan(mean_individual_frequency) and not np.isnan(naive_frequency_bias) and mean_individual_frequency > 0:
            naive_frequency_reduction = ((mean_individual_frequency - naive_frequency_bias) / mean_individual_frequency) * 100
        
        # Calculate reduction percentages vs MLP
        mlp_framing_reduction = np.nan
        mlp_frequency_reduction = np.nan
        
        if not np.isnan(mean_individual_framing) and not np.isnan(mlp_framing_bias) and mean_individual_framing > 0:
            mlp_framing_reduction = ((mean_individual_framing - mlp_framing_bias) / mean_individual_framing) * 100
        
        if not np.isnan(mean_individual_frequency) and not np.isnan(mlp_frequency_bias) and mean_individual_frequency > 0:
            mlp_frequency_reduction = ((mean_individual_frequency - mlp_frequency_bias) / mean_individual_frequency) * 100
        
        return {
            # Individual judge metrics
            'individual_judges': {
                'count': len(individual_results),
                'mean_framing_bias': mean_individual_framing,
                'mean_frequency_bias': mean_individual_frequency,
                'framing_biases': individual_framing_biases,
                'frequency_biases': individual_frequency_biases
            },
            
            # Naive averaging metrics  
            'naive_average': {
                'framing_bias': naive_framing_bias,
                'frequency_bias': naive_frequency_bias,
                'framing_reduction_percent': naive_framing_reduction,
                'frequency_reduction_percent': naive_frequency_reduction
            },
            
            # MLP aggregator metrics
            'mlp_aggregator': {
                'framing_bias': mlp_framing_bias,
                'frequency_bias': mlp_frequency_bias,
                'framing_reduction_percent': mlp_framing_reduction,
                'frequency_reduction_percent': mlp_frequency_reduction
            }
        }
    
    def generate_complete_report(self,
                               token_dataset: pd.DataFrame,
                               scores_dataset: pd.DataFrame, 
                               analysis_results: Dict,
                               bias_reduction_results: Dict,
                               experiment_config: Dict) -> Dict[str, Any]:
        """
        Generate complete detailed report
        
        Returns:
            Complete report dictionary
        """
        logger.info("Generating complete detailed report...")
        
        complete_report = {
            'experiment_metadata': {
                'name': 'Experiment 4C: Framing Effects and Bias Transfer',
                'timestamp': self.timestamp,
                'run_date': datetime.now().isoformat(),
                'version': '2.0-cleaned',
                **experiment_config
            },
            
            'dataset_info': {
                'token_dataset': {
                    'total_tokens': len(token_dataset),
                    'positive_tokens': int(token_dataset['is_positive'].sum()) if 'is_positive' in token_dataset.columns else 0,
                    'negative_tokens': int(token_dataset['is_negative'].sum()) if 'is_negative' in token_dataset.columns else 0,
                    'neutral_control_tokens': int(token_dataset['is_neutral_control'].sum()) if 'is_neutral_control' in token_dataset.columns else 0
                },
                'scores_dataset': {
                    'total_score_records': len(scores_dataset),
                    'unique_tokens_scored': int(scores_dataset['token'].nunique()) if 'token' in scores_dataset.columns else 0,
                    'prompt_types': list(scores_dataset['prompt_type'].unique()) if 'prompt_type' in scores_dataset.columns else []
                }
            },
            
            'bias_analysis_results': {
                'framing_effects': analysis_results.get('framing_results', {}),
                'frequency_bias': analysis_results.get('frequency_results', {}), 
                'significance_tests': analysis_results.get('significance_results', {})
            },
            
            'bias_reduction_analysis': bias_reduction_results,
            
            'key_findings': self._extract_key_findings(bias_reduction_results),
            
            'model_performance': {
                'individual_judges_analyzed': bias_reduction_results['individual_judges']['count'],
                'naive_average_analyzed': bool(bias_reduction_results['naive_average']),
                'mlp_aggregator_analyzed': bool(bias_reduction_results['mlp_aggregator'])
            }
        }
        
        self.results = complete_report
        return complete_report
    
    def _extract_key_findings(self, bias_reduction_results: Dict) -> List[str]:
        """Extract key findings from bias reduction results"""
        findings = []
        
        # MLP findings
        mlp_framing = bias_reduction_results['mlp_aggregator']['framing_reduction_percent']
        mlp_frequency = bias_reduction_results['mlp_aggregator']['frequency_reduction_percent']
        
        # Naive findings
        naive_framing = bias_reduction_results['naive_average']['framing_reduction_percent']
        naive_frequency = bias_reduction_results['naive_average']['frequency_reduction_percent']
        
        # MLP framing bias findings
        if not np.isnan(mlp_framing):
            if mlp_framing > 0:
                findings.append(f"MLP reduces framing bias by {mlp_framing:.1f}%")
            else:
                findings.append(f"MLP increases framing bias by {abs(mlp_framing):.1f}%")
        else:
            findings.append("MLP framing bias reduction could not be calculated")
        
        # MLP frequency bias findings
        if not np.isnan(mlp_frequency):
            if mlp_frequency > 0:
                findings.append(f"MLP reduces frequency bias by {mlp_frequency:.1f}%")
            else:
                findings.append(f"MLP increases frequency bias by {abs(mlp_frequency):.1f}%")
        else:
            findings.append("MLP frequency bias reduction could not be calculated")
        
        # Naive comparison findings
        if not np.isnan(naive_framing):
            if naive_framing > 0:
                findings.append(f"Naive averaging reduces framing bias by {naive_framing:.1f}%")
            else:
                findings.append(f"Naive averaging increases framing bias by {abs(naive_framing):.1f}%")
        
        if not np.isnan(naive_frequency):
            if naive_frequency > 0:
                findings.append(f"Naive averaging reduces frequency bias by {naive_frequency:.1f}%")
            else:
                findings.append(f"Naive averaging increases frequency bias by {abs(naive_frequency):.1f}%")
        
        # Comparison between MLP and naive
        if not np.isnan(mlp_framing) and not np.isnan(naive_framing):
            if mlp_framing > naive_framing:
                findings.append("MLP outperforms naive averaging on framing bias")
            elif mlp_framing < naive_framing:
                findings.append("Naive averaging outperforms MLP on framing bias")
            else:
                findings.append("MLP and naive averaging perform equally on framing bias")
        
        if not np.isnan(mlp_frequency) and not np.isnan(naive_frequency):
            if mlp_frequency > naive_frequency:
                findings.append("MLP outperforms naive averaging on frequency bias")
            elif mlp_frequency < naive_frequency:
                findings.append("Naive averaging outperforms MLP on frequency bias")
            else:
                findings.append("MLP and naive averaging perform equally on frequency bias")
        
        return findings
    
    def generate_clean_summary(self, complete_report: Dict) -> str:
        """
        Generate clean markdown summary
        
        Returns:
            Markdown summary string
        """
        logger.info("Generating clean summary...")
        
        metadata = complete_report['experiment_metadata']
        dataset_info = complete_report['dataset_info']
        bias_reduction = complete_report['bias_reduction_analysis']
        individual = bias_reduction['individual_judges']
        naive = bias_reduction['naive_average']
        mlp = bias_reduction['mlp_aggregator']
        key_findings = complete_report['key_findings']
        
        summary = f"""# Experiment 4C: Framing Effects and Bias Transfer - Clean Results

## Experiment Overview
- **Run Date**: {metadata['run_date'][:19]}
- **Mode**: {'Quick' if metadata.get('quick_mode', False) else 'Full'}
- **Judge Scoring**: {'Real' if metadata.get('use_real_judges', False) else 'Mock'}
- **Score Normalization**: {'Enabled' if metadata.get('normalize_scores', False) else 'Disabled'}

## Dataset Summary
- **Total Tokens Analyzed**: {dataset_info['token_dataset']['total_tokens']:,}
- **Positive Sentiment**: {dataset_info['token_dataset']['positive_tokens']}
- **Negative Sentiment**: {dataset_info['token_dataset']['negative_tokens']}
- **Neutral Control**: {dataset_info['token_dataset']['neutral_control_tokens']}
- **Total Score Records**: {dataset_info['scores_dataset']['total_score_records']:,}

## Model Analysis
- **Individual Judges**: {individual['count']}
- **Aggregation Methods**: Naive Average + MLP Aggregator

## Bias Reduction Results

### Framing Bias
- **Individual Judge Mean**: {individual['mean_framing_bias']:.3f}
- **Naive Average**: {naive['framing_bias']:.3f} ({naive['framing_reduction_percent']:.1f}% reduction)
- **MLP Aggregator**: {mlp['framing_bias']:.3f} ({mlp['framing_reduction_percent']:.1f}% reduction)

### Frequency Bias
- **Individual Judge Mean**: {individual['mean_frequency_bias']:.3f}
- **Naive Average**: {naive['frequency_bias']:.3f} ({naive['frequency_reduction_percent']:.1f}% reduction)
- **MLP Aggregator**: {mlp['frequency_bias']:.3f} ({mlp['frequency_reduction_percent']:.1f}% reduction)

## Key Findings

"""
        
        # Key findings
        for finding in key_findings:
            summary += f"- {finding}\n"
        
        # Files generated
        summary += f"""
## Files Generated
- **Complete Report**: `{self.timestamp}_COMPLETE_REPORT.json`
- **Clean Summary**: `{self.timestamp}_CLEAN_SUMMARY.md`
- **Bias Reduction Plots**: `{self.timestamp}_bias_reduction_plots.png`

## Interpretation
The bias reduction percentages show how much each aggregation method reduces the cognitive biases present in individual judges:
- **Positive values** = Bias reduction (desirable)
- **Negative values** = Bias amplification (concerning)
- **Comparison**: MLP vs Naive averaging performance
"""
        
        return summary
    
    def create_bias_reduction_plots(self, 
                                  bias_reduction_results: Dict,
                                  save_path: str = None) -> str:
        """
        Create clean bias reduction visualization
        
        Returns:
            Path to saved plot
        """
        logger.info("Creating bias reduction plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experiment 4C: Bias Reduction Analysis', fontsize=16, fontweight='bold')
        
        individual = bias_reduction_results['individual_judges']
        naive = bias_reduction_results['naive_average']
        mlp = bias_reduction_results['mlp_aggregator']
        
        # Plot 1: Framing Bias Comparison
        framing_values = [individual['mean_framing_bias'], naive['framing_bias'], mlp['framing_bias']]
        framing_labels = ['Individual\nJudges', 'Naive\nAverage', 'MLP\nAggregator']
        framing_colors = ['lightblue', 'orange', 'lightgreen']
        
        # Filter out NaN values
        framing_plot_values = []
        framing_plot_labels = []
        framing_plot_colors = []
        
        for val, label, color in zip(framing_values, framing_labels, framing_colors):
            if not np.isnan(val):
                framing_plot_values.append(val)
                framing_plot_labels.append(label)
                framing_plot_colors.append(color)
        
        if framing_plot_values:
            bars = ax1.bar(framing_plot_labels, framing_plot_values, color=framing_plot_colors, alpha=0.7)
            ax1.set_title('Framing Bias Comparison')
            ax1.set_ylabel('Absolute Framing Bias')
            
            # Add value labels on bars
            for bar, value in zip(bars, framing_plot_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'No framing bias data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Framing Bias Comparison')
        
        # Plot 2: Frequency Bias Comparison  
        frequency_values = [individual['mean_frequency_bias'], naive['frequency_bias'], mlp['frequency_bias']]
        frequency_labels = ['Individual\nJudges', 'Naive\nAverage', 'MLP\nAggregator']
        frequency_colors = ['lightblue', 'orange', 'lightgreen']
        
        # Filter out NaN values
        frequency_plot_values = []
        frequency_plot_labels = []
        frequency_plot_colors = []
        
        for val, label, color in zip(frequency_values, frequency_labels, frequency_colors):
            if not np.isnan(val):
                frequency_plot_values.append(val)
                frequency_plot_labels.append(label)
                frequency_plot_colors.append(color)
        
        if frequency_plot_values:
            bars = ax2.bar(frequency_plot_labels, frequency_plot_values, color=frequency_plot_colors, alpha=0.7)
            ax2.set_title('Frequency Bias Comparison')
            ax2.set_ylabel('Absolute Frequency Bias')
            
            # Add value labels on bars
            for bar, value in zip(bars, frequency_plot_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No frequency bias data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Frequency Bias Comparison')
        
        # Plot 3: Bias Reduction Percentages
        naive_framing_red = naive['framing_reduction_percent']
        naive_frequency_red = naive['frequency_reduction_percent']
        mlp_framing_red = mlp['framing_reduction_percent']
        mlp_frequency_red = mlp['frequency_reduction_percent']
        
        # Create grouped bar chart
        x = np.arange(2)  # Framing and Frequency
        width = 0.35
        
        naive_reductions = [naive_framing_red if not np.isnan(naive_framing_red) else 0,
                           naive_frequency_red if not np.isnan(naive_frequency_red) else 0]
        mlp_reductions = [mlp_framing_red if not np.isnan(mlp_framing_red) else 0,
                         mlp_frequency_red if not np.isnan(mlp_frequency_red) else 0]
        
        bars1 = ax3.bar(x - width/2, naive_reductions, width, label='Naive Average', 
                       color='orange', alpha=0.7)
        bars2 = ax3.bar(x + width/2, mlp_reductions, width, label='MLP Aggregator', 
                       color='lightgreen', alpha=0.7)
        
        ax3.set_ylabel('Reduction Percentage (%)')
        ax3.set_title('Bias Reduction Percentages')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Framing Bias', 'Frequency Bias'])
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, naive_reductions):
            if value != 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        for bar, value in zip(bars2, mlp_reductions):
            if value != 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Plot 4: Method Performance Comparison
        methods = ['Naive Average', 'MLP Aggregator']
        framing_performance = [naive_framing_red if not np.isnan(naive_framing_red) else 0,
                              mlp_framing_red if not np.isnan(mlp_framing_red) else 0]
        frequency_performance = [naive_frequency_red if not np.isnan(naive_frequency_red) else 0,
                                mlp_frequency_red if not np.isnan(mlp_frequency_red) else 0]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, framing_performance, width, label='Framing Bias Reduction', 
                       color='lightcoral', alpha=0.7)
        bars2 = ax4.bar(x + width/2, frequency_performance, width, label='Frequency Bias Reduction', 
                       color='lightblue', alpha=0.7)
        
        ax4.set_ylabel('Reduction Percentage (%)')
        ax4.set_title('Method Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods)
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, framing_performance):
            if value != 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        for bar, value in zip(bars2, frequency_performance):
            if value != 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                        f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = f"{self.timestamp}_bias_reduction_plots.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bias reduction plots saved to: {save_path}")
        return save_path
    
    def save_complete_report(self, complete_report: Dict, save_path: str = None) -> str:
        """Save complete report as JSON"""
        if save_path is None:
            save_path = f"{self.timestamp}_COMPLETE_REPORT.json"
        
        with open(save_path, 'w') as f:
            json.dump(complete_report, f, indent=2, default=str)
        
        logger.info(f"Complete report saved to: {save_path}")
        return save_path
    
    def save_clean_summary(self, summary: str, save_path: str = None) -> str:
        """Save clean summary as Markdown"""
        if save_path is None:
            save_path = f"{self.timestamp}_CLEAN_SUMMARY.md"
        
        with open(save_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Clean summary saved to: {save_path}")
        return save_path


def main():
    """Example usage"""
    reporter = CleanResultsReporter()
    print("Clean Results Reporter initialized")
    print(f"Timestamp: {reporter.timestamp}")


if __name__ == "__main__":
    main()