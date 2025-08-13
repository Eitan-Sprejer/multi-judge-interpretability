#!/usr/bin/env python3
"""
Analysis Script for Experiment 4C: Real Judge Results

Generates visualizations and analysis for bias transfer experiment with real judges.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import glob

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealJudgeAnalyzer:
    """Analyzes and visualizes real judge experiment results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.plots_dir = results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def find_latest_results(self):
        """Find the most recent experiment results"""
        pattern = str(self.results_dir / "*_FINAL_REPORT.json")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError("No FINAL_REPORT.json files found in results directory")
        
        # Get the most recent file
        latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        logger.info(f"Using latest results: {latest_file}")
        
        return latest_file
    
    def load_results(self, results_file: str = None):
        """Load experiment results from JSON file"""
        if results_file is None:
            results_file = self.find_latest_results()
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info("Results loaded successfully")
        return self.results
    
    def create_bias_reduction_plot(self):
        """Create bias reduction comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract bias reduction data
        comparisons = self.results['detailed_results']['comparisons']
        
        # Frequency bias reduction
        methods = []
        freq_reductions = []
        
        if 'naive_average_frequency_reduction' in comparisons:
            methods.append('Naive Average')
            freq_reductions.append(comparisons['naive_average_frequency_reduction'])
        
        if 'mlp_aggregator_frequency_reduction' in comparisons:
            methods.append('MLP Aggregator')
            freq_reductions.append(comparisons['mlp_aggregator_frequency_reduction'])
        
        if 'gam_aggregator_frequency_reduction' in comparisons and not np.isnan(comparisons['gam_aggregator_frequency_reduction']):
            methods.append('GAM Aggregator')
            freq_reductions.append(comparisons['gam_aggregator_frequency_reduction'])
        
        # Plot frequency bias reduction
        bars1 = ax1.bar(methods, freq_reductions, color='lightcoral', alpha=0.8)
        ax1.set_ylabel('Frequency Bias Reduction (%)')
        ax1.set_title('Frequency Bias Reduction by Method')
        ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Target: 25%')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Framing effects reduction
        framing_methods = []
        framing_reductions = []
        
        if 'naive_average_framing_reduction' in comparisons:
            framing_methods.append('Naive Average')
            framing_reductions.append(comparisons['naive_average_framing_reduction'])
        
        if 'mlp_aggregator_framing_reduction' in comparisons:
            framing_methods.append('MLP Aggregator')
            framing_reductions.append(comparisons['mlp_aggregator_framing_reduction'])
        
        if 'gam_aggregator_framing_reduction' in comparisons and not np.isnan(comparisons['gam_aggregator_framing_reduction']):
            framing_methods.append('GAM Aggregator')
            framing_reductions.append(comparisons['gam_aggregator_framing_reduction'])
        
        # Plot framing effects reduction
        bars2 = ax2.bar(framing_methods, framing_reductions, color='lightblue', alpha=0.8)
        ax2.set_ylabel('Framing Effects Reduction (%)')
        ax2.set_title('Framing Effects Reduction by Method')
        ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Target: 30%')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'bias_reduction_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Created bias reduction comparison plot")
    
    def create_individual_judge_analysis(self):
        """Create individual judge bias analysis"""
        frequency_results = self.results['detailed_results']['frequency_results']
        
        # Extract individual judge frequency biases
        judge_names = []
        frequency_biases = []
        
        for judge_name, results in frequency_results.items():
            if judge_name.startswith('judge_'):
                judge_names.append(judge_name.replace('judge_', 'Judge '))
                frequency_biases.append(results['overall_frequency_bias'])
        
        if not judge_names:
            logger.warning("No individual judge results found")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(judge_names, frequency_biases, color='skyblue', alpha=0.8)
        ax.set_ylabel('Frequency Bias Correlation')
        ax.set_title('Individual Judge Frequency Bias Patterns')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'individual_judge_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Created individual judge analysis plot")
    
    def create_summary_report(self):
        """Create summary analysis report"""
        summary = self.results['analysis_summary']
        experiment_info = self.results['experiment_info']
        
        # Create text summary
        report_text = f"""# Experiment 4C: Real Judge Results Summary

## Experiment Information
- **Date**: {experiment_info.get('run_date', 'Unknown')[:19]}
- **Real Judge Mode**: {'Yes' if experiment_info.get('use_real_judges', False) else 'No'}
- **Tokens Analyzed**: {self.results['dataset_info'].get('unique_tokens_scored', 'Unknown')}
- **Total Score Records**: {self.results['dataset_info'].get('total_score_records', 'Unknown')}

## Key Findings

### Frequency Bias Results
"""
        
        # Add frequency bias results
        comparisons = self.results['detailed_results']['comparisons']
        if 'naive_average_frequency_reduction' in comparisons:
            report_text += f"- **Naive Average**: {comparisons['naive_average_frequency_reduction']:.1f}% reduction\n"
        if 'mlp_aggregator_frequency_reduction' in comparisons:
            report_text += f"- **MLP Aggregator**: {comparisons['mlp_aggregator_frequency_reduction']:.1f}% reduction\n"
        
        report_text += "\n### Framing Effects Results\n"
        if 'naive_average_framing_reduction' in comparisons:
            report_text += f"- **Naive Average**: {comparisons['naive_average_framing_reduction']:.1f}% reduction\n"
        if 'mlp_aggregator_framing_reduction' in comparisons:
            report_text += f"- **MLP Aggregator**: {comparisons['mlp_aggregator_framing_reduction']:.1f}% reduction\n"
        
        report_text += "\n## Conclusions\n"
        for conclusion in summary.get('conclusions', []):
            report_text += f"- {conclusion}\n"
        
        # Save report
        report_path = self.results_dir / 'ANALYSIS_SUMMARY.md'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Created summary report: {report_path}")
    
    def run_full_analysis(self, results_file: str = None):
        """Run complete analysis pipeline"""
        logger.info("Starting real judge results analysis...")
        
        # Load results
        self.load_results(results_file)
        
        # Generate all visualizations
        self.create_bias_reduction_plot()
        self.create_individual_judge_analysis()
        self.create_summary_report()
        
        logger.info("Analysis complete!")
        logger.info(f"Results saved in: {self.plots_dir}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Analyze Experiment 4C real judge results"
    )
    
    parser.add_argument(
        '--results-file',
        type=str,
        help='Path to specific FINAL_REPORT.json file (defaults to latest)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1
    
    analyzer = RealJudgeAnalyzer(results_dir)
    
    try:
        analyzer.run_full_analysis(args.results_file)
        print(f"\n✅ Analysis complete! Check results in: {results_dir}/plots/")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())