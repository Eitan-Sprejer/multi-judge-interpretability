#!/usr/bin/env python3
"""
Complete Contamination Analysis Runner

Pragmatic pipeline that uses existing computed results to measure
degradation with contaminated judge mixtures.

Usage:
    python run_contamination_analysis.py --quick        # Fast analysis 
    python run_contamination_analysis.py --full         # Comprehensive analysis
    python run_contamination_analysis.py --visualize    # Generate plots
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from contamination_mixture_generator import ContaminatedJudgeMixtureGenerator
from metric_degradation_analyzer import MetricDegradationAnalyzer

logger = logging.getLogger(__name__)

class ContaminationAnalysisRunner:
    """Complete contamination analysis runner."""
    
    def __init__(self, output_dir: str = "contamination_analysis_results"):
        """Initialize runner with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"analysis_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized contamination analysis runner")
        logger.info(f"Results will be saved to: {self.experiment_dir}")
    
    def run_quick_analysis(self) -> dict:
        """Run quick contamination analysis with limited parameter space."""
        logger.info("🚀 Running QUICK contamination analysis...")
        
        # Generate limited contaminated mixtures
        generator = ContaminatedJudgeMixtureGenerator()
        
        mixtures = generator.generate_judge_mixtures(
            contamination_rates=[0.0, 0.2, 0.5],  # Limited rates
            contamination_strategies=['inversion', 'noise'],  # 2 strategies  
            judge_mixture_rates=[0.2, 0.5],  # 2 judge mixture rates
            random_seed=42
        )
        
        # Save mixtures
        mixtures_path = self.experiment_dir / "contaminated_mixtures.pkl"
        generator.save_mixtures(mixtures, str(mixtures_path))
        
        # Run degradation analysis
        analyzer = MetricDegradationAnalyzer()
        results = analyzer.analyze_mixture_degradation(mixtures, test_size=0.2, random_seed=42)
        
        # Save results
        results_path = self.experiment_dir / "degradation_analysis.pkl"
        analyzer.save_results(results, str(results_path))
        
        return {
            'mixtures_generated': len(mixtures),
            'mixtures_path': str(mixtures_path),
            'results_path': str(results_path),
            'experiment_dir': str(self.experiment_dir)
        }
    
    def run_full_analysis(self) -> dict:
        """Run comprehensive contamination analysis."""
        logger.info("🚀 Running FULL contamination analysis...")
        
        # Generate comprehensive contaminated mixtures
        generator = ContaminatedJudgeMixtureGenerator()
        
        mixtures = generator.generate_judge_mixtures(
            contamination_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            contamination_strategies=['inversion', 'noise', 'systematic_bias', 'random_uniform'],
            judge_mixture_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            random_seed=42
        )
        
        # Save mixtures
        mixtures_path = self.experiment_dir / "contaminated_mixtures.pkl"
        generator.save_mixtures(mixtures, str(mixtures_path))
        
        # Run degradation analysis
        analyzer = MetricDegradationAnalyzer()
        results = analyzer.analyze_mixture_degradation(mixtures, test_size=0.2, random_seed=42)
        
        # Save results
        results_path = self.experiment_dir / "degradation_analysis.pkl"
        analyzer.save_results(results, str(results_path))
        
        return {
            'mixtures_generated': len(mixtures),
            'mixtures_path': str(mixtures_path),
            'results_path': str(results_path),
            'experiment_dir': str(self.experiment_dir)
        }
    
    def generate_visualizations(self, results_path: str) -> dict:
        """Generate visualizations from analysis results."""
        logger.info("📊 Generating visualizations...")
        
        # Load results
        import pickle
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        generated_plots = []
        
        try:
            # 1. Method Degradation Heatmap
            fig, ax = self._create_degradation_heatmap(results)
            plot_path = plots_dir / "method_degradation_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            logger.info(f"Generated: {plot_path}")
            
            # 2. Contamination vs Performance Curves
            fig, ax = self._create_performance_curves(results)
            plot_path = plots_dir / "contamination_performance_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            logger.info(f"Generated: {plot_path}")
            
            # 3. Strategy Comparison
            fig, ax = self._create_strategy_comparison(results)
            plot_path = plots_dir / "strategy_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            logger.info(f"Generated: {plot_path}")
            
            # 4. Robustness Summary Dashboard
            fig = self._create_robustness_dashboard(results)
            plot_path = plots_dir / "robustness_dashboard.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
            logger.info(f"Generated: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate some visualizations: {e}")
        
        return {
            'plots_generated': len(generated_plots),
            'plots_dir': str(plots_dir),
            'generated_plots': generated_plots
        }
    
    def _create_degradation_heatmap(self, results: dict):
        """Create heatmap showing performance degradation by method and contamination."""
        # Extract degradation data
        degradation_data = []
        
        for mixture_key, mixture_result in results['mixture_results'].items():
            if 'method_results' not in mixture_result:
                continue
                
            mixture_info = mixture_result['mixture_info']
            
            for method_name, method_result in mixture_result['method_results'].items():
                if 'metrics' in method_result:
                    degradation_data.append({
                        'method': method_name,
                        'strategy': mixture_info['strategy'],
                        'contamination_rate': mixture_info['contamination_rate'],
                        'judge_mixture_rate': mixture_info['judge_mixture_rate'],
                        'r2': method_result['metrics']['r2']
                    })
        
        if not degradation_data:
            raise ValueError("No degradation data available for visualization")
        
        df = pd.DataFrame(degradation_data)
        
        # Create pivot table for heatmap
        pivot_data = df.groupby(['method', 'contamination_rate'])['r2'].mean().reset_index()
        heatmap_data = pivot_data.pivot(index='method', columns='contamination_rate', values='r2')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
        ax.set_title('Performance Degradation by Method and Contamination Rate', fontsize=14)
        ax.set_xlabel('Contamination Rate', fontsize=12)
        ax.set_ylabel('Baseline Method', fontsize=12)
        
        return fig, ax
    
    def _create_performance_curves(self, results: dict):
        """Create curves showing performance vs contamination rate."""
        degradation_data = []
        
        for mixture_key, mixture_result in results['mixture_results'].items():
            if 'method_results' not in mixture_result:
                continue
                
            mixture_info = mixture_result['mixture_info']
            
            for method_name, method_result in mixture_result['method_results'].items():
                if 'metrics' in method_result:
                    degradation_data.append({
                        'method': method_name,
                        'contamination_rate': mixture_info['contamination_rate'],
                        'r2': method_result['metrics']['r2']
                    })
        
        df = pd.DataFrame(degradation_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot curves for each method
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            method_summary = method_data.groupby('contamination_rate')['r2'].agg(['mean', 'std']).reset_index()
            
            ax.plot(method_summary['contamination_rate'], method_summary['mean'], 
                   marker='o', label=method, linewidth=2)
            
            # Add error bars if we have multiple data points
            if 'std' in method_summary.columns and not method_summary['std'].isna().all():
                ax.fill_between(method_summary['contamination_rate'], 
                               method_summary['mean'] - method_summary['std'],
                               method_summary['mean'] + method_summary['std'], 
                               alpha=0.2)
        
        ax.set_xlabel('Contamination Rate', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Performance vs Contamination Rate by Method', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def _create_strategy_comparison(self, results: dict):
        """Compare different contamination strategies."""
        strategy_data = []
        
        for mixture_key, mixture_result in results['mixture_results'].items():
            if 'method_results' not in mixture_result:
                continue
                
            mixture_info = mixture_result['mixture_info']
            
            # Average across methods for strategy comparison
            method_r2s = []
            for method_name, method_result in mixture_result['method_results'].items():
                if 'metrics' in method_result:
                    method_r2s.append(method_result['metrics']['r2'])
            
            if method_r2s:
                strategy_data.append({
                    'strategy': mixture_info['strategy'],
                    'contamination_rate': mixture_info['contamination_rate'],
                    'avg_r2': np.mean(method_r2s)
                })
        
        df = pd.DataFrame(strategy_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot by strategy
        sns.boxplot(data=df, x='strategy', y='avg_r2', ax=ax)
        ax.set_title('Performance Distribution by Contamination Strategy', fontsize=14)
        ax.set_xlabel('Contamination Strategy', fontsize=12)
        ax.set_ylabel('Average R² Score', fontsize=12)
        
        return fig, ax
    
    def _create_robustness_dashboard(self, results: dict):
        """Create comprehensive robustness dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Contamination Robustness Analysis Dashboard', fontsize=16)
        
        # Collect data
        all_data = []
        for mixture_key, mixture_result in results['mixture_results'].items():
            if 'method_results' not in mixture_result:
                continue
                
            mixture_info = mixture_result['mixture_info']
            
            for method_name, method_result in mixture_result['method_results'].items():
                if 'metrics' in method_result:
                    all_data.append({
                        'method': method_name,
                        'strategy': mixture_info['strategy'],
                        'contamination_rate': mixture_info['contamination_rate'],
                        'judge_mixture_rate': mixture_info['judge_mixture_rate'],
                        'r2': method_result['metrics']['r2'],
                        'mae': method_result['metrics']['mae']
                    })
        
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            axes[0,0].text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        # 1. R² vs Judge Mixture Rate
        ax1 = axes[0,0]
        for method in df['method'].unique()[:5]:  # Top 5 methods
            method_data = df[df['method'] == method]
            method_summary = method_data.groupby('judge_mixture_rate')['r2'].mean().reset_index()
            ax1.plot(method_summary['judge_mixture_rate'], method_summary['r2'], 
                    marker='o', label=method[:15])
        ax1.set_xlabel('Fraction of Judges Contaminated')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Performance vs Judge Contamination Fraction')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Strategy effectiveness
        ax2 = axes[0,1] 
        strategy_summary = df.groupby('strategy')['r2'].agg(['mean', 'std']).reset_index()
        bars = ax2.bar(strategy_summary['strategy'], strategy_summary['mean'], 
                      yerr=strategy_summary['std'], capsize=5)
        ax2.set_xlabel('Contamination Strategy')
        ax2.set_ylabel('Mean R² Score')
        ax2.set_title('Average Performance by Strategy')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Contamination rate impact
        ax3 = axes[1,0]
        contam_summary = df.groupby('contamination_rate')['r2'].agg(['mean', 'std']).reset_index()
        ax3.plot(contam_summary['contamination_rate'], contam_summary['mean'], 
                marker='o', linewidth=2, markersize=8)
        ax3.fill_between(contam_summary['contamination_rate'], 
                        contam_summary['mean'] - contam_summary['std'],
                        contam_summary['mean'] + contam_summary['std'], 
                        alpha=0.2)
        ax3.set_xlabel('Contamination Severity')
        ax3.set_ylabel('Mean R² Score')
        ax3.set_title('Impact of Contamination Severity')
        ax3.grid(True, alpha=0.3)
        
        # 4. Method robustness ranking
        ax4 = axes[1,1]
        method_robustness = df.groupby('method')['r2'].agg(['mean', 'std']).reset_index()
        method_robustness['robustness'] = method_robustness['mean'] / (method_robustness['std'] + 0.001)  # Higher is more robust
        method_robustness = method_robustness.sort_values('robustness', ascending=True).tail(10)
        
        bars = ax4.barh(method_robustness['method'], method_robustness['robustness'])
        ax4.set_xlabel('Robustness Score (Mean/Std)')
        ax4.set_title('Method Robustness Ranking')
        ax4.tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, analysis_result: dict, visualization_result: dict = None) -> str:
        """Generate comprehensive analysis report."""
        report_path = self.experiment_dir / "CONTAMINATION_ANALYSIS_REPORT.md"
        
        with open(report_path, 'w') as f:
            f.write("# Contamination Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Experiment ID**: {self.timestamp}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Contaminated Mixtures Generated**: {analysis_result['mixtures_generated']}\n")
            f.write(f"- **Analysis Type**: {'Quick' if analysis_result['mixtures_generated'] < 50 else 'Comprehensive'}\n")
            
            if visualization_result:
                f.write(f"- **Visualizations Generated**: {visualization_result['plots_generated']}\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("This analysis measures performance degradation when using contaminated judge mixtures ")
            f.write("compared to clean baseline performance from existing experiments.\n\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write("1. **Baseline Performance**: Uses existing computed results from clean judge evaluations\n")
            f.write("2. **Contamination Strategies**: Tests multiple contamination approaches (inversion, noise, bias)\n")
            f.write("3. **Judge Mixtures**: Varies fraction of contaminated judges (10% to 100%)\n")
            f.write("4. **Performance Measurement**: Applies same baseline methods to contaminated data\n")
            f.write("5. **Degradation Analysis**: Compares contaminated vs clean performance metrics\n\n")
            
            # Files Generated
            f.write("## Generated Files\n\n")
            f.write(f"- **Raw Results**: `{Path(analysis_result['results_path']).name}`\n")
            f.write(f"- **Contaminated Mixtures**: `{Path(analysis_result['mixtures_path']).name}`\n")
            
            if visualization_result:
                f.write("- **Visualizations**:\n")
                for plot_path in visualization_result['generated_plots']:
                    f.write(f"  - `{Path(plot_path).name}`\n")
            
            f.write("\n## Usage\n\n")
            f.write("To extend this analysis:\n\n")
            f.write("1. **Load Results**: Use pickle to load the results file\n")
            f.write("2. **Custom Analysis**: Extract specific degradation patterns\n") 
            f.write("3. **Additional Strategies**: Modify contamination_mixture_generator.py\n")
            f.write("4. **New Baselines**: Add methods to metric_degradation_analyzer.py\n")
            
            # Technical Details
            f.write("\n## Technical Details\n\n")
            f.write("- **Random Seed**: 42 (for reproducibility)\n")
            f.write("- **Train/Test Split**: 80/20\n")
            f.write("- **Evaluation Metrics**: R², MSE, MAE\n")
            f.write("- **Baseline Methods**: Linear scaling, best judge, UltraFeedback 4-judge, StandardScaler+LR\n")
        
        logger.info(f"Generated analysis report: {report_path}")
        return str(report_path)


def main():
    """Main entry point for contamination analysis."""
    parser = argparse.ArgumentParser(
        description="Pragmatic contamination analysis using existing computed results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # Fast analysis with limited parameter space
  %(prog)s --full                     # Comprehensive analysis (recommended)
  %(prog)s --full --visualize         # Full analysis with visualizations
  %(prog)s --output my_analysis       # Custom output directory
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Run quick analysis with limited parameters')
    parser.add_argument('--full', action='store_true',
                        help='Run comprehensive analysis')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--output', default='contamination_analysis_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Default to full analysis if neither specified
    if not args.quick and not args.full:
        args.full = True
    
    try:
        # Initialize runner
        runner = ContaminationAnalysisRunner(output_dir=args.output)
        
        # Run analysis
        if args.quick:
            analysis_result = runner.run_quick_analysis()
            logger.info("✅ Quick contamination analysis completed")
        else:
            analysis_result = runner.run_full_analysis()
            logger.info("✅ Full contamination analysis completed")
        
        # Generate visualizations if requested
        visualization_result = None
        if args.visualize:
            visualization_result = runner.generate_visualizations(analysis_result['results_path'])
            logger.info("✅ Visualizations generated")
        
        # Generate report
        report_path = runner.generate_report(analysis_result, visualization_result)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 CONTAMINATION ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📊 Analysis Results:")
        print(f"   • Contaminated mixtures generated: {analysis_result['mixtures_generated']}")
        print(f"   • Analysis type: {'Quick' if analysis_result['mixtures_generated'] < 50 else 'Comprehensive'}")
        
        if visualization_result:
            print(f"   • Visualizations generated: {visualization_result['plots_generated']}")
        
        print(f"\n📁 Results Directory: {analysis_result['experiment_dir']}")
        print(f"📋 Analysis Report: {report_path}")
        print(f"📊 Raw Results: {analysis_result['results_path']}")
        
        if visualization_result:
            print(f"📈 Plots Directory: {visualization_result['plots_dir']}")
        
        print("\n🚀 Next Steps:")
        print("   • Review the analysis report for key findings")
        print("   • Examine visualizations (if generated) for degradation patterns")
        print("   • Load raw results in Python for custom analysis")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())