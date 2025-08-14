#!/usr/bin/env python3
"""
Run Clean Analysis for Experiment 4C

Generates clean, organized bias reduction results using existing data.
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project paths
experiment_src = Path(__file__).parent / "src"
sys.path.append(str(experiment_src))

# Import our modules
from bias_analysis import BiasAnalyzer
from results_reporter import CleanResultsReporter
from supplementary_analysis import SupplementaryAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_existing_data(token_dataset_path: str, scores_dataset_path: str):
    """Load existing datasets"""
    logger.info("Loading existing datasets...")
    
    # Load token dataset
    logger.info(f"Loading token dataset from: {token_dataset_path}")
    token_dataset = pd.read_pickle(token_dataset_path)
    logger.info(f"Token dataset loaded: {len(token_dataset)} tokens")
    
    # Load scores dataset
    logger.info(f"Loading scores dataset from: {scores_dataset_path}")
    scores_dataset = pd.read_pickle(scores_dataset_path)
    logger.info(f"Scores dataset loaded: {len(scores_dataset)} score records")
    
    return token_dataset, scores_dataset


def identify_model_columns(scores_df):
    """Identify which columns contain model scores"""
    model_columns = []
    
    # Individual judges
    judge_cols = [col for col in scores_df.columns if col.startswith('judge_')]
    model_columns.extend(judge_cols)
    
    # Aggregators
    if 'naive_average' in scores_df.columns:
        model_columns.append('naive_average')
    if 'mlp_aggregator' in scores_df.columns:
        model_columns.append('mlp_aggregator')
    if 'gam_aggregator' in scores_df.columns:
        model_columns.append('gam_aggregator')
    
    logger.info(f"Identified model columns: {model_columns}")
    return model_columns


def run_clean_analysis(token_dataset_path: str, scores_dataset_path: str, 
                      normalize_scores: bool = False, results_dir: str = None):
    """
    Run complete clean analysis pipeline
    
    Args:
        token_dataset_path: Path to token dataset pickle
        scores_dataset_path: Path to scores dataset pickle
        normalize_scores: Whether to normalize scores
        results_dir: Output directory for results
    """
    
    # Set up results directory
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    else:
        results_dir = Path(results_dir)
    
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting clean analysis - Timestamp: {timestamp}")
    
    # Load data
    token_dataset, scores_dataset = load_existing_data(token_dataset_path, scores_dataset_path)
    
    # Initialize analyzer and reporter
    bias_analyzer = BiasAnalyzer(normalize_scores=normalize_scores)
    reporter = CleanResultsReporter(timestamp=timestamp)
    
    # Identify model columns
    model_columns = identify_model_columns(scores_dataset)
    
    if not model_columns:
        logger.error("No model columns found in scores dataset!")
        return None
    
    # Run bias analysis
    logger.info("Running framing effects analysis...")
    framing_results = bias_analyzer.analyze_framing_effects(scores_dataset, model_columns)
    
    logger.info("Running frequency bias analysis...")
    frequency_results = bias_analyzer.analyze_frequency_bias(scores_dataset, model_columns)
    
    logger.info("Running significance tests...")
    significance_results = bias_analyzer.run_significance_tests(scores_dataset, model_columns)
    
    # Extract clean results
    logger.info("Extracting clean results...")
    clean_results = bias_analyzer.extract_clean_results()
    individual_results = clean_results['individual_results']
    naive_results = clean_results['naive_results']
    mlp_results = clean_results['mlp_results']
    
    logger.info(f"Individual judges: {len(individual_results)}")
    logger.info(f"Naive results available: {'Yes' if naive_results else 'No'}")
    logger.info(f"MLP results available: {'Yes' if mlp_results else 'No'}")
    
    # Calculate bias reduction
    logger.info("Calculating bias reduction...")
    bias_reduction_results = reporter.calculate_bias_reduction(individual_results, naive_results, mlp_results)
    
    # Generate complete report
    logger.info("Generating complete report...")
    experiment_config = {
        'normalize_scores': normalize_scores,
        'use_real_judges': True,  # Inferred from existing data
        'quick_mode': False,  # Inferred from data size
        'token_dataset_path': token_dataset_path,
        'scores_dataset_path': scores_dataset_path
    }
    
    analysis_results = {
        'framing_results': framing_results,
        'frequency_results': frequency_results,
        'significance_results': significance_results
    }
    
    complete_report = reporter.generate_complete_report(
        token_dataset, scores_dataset, analysis_results, 
        bias_reduction_results, experiment_config
    )
    
    # Generate clean summary
    logger.info("Generating clean summary...")
    clean_summary = reporter.generate_clean_summary(complete_report)
    
    # Create visualizations
    logger.info("Creating bias reduction plots...")
    plots_path = results_dir / f"{timestamp}_bias_reduction_plots.png"
    reporter.create_bias_reduction_plots(bias_reduction_results, str(plots_path))
    
    # Save all files
    logger.info("Saving results...")
    
    complete_report_path = results_dir / f"{timestamp}_COMPLETE_REPORT.json"
    reporter.save_complete_report(complete_report, str(complete_report_path))
    
    clean_summary_path = results_dir / f"{timestamp}_CLEAN_SUMMARY.md"
    reporter.save_clean_summary(clean_summary, str(clean_summary_path))
    
    # Generate supplementary analysis
    logger.info("Generating supplementary analysis...")
    supplementary_analyzer = SupplementaryAnalyzer(timestamp)
    supplementary_dir = results_dir / "supplementary"
    supplementary_dir.mkdir(exist_ok=True)
    
    # Create individual judge analysis
    individual_files = supplementary_analyzer.create_individual_judge_analysis(
        bias_reduction_results, str(supplementary_dir)
    )
    
    # Create method comparison details
    comparison_files = supplementary_analyzer.create_method_comparison_details(
        bias_reduction_results, str(supplementary_dir)
    )
    
    # Generate README for supplementary folder
    readme_path = supplementary_analyzer.generate_supplementary_readme(
        individual_files + comparison_files, str(supplementary_dir)
    )
    
    all_supplementary_files = individual_files + comparison_files + [readme_path]
    
    # Print summary
    print("\n" + "="*60)
    print("CLEAN BIAS ANALYSIS COMPLETED!")
    print("="*60)
    
    print(f"\nFiles generated:")
    print(f"📄 Complete Report: {complete_report_path}")
    print(f"📋 Clean Summary:   {clean_summary_path}")
    print(f"📊 Bias Plots:      {plots_path}")
    print(f"📁 Supplementary:   {supplementary_dir}/ ({len(all_supplementary_files)} files)")
    
    print(f"\nBias Reduction Results:")
    
    naive = bias_reduction_results['naive_average']
    mlp = bias_reduction_results['mlp_aggregator']
    
    print(f"\n📊 Naive Average:")
    if not np.isnan(naive['framing_reduction_percent']):
        print(f"   🎯 Framing Bias:    {naive['framing_reduction_percent']:.1f}% reduction")
    else:
        print(f"   🎯 Framing Bias:    Could not calculate")
        
    if not np.isnan(naive['frequency_reduction_percent']):
        print(f"   🔁 Frequency Bias:  {naive['frequency_reduction_percent']:.1f}% reduction")
    else:
        print(f"   🔁 Frequency Bias:  Could not calculate")
    
    print(f"\n🧠 MLP Aggregator:")
    if not np.isnan(mlp['framing_reduction_percent']):
        print(f"   🎯 Framing Bias:    {mlp['framing_reduction_percent']:.1f}% reduction")
    else:
        print(f"   🎯 Framing Bias:    Could not calculate")
        
    if not np.isnan(mlp['frequency_reduction_percent']):
        print(f"   🔁 Frequency Bias:  {mlp['frequency_reduction_percent']:.1f}% reduction")
    else:
        print(f"   🔁 Frequency Bias:  Could not calculate")
    
    # Compare methods
    naive_better_framing = (not np.isnan(naive['framing_reduction_percent']) and 
                           not np.isnan(mlp['framing_reduction_percent']) and 
                           naive['framing_reduction_percent'] > mlp['framing_reduction_percent'])
    
    mlp_better_framing = (not np.isnan(naive['framing_reduction_percent']) and 
                         not np.isnan(mlp['framing_reduction_percent']) and 
                         mlp['framing_reduction_percent'] > naive['framing_reduction_percent'])
    
    naive_better_frequency = (not np.isnan(naive['frequency_reduction_percent']) and 
                             not np.isnan(mlp['frequency_reduction_percent']) and 
                             naive['frequency_reduction_percent'] > mlp['frequency_reduction_percent'])
    
    mlp_better_frequency = (not np.isnan(naive['frequency_reduction_percent']) and 
                           not np.isnan(mlp['frequency_reduction_percent']) and 
                           mlp['frequency_reduction_percent'] > naive['frequency_reduction_percent'])
    
    print(f"\n🏆 Performance Comparison:")
    if mlp_better_framing:
        print("   📈 MLP outperforms Naive on framing bias")
    elif naive_better_framing:
        print("   📈 Naive outperforms MLP on framing bias")
    else:
        print("   📈 Framing bias: Similar performance")
        
    if mlp_better_frequency:
        print("   📈 MLP outperforms Naive on frequency bias")
    elif naive_better_frequency:
        print("   📈 Naive outperforms MLP on frequency bias")
    else:
        print("   📈 Frequency bias: Similar performance")
    
    print("="*60)
    
    return {
        'complete_report_path': complete_report_path,
        'clean_summary_path': clean_summary_path,
        'plots_path': plots_path,
        'supplementary_dir': supplementary_dir,
        'supplementary_files': all_supplementary_files,
        'bias_reduction_results': bias_reduction_results
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run clean bias analysis on existing Experiment 4C data"
    )
    
    parser.add_argument(
        '--token-dataset',
        required=True,
        help='Path to token dataset pickle file'
    )
    
    parser.add_argument(
        '--scores-dataset', 
        required=True,
        help='Path to scores dataset pickle file'
    )
    
    parser.add_argument(
        '--normalize-scores',
        action='store_true',
        help='Normalize model scores to [0,1] range'
    )
    
    parser.add_argument(
        '--results-dir',
        help='Output directory for results (default: ./results/)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.token_dataset).exists():
        print(f"Error: Token dataset file not found: {args.token_dataset}")
        return 1
    
    if not Path(args.scores_dataset).exists():
        print(f"Error: Scores dataset file not found: {args.scores_dataset}")
        return 1
    
    try:
        result = run_clean_analysis(
            args.token_dataset,
            args.scores_dataset,
            args.normalize_scores,
            args.results_dir
        )
        
        if result is None:
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Clean analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())