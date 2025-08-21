#!/usr/bin/env python3
"""
Main Script for Experiment 2b: Aggregator Validation with Less Varied Data

This script runs the complete experiment to test whether the aggregator's 
baseline R² score of ~0.58 is due to varied simulated human preference data.

Runs 11 experiments total:
1. Mixed personas (baseline replication)
2. UltraFeedback overall_score (GPT-4 generated)
3-10. Individual personas (8 experiments)
11. Mean of all personas (bonus experiment)

Usage:
    python run_experiment.py [--quick-test] [--output-dir custom_dir]
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiment_runner import ExperimentRunner
from visualizations import create_all_visualizations
from data_preparation import load_experiment_data, prepare_all_targets


def print_experiment_header():
    """Print experiment header with information."""
    print("=" * 80)
    print("🧪 EXPERIMENT 2B: AGGREGATOR VALIDATION WITH LESS VARIED DATA")
    print("=" * 80)
    print()
    print("HYPOTHESIS:")
    print("   The baseline R² score of ~0.58 is limited by ground truth variance,")
    print("   not by fundamental aggregation approach limitations.")
    print()
    print("EXPECTATIONS:")
    print("   • UltraFeedback (GPT-4): R² > 0.70 (low variance)")
    print("   • Individual personas: Mean R² > 0.65 (consistent perspectives)")
    print("   • Variance order: Individual < UltraFeedback < Persona Mean < Mixed")
    print()
    print("EXPERIMENTS:")
    print("   1. Mixed personas (baseline replication)")
    print("   2. UltraFeedback overall_score") 
    print("   3-16. Individual personas (14 experiments)")
    print("   17. Mean of all personas (bonus)")
    print()


def validate_hypothesis(results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate the experimental hypotheses against results.
    
    Args:
        results: Dictionary with all experiment results
        
    Returns:
        Dictionary with hypothesis validation results
    """
    print("\n🔍 HYPOTHESIS VALIDATION")
    print("-" * 50)
    
    hypothesis_results = results['summary']['hypothesis_validation']
    
    # Print each hypothesis result
    for key, result in hypothesis_results.items():
        hypothesis = result['hypothesis']
        passed = result['passed']
        value = result.get('result', 'N/A')
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {hypothesis}")
        if isinstance(value, (int, float)):
            print(f"         Result: {value:.4f}")
        print()
    
    # Overall assessment
    total_passed = sum(result['passed'] for result in hypothesis_results.values())
    total_tests = len(hypothesis_results)
    
    print(f"OVERALL: {total_passed}/{total_tests} hypotheses validated")
    
    if total_passed == total_tests:
        print("🎉 ALL HYPOTHESES CONFIRMED! Variance is the limiting factor.")
    elif total_passed >= total_tests * 0.67:
        print("✅ STRONG EVIDENCE that variance limits performance.")
    else:
        print("⚠️  MIXED RESULTS - variance may not be the only factor.")
    
    return {key: result['passed'] for key, result in hypothesis_results.items()}


def print_key_findings(results: Dict[str, Any]):
    """Print key experimental findings."""
    print("\n📊 KEY FINDINGS")
    print("-" * 50)
    
    main_comparison = results['summary']['main_comparison']
    
    # R² comparison
    print("R² SCORE COMPARISON:")
    for exp_name, exp_data in main_comparison.items():
        name = exp_name.replace('_', ' ').title()
        gam_r2 = exp_data['gam_r2']
        mlp_r2 = exp_data['mlp_r2']
        variance = exp_data['variance']
        best_r2 = max(gam_r2, mlp_r2)
        best_model = "GAM" if gam_r2 > mlp_r2 else "MLP"
        
        print(f"   {name:25}: {best_r2:.4f} ({best_model}) | Variance: {variance:.3f}")
    
    # Performance improvement
    mixed_best = max(main_comparison['mixed_personas']['gam_r2'], 
                    main_comparison['mixed_personas']['mlp_r2'])
    uf_best = max(main_comparison['ultrafeedback']['gam_r2'], 
                 main_comparison['ultrafeedback']['mlp_r2'])
    ind_best = max(main_comparison['individual_personas_mean']['gam_r2'], 
                  main_comparison['individual_personas_mean']['mlp_r2'])
    
    print(f"\nPERFORMANCE IMPROVEMENTS:")
    print(f"   UltraFeedback vs Mixed:     +{(uf_best - mixed_best):.4f} R² ({(uf_best/mixed_best - 1)*100:+.1f}%)")
    print(f"   Individual vs Mixed:        +{(ind_best - mixed_best):.4f} R² ({(ind_best/mixed_best - 1)*100:+.1f}%)")
    
    # Variance analysis
    variance_analysis = results['summary']['variance_analysis']
    correlation = variance_analysis['variance_vs_r2_correlation']
    
    print(f"\nVARIANCE ANALYSIS:")
    print(f"   Variance-R² Correlation:    {correlation:.3f}")
    print(f"   Relationship:               {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}")
    
    # Best performing personas
    individual_results = results['individual_personas']['individual_results']
    best_persona = max(individual_results.items(), key=lambda x: x[1]['summary']['best_r2'])
    worst_persona = min(individual_results.items(), key=lambda x: x[1]['summary']['best_r2'])
    
    print(f"\nINDIVIDUAL PERSONA INSIGHTS:")
    print(f"   Best Performer:             {best_persona[0]} (R² = {best_persona[1]['summary']['best_r2']:.4f})")
    print(f"   Worst Performer:            {worst_persona[0]} (R² = {worst_persona[1]['summary']['best_r2']:.4f})")
    print(f"   Performance Range:          {worst_persona[1]['summary']['best_r2']:.4f} - {best_persona[1]['summary']['best_r2']:.4f}")


def run_complete_experiment(
    data_path: str,
    output_dir: str = None,
    quick_test: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Run the complete experiment with all components.
    
    Args:
        data_path: Path to the experiment data
        output_dir: Output directory (auto-generated if None)
        quick_test: If True, run with smaller dataset for testing
        test_size: Test set fraction
        random_seed: Random seed for reproducibility
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with all results
    """
    start_time = time.time()
    
    # Print header
    print_experiment_header()
    
    # Initialize experiment runner
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{timestamp}"
    
    runner = ExperimentRunner(
        data_path=data_path,
        output_dir=output_dir,
        test_size=test_size,
        random_seed=random_seed,
        normalize=normalize
    )
    
    print(f"⚙️  CONFIGURATION:")
    print(f"   Data source: {data_path}")
    print(f"   Output directory: {runner.output_dir}")
    print(f"   Test size: {test_size}")
    print(f"   Random seed: {random_seed}")
    print(f"   Normalize features: {normalize}")
    print(f"   Quick test mode: {quick_test}")
    
    if quick_test:
        print("\n⚠️  QUICK TEST MODE: Using subset of data for testing")
    
    # Run all experiments
    print(f"\n🚀 STARTING EXPERIMENTS...")
    results = runner.run_all_experiments()
    
    # Load prepared targets for visualizations
    targets_path = runner.output_dir / "prepared_targets.pkl"
    with open(targets_path, 'rb') as f:
        targets = pickle.load(f)
    
    # Create visualizations
    print("\n🎨 CREATING VISUALIZATIONS...")
    plot_paths = create_all_visualizations(results, targets, str(runner.output_dir / "plots"))
    
    # Add plot paths to results
    results['plot_paths'] = plot_paths
    
    # Save final results with plots
    final_results_path = runner.output_dir / "final_results_with_plots.pkl"
    with open(final_results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    results['execution_info'] = {
        'execution_time_seconds': execution_time,
        'execution_time_formatted': f"{execution_time // 60:.0f}m {execution_time % 60:.0f}s"
    }
    
    # Print results
    print_key_findings(results)
    hypothesis_validation = validate_hypothesis(results)
    
    print(f"\n⏱️  EXECUTION TIME: {execution_time // 60:.0f}m {execution_time % 60:.0f}s")
    print(f"📁 RESULTS SAVED TO: {runner.output_dir}")
    print(f"🎨 PLOTS SAVED TO: {runner.output_dir / 'plots'}")
    
    print("\n" + "=" * 80)
    print("🎉 EXPERIMENT 2B COMPLETE!")
    print("=" * 80)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Experiment 2b: Aggregator Validation")
    
    parser.add_argument(
        '--data-path',
        default="/Users/eitu/Documents/Eitu/AI Safety/AIS_hackathons/model_routing/multi-judge-interpretability/results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023/data/data_with_judge_scores_and_ultrafeedback.pkl",
        help="Path to experiment data with judge scores and ultrafeedback"
    )
    
    parser.add_argument(
        '--output-dir',
        help="Output directory for results (default: auto-generated)"
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help="Run quick test with subset of data"
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help="Disable feature normalization"
    )
    
    args = parser.parse_args()
    
    # Run complete experiment
    try:
        results = run_complete_experiment(
            data_path=args.data_path,
            output_dir=args.output_dir,
            quick_test=args.quick_test,
            test_size=args.test_size,
            random_seed=args.random_seed,
            normalize=not args.no_normalize
        )
        
        print("\n✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()