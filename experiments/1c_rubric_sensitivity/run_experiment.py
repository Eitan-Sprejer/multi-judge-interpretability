#!/usr/bin/env python3
"""
Experiment 1C: Rubric Sensitivity

Tests robustness of aggregation models to semantically equivalent but differently 
phrased judge rubrics. Robust aggregators should show <5% variance across 
equivalent rubrics.

Usage:
    python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl
    python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl --quick
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.experiment_runner import RubricSensitivityExperiment


def main():
    parser = argparse.ArgumentParser(description="Run Rubric Sensitivity Experiment")
    parser.add_argument("--data", required=True, help="Path to dataset with judge scores")
    parser.add_argument("--quick", action="store_true", help="Run quick test with 100 samples")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--model-path", help="Path to trained aggregation model")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize experiment
    experiment = RubricSensitivityExperiment(
        data_path=args.data,
        output_dir=output_dir,
        model_path=args.model_path,
        quick_mode=args.quick
    )
    
    # Run experiment
    print("Starting Rubric Sensitivity Experiment...")
    results = experiment.run()
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    metrics = results['key_metrics']
    success = results['success_criteria']
    
    print(f"Learned Aggregator Variance: {metrics['learned_variance']:.4f}")
    print(f"Baseline Mean Variance: {metrics['baseline_variance']:.4f}")
    print(f"Improvement Factor: {metrics['improvement_factor']:.2f}x")
    print(f"Correlation (Learned): {metrics['learned_correlation']:.4f}")
    print(f"Correlation (Baseline): {metrics['baseline_correlation']:.4f}")
    
    if success['variance_below_5_percent']:
        print("✅ SUCCESS: Learned aggregator shows <5% variance")
    else:
        print("❌ ISSUE: Learned aggregator variance exceeds 5%")
        
    if success['correlation_above_95_percent']:
        print("✅ SUCCESS: High correlation maintained (>0.95)")
    else:
        print("❌ ISSUE: Correlation below target (0.95)")
    
    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    main()
