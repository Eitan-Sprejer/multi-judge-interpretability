#!/usr/bin/env python3
"""
Main script to run the Persona Poisoning experiment
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from simple_runner import run_contamination_experiment
import json
from datetime import datetime

def main():
    """Run the persona poisoning experiment with configurable parameters."""
    
    parser = argparse.ArgumentParser(description="Persona Poisoning Experiment")
    parser.add_argument('--data', type=str, 
                       help='Path to dataset with judge scores',
                       default=None)
    parser.add_argument('--strategy', type=str, 
                       choices=['inverse', 'random', 'extreme', 'safety_inverse'],
                       default='inverse',
                       help='Troll strategy to use')
    parser.add_argument('--rates', type=float, nargs='+',
                       default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                       help='Contamination rates to test')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/1b_persona_poisoning/results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Auto-detect dataset if not provided
    if args.data is None:
        # Try common dataset paths from new pipeline structure
        possible_paths = [
            'dataset/data_with_judge_scores.pkl',
            'full_experiment_runs/latest/data_with_judge_scores.pkl',
            'full_experiment_runs/data_with_judge_scores.pkl'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                args.data = path
                break
        
        if args.data is None:
            print("❌ Could not find dataset. Please specify --data path.")
            print("   Looked for:")
            for path in possible_paths:
                print(f"   - {path}")
            return 1
    
    if not Path(args.data).exists():
        print(f"❌ Dataset not found: {args.data}")
        return 1
    
    print("="*60)
    print("PERSONA POISONING EXPERIMENT")
    print("="*60)
    print(f"Dataset: {args.data}")
    print(f"Testing {len(args.rates)} contamination rates")
    print(f"Strategy: {args.strategy} ({'good→bad, bad→good' if args.strategy == 'inverse' else args.strategy})")
    print("="*60)
    
    # Run experiment
    results = run_contamination_experiment(
        args.data,
        args.rates,
        args.strategy
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_file = output_dir / f'experiment_{args.strategy}_{timestamp}.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    baseline_r2 = results[0.0]['r2']
    print(f"Baseline R² (0% contamination): {baseline_r2:.3f}")
    
    if 0.25 in results:
        r2_25 = results[0.25]['r2']
        drop_25 = (baseline_r2 - r2_25) / baseline_r2 * 100
        print(f"At 25% contamination: R² = {r2_25:.3f} (drop: {drop_25:.1f}%)")
    
    # Find breaking point
    for rate in sorted(results.keys()):
        if results[rate]['r2'] < 0.3:
            print(f"Breaking point (R² < 0.3): {rate*100:.0f}% contamination")
            break
    
    print("\nRun analyze_with_baselines.py to generate figures and comparisons")
    return 0

if __name__ == "__main__":
    exit(main())