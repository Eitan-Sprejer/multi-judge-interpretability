#!/usr/bin/env python3
"""
Main script to run the Persona Poisoning experiment
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from simple_runner import run_contamination_experiment
import json
from datetime import datetime

def main():
    """Run the persona poisoning experiment with inverse strategy."""
    
    # Define contamination rates to test
    contamination_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    print("="*60)
    print("PERSONA POISONING EXPERIMENT")
    print("="*60)
    print(f"Testing {len(contamination_rates)} contamination rates")
    print("Strategy: Inverse (good→bad, bad→good)")
    print("="*60)
    
    # Run experiment
    results = run_contamination_experiment(
        '../../dataset/data_with_judge_scores.pkl',
        contamination_rates,
        'inverse'
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/experiment_{timestamp}.json'
    
    Path('results').mkdir(parents=True, exist_ok=True)
    
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

if __name__ == "__main__":
    main()