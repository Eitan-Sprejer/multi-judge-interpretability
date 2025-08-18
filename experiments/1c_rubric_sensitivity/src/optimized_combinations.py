"""
Optimized Judge Combinations for Rubric Sensitivity Testing

This module generates a smaller, more meaningful set of judge combinations
that test specific robustness properties while requiring fewer API calls.
"""

from typing import List, Dict
import random
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS


def generate_optimized_combinations(seed: int = 42) -> List[Dict]:
    """
    Generate optimized judge combinations for rubric sensitivity testing.
    
    Returns 5-7 key combinations instead of 44, focusing on:
    1. Baseline (all original)
    2. Systematic variations (all same variant)
    3. Mixed variations (realistic contamination scenarios)
    4. Single contamination (robustness to individual bad judges)
    
    Args:
        seed: Random seed for reproducible combinations
        
    Returns:
        List of combination dictionaries with name and judge mappings
    """
    random.seed(seed)
    judges = list(JUDGE_RUBRICS.keys())
    combinations = []
    
    # 1. BASELINE: All original judges
    combinations.append({
        'name': 'baseline_original',
        'combination': {judge: 'original' for judge in judges},
        'description': 'Baseline with all original judge rubrics'
    })
    
    # 2. SYSTEMATIC STRICT: All judges use strict variant
    combinations.append({
        'name': 'systematic_strict',
        'combination': {judge: 'strict' for judge in judges},
        'description': 'All judges use stricter criteria'
    })
    
    # 3. SYSTEMATIC LENIENT: All judges use lenient variant
    combinations.append({
        'name': 'systematic_lenient',
        'combination': {judge: 'lenient' for judge in judges},
        'description': 'All judges use more lenient criteria'
    })
    
    # 4. MIXED BALANCED: Half strict, half lenient (realistic scenario)
    half_point = len(judges) // 2
    shuffled_judges = judges.copy()
    random.shuffle(shuffled_judges)
    mixed_combination = {}
    for i, judge in enumerate(shuffled_judges):
        mixed_combination[judge] = 'strict' if i < half_point else 'lenient'
    
    combinations.append({
        'name': 'mixed_balanced',
        'combination': mixed_combination,
        'description': 'Half judges strict, half lenient (mixed rubrics)'
    })
    
    # 5. SINGLE CONTAMINATION: One judge variant, rest original
    # This tests robustness to a single "bad" judge
    contaminated_judge = random.choice(judges)
    combinations.append({
        'name': f'single_contaminated_{contaminated_judge.split("-")[0]}',
        'combination': {
            judge: 'strict' if judge == contaminated_judge else 'original'
            for judge in judges
        },
        'description': f'Single contaminated judge ({contaminated_judge}) with strict rubric'
    })
    
    # Optional additional combinations for more thorough testing:
    
    # 6. INTERVAL SCALING: Test bottom-heavy vs top-heavy
    combinations.append({
        'name': 'interval_bottom_heavy',
        'combination': {judge: 'bottom_heavy' for judge in judges},
        'description': 'All judges use bottom-heavy score intervals'
    })
    
    # 7. RANDOM MIXED: Random assignment (stress test)
    variants = ['original', 'strict', 'lenient', 'bottom_heavy', 'top_heavy']
    combinations.append({
        'name': 'random_mixed',
        'combination': {
            judge: random.choice(variants) for judge in judges
        },
        'description': 'Random variant assignment (stress test)'
    })
    
    return combinations


def generate_minimal_combinations() -> List[Dict]:
    """
    Generate minimal set of combinations for quick testing.
    
    Only 3 combinations:
    1. Baseline (original)
    2. All strict
    3. All lenient
    
    Returns:
        List of 3 essential combinations
    """
    judges = list(JUDGE_RUBRICS.keys())
    
    return [
        {
            'name': 'baseline_original',
            'combination': {judge: 'original' for judge in judges},
            'description': 'Baseline with all original judge rubrics'
        },
        {
            'name': 'systematic_strict',
            'combination': {judge: 'strict' for judge in judges},
            'description': 'All judges use stricter criteria'
        },
        {
            'name': 'systematic_lenient',
            'combination': {judge: 'lenient' for judge in judges},
            'description': 'All judges use more lenient criteria'
        }
    ]


def calculate_api_calls(n_combinations: int, n_examples: int, n_judges: int = 10) -> Dict:
    """
    Calculate API call requirements for different configurations.
    
    Args:
        n_combinations: Number of judge combinations
        n_examples: Number of examples to evaluate
        n_judges: Number of judges per combination (default: 10)
        
    Returns:
        Dictionary with API call statistics
    """
    total_calls = n_combinations * n_examples * n_judges
    
    # Estimate time (assuming ~1 second per API call with parallelization)
    time_seconds = total_calls / 10  # With 10 parallel workers
    time_minutes = time_seconds / 60
    time_hours = time_minutes / 60
    
    # Estimate cost (rough approximation)
    # Assuming $0.001 per call (adjust based on actual pricing)
    estimated_cost = total_calls * 0.001
    
    return {
        'total_api_calls': total_calls,
        'calls_per_combination': n_examples * n_judges,
        'estimated_time_minutes': round(time_minutes, 1),
        'estimated_time_hours': round(time_hours, 2),
        'estimated_cost_usd': round(estimated_cost, 2),
        'configuration': {
            'combinations': n_combinations,
            'examples': n_examples,
            'judges': n_judges
        }
    }


def print_comparison():
    """Print comparison of different configuration options."""
    print("="*60)
    print("RUBRIC SENSITIVITY EXPERIMENT - CONFIGURATION COMPARISON")
    print("="*60)
    
    configs = [
        ("Original (44 combos, 100 examples)", 44, 100),
        ("Optimized (7 combos, 500 examples)", 7, 500),
        ("Minimal (3 combos, 1000 examples)", 3, 1000),
        ("Balanced (5 combos, 600 examples)", 5, 600),
    ]
    
    for name, n_combos, n_examples in configs:
        stats = calculate_api_calls(n_combos, n_examples)
        print(f"\n{name}:")
        print(f"  Total API calls: {stats['total_api_calls']:,}")
        print(f"  Training samples per model: {int(n_examples * 0.8)}")
        print(f"  Test samples per model: {int(n_examples * 0.2)}")
        print(f"  Estimated time: {stats['estimated_time_hours']:.1f} hours")
        print(f"  Estimated cost: ${stats['estimated_cost_usd']:.2f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Balanced (5 combos, 600 examples)")
    print("- Sufficient training data (480 samples per model)")
    print("- Tests key robustness properties")
    print("- Reasonable API calls (30,000)")
    print("- ~3 hours runtime with parallelization")
    print("="*60)


if __name__ == "__main__":
    # Test the generation
    print("Generating optimized combinations...")
    
    optimized = generate_optimized_combinations()
    print(f"\nOptimized combinations ({len(optimized)}):")
    for combo in optimized:
        print(f"  - {combo['name']}: {combo['description']}")
    
    minimal = generate_minimal_combinations()
    print(f"\nMinimal combinations ({len(minimal)}):")
    for combo in minimal:
        print(f"  - {combo['name']}: {combo['description']}")
    
    print("\n")
    print_comparison()