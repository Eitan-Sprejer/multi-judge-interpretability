#!/usr/bin/env python3
"""
Analyze experiment results with proper baselines and create figures
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import judge configuration
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS

# Judge IDs
JUDGE_IDS = list(JUDGE_RUBRICS.keys())

def main():
    parser = argparse.ArgumentParser(description="Analyze persona poisoning experiment results")
    parser.add_argument('--results', type=str, required=True,
                       help='Path to experiment results JSON file')
    parser.add_argument('--data', type=str, 
                       help='Path to dataset with judge scores')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/1b_persona_poisoning/results',
                       help='Output directory for figures and analysis')
    
    args = parser.parse_args()
    
    # Load the experiment results
    with open(args.results, 'r') as f:
        aggregator_results = json.load(f)
    
    # Auto-detect dataset if not provided
    if args.data is None:
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
            return 1
    
    # Load the dataset to compute single judge baselines
    with open(args.data, 'rb') as f:
        df = pickle.load(f)

    # Prepare data - handle different column formats
    if 'judge_scores' not in df.columns:
        if 'scores' in df.columns:
            df['judge_scores'] = df['scores']
        else:
            print("❌ Dataset must have 'judge_scores' or 'scores' column")
            return 1
    
    # Handle human feedback columns
    if 'human_feedback_score' not in df.columns:
        if 'score' in df.columns:
            df['human_feedback_score'] = df['score']
        elif 'human_feedback' in df.columns:
            def extract_score(hf):
                if isinstance(hf, dict):
                    return hf.get('score', hf.get('average_score', 5.0))
                return 5.0
            df['human_feedback_score'] = df['human_feedback'].apply(extract_score)
        else:
            print("❌ Dataset must have human feedback score column")
            return 1
    
    df = df.dropna(subset=['human_feedback_score'])
    expected_judge_count = len(JUDGE_IDS)
    df = df[df['judge_scores'].apply(lambda x: isinstance(x, list) and len(x) == expected_judge_count)]

    # Split data (same as in experiment)
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[:test_size].copy().reset_index(drop=True)
    df_train_base = df.iloc[test_size:].copy().reset_index(drop=True)
    
    X_test = np.array(df_test['judge_scores'].tolist())
    y_test = np.array(df_test['human_feedback_score'].values, dtype=np.float32)
    
    print("Computing baseline performances...")

    # Use current judge names from rubrics
    judge_names = [jid.replace('-judge', '') for jid in JUDGE_IDS]

    def contaminate_training_data(df_train, contamination_rate):
        """Contaminate training data by inverting scores."""
        if contamination_rate == 0:
            return df_train.copy()
        
        df_contaminated = df_train.copy()
        n_contaminate = int(len(df_contaminated) * contamination_rate)
        
        np.random.seed(42)  # Same seed as experiment
        contaminate_indices = np.random.choice(len(df_contaminated), n_contaminate, replace=False)
        
        for idx in contaminate_indices:
            # Invert the human feedback score (0-10 scale)
            original = df_contaminated.iloc[idx]['human_feedback_score']
            df_contaminated.at[idx, 'human_feedback_score'] = 10 - original
        
        return df_contaminated

    def evaluate_single_judge(judge_idx, contamination_rates):
        """Evaluate a single judge's performance across contamination rates."""
        results = {}
        
        for rate in contamination_rates:
            # Contaminate training data
            df_train = contaminate_training_data(df_train_base, rate)
            
            # Train simple linear model: y = a * judge_score + b
            X_train_single = df_train['judge_scores'].apply(lambda x: x[judge_idx]).values.reshape(-1, 1)
            y_train = df_train['human_feedback_score'].values
            
            lr = LinearRegression()
            lr.fit(X_train_single, y_train)
            
            # Evaluate on clean test set
            X_test_single = X_test[:, judge_idx].reshape(-1, 1)
            y_pred = lr.predict(X_test_single)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            results[rate] = {'r2': r2, 'mse': mse}
        
        return results

    def evaluate_mean_baseline(contamination_rates):
        """Evaluate mean of all judges as baseline."""
        results = {}
        
        for rate in contamination_rates:
            # Contaminate training data
            df_train = contaminate_training_data(df_train_base, rate)
            
            # Train linear model on mean of judges
            X_train_mean = np.array(df_train['judge_scores'].tolist()).mean(axis=1).reshape(-1, 1)
            y_train = df_train['human_feedback_score'].values
            
            lr = LinearRegression()
            lr.fit(X_train_mean, y_train)
            
            # Evaluate on clean test set
            X_test_mean = X_test.mean(axis=1).reshape(-1, 1)
            y_pred = lr.predict(X_test_mean)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            results[rate] = {'r2': r2, 'mse': mse}
        
        return results

    # Get contamination rates from aggregator results
    contamination_rates = sorted([float(k) for k in aggregator_results.keys()])

    # Compute baselines
    print("Evaluating single judges...")
    single_judge_results = {}
    for i, name in enumerate(judge_names):
        print(f"  {name}...")
        single_judge_results[name] = evaluate_single_judge(i, contamination_rates)
    
    print("Evaluating mean baseline...")
    mean_baseline_results = evaluate_mean_baseline(contamination_rates)

    # Find best single judge (at 0% contamination)
    best_judge_name = max(judge_names, 
                          key=lambda j: single_judge_results[j][0.0]['r2'])
    best_judge_results = single_judge_results[best_judge_name]
    
    print(f"\nBest single judge: {best_judge_name} (R²={best_judge_results[0.0]['r2']:.3f} at 0%)")

# Create comprehensive figures
fig = plt.figure(figsize=(15, 10))

# Convert rates to percentages for plotting
rates_pct = [r * 100 for r in contamination_rates]

# Plot 1: R² comparison
ax1 = plt.subplot(2, 3, 1)
aggregator_r2 = [aggregator_results[str(r)]['r2'] for r in contamination_rates]
best_judge_r2 = [best_judge_results[r]['r2'] for r in contamination_rates]
mean_baseline_r2 = [mean_baseline_results[r]['r2'] for r in contamination_rates]

ax1.plot(rates_pct, aggregator_r2, 'b-o', linewidth=2, markersize=8, label='Learned Aggregator')
ax1.plot(rates_pct, best_judge_r2, 'g--s', linewidth=2, markersize=6, label=f'Best Single Judge ({best_judge_name})')
ax1.plot(rates_pct, mean_baseline_r2, 'r-.^', linewidth=1.5, markersize=6, label='Mean of Judges')
ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Acceptable threshold')
ax1.set_xlabel('Contamination Rate (%)', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Model Performance vs Contamination', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-0.1, 0.6])

# Plot 2: Relative degradation
ax2 = plt.subplot(2, 3, 2)
baseline_agg = aggregator_r2[0]
baseline_judge = best_judge_r2[0]
baseline_mean = mean_baseline_r2[0]

rel_deg_agg = [(baseline_agg - r2) / baseline_agg * 100 for r2 in aggregator_r2]
rel_deg_judge = [(baseline_judge - r2) / baseline_judge * 100 for r2 in best_judge_r2]
rel_deg_mean = [(baseline_mean - r2) / baseline_mean * 100 for r2 in mean_baseline_r2]

ax2.plot(rates_pct, rel_deg_agg, 'b-o', linewidth=2, markersize=8, label='Learned Aggregator')
ax2.plot(rates_pct, rel_deg_judge, 'g--s', linewidth=2, markersize=6, label=f'Best Single Judge')
ax2.plot(rates_pct, rel_deg_mean, 'r-.^', linewidth=1.5, markersize=6, label='Mean of Judges')
ax2.set_xlabel('Contamination Rate (%)', fontsize=12)
ax2.set_ylabel('Performance Drop (%)', fontsize=12)
ax2.set_title('Relative Performance Degradation', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: All single judges comparison
ax3 = plt.subplot(2, 3, 3)
for name in judge_names[:5]:  # Plot top 5 judges for clarity
    judge_r2 = [single_judge_results[name][r]['r2'] for r in contamination_rates]
    ax3.plot(rates_pct, judge_r2, alpha=0.6, linewidth=1, label=name[:8])
ax3.plot(rates_pct, aggregator_r2, 'b-', linewidth=3, label='Aggregator', alpha=0.8)
ax3.set_xlabel('Contamination Rate (%)', fontsize=12)
ax3.set_ylabel('R² Score', fontsize=12)
ax3.set_title('Individual Judges vs Aggregator', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: MSE comparison
ax4 = plt.subplot(2, 3, 4)
aggregator_mse = [aggregator_results[str(r)]['mse'] for r in contamination_rates]
best_judge_mse = [best_judge_results[r]['mse'] for r in contamination_rates]
mean_baseline_mse = [mean_baseline_results[r]['mse'] for r in contamination_rates]

ax4.plot(rates_pct, aggregator_mse, 'b-o', linewidth=2, markersize=8, label='Learned Aggregator')
ax4.plot(rates_pct, best_judge_mse, 'g--s', linewidth=2, markersize=6, label=f'Best Single Judge')
ax4.plot(rates_pct, mean_baseline_mse, 'r-.^', linewidth=1.5, markersize=6, label='Mean of Judges')
ax4.set_xlabel('Contamination Rate (%)', fontsize=12)
ax4.set_ylabel('Mean Squared Error', fontsize=12)
ax4.set_title('Prediction Error vs Contamination', fontsize=14, fontweight='bold')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

# Plot 5: Robustness comparison bar chart
ax5 = plt.subplot(2, 3, 5)
key_rates = [0.1, 0.25, 0.5]
x_pos = np.arange(len(key_rates))
width = 0.25

agg_r2_at_rates = [aggregator_results[str(r)]['r2'] for r in key_rates]
judge_r2_at_rates = [best_judge_results[r]['r2'] for r in key_rates]
mean_r2_at_rates = [mean_baseline_results[r]['r2'] for r in key_rates]

bars1 = ax5.bar(x_pos - width, agg_r2_at_rates, width, label='Aggregator', color='blue', alpha=0.8)
bars2 = ax5.bar(x_pos, judge_r2_at_rates, width, label='Best Judge', color='green', alpha=0.8)
bars3 = ax5.bar(x_pos + width, mean_r2_at_rates, width, label='Mean', color='red', alpha=0.8)

ax5.set_xlabel('Contamination Rate', fontsize=12)
ax5.set_ylabel('R² Score', fontsize=12)
ax5.set_title('Performance at Key Contamination Levels', fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f'{int(r*100)}%' for r in key_rates])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Advantage of aggregator
ax6 = plt.subplot(2, 3, 6)
advantage = [(aggregator_r2[i] - best_judge_r2[i]) for i in range(len(contamination_rates))]
ax6.plot(rates_pct, advantage, 'b-o', linewidth=2, markersize=8)
ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax6.fill_between(rates_pct, 0, advantage, where=[a > 0 for a in advantage], 
                  alpha=0.3, color='green', label='Aggregator Better')
ax6.fill_between(rates_pct, 0, advantage, where=[a <= 0 for a in advantage], 
                  alpha=0.3, color='red', label='Judge Better')
ax6.set_xlabel('Contamination Rate (%)', fontsize=12)
ax6.set_ylabel('R² Difference (Aggregator - Best Judge)', fontsize=12)
ax6.set_title('Aggregator Advantage Over Best Single Judge', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('Persona Poisoning Experiment: Robustness Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

    # Save figure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / 'contamination_analysis.png'
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {figure_path}")

# Generate summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\nClean Performance (0% contamination):")
print(f"  Learned Aggregator: R² = {aggregator_r2[0]:.3f}")
print(f"  Best Single Judge:  R² = {best_judge_r2[0]:.3f}")
print(f"  Mean of Judges:     R² = {mean_baseline_r2[0]:.3f}")

print("\nAt 25% contamination:")
print(f"  Learned Aggregator: R² = {aggregator_results['0.25']['r2']:.3f} (drop: {rel_deg_agg[5]:.1f}%)")
print(f"  Best Single Judge:  R² = {best_judge_results[0.25]['r2']:.3f} (drop: {rel_deg_judge[5]:.1f}%)")
print(f"  Mean of Judges:     R² = {mean_baseline_results[0.25]['r2']:.3f} (drop: {rel_deg_mean[5]:.1f}%)")

print("\nBreaking points (R² < 0.3):")
for name, results in [('Aggregator', aggregator_results), 
                      ('Best Judge', best_judge_results),
                      ('Mean', mean_baseline_results)]:
    breaking_point = None
    for rate in contamination_rates:
        r2 = results[str(rate) if name == 'Aggregator' else rate]['r2']
        if r2 < 0.3:
            breaking_point = rate
            break
    print(f"  {name:12}: {breaking_point*100:.0f}% contamination" if breaking_point else f"  {name:12}: No breaking point")

# Save detailed results
results_summary = {
    'aggregator': {str(r): aggregator_results[str(r)] for r in contamination_rates},
    'best_single_judge': {
        'name': best_judge_name,
        'results': {str(r): best_judge_results[r] for r in contamination_rates}
    },
    'mean_baseline': {str(r): mean_baseline_results[r] for r in contamination_rates},
    'all_judges': {name: {str(r): single_judge_results[name][r] for r in contamination_rates} 
                   for name in judge_names}
}

    complete_analysis_path = output_dir / 'complete_analysis.json'
    with open(complete_analysis_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nComplete results saved to: {complete_analysis_path}")
    return 0


if __name__ == "__main__":
    exit(main())