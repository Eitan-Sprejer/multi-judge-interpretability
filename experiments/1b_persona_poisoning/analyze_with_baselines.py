#!/usr/bin/env python3
"""
Analyze experiment results with proper baselines and create figures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the experiment results (corrected methodology)
with open('results/experiment_20250817_144956.json', 'r') as f:
    aggregator_results = json.load(f)

# Load the dataset to compute single judge baselines
with open('../../dataset/data_with_judge_scores.pkl', 'rb') as f:
    df = pickle.load(f)

# Prepare data
# Extract human feedback score from dict if needed
if 'human_feedback_score' not in df.columns:
    if 'human_feedback' in df.columns:
        df['human_feedback_score'] = df['human_feedback'].apply(
            lambda x: x.get('score', x.get('average_score', 5.0)) if isinstance(x, dict) else 5.0
        )

df = df.dropna(subset=['human_feedback_score'])
df = df[df['judge_scores'].apply(lambda x: isinstance(x, list) and len(x) == 10)]

# Use CORRECTED methodology: random split + normalization (same as corrected experiment)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.array(df['judge_scores'].tolist())
y = np.array(df['human_feedback_score'].values, dtype=np.float32)

# Use SAME random split as corrected experiment (seed=42)
X_train_base, X_test_raw, y_train_base, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply normalization (same as corrected experiment)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test_raw.reshape(-1, 10)).astype(np.float32)
y_test = y_test_raw.astype(np.float32)

print("Using CORRECTED methodology: random split + normalization")
print(f"Train size: {len(X_train_base)}, Test size: {len(X_test)}")

print("Computing baseline performances...")

# Define judge names
judge_names = [
    'harmlessness', 'privacy', 'factual-accuracy', 'prompt-faithfulness',
    'calibration', 'bias-fairness', 'reasoning', 'discourse',
    'conciseness', 'style-formatting'
]

def contaminate_arrays(X_train, y_train, contamination_rate):
    """Contaminate training arrays by inverting scores (same as corrected experiment)."""
    if contamination_rate == 0:
        return X_train.copy(), y_train.copy()
    
    X_contaminated = X_train.copy()
    y_contaminated = y_train.copy()
    n_contaminate = int(len(X_contaminated) * contamination_rate)
    
    np.random.seed(42)  # Same seed as experiment
    contaminate_indices = np.random.choice(len(X_contaminated), n_contaminate, replace=False)
    
    for idx in contaminate_indices:
        # Invert the human feedback score (same as troll persona logic)
        original = y_contaminated[idx]
        y_contaminated[idx] = 10 - original  # Simple inversion for analysis
    
    return X_contaminated, y_contaminated

def evaluate_single_judge(judge_idx, contamination_rates):
    """Evaluate a single judge's performance across contamination rates."""
    results = {}
    
    for rate in contamination_rates:
        # Contaminate training data
        X_train_contaminated, y_train_contaminated = contaminate_arrays(X_train_base, y_train_base, rate)
        
        # Apply normalization (fit on contaminated training data)
        scaler_train = StandardScaler()
        X_train_normalized = scaler_train.fit_transform(X_train_contaminated)
        
        # Re-normalize test data with training scaler
        X_test_normalized = scaler_train.transform(X_test_raw)
        
        # Train simple linear model: y = a * judge_score + b
        X_train_single = X_train_normalized[:, judge_idx].reshape(-1, 1)
        y_train = y_train_contaminated
        
        lr = LinearRegression()
        lr.fit(X_train_single, y_train)
        
        # Evaluate on clean test set
        X_test_single = X_test_normalized[:, judge_idx].reshape(-1, 1)
        y_pred = lr.predict(X_test_single)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results[rate] = {'r2': r2, 'mse': mse}
    
    return results

def evaluate_mean_baseline(contamination_rates):
    """Evaluate mean of all judges as baseline."""
    results = {}
    
    for rate in contamination_rates:
        # Contaminate training data (same as single judge evaluation)
        X_train_contaminated, y_train_contaminated = contaminate_arrays(X_train_base, y_train_base, rate)
        
        # Apply normalization (fit on contaminated training data)
        scaler_train = StandardScaler()
        X_train_normalized = scaler_train.fit_transform(X_train_contaminated)
        
        # Re-normalize test data with training scaler
        X_test_normalized = scaler_train.transform(X_test_raw)
        
        # Train linear model on mean of judges
        X_train_mean = X_train_normalized.mean(axis=1).reshape(-1, 1)
        y_train = y_train_contaminated
        
        lr = LinearRegression()
        lr.fit(X_train_mean, y_train)
        
        # Evaluate on clean test set
        X_test_mean = X_test_normalized.mean(axis=1).reshape(-1, 1)
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

# Create cleaner 2x2 figure
fig = plt.figure(figsize=(12, 8))

# Convert rates to percentages for plotting
rates_pct = [r * 100 for r in contamination_rates]

# Plot 1: R² comparison (top-left)
ax1 = plt.subplot(2, 2, 1)
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
ax1.set_ylim([-0.1, 0.7])

# Plot 2: All single judges comparison (top-right)
ax2 = plt.subplot(2, 2, 2)
for name in judge_names:  # Plot all judges
    judge_r2 = [single_judge_results[name][r]['r2'] for r in contamination_rates]
    ax2.plot(rates_pct, judge_r2, alpha=0.6, linewidth=1, label=name[:8])
ax2.plot(rates_pct, aggregator_r2, 'b-', linewidth=3, label='Aggregator', alpha=0.8)
ax2.set_xlabel('Contamination Rate (%)', fontsize=12)
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_title('Individual Judges vs Aggregator', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

# Calculate baselines for degradation
baseline_agg = aggregator_r2[0]
baseline_judge = best_judge_r2[0]
baseline_mean = mean_baseline_r2[0]

rel_deg_agg = [(baseline_agg - r2) / baseline_agg * 100 for r2 in aggregator_r2]
rel_deg_judge = [(baseline_judge - r2) / baseline_judge * 100 for r2 in best_judge_r2]
rel_deg_mean = [(baseline_mean - r2) / baseline_mean * 100 for r2 in mean_baseline_r2]

# Plot 3: Robustness comparison bar chart (bottom-left)
ax3 = plt.subplot(2, 2, 3)
key_rates = [0.1, 0.25, 0.5]
x_pos = np.arange(len(key_rates))
width = 0.25

agg_r2_at_rates = [aggregator_results[str(r)]['r2'] for r in key_rates]
judge_r2_at_rates = [best_judge_results[r]['r2'] for r in key_rates]
mean_r2_at_rates = [mean_baseline_results[r]['r2'] for r in key_rates]

bars1 = ax3.bar(x_pos - width, agg_r2_at_rates, width, label='Aggregator', color='blue', alpha=0.8)
bars2 = ax3.bar(x_pos, judge_r2_at_rates, width, label='Best Judge', color='green', alpha=0.8)
bars3 = ax3.bar(x_pos + width, mean_r2_at_rates, width, label='Mean', color='red', alpha=0.8)

ax3.set_xlabel('Contamination Rate', fontsize=12)
ax3.set_ylabel('R² Score', fontsize=12)
ax3.set_title('Performance at Key Contamination Levels', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'{int(r*100)}%' for r in key_rates])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Advantage of aggregator over mean of judges (bottom-right)
ax4 = plt.subplot(2, 2, 4)
advantage = [(aggregator_r2[i] - mean_baseline_r2[i]) for i in range(len(contamination_rates))]
ax4.plot(rates_pct, advantage, 'b-o', linewidth=2, markersize=8)
ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax4.fill_between(rates_pct, 0, advantage, where=[a > 0 for a in advantage], 
                  alpha=0.3, color='green', label='Aggregator Better')
ax4.fill_between(rates_pct, 0, advantage, where=[a <= 0 for a in advantage], 
                  alpha=0.3, color='red', label='Mean Better')
ax4.set_xlabel('Contamination Rate (%)', fontsize=12)
ax4.set_ylabel('R² Difference (Aggregator - Mean)', fontsize=12)
ax4.set_title('Aggregator Advantage Over Mean of Judges', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Persona Poisoning Experiment: Robustness Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
figure_path = 'results/contamination_analysis.png'
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

with open('results/complete_analysis.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nComplete results saved to: results/complete_analysis.json")