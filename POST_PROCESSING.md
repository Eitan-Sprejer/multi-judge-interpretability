# Post-Processing Guide for Multi-Judge Interpretability Experiments

This guide documents how to analyze results, generate plots, and prepare data for publication after running the full experiment pipeline.

## 📊 Experiment Output Structure

After running an experiment, results are organized in:
```
results/full_experiments/{run_name}/
├── data/
│   ├── experiment_subset.pkl         # Original data subset
│   ├── data_with_personas.pkl        # Data + persona scores
│   └── data_with_judge_scores.pkl    # Final dataset with all scores
├── results/
│   ├── correlation_analysis.json     # Judge-persona correlations
│   ├── baseline_results.json         # Baseline model performance
│   ├── model_results.json           # Aggregation model results
│   └── cross_correlation_analysis.json # Detailed correlation matrices
├── plots/
│   ├── experiment_analysis.png       # Main results summary
│   ├── cross_correlation_heatmaps.png # Correlation visualizations
│   └── baseline_comparison_comprehensive.png # Model comparisons
├── checkpoints/                      # Recovery checkpoints
└── logs/                            # Execution logs
```

## 🔧 Key Analysis Tasks

### 1. Correlation Analysis

**Purpose**: Understand how well AI judges align with human persona preferences.

**Key Files**:
- `results/correlation_analysis.json` - Overall correlation metrics
- `results/cross_correlation_analysis.json` - Detailed correlation matrices
- `plots/cross_correlation_heatmaps.png` - Visualization

**Interpretation**:
- **Overall Correlation > 0.5**: Strong alignment between judges and personas
- **Overall Correlation 0.3-0.5**: Moderate alignment, interesting research findings
- **Overall Correlation < 0.3**: Weak alignment, potential judge-human misalignment

**Key Metrics**:
```python
{
    "overall_correlation": 0.478,           # Primary research metric
    "judge_correlations": {...},           # Individual judge performance
    "persona_correlations": {...},         # Individual persona consistency
    "judge_range": [1.93, 3.59],          # Judge score distribution
    "persona_range": [1.40, 8.13]         # Persona score distribution
}
```

### 2. Model Performance Analysis

**Purpose**: Evaluate how well learned aggregation models combine judge scores.

**Key Files**:
- `results/baseline_results.json` - Simple baseline performance
- `results/model_results.json` - Advanced model results

**Key Metrics**:
- **R² Score**: Explained variance (higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Comparison to Baselines**: GAM vs. MLP vs. Naive Mean

**Expected Performance Hierarchy**:
1. Best GAM/MLP model (target R² > 0.6)
2. Best single judge (target R² > 0.4)
3. Naive mean aggregation (baseline)

### 3. Sampling Method Comparison

**Purpose**: Compare uniform vs. confidence-weighted ground truth sampling.

**Analysis Script**:
```python
# Compare sampling methods
python run_full_experiment.py --sampling-method uniform --run-name exp_uniform
python run_full_experiment.py --sampling-method confidence_weighted --run-name exp_confidence

# Analyze differences
python compare_sampling_methods.py --exp1 exp_uniform --exp2 exp_confidence
```

**Expected Findings**:
- Confidence-weighted sampling should improve model performance
- High-confidence personas should show stronger correlations
- Diversity maintained across persona types

### 4. Certainty Distribution Analysis

**Purpose**: Understand persona confidence patterns and their impact.

**Analysis**:
```python
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment data
with open('results/full_experiments/{run_name}/data/data_with_personas.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract certainty scores
certainty_data = []
for _, row in data.iterrows():
    personas = row['human_feedback']['personas']
    for persona_name, feedback in personas.items():
        if 'certainty' in feedback:
            certainty_data.append({
                'persona': persona_name,
                'certainty': feedback['certainty'],
                'score': feedback['score']
            })

certainty_df = pd.DataFrame(certainty_data)

# Analyze patterns
print("Certainty by Persona:")
print(certainty_df.groupby('persona')['certainty'].agg(['mean', 'std', 'count']))

# Plot distribution
plt.figure(figsize=(12, 8))
certainty_df.boxplot(column='certainty', by='persona', rot=45)
plt.title('Certainty Distribution by Persona')
plt.tight_layout()
plt.savefig('certainty_analysis.png', dpi=300, bbox_inches='tight')
```

### 5. Judge-Persona Correlation Deep Dive

**Purpose**: Understand which judges align with which personas.

**Analysis**:
```python
import json
import numpy as np
import seaborn as sns

# Load correlation data
with open('results/cross_correlation_analysis.json', 'r') as f:
    correlations = json.load(f)

# Extract judge-persona correlations
judge_persona_corr = correlations['judge_persona_corr_matrix']

# Find strongest correlations
strong_correlations = []
for judge, persona_corrs in judge_persona_corr.items():
    for persona, corr in persona_corrs.items():
        if abs(corr) > 0.5:  # Strong correlation threshold
            strong_correlations.append({
                'judge': judge,
                'persona': persona,
                'correlation': corr
            })

# Sort by correlation strength
strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

print("Strongest Judge-Persona Correlations:")
for item in strong_correlations[:10]:
    print(f"{item['judge']} <-> {item['persona']}: {item['correlation']:.3f}")
```

## 📈 Visualization Scripts

### Generate All Plots
```bash
# Main experiment plots (generated automatically)
# Check: results/full_experiments/{run_name}/plots/

# Custom analysis plots
python -c "
import sys
sys.path.append('analysis_scripts')
from plot_generator import generate_all_plots
generate_all_plots('results/full_experiments/{run_name}')
"
```

### Publication-Ready Figures
```python
# Create high-quality figures for papers
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Generate paper figures
generate_correlation_heatmap(output='paper_correlation_heatmap.pdf')
generate_performance_comparison(output='paper_model_comparison.pdf')
generate_certainty_analysis(output='paper_certainty_distribution.pdf')
```

## 📊 Statistical Analysis

### Significance Testing
```python
from scipy import stats

# Test correlation significance
def test_correlation_significance(r, n):
    """Test if correlation is statistically significant."""
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    return p_value

# Load correlation data
with open('results/correlation_analysis.json', 'r') as f:
    corr_data = json.load(f)

overall_corr = corr_data['overall_correlation']
n_samples = corr_data['num_samples']
p_value = test_correlation_significance(overall_corr, n_samples)

print(f"Overall correlation: {overall_corr:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")
```

### Effect Size Analysis
```python
# Cohen's conventions for correlation effect sizes
def interpret_correlation(r):
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"

print(f"Effect size: {interpret_correlation(overall_corr)}")
```

## 📝 Reporting Templates

### Results Summary Template
```
## Multi-Judge Interpretability Results

**Dataset**: {data_source} ({data_size} samples)
**Sampling Method**: {sampling_method}
**Date**: {experiment_date}

### Key Findings
- **Judge-Persona Correlation**: {overall_correlation:.3f} ({effect_size} effect)
- **Statistical Significance**: p < {p_value:.3f}
- **Best Model Performance**: R² = {best_r2:.3f}
- **Model Improvement over Baseline**: +{improvement:.3f}

### Judge Performance
- **Best Performing Judge**: {best_judge} (r = {best_judge_corr:.3f})
- **Most Consistent Judge**: {most_consistent_judge}
- **Least Aligned Judge**: {worst_judge}

### Persona Insights
- **Most Predictable Persona**: {most_predictable_persona}
- **Highest Average Certainty**: {highest_certainty_persona}
- **Most Variable Persona**: {most_variable_persona}

### Research Implications
[Interpretation based on correlation strength and patterns]
```

### Paper Export Scripts
```python
# Export data for LaTeX tables
def export_correlation_table():
    """Export correlation matrix as LaTeX table."""
    # Implementation for publication-ready tables

def export_model_comparison_table():
    """Export model performance comparison."""
    # Implementation for model comparison tables

def export_summary_statistics():
    """Export key statistics for paper."""
    # Implementation for summary stats
```

## 🔄 Reproduction & Verification

### Validate Results
```bash
# Re-run with same seed to verify reproducibility
python run_full_experiment.py \
    --data-source ultrafeedback \
    --data-size 10000 \
    --random-seed 42 \
    --run-name validation_run

# Compare results
python compare_experiments.py \
    --exp1 baseline_10k_personas_15_enhanced \
    --exp2 validation_run
```

### Cross-Validation Analysis
```python
# Run multiple seeds for robustness testing
seeds = [42, 123, 456, 789, 999]
results = []

for seed in seeds:
    # Run experiment with different seed
    result = run_experiment(random_seed=seed)
    results.append(result['overall_correlation'])

# Analyze stability
mean_corr = np.mean(results)
std_corr = np.std(results)
ci_95 = 1.96 * std_corr / np.sqrt(len(results))

print(f"Mean correlation: {mean_corr:.3f} ± {ci_95:.3f}")
```

## 🚀 Next Steps

1. **Validate with 10k sample run**
2. **Compare against UltraFeedback ground truth**
3. **Test robustness with contaminated judges**
4. **Prepare submission for NeurIPS Interpretability Workshop**

## 📞 Troubleshooting

**Issue**: Low correlations (< 0.3)
**Solution**: Check persona simulation quality, verify judge deployment, analyze failure rates

**Issue**: Model overfitting 
**Solution**: Increase regularization, use cross-validation, check training curves

**Issue**: Missing visualizations
**Solution**: Check plot generation logs, verify matplotlib backend, ensure sufficient disk space

## 🎯 Success Criteria

- **Primary**: Overall correlation > 0.4 with statistical significance
- **Secondary**: GAM interpretability analysis complete
- **Tertiary**: Publication-ready visualizations generated