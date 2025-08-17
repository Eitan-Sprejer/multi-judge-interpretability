# Full Experiment Runs

This directory contains complete experimental runs for the multi-judge interpretability project. Each subdirectory represents a full end-to-end experiment with different configurations and sample sizes.

## 🏆 Best Results So Far

**`baseline_ultrafeedback_2000samples_20250816_213023/`** contains our best performing experiment to date:
- **Overall Correlation**: 0.799 (79.9%)
- **Best Model R²**: 0.539 (53.9%)
- **Sample Size**: 2,000 samples
- **Run Date**: August 16, 2025

This baseline ultrafeedback experiment demonstrates strong correlation between judge predictions and actual human preferences, making it our current benchmark for performance.

## Directory Structure

Each experiment run follows a standardized structure:

```
experiment_name_samplesize_timestamp/
├── config.json                    # Experiment configuration
├── experiment_results.pkl         # Raw experimental data
├── experiment_summary.json        # Key metrics and summary
├── checkpoints/                   # Training checkpoints
│   └── checkpoint_*.pkl
├── data/                          # Processed datasets
│   ├── data_with_judge_scores.pkl
│   ├── data_with_personas.pkl
│   └── experiment_subset.pkl
├── logs/                          # Detailed execution logs
│   ├── debug_*.log
│   ├── full_experiment_*.log
│   └── progress_*.log
├── plots/                         # Visualization outputs
│   └── experiment_analysis.png
└── results/                       # Analysis results
    ├── correlation_analysis.json
    └── model_results.json
```

## Experiment Types

### 1. Baseline Ultrafeedback Experiments
- **Prefix**: `baseline_ultrafeedback_*`
- **Description**: Baseline experiments using ultrafeedback dataset
- **Best Performance**: 2000 samples (79.9% correlation)

### 2. Ultrafeedback Experiments
- **Prefix**: `ultrafeedback_*`
- **Description**: Various ultrafeedback configurations with different sample sizes
- **Sample Sizes**: 5, 10, 100, 10000 samples

### 3. Persona-based Experiments
- **Prefix**: `personas_*`
- **Description**: Experiments using persona simulation approach
- **Sample Sizes**: 2, 5, 10, 20, 100 samples
- **Note**: Smaller sample sizes show very high correlation (94%+) but may be overfit

## Key Metrics

Each experiment tracks:
- **Overall Correlation**: Correlation between judge predictions and ground truth
- **Best Model R²**: Coefficient of determination for the best performing model
- **Normalization Helps**: Whether data normalization improved performance
- **Samples Processed**: Number of samples used in the experiment

## Performance Comparison

| Experiment Type | Sample Size | Correlation | R² Score | Status |
|----------------|-------------|-------------|----------|---------|
| **baseline_ultrafeedback** | **2000** | **0.799** | **0.539** | **✅ Best** |
| personas | 10 | 0.945 | 0.947 | ⚠️ Likely overfit |
| ultrafeedback | 10 | 0.416 | -33.99 | ❌ Poor |

## Usage

To reproduce any experiment:
1. Check the `config.json` for experiment parameters
2. Review `experiment_summary.json` for key results
3. Examine plots in `plots/` directory for visualizations
4. Check detailed logs in `logs/` directory for debugging

## Notes

- **High correlation with small samples**: Experiments with very small sample sizes (≤10) often show artificially high correlations due to overfitting
- **Baseline performance**: The 2000-sample baseline experiment provides the most reliable performance benchmark
- **Scalability**: Larger sample sizes generally provide more robust and generalizable results
