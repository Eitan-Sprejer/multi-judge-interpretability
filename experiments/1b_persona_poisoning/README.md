# Experiment 1B: Persona Poisoning

## Overview

This experiment tests the robustness of learned aggregation functions against contaminated training data ("troll personas"). We train separate models on increasingly contaminated data and evaluate all models on clean test data.

## Key Finding

**Learned aggregators are significantly more robust than single judges to training data contamination.**

- At 25% contamination: Aggregator retains 79% performance vs. Single judge retains 58% performance
- Breaking point: Aggregator at 30% vs. Single judge at 25%

## How to Run

### 1. Run the main experiment
```bash
cd experiments/1b_persona_poisoning
python run_experiment.py
```
This trains 9 models (one for each contamination level) and saves results.

### 2. Generate analysis and figures
```bash
python analyze_with_baselines.py
```
This compares against single judge baselines and creates visualization.

## Files

### Scripts
- `run_experiment.py` - Main experiment runner
- `analyze_with_baselines.py` - Analysis and figure generation
- `src/simple_runner.py` - Core experiment logic
- `src/troll_generator.py` - Contamination generation

### Results
- `results/FINAL_REPORT.md` - Complete analysis report
- `results/contamination_analysis.png` - 6-panel figure with all comparisons
- `results/complete_analysis.json` - Raw data for all methods

## Method

1. **Contamination**: For each rate (0% to 50%), we randomly select training samples and invert their human feedback scores
2. **Training**: Train a new MLP aggregator on the contaminated training set
3. **Evaluation**: Test ALL models on the same clean test set
4. **Comparison**: Compare against single judges and mean baseline

## Results Summary

| Method | Clean R² | 25% Contaminated R² | Performance Drop |
|--------|----------|-------------------|------------------|
| **Learned Aggregator** | 0.514 | 0.404 | 21.4% |
| Mean of Judges | 0.518 | 0.334 | 35.6% |
| Best Single Judge | 0.466 | 0.271 | 41.9% |

## Contribution to Paper

This experiment provides empirical evidence that multi-judge aggregation offers superior robustness compared to single judges, supporting the main thesis of the Multi-Judge Interpretability framework.