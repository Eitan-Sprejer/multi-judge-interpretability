# Aggregator Robustness Study

**Research Question**: How robust is the learned MLP aggregator to realistic human feedback contamination?

## Experiment Design

**Approach**: Train MLP aggregator on contaminated human feedback, evaluate on clean test set.

**Contamination Strategies**:
- **Random Noise**: ±3 random error per rating (inconsistent annotators)
- **Systematic Bias**: Consistent +2/-2 offset per annotator (scale misalignment)  
- **Scaled Down**: Compress [0,10] → [3,7] (annotators avoiding extremes)

**Contamination Rates**: 0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%

## Key Results

| Strategy | Clean R² | 50% Contamination | Degradation |
|----------|----------|-------------------|-------------|
| Random Noise | 0.537 | 0.524 | **2.4%** |
| Systematic Bias | 0.532 | 0.480 | **9.8%** |  
| Scaled Down | 0.517 | 0.392 | **24.1%** |

**Finding**: Aggregator maintains reasonable performance (R² > 0.39) even under heavy contamination. Most robust to noise, most vulnerable to scale compression.

## Files

- `run_aggregator_robustness.py` - Main experiment script
- `results/aggregator_robustness_analysis.png` - Visualization
- `results/aggregator_robustness_20250818_131617.json` - Full results

## Usage

```bash
python run_aggregator_robustness.py --data ../../dataset/data_with_judge_scores.pkl
```

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