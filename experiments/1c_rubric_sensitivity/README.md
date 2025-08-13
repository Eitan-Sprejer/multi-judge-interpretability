# Experiment 1C: Rubric Sensitivity

## Hypothesis
Robust aggregation models should maintain consistent scoring despite semantically equivalent but differently phrased judge rubrics. We expect robust aggregators to show <5% variance across equivalent rubrics and demonstrate superior stability compared to naive baselines.

## Overview
This experiment tests whether learned aggregation functions are more robust to rubric variations than simple baseline methods by:
1. Creating multiple semantic variations of judge rubrics (formal, casual, restructured)
2. Scoring examples through all rubric variants 
3. Comparing robustness across aggregation methods: **Learned GAM/MLP**, **Mean baseline**, **Single best judge**
4. Generating visualizations and statistical analysis

## Prerequisites
- **Trained aggregation models**: Ensure you have trained GAM/MLP models from the main project
- **Judge setup**: 10 specialized judges created via Martian API
- **Data**: UltraFeedback dataset with judge scores (`../../dataset/data_with_judge_scores.pkl`)

## How to Run

### Full Experiment (1000 examples)
```bash
# Run complete experiment with model comparison
python run_experiment.py --model path/to/trained_model.pt

# Or run without model comparison (rubric sensitivity only)
python run_experiment.py
```

### Quick Test (50 examples)
```bash
python run_experiment.py --quick --model path/to/trained_model.pt
```

## Implementation Details

### Rubric Variations Generated
For each original judge rubric, creates 3 variants:
- **Original**: Baseline rubric from judge creation
- **Formal**: Academic/technical language style
- **Casual**: Conversational/informal style

### Aggregation Methods Compared
1. **Learned Aggregator**: Trained GAM/MLP model (`f_θ`)
2. **Mean Baseline**: Simple average of all judge scores
3. **Single Best Judge**: Highest-performing individual judge

### Robustness Metrics
- **Score Variance**: Consistency across rubric variations
- **Cross-Rubric Correlation**: Pearson & Spearman correlations
- **Rank Consistency**: Kendall's tau for ranking stability
- **Improvement Factor**: Learned vs baseline variance ratio

## Output Files
Results saved to `results/` directory:
- `scores.csv` - Raw judge scores across all variants
- `analysis_results.json` - Statistical metrics and comparisons
- `robustness_report.md` - Human-readable analysis summary
- `robustness_plots.png` - Visualization of variance by method

## Success Criteria
- **Learned aggregator** shows <5% variance across rubric variations
- **Learned aggregator** maintains >95% correlation vs <90% for baselines  
- **Improvement factor** >2x variance reduction over mean baseline

## Expected Results
- Quantitative demonstration that learned aggregation provides robustness benefits
- Identification of most/least stable rubric formulations
- Visual evidence of aggregator superiority over naive methods
- Statistical validation of robustness hypothesis

## Architecture
- `src/rubric_variations.py` - Generates semantic rubric variants
- `src/judge_variant_creator.py` - Creates judges via Martian API
- `src/scoring_pipeline.py` - Scores examples through all variants
- `src/robustness_metrics.py` - Calculates metrics and model comparisons
- `src/experiment_runner.py` - Main orchestrator and reporting
