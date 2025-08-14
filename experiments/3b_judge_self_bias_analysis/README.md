# Experiment 3b: Judge Self-Bias Analysis

## Overview

This experiment investigates whether LLM judges show systematic bias towards responses from the same model family. For example, do GPT-based judges systematically favor GPT-generated responses over those from Claude or other models?

## Research Questions

1. **Primary Question**: Do judges from a specific model family systematically favor responses from the same family?
2. **Secondary Questions**:
   - How does this bias compare across different model families (GPT, Claude, Llama, Gemini)?
   - What is the magnitude and statistical significance of detected biases?
   - What are the implications for judge selection in multi-judge systems?

## Background

Model bias in evaluation systems can lead to unfair assessments and reduced system reliability. If judges consistently favor responses from their own model family, this could create systematic advantages that undermine the fairness of multi-judge evaluation systems.

## Methodology

### Data Requirements
- Dataset containing judge scores for responses from various models
- Columns: `judge_model`, `response_model`, `judge_score`, `prompt_id`, `response_id`
- Minimum sample sizes per judge-response combination

### Analysis Approach
1. **Model Family Classification**: Group models into families (GPT, Claude, Llama, Gemini, Other)
2. **Bias Detection**: Compare scores within vs. across model families
3. **Statistical Testing**: Calculate confidence intervals and significance levels
4. **Effect Size Analysis**: Quantify bias magnitude using standardized metrics

### Bias Detection Criteria
- **Threshold**: Bias detected when mean score difference > 0.1
- **Confidence Level**: 95% confidence intervals
- **Minimum Sample Size**: 50 samples per combination

## Expected Results

### Hypotheses
- **H1**: Judges will show higher scores for responses from their own model family
- **H2**: Bias magnitude will vary across model families
- **H3**: Self-bias will be statistically significant in most cases

### Expected Outcomes
- Identification of systematic bias patterns
- Quantification of bias magnitude and significance
- Recommendations for bias mitigation strategies

## Usage

### Quick Test Mode
```bash
cd experiments/3b_judge_self_bias_analysis
python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl --quick
```

### Full Experiment
```bash
python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl
```

### Custom Configuration
```bash
python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl --config configs/custom.yaml
```

## Output Files

### Results
- `[timestamp]_self_bias_results.json`: Raw experimental data and analysis
- `[timestamp]_self_bias_report.md`: Human-readable analysis report
- `experiment.log`: Detailed execution log

### Key Metrics
- **Self-Bias Score**: Difference between same-family and cross-family scores
- **Bias Magnitude**: Categorized as negligible, small, medium, or large
- **Statistical Significance**: p-values and confidence intervals
- **Effect Size**: Standardized bias measures

## Configuration

The experiment uses `configs/default_config.yaml` for settings:

- **Data Settings**: Sample sizes, column mappings, model family definitions
- **Analysis Settings**: Bias thresholds, confidence levels, statistical tests
- **Output Settings**: Plot generation, report formats, logging levels

## Dependencies

### Core Dependencies
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical testing (optional, for confidence intervals)
- `scikit-learn`: Machine learning utilities
- `matplotlib` & `seaborn`: Visualization (optional)
- `pyyaml`: Configuration file parsing

### Pipeline Integration
- `pipeline.core.*`: Core pipeline components
- `pipeline.utils.*`: Utility functions

## Data Format Requirements

Your dataset should contain these columns:

```python
{
    'judge_model': 'gpt-4',           # Model used as judge
    'response_model': 'claude-3',      # Model that generated response
    'judge_score': 0.85,              # Score assigned by judge (0-1 scale)
    'prompt_id': 'prompt_001',        # Unique prompt identifier
    'response_id': 'response_001',     # Unique response identifier
    # ... other columns as needed
}
```

## Model Family Classification

The experiment automatically classifies models into families:

- **GPT**: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o
- **Claude**: claude-3-sonnet, claude-3-opus, claude-3-haiku, claude-2.1
- **Llama**: llama-2-7b, llama-2-13b, llama-2-70b, llama-3-8b, llama-3-70b
- **Gemini**: gemini-pro, gemini-flash, gemini-1.5-pro
- **Other**: Unidentified or custom models

## Interpretation Guide

### Bias Magnitude
- **Negligible** (< 0.1): No practical significance
- **Small** (0.1-0.3): Minor bias, monitor closely
- **Medium** (0.3-0.5): Moderate bias, consider mitigation
- **Large** (0.5-0.7): Significant bias, requires attention
- **Very Large** (> 0.7): Critical bias, immediate action needed

### Statistical Significance
- **High** (p < 0.01): Strong evidence of bias
- **Medium** (p < 0.05): Moderate evidence of bias
- **Low** (p < 0.10): Weak evidence of bias
- **None** (p ≥ 0.10): No significant evidence of bias

## Limitations

1. **Data Quality**: Results depend on dataset representativeness
2. **Model Coverage**: Limited to models present in the dataset
3. **Score Scale**: Assumes 0-1 normalized scoring scale
4. **Statistical Assumptions**: Relies on normal distribution assumptions

## Future Work

1. **Bias Mitigation**: Develop strategies to reduce detected biases
2. **Cross-Dataset Validation**: Test findings on additional datasets
3. **Dynamic Bias Monitoring**: Real-time bias detection in production
4. **Judge Selection Optimization**: Algorithmic judge selection to minimize bias

## Related Experiments

- **1b_persona_poisoning**: Tests robustness to adversarial inputs
- **1c_rubric_sensitivity**: Evaluates semantic robustness
- **4c_bias_transfer**: Analyzes bias propagation in training

## Contact

For questions about this experiment, contact the Multi-Judge Interpretability team or refer to the main project README.

## Citation

If you use this experiment in your research, please cite:

```bibtex
@misc{judge_self_bias_analysis_2025,
  title={Judge Self-Bias Analysis in Multi-Judge Systems},
  author={Multi-Judge Interpretability Team},
  year={2025},
  note={Experiment 3b from the Multi-Judge Interpretability Project}
}
```
