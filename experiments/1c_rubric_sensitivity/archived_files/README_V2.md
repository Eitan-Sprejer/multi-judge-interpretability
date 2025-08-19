# Rubric Sensitivity Experiment V2 - Real API Implementation

## Overview

This is the enhanced version of the rubric sensitivity experiment that uses **real Martian API calls** to create and evaluate judge variants with modified rubrics, replacing the previous mathematical simulation approach.

## Key Improvements

### 1. **Real Judge API Calls** ✅
- Creates actual judge variants with modified rubrics using the Martian API
- Each variant judge has a genuinely different rubric (strict, lenient, bottom_heavy, top_heavy)
- Real evaluation scores from actual judge API calls, not mathematical transformations

### 2. **Proper Parallelization** ✅
- Adapted from `run_full_experiment.py` patterns
- Configurable worker pools (default: 10 parallel workers)
- Efficient handling of thousands of API calls
- Built-in retry logic with exponential backoff

### 3. **Scalability** ✅
- Supports 100+ examples with all judge variants
- Checkpoint system for long-running experiments
- Estimated 44,000 API calls for full experiment (100 examples × 44 combinations × 10 judges)

## Files Structure

```
src/
├── variant_judge_pipeline.py      # Real API calls with Martian SDK
├── experiment_runner_v2.py        # Main experiment orchestrator
├── scoring_criteria_variations.py # Rubric variation generator
├── robustness_metrics.py          # Analysis and metrics
└── scoring_reuse_pipeline.py      # Fallback simulation mode

test_real_api.py                   # Test script for small-scale validation
run_full_experiment.sh              # Full experiment runner (100 examples)
```

## Usage

### 1. Test with Small Scale First

```bash
# Test with simulation (no API calls)
python test_real_api.py --mode simulation

# Test with real API (5 examples, ~500 API calls)
python test_real_api.py --mode real
```

### 2. Run Full Experiment

```bash
# Run full experiment (100 examples, ~44,000 API calls)
./run_full_experiment.sh

# Or with custom parameters
python src/experiment_runner_v2.py \
    --examples 100 \
    --workers 10 \
    --output ../results_full
```

### 3. Quick Mode for Development

```bash
# Quick mode with fewer combinations
python src/experiment_runner_v2.py \
    --examples 20 \
    --workers 5 \
    --quick \
    --output ../results_quick
```

## API Call Estimation

For full experiment (100 examples):
- **Single variations**: 10 judges × 4 variants = 40 combinations
- **Systematic variations**: 4 combinations (all_strict, all_lenient, etc.)
- **Total combinations**: 44
- **API calls per combination**: 100 examples × 10 judges = 1,000
- **Total API calls**: 44 × 1,000 = **44,000 API calls**

## Key Components

### VariantJudgePipeline
- Creates judge variants with modified rubrics dynamically
- Manages judge lifecycle (creation, evaluation, cleanup)
- Handles parallel evaluation with retry logic
- Real API calls using Martian SDK

### RubricSensitivityExperimentV2
- Orchestrates the complete experiment workflow
- Manages parallelization and checkpointing
- Generates comprehensive analysis and visualizations
- Supports both real API and simulation modes

### Parallelization Strategy
- **Row-level concurrency**: Process multiple examples in parallel
- **Judge-level parallelization**: Evaluate all 10 judges for an example simultaneously
- **Configurable workers**: Adjust based on API rate limits
- **Exponential backoff**: Automatic retry on API failures

## Results

The experiment produces:
- **raw_scores.pkl**: All judge scores for all combinations
- **robustness_report.pkl**: Comprehensive analysis results
- **SUMMARY.txt**: High-level experiment summary
- **plots/**: Visualizations of variance and correlation
- **experiment.log**: Detailed execution log

## Important Notes

1. **API Costs**: The full experiment makes ~44,000 API calls. Test with small scale first!

2. **Judge Cleanup**: Created variant judges are tracked but deletion is not implemented in the SDK. You may need to manually clean up test judges.

3. **Rate Limiting**: The default configuration uses 10 parallel workers. Adjust based on your API rate limits.

4. **Checkpointing**: The system saves checkpoints every 5 combinations to allow resuming if interrupted.

## Expected Outcomes

A successful experiment should show:
- **Lower variance** for learned aggregators compared to naive mean
- **High correlation** (>0.85) across rubric variations
- **Meaningful differences** between variant types (strict vs lenient)
- **Robustness** to rubric phrasing changes

## Troubleshooting

1. **API Authentication Failed**
   - Ensure MARTIAN_API_KEY is set in environment
   - Check API credentials are valid

2. **Judge Not Found**
   - Verify original judges are deployed
   - Check judge IDs match those in JUDGE_RUBRICS

3. **Rate Limiting Errors**
   - Reduce max_workers parameter
   - Increase retry delays in variant_judge_pipeline.py

4. **Out of Memory**
   - Process in smaller batches
   - Reduce n_examples parameter