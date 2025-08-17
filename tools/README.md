# Tools Directory

This directory contains utility scripts for the Multi-Judge Interpretability project.

## Available Tools

### Development & Testing Utilities

- **`test_early_stopping.py`** - Validate early stopping improvements and compare against baseline
- **`test_best_config.py`** - Train single optimal configuration with full epochs for production use

## Usage

All tools should be run from the project root directory:

```bash
# Test early stopping effectiveness
python tools/test_early_stopping.py

# Train best configuration for production
python tools/test_best_config.py --config config.json
```

## Main Scripts

For the primary experiment workflows, use the main scripts in the project root:

- **`run_full_experiment.py`** - Complete multi-judge experiment pipeline
- **`hyperparameter_tuning.py`** - Systematic hyperparameter optimization  
- **`rapid_tune.py`** - Quick configuration testing and validation

## Output Structure

All tools write outputs to the organized `results/` directory:

- `results/full_experiments/` - Main experiment outputs
- `results/hyperparameter_search/` - Hyperparameter tuning results
- `results/quick_tests/` - Rapid tuning and utility outputs
- `results/single_configs/` - Single configuration training results