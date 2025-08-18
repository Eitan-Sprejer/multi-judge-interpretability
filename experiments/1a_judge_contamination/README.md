# Experiment 1A: Judge Contamination

This experiment tests how well the aggregator performs when some judges are deliberately flawed or "contaminated."

## Overview

The goal is to create deliberately flawed judges and test whether the learned aggregator can:

1. Identify unreliable judges
2. Assign near-zero or negative weights to bad judges
3. Maintain performance with <10% degradation vs 20-30% for naive methods

## Contamination Types

### 1. Inverted Scorer

- **Purpose**: Rates bad responses highly (opposite of normal scoring)
- **Implementation**: Inverts the 0-4 scoring scale completely
- **Expected Behavior**: Should receive negative weights from aggregator

### 2. Random Noise Judge

- **Purpose**: Provides random scores with no meaningful signal
- **Implementation**: Generates arbitrary scores regardless of response quality
- **Expected Behavior**: Should receive near-zero weights from aggregator

### 3. Safety-Blind Judge

- **Purpose**: Ignores safety violations and harmful content
- **Implementation**: Focuses only on technical quality, ignores safety concerns
- **Expected Behavior**: Should receive low weights from aggregator

## Usage

### Create Contaminated Judges

```bash
# Create 1 contaminated judge (default)
python run_experiment.py --create-judges

# Create specific number of contaminated judges
python run_experiment.py --create-judges --n-contaminated 2

# Create all types of contaminated judges
python run_experiment.py --create-judges --n-contaminated 3
```

### Run the Experiment

```bash
# Run with 1 contaminated judge
python run_experiment.py --run-experiment --n-contaminated 1

# Run with 2 contaminated judges
python run_experiment.py --run-experiment --n-contaminated 2

# Run with 3 contaminated judges
python run_experiment.py --run-experiment --n-contaminated 3

# Quick test mode
python run_experiment.py --run-experiment --n-contaminated 1 --quick
```

### Run Complete Pipeline

```bash
# Run everything (create judges + experiment)
python run_experiment.py --n-contaminated 2
```

## Expected Results

### Success Criteria

- **Performance Degradation**: <10% vs naive methods
- **Contamination Detection**: >70% accuracy in identifying bad judges
- **Robustness**: >80% performance on clean datasets

### Key Metrics

- **Performance Degradation**: How much performance drops with contaminated judges
- **Contamination Detection Score**: How well the aggregator identifies bad judges
- **Robustness Score**: How well the system performs on clean data after training

## Implementation Details

### Judge Creation

The experiment creates contaminated judges using the Martian API:

- Each contaminated judge gets a unique ID (e.g., `harmlessness-judge-inverted_scorer`)
- Rubrics are modified to implement the specific contamination strategy
- Judges are marked as "DELIBERATELY FLAWED" in their descriptions

### Experiment Flow

1. **Setup**: Create contaminated judges
2. **Baseline**: Test performance with clean judges only
3. **Contamination**: Test performance with mixed clean/contaminated judges
4. **Training**: Train aggregator on contaminated data
5. **Analysis**: Examine learned weights and robustness
6. **Evaluation**: Calculate key metrics and success criteria

### Data Collection

- Judge scores on test queries
- Learned weights from GAM model
- Performance metrics on clean vs contaminated data
- Robustness testing on held-out clean datasets

## Files

- `run_experiment.py`: Main entry point
- `src/inverted_judge_creator.py`: Creates contaminated judges
- `src/contamination_experiment.py`: Runs the main experiment
- `README.md`: This documentation

## Dependencies

- Martian API SDK
- Core pipeline components
- Judge rubrics and evaluation tools
- Aggregator training pipeline

## Notes

- This experiment directly tests the robustness of the learned aggregator
- Results will inform the design of more robust aggregation methods
- The experiment can be scaled by varying the number of contaminated judges
- All contaminated judges are clearly marked to avoid confusion in production
