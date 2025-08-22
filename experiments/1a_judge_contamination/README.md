# Experiment 1A: Judge Contamination

## Hypothesis

Judges with intentionally inverted or corrupted rubrics will produce systematically biased scores that can be detected and measured. This tests the robustness of aggregation methods to judge contamination scenarios.

## Overview

This experiment creates "poisoned" judges with inverted rubrics to simulate real-world judge contamination scenarios:

1. **Inverted Rubric Judges**: Judges that score opposite to their intended criteria
2. **Contamination Detection**: Measure how well aggregation methods handle contaminated judges
3. **Robustness Testing**: Compare learned aggregators vs baselines under contamination

## Prerequisites

- **Martian API Access**: For judge creation and management
- **Pipeline Setup**: Core judge creation and evaluation modules
- **Existing Judges**: Base judges to contaminate

## How to Run

### Full Experiment

```bash
python run_experiment.py --contamination-type inverted
```

### Quick Test

```bash
python run_experiment.py --quick --contamination-type inverted
```

### Multiple Contamination Types

```bash
python run_experiment.py --contamination-type all --num-samples 100
```

## Contamination Types

- **inverted**: Rubrics that score opposite to intended criteria
- **noise**: Random scoring variations
- **bias**: Systematic scoring bias in one direction

## Success Criteria

- Successfully create contaminated judges via Martian API
- Detect contamination through score analysis
- Demonstrate aggregator robustness (or lack thereof)
- Generate contamination detection metrics

## Expected Results

- Contaminated judge performance metrics
- Score distribution analysis showing contamination effects
- Aggregator robustness assessment under contamination
- Recommendations for contamination detection methods
