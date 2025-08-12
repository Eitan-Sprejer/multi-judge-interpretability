# Experiments Directory

This directory contains all research experiments for the Multi-Judge Interpretability project. Each experiment is self-contained with its own source code, configuration, and results.

## Experiment Naming Convention

```
[track]_[id]_[descriptive_name]/
```

Examples:
- `1a_judge_contamination/` - Track 1, Experiment A
- `1b_persona_poisoning/` - Track 1, Experiment B
- `2_ultrafeedback_validation/` - Track 2, Main experiment
- `3_moj_comparison/` - Track 3, Mixture of Judges comparison
- `4_interpretability/` - Track 4, Interpretability analysis

## Standard Experiment Structure

Each experiment follows this structure:

```
experiment_folder/
├── src/                    # Experiment-specific source code
│   ├── main_logic.py      # Core experiment implementation
│   └── utils.py           # Helper functions
├── configs/               # Configuration files
│   └── default_config.yaml
├── results/               # Output directory (auto-created)
│   ├── models/           # Trained models
│   ├── data/             # Generated datasets
│   └── reports/          # Analysis reports
├── run_experiment.py      # Main entry point
└── README.md             # Experiment documentation
```

## Creating a New Experiment

1. **Copy the template**:
   ```bash
   cp -r experiments/experiment_template experiments/[your_experiment_name]
   ```

2. **Update the runner**:
   - Edit `run_experiment.py` with your experiment logic
   - Update the docstring and experiment name

3. **Add source files**:
   - Place experiment-specific code in `src/`
   - Import pipeline components as needed

4. **Configure**:
   - Edit `configs/default_config.yaml`
   - Add experiment-specific parameters

5. **Document**:
   - Update the README.md with experiment details
   - Include expected results and research questions

## Experiment Tracks (from APART Proposal)

### Track 1: Robustness Analysis (Priority)
- **1a_judge_contamination**: Test with deliberately flawed judges
- **1b_persona_poisoning**: Include troll personas in training ✅
- **1c_rubric_sensitivity**: Evaluate semantic robustness

### Track 2: Ground Truth Validation (Priority)
- **2_ultrafeedback_validation**: Use UltraFeedback multi-dimensional ratings

### Track 3: Architectural Comparisons (Secondary)
- **3_moj_comparison**: Compare against Mixture of Judges
- **3b_self_bias**: Test if LLM judges favor same model family

### Track 4: Interpretability Deep Dive (Secondary)
- **4_interpretability**: Systematic interpretability analysis
- **4b_distillation**: Sparse additive model distillation

## Running Experiments

### Quick Test Mode
Most experiments support a `--quick` flag for rapid testing:

```bash
cd experiments/1b_persona_poisoning
python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl --quick
```

### Full Experiment
Run without `--quick` for complete analysis:

```bash
python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl
```

### With Custom Config
Use a specific configuration:

```bash
python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl --config configs/custom.yaml
```

## Accessing Pipeline Components

All experiments can import the core pipeline:

```python
from pipeline.core.aggregator_training import train_model
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.utils.data_merger import DataMerger
```

## Results Organization

Results are saved in each experiment's `results/` directory:

```
results/
├── [timestamp]_results.json    # Raw experimental data
├── experiment_report.md        # Human-readable report
├── models/                     # Trained models
├── plots/                      # Generated visualizations
└── data/                       # Intermediate datasets
```

## Best Practices

1. **Self-Contained**: Each experiment should be runnable independently
2. **Reproducible**: Use fixed random seeds and save configurations
3. **Well-Documented**: Include clear README with research questions
4. **Pipeline Integration**: Reuse core components, don't duplicate
5. **Version Control**: Commit experiment code before running
6. **Results Tracking**: Save all outputs with timestamps

## Current Experiments Status

| Experiment | Status | Priority | Key Finding |
|------------|--------|----------|-------------|
| 1b_persona_poisoning | ✅ Ready | High | Tests robustness to troll feedback |
| 1a_judge_contamination | 🔄 Planned | High | - |
| 2_ultrafeedback_validation | 🔄 Planned | High | - |
| 3_moj_comparison | 📝 Design | Medium | - |
| 4_interpretability | 📝 Design | Low | - |

## Dependencies

All experiments share common dependencies:
- Core pipeline (`pipeline/`)
- Shared datasets (`dataset/`)
- Common models (`models/`)

Experiment-specific dependencies should be documented in each experiment's README.

## Contributing New Experiments

1. Follow the standard structure
2. Document research questions clearly
3. Include expected results
4. Add to the experiments table above
5. Test with `--quick` mode first

## Paper Integration

Results from these experiments will be synthesized for the NeurIPS Interpretability Workshop submission (deadline: August 22, 2025).