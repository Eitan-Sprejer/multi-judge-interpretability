# Multi-Judge Interpretability: Quick Start Guide

**TL;DR for collaborators who want to get started fast**

## 🏗️ Project Structure

```
multi-judge-interpretability/
├── 📦 pipeline/              # Reusable components (DON'T modify)
│   ├── core/                # judge_creation.py, judge_evaluation.py, etc.
│   └── utils/               # judge_rubrics.py, data_merger.py
├── 🔬 experiments/          # Your research experiments (ADD new ones here)
└── 📊 dataset/              # Shared datasets
```

## 🚀 Run Existing Experiment (2 minutes)

```bash
# Try the working example
cd experiments/1b_persona_poisoning
python run_experiment.py        # Runs experiment
python analyze_with_baselines.py # Generates figures
cat results/FINAL_REPORT.md     # See results
```

**Key Result**: Aggregators are 2x more robust than single judges to contaminated data.

## 🆕 Create New Experiment (10 minutes)

### 1. Set up folder
```bash
mkdir experiments/[exp_number]_your_experiment
cd experiments/[exp_number]_your_experiment
mkdir src results
```

### 2. Create README.md
```markdown
# Experiment 1C: Your Research Question

## Hypothesis
What you're testing and why.

## Method  
1. Step one
2. Step two
3. Compare against baselines

## How to Run
```bash
python run_experiment.py
```

### 3. Write run_experiment.py
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import pipeline components
from pipeline.core.aggregator_training import SingleLayerMLP
from pipeline.core.judge_evaluation import JudgeEvaluator

# Your experiment logic here
def main():
    # 1. Load data
    # 2. Apply experimental manipulation  
    # 3. Train/evaluate models
    # 4. Save results
    pass

if __name__ == "__main__":
    main()
```

## 📊 Pipeline Components You Can Use

| Component | Purpose | Import |
|-----------|---------|--------|
| Judge Creation | Create 10 specialized judges | `from pipeline.core.judge_creation import JudgeCreator` |
| Judge Evaluation | Get scores from all judges | `from pipeline.core.judge_evaluation import JudgeEvaluator` |
| Persona Simulation | Generate human feedback | `from pipeline.core.persona_simulation import PersonaSimulator` |
| Aggregator Training | Train GAM/MLP models | `from pipeline.core.aggregator_training import SingleLayerMLP` |

## 🎯 Research Tracks

1. **Track 1: Robustness** (Priority) - Test against contaminated judges/personas
2. **Track 2: Ground Truth** (Priority) - Use UltraFeedback ratings  
3. **Track 3: Comparisons** (Secondary) - Compare against Mixture of Judges
4. **Track 4: Interpretability** (Secondary) - Feature importance analysis

## 💡 Quick Tips

- **Naming**: Use `[track][letter]_[name]` (e.g., `1c_noise_robustness`)
- **Imports**: Always add `sys.path.append(str(Path(__file__).parent.parent.parent))`
- **Test Small**: Try 100-1000 samples first before full runs
- **Save Everything**: Results as JSON, figures as PNG, reports as MD
- **Set Seeds**: `np.random.seed(42)`, `torch.manual_seed(42)` for reproducibility


## 📈 Success Checklist

- [ ] Clear README with hypothesis
- [ ] Working `run_experiment.py`  
- [ ] Results saved to `results/` folder
- [ ] Comparison against baselines
- [ ] Final report with key findings

## 🤝 Need Help?

1. Look at `experiments/1b_persona_poisoning/` as example
2. Check the full [COLLABORATION_GUIDE.md](COLLABORATION_GUIDE.md) for details
3. Ask team or create GitHub issue
