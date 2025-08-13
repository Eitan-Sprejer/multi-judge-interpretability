# Multi-Judge Interpretability: Collaboration Guide

**For collaborators working on the NeurIPS Interpretability Workshop submission**

## 🏗️ Project Architecture

This project is organized around **two key principles**:
1. **Reusable Pipeline**: Core components that all experiments can use
2. **Self-Contained Experiments**: Individual research questions with their own code and results

```
multi-judge-interpretability/
├── 📦 pipeline/              # Reusable components (DON'T modify lightly)
│   ├── core/                # Main pipeline steps
│   └── utils/               # Shared utilities
├── 🔬 experiments/          # Research experiments (ADD new ones here)
├── 📊 dataset/              # Shared datasets
├── 🤖 models/               # Shared trained models
└── 📚 docs/                 # Documentation
```

## 🔄 Pipeline: The Foundation

The `pipeline/` folder contains **battle-tested, reusable components** that power all experiments:

### Core Components (`pipeline/core/`)

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `judge_creation.py` | Create/manage 10 specialized judges via Martian API | Setting up judges for new datasets |
| `judge_evaluation.py` | Evaluate samples using all judges | Getting judge scores for any dataset |
| `persona_simulation.py` | Simulate 8 diverse human personas | Generating human feedback data |
| `aggregator_training.py` | Train GAM/MLP models to combine judge scores | Training aggregation models |

### Utilities (`pipeline/utils/`)

- `judge_rubrics.py` - Full rubrics for all 10 judges (3600+ chars each)
- `data_merger.py` - Utilities for combining data from different pipeline stages

### 🚨 Pipeline Guidelines

**DO:**
- Import pipeline components in your experiments
- Use the provided APIs and interfaces
- Report bugs or feature requests as GitHub issues

**DON'T:**
- Modify pipeline code without team discussion
- Create duplicate functionality
- Break backwards compatibility

## 🔬 Experiments: Your Research Playground

Each experiment is a **self-contained research project** testing specific hypotheses.

### Experiment Structure

```
experiments/[track]_[name]/
├── README.md                 # What, why, how
├── run_experiment.py         # Main entry point
├── analyze_[topic].py        # Analysis scripts
├── src/                      # Experiment-specific code
│   ├── custom_logic.py      
│   └── utilities.py         
└── results/                  # All outputs
    ├── FINAL_REPORT.md      # Key findings
    ├── figures.png          
    └── data.json            
```

### Naming Convention

Use this format: `[track][letter]_[descriptive_name]`

**Examples:**
- `1a_judge_contamination` - Track 1, Experiment A
- `1b_persona_poisoning` - Track 1, Experiment B  
- `2_ultrafeedback_validation` - Track 2, Main experiment
- `3_moj_comparison` - Track 3, Mixture of Judges comparison

### Research Tracks

Based on our APART proposal:

1. **Track 1: Robustness Analysis** (Priority)
   - Test against contaminated judges and personas
   - Evaluate semantic robustness

2. **Track 2: Ground Truth Validation** (Priority)  
   - Use UltraFeedback multi-dimensional ratings
   - Compare against single-judge baselines

3. **Track 3: Architectural Comparisons** (Secondary)
   - Compare against Mixture of Judges
   - Test judge self-bias

4. **Track 4: Interpretability** (Secondary)
   - Systematic interpretability analysis
   - Feature importance and partial dependence

## 🚀 How to Run Existing Experiments

### Example: Persona Poisoning (1b)

```bash
# Navigate to experiment
cd experiments/1b_persona_poisoning

# Read the README first!
cat README.md

# Run the main experiment
python run_experiment.py

# Generate analysis and figures  
python analyze_with_baselines.py

# Check results
ls results/
cat results/FINAL_REPORT.md
```

### What Each Script Does

- `run_experiment.py` - Runs the core experiment logic
- `analyze_*.py` - Generates figures, comparisons, and analysis
- `src/*.py` - Contains experiment-specific logic
- `results/` - All outputs, figures, and reports

## 🆕 How to Create a New Experiment

### Step 1: Choose Your Research Question

Pick a specific, testable hypothesis. Examples:
- "How robust are aggregators to random noise in judge scores?"
- "Do LLM judges show bias toward responses from the same model family?"
- "Can we detect contaminated training data through performance monitoring?"

### Step 2: Set Up the Experiment Folder

```bash
# Create experiment folder (follow naming convention)
mkdir experiments/1c_noise_robustness

# Create basic structure
cd experiments/1c_noise_robustness
mkdir src results
touch README.md run_experiment.py src/noise_generator.py
```

### Step 3: Write the README

**Template:**
```markdown
# Experiment 1C: Noise Robustness

## Research Question
How do learned aggregators perform when judge scores contain random noise?

## Hypothesis
Aggregators should be more robust to noise than single judges due to averaging effects.

## Method
1. Add Gaussian noise to judge scores at different levels (σ = 0.1, 0.5, 1.0)
2. Train aggregators on noisy data
3. Evaluate on clean test set
4. Compare against single judge baselines

## Expected Results
- Aggregator performance should degrade gracefully
- Should outperform single judges at high noise levels

## How to Run
```bash
python run_experiment.py
python analyze_results.py
```
```

### Step 4: Implement the Experiment

**Import from pipeline:**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import pipeline components
from pipeline.core.aggregator_training import SingleLayerMLP
from pipeline.core.judge_evaluation import JudgeEvaluator
```

**Follow the pattern:**
1. Load data
2. Apply experimental manipulation
3. Train/evaluate models  
4. Save results
5. Generate report

### Step 5: Create Analysis Script

Generate figures comparing:
- Your method vs baselines
- Performance across conditions
- Key metrics (R², MSE, etc.)

Save everything to `results/` folder.

## 📊 Data Management

### Shared Datasets (`dataset/`)

Current datasets:
- `data.pkl` - Base UltraFeedback dataset (255K samples)
- `data_with_judge_scores.pkl` - With 10 judge evaluations (1K samples)  
- `data_with_human_feedback.pickle` - With human persona ratings

### Adding New Datasets

1. **Follow naming convention**: `data_with_[feature].pkl`
2. **Document the schema**: What columns, what format
3. **Add to pipeline if reusable**: Update data_merger.py if needed

### Results Storage

**Always save:**
- Raw data as JSON/pickle
- Figures as PNG (high DPI)
- Final report as Markdown
- Summary metrics

## 🔍 Code Quality Standards

### Imports
```python
# Project structure
sys.path.append(str(Path(__file__).parent.parent.parent))
from pipeline.core.something import Something

# Local modules  
sys.path.append(str(Path(__file__).parent))
from my_module import my_function
```

### Documentation
- **README for every experiment**
- **Docstrings for all functions**
- **Comments for non-obvious logic**
- **Final report with key findings**

### Testing
- **Run your experiment on small data first**
- **Verify results make sense**
- **Check that figures are readable**
- **Ensure reproducibility with fixed seeds**

## 🤝 Collaboration Workflow

### Before Starting Work
1. **Check existing experiments** - avoid duplication
2. **Discuss your idea** - get team input
3. **Plan the experiment** - clear hypothesis and method

### While Working
1. **Commit early and often**
2. **Document as you go**
3. **Ask for help** - pipeline team can assist
4. **Share preliminary results**

### When Finished
1. **Clean up your code**
2. **Write clear final report**
3. **Add to main experiments README**
4. **Present findings to team**

## 🛠️ Common Pitfalls and Solutions

### "Import Error: No module named 'pipeline'"
**Solution:** Check your sys.path setup
```python
sys.path.append(str(Path(__file__).parent.parent.parent))
```

### "My experiment is too slow"
**Solutions:**
- Test on small data first (100-1000 samples)
- Use fewer epochs for quick iteration
- Implement checkpointing for long runs

### "I can't reproduce results"
**Solutions:**
- Set random seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- Document exact package versions
- Save all hyperparameters

### "My figures are ugly"
**Solutions:**
- Use consistent color schemes
- Add proper labels and titles
- Save high DPI: `plt.savefig('plot.png', dpi=150)`
- Check the persona_poisoning example

## 📈 Success Metrics

Your experiment is successful if:
- ✅ **Reproducible**: Others can run your code and get same results
- ✅ **Well-documented**: Clear README and final report  
- ✅ **Scientifically sound**: Good baselines, proper statistics
- ✅ **Relevant**: Addresses key research questions
- ✅ **Clean**: No dead code, clear structure

## 🎯 Example: Learning from 1b_persona_poisoning

This experiment demonstrates all best practices:

**Good structure:**
- Clear README with hypothesis
- Simple run_experiment.py entry point
- Comprehensive analysis with baselines
- Beautiful 6-panel figure
- Detailed final report

**Key insights:**
- Aggregators are 2x more robust than single judges
- Breaking point at 30% vs 25% contamination
- Clear evidence for multi-judge value

**Reusable patterns:**
- Contamination methodology
- Baseline comparison approach
- Figure generation code
- Statistical analysis

---

## 🚀 Quick Start Checklist

For new collaborators:

- [ ] Read this guide
- [ ] Look at `experiments/1b_persona_poisoning/` example
- [ ] Run the example experiment
- [ ] Choose your research question
- [ ] Create experiment folder with proper naming
- [ ] Write README with hypothesis and method
- [ ] Implement and test on small data
- [ ] Generate analysis and figures
- [ ] Write final report
- [ ] Share with team

**Questions?** Ask in our team channel or create a GitHub issue!

---

*This guide will evolve as we add more experiments. Please suggest improvements!*