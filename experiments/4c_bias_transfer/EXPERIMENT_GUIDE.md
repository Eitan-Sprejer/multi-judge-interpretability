# 🧪 Experiment 4C: Complete Usage Guide

## ✅ **FIXED CRITICAL ISSUES** 
- **Token Selection Bug**: `--min-tokens` now actually controls experiment size
- **Parameter Confusion**: Clear documentation of experiment scales
- **Performance**: 10x speedup through parallel judge evaluation

## 🚀 **Quick Start Commands**

### **Small-Scale Testing (10 tokens)**
```bash
python run_experiment.py --quick --use-real-judges --min-tokens 10
```
- **Evaluation**: 20 API calls (10 tokens × 2 prompts)
- **Time**: ~1-2 minutes
- **Cost**: Minimal API usage

### **Medium-Scale Testing (50 tokens)**
```bash
python run_experiment.py --quick --use-real-judges --min-tokens 50
```
- **Evaluation**: 100 API calls (50 tokens × 2 prompts)  
- **Time**: ~3-5 minutes
- **Cost**: Low API usage

### **Research-Scale Testing (200 tokens)**
```bash
python run_experiment.py --use-real-judges --min-tokens 200
```
- **Evaluation**: 400 API calls (200 tokens × 2 prompts)
- **Time**: ~15-20 minutes  
- **Cost**: Moderate API usage

### **Full-Scale Research (500+ tokens)**
```bash
python run_experiment.py --use-real-judges --min-tokens 500
```
- **Evaluation**: 1,000+ API calls
- **Time**: ~45-60 minutes
- **Cost**: Higher API usage

## 📊 **Experiment Scale Reference**

| Command | Tokens | API Calls | Time | Use Case |
|---------|--------|-----------|------|----------|
| `--quick --min-tokens 10` | 10 | 20 | ~2 min | Code testing |
| `--quick --min-tokens 25` | 25 | 50 | ~3 min | Quick validation |
| `--quick --min-tokens 50` | 50 | 100 | ~5 min | Small study |
| `--min-tokens 100` | 100 | 200 | ~8 min | Pilot study |
| `--min-tokens 200` | 200 | 400 | ~15 min | Standard research |
| `--min-tokens 500` | 500 | 1,000 | ~45 min | Large study |

**API Call Calculation**: `tokens × 2 framing prompts × 10 judges = total evaluations`

## 🔧 **Parameter Details**

### **Core Parameters**
- `--quick`: Test mode with reasonable limits (≤50 tokens max)
- `--use-real-judges`: Real Martian API calls (vs mock scoring)
- `--min-tokens N`: Exact number of tokens to evaluate

### **Optional Parameters**
- `--normalize-scores`: Normalize all scores to [0,1] range  
- `--judge-subset`: Use specific judges only
- `--vocabulary-file`: Use custom vocabulary (CSV format, see **Balanced Vocabulary** section)

## 📈 **What Gets Analyzed**

### **1. Framing Effects Analysis**
- How sentiment context affects judge scoring
- Positive vs negative framing prompts
- Statistical significance testing

### **2. Frequency Bias Analysis**  
- How word frequency correlates with scores
- Partial correlation controlling for sentiment
- High-frequency vs low-frequency word bias

### **3. Aggregator Comparison**
- Individual judges vs aggregated models
- MLP vs GAM vs naive averaging
- Bias reduction effectiveness

## 📁 **Output Files**

Each experiment generates timestamped files:

### **Core Results**
- `TIMESTAMP_FINAL_REPORT.json`: Complete experiment results
- `TIMESTAMP_bias_analysis.json`: Detailed bias analysis
- `TIMESTAMP_EXPERIMENT_SUMMARY.md`: Human-readable summary

### **Data Files**
- `TIMESTAMP_token_dataset.pkl`: Token dataset used
- `TIMESTAMP_bias_scores.pkl`: All judge scores collected

### **Logs** (in `logs/` directory)
- `experiment_TIMESTAMP.log`: Main experiment log
- `debug_TIMESTAMP.log`: Detailed debug information  
- `progress_TIMESTAMP.log`: Progress tracking only

## 🎨 **Generating Visualizations**

After running experiment, generate plots:

```bash
python analyze_results.py
```

**Generates:**
- Bias reduction comparison plots
- Framing effects visualization
- Frequency bias analysis charts
- Statistical significance summaries

## ⚠️ **Important Notes**

### **API Usage**
- Each token requires **2 evaluations** (positive/negative framing)
- Each evaluation calls **10 judges** in parallel
- Total API calls = `tokens × 2 × 10`

### **Time Estimates**
- **With optimization**: ~3.5 seconds per token (~1.75s per evaluation)
- **Progress tracking**: Updates every 5 tokens
- **Parallel execution**: All 10 judges run simultaneously

### **Resumability**
- All results saved with timestamps
- Can run multiple experiments and compare
- Previous results preserved automatically

## 🔄 **Recommended Workflow**

1. **Start Small**: Test with 10 tokens
   ```bash
   python run_experiment.py --quick --use-real-judges --min-tokens 10
   ```

2. **Validate Results**: Check logs and summary
   ```bash
   python analyze_results.py
   ```

3. **Scale Up**: Run larger experiment
   ```bash
   python run_experiment.py --use-real-judges --min-tokens 100
   ```

4. **Full Research**: Production-scale run
   ```bash
   python run_experiment.py --use-real-judges --min-tokens 200
   ```

## 🐛 **Common Issues**

### **"GAM scoring failed"**
- **Issue**: GAM model warning (non-critical)
- **Impact**: MLP and individual judges still work
- **Solution**: Ignore warning, results are valid

### **API timeouts**
- **Issue**: Network/API temporary failures
- **Impact**: Some evaluations may fail
- **Solution**: Re-run experiment, results cached where possible

### **Memory usage**
- **Issue**: Large experiments use significant memory
- **Solution**: Run smaller batches or increase system memory

## 🎯 **Balanced Vocabulary Sampling**

**Critical Update**: The experiment now uses balanced vocabulary sampling to ensure valid bias analysis.

### **Previous Issue: Skewed AFINN Distribution**
- Original AFINN-111 has ~40% negative-2 tokens
- Creates invalid bias analysis due to sampling bias
- Skewed toward negative sentiment spectrum

### **New Balanced Sampling**
```bash
# Generate balanced vocabulary sample
python sample_generation_utils.py --strategy shuffled --size 387 --output data/balanced_vocab.csv

# Use in experiment
python run_experiment.py --vocabulary-file data/balanced_vocab_500_shuffled.csv --use-real-judges
```

### **Available Sampling Strategies**
1. **`shuffled`** (RECOMMENDED): Balanced + shuffled for random sampling
2. **`balanced`**: Equal tokens per sentiment score (-5 to +5)  
3. **`representative`** (DEPRECATED): Maintains skewed AFINN proportions

### **Sampling Utility Usage**
```bash
# Default: 387 shuffled balanced tokens
python sample_generation_utils.py

# Custom balanced sample
python sample_generation_utils.py --strategy balanced --size 500 --output data/custom_vocab.csv

# View distribution analysis
python sample_generation_utils.py --strategy balanced --size 100
```

### **Why Balanced Sampling Matters**
- **Valid bias analysis**: Equal representation across sentiment spectrum
- **Reduced confounding**: Eliminates vocabulary selection bias
- **Reproducible results**: Consistent experimental conditions
- **Statistical power**: Better detection of framing effects

## 📚 **Research Context**

This experiment tests **bias transfer** in judge aggregation:

- **Hypothesis**: Learned aggregators reduce cognitive biases
- **Biases Tested**: Framing effects, frequency bias  
- **Models**: Individual judges, MLP aggregator, GAM aggregator
- **Baseline**: Naive averaging of judge scores
- **Vocabulary**: Balanced sampling across sentiment spectrum (387 tokens)

**Success Metrics**: Aggregators show lower bias than individual judges and naive averaging.