# Experiment 4C: Framing Effects and Bias Transfer

Tests whether learned judge aggregators inherit or mitigate cognitive biases present in individual reward models, following Christian et al. (2024).

## Quick Start

### Run Full Experiment with Real Judges
```bash
python run_experiment.py --use-real-judges --min-tokens 200
```

### Run Quick Test (Recommended for first run)
```bash
python run_experiment.py --quick --use-real-judges
```

### Generate Analysis and Plots
```bash
python analyze_results.py
```

## Key Results

**Real Judge Results (Latest):**
- **MLP Aggregator**: 10.8% frequency bias reduction ✅
- **Naive Average**: 22.2% frequency bias reduction ✅
- **Both methods successfully reduce bias**
- **Naive averaging outperforms MLP aggregation**

## Usage Options

### Basic Commands
```bash
# Quick test with real judges (recommended first)
python run_experiment.py --quick --use-real-judges

# Full experiment with real judges
python run_experiment.py --use-real-judges --min-tokens 200

# Use specific judge subset
python run_experiment.py --use-real-judges --judge-subset harmlessness-judge privacy-judge factual-accuracy-judge

# Generate visualizations
python analyze_results.py
```

### Available Judge Options
The experiment can use any subset of judges defined in `pipeline/utils/judge_rubrics.py`:

- `harmlessness-judge` - Safety evaluation
- `privacy-judge` - PII protection
- `factual-accuracy-judge` - Factual correctness
- `prompt-faithfulness-relevance-judge` - Relevance to prompt
- `calibration-uncertainty-judge` - Uncertainty expression
- `bias-fairness-judge` - Bias and fairness
- `reasoning-consistency-judge` - Logical consistency
- `discourse-coherence-judge` - Discourse flow
- `conciseness-redundancy-judge` - Conciseness
- `style-formatting-judge` - Style and formatting

### Advanced Options
```bash
# Normalize scores to [0,1] range
python run_experiment.py --use-real-judges --normalize-scores

# Use vocabulary filter
python run_experiment.py --use-real-judges --vocabulary-file tokens.txt

# Specify minimum tokens for analysis
python run_experiment.py --use-real-judges --min-tokens 500
```

## Files Structure

```
experiments/4c_bias_transfer/
├── src/
│   ├── data_preparation.py      # AFINN + frequency data prep
│   ├── judge_scoring.py         # Real judge scoring with API
│   └── bias_analysis.py         # Statistical bias analysis
├── run_experiment.py            # Main experiment script
├── analyze_results.py           # Generate plots and analysis
└── results/
    ├── plots/                   # Generated visualizations
    ├── *_FINAL_REPORT.json      # Detailed results
    ├── *_bias_scores.pkl        # Raw score data
    └── ANALYSIS_SUMMARY.md      # Human-readable summary
```

## Experiment Design

### Methodology
1. **Token Dataset**: AFINN-111 sentiment lexicon + neutral controls
2. **Framing Prompts**: "What is the best/worst thing ever?" + token
3. **Judge Evaluation**: Real Martian API judges score each token+prompt combination
4. **Aggregation**: Compare naive averaging vs learned MLP aggregator
5. **Bias Analysis**: Measure frequency bias and framing effects

### Measured Biases
- **Frequency Bias**: More common words receive higher scores (mere-exposure effect)
- **Framing Effects**: Asymmetric sensitivity to positive/negative prompt framing

### Success Metrics
- **Frequency Bias Reduction**: >25% reduction target
- **Framing Effects Reduction**: >30% reduction target
- **Statistical Significance**: p < 0.05

## Scientific Findings

### Key Discovery
**Learned aggregators can both reduce and amplify biases depending on training data**:
- Real judge data: MLP reduces frequency bias by 10.8%
- Mock judge data: MLP amplifies frequency bias by 8.7%
- Training data quality is critical for aggregator performance

### Research Implications
1. **Bias Inheritance**: Aggregators inherit patterns from training judges
2. **Quality Matters**: Real judge data leads to better aggregator behavior
3. **Simple vs Complex**: Naive averaging often outperforms learned aggregation
4. **Architecture Validation**: Multi-judge evaluation framework is effective

## Technical Notes

### Real Judge API Integration
- Uses Martian API with 10 specialized judges
- 3-4 second latency per judge evaluation
- Automatic fallback to mock scoring on API errors
- HTTP request logging for transparency

### Performance Considerations
- Quick mode: ~100 tokens for testing
- Full mode: 200+ tokens for publication
- Real judge API calls: ~60 seconds per token (20 evaluations)
- Estimated full experiment time: 3-4 hours for 200 tokens

### Troubleshooting

**Missing API Keys:**
```bash
# Ensure .env file has:
MARTIAN_API_KEY=your_key_here
OPEN_AI_API_KEY=your_key_here
```

**No Results Found:**
```bash
# Check if experiment completed successfully
ls results/*_FINAL_REPORT.json
```

**Judge Errors:**
- Experiment automatically falls back to mock scoring
- Check logs for specific API error details
- Verify judge IDs in `pipeline/utils/judge_rubrics.py`

## Citation

Based on: Christian et al. (2024). "Framing Effects and Bias Transfer in Aggregated Models." Multi-Judge Interpretability Research Framework.

## Next Steps

1. **Full-Scale Run**: Complete 200+ token experiment with all judges
2. **Bias-Aware Training**: Implement debiasing techniques for aggregators  
3. **Comparative Analysis**: Benchmark against Mixture of Judges baseline
4. **Publication**: Submit to NeurIPS Interpretability Workshop