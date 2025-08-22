# Experiment 1A: Enhanced Judge Contamination Analysis

## Research Question

How robust are multi-judge aggregation systems to various forms of judge contamination, and what statistical methods can effectively detect and quantify contamination effects?

## Enhanced Overview

This experiment provides a comprehensive framework for testing judge contamination scenarios with advanced statistical analysis, visualization, and publication-ready reporting:

### Core Capabilities
1. **Multiple Contamination Strategies**: Inverted rubrics, noise injection, systematic bias, rubric drift
2. **Advanced Statistical Analysis**: Correlation analysis, significance testing, effect size computation, power analysis
3. **Comprehensive Visualization Suite**: Distribution plots, correlation heatmaps, robustness curves, publication figures
4. **Robustness Testing**: Performance degradation analysis across contamination rates
5. **Publication-Ready Outputs**: LaTeX tables, CSV exports, executive summaries

### Key Innovations
- **Statistical Rigor**: Multiple comparison corrections, confidence intervals, power analysis
- **Automated Insights**: Key finding extraction and recommendation generation
- **Quality Assessment**: Experimental validity scoring and reliability metrics
- **Historical Tracking**: SQLite database for longitudinal analysis
- **Modular Architecture**: Reusable components for other contamination studies

## Prerequisites

- **Python 3.10+**: With scientific computing packages (numpy, pandas, scipy, sklearn, matplotlib)
- **Martian API Access**: For judge creation and management
- **Pipeline Setup**: Core judge creation and evaluation modules
- **Data Access**: Existing judge scores and human feedback data

## How to Run

### Integration Validation (Recommended First Step)

```bash
# Run comprehensive integration tests
python test_integration.py
```

### Quick Test (Development/Testing)

```bash
# Basic functionality test with minimal samples
python run_experiment.py --quick
```

### Standard Analysis

```bash
# Full contamination analysis with robustness testing
python run_experiment.py --num-samples 500 --enable-robustness
```

### Publication-Ready Analysis

```bash
# Complete analysis with high-quality visualizations
python run_experiment.py --generate-publication-figures --num-samples 1000
```

### Advanced Options

```bash
# Multiple contamination types
python run_experiment.py --contamination-types all --enable-robustness

# Custom contamination rates for robustness testing
python run_experiment.py --contamination-rates 0.0 0.05 0.1 0.2 0.3 0.5

# Disable visualizations for faster execution
python run_experiment.py --disable-visualizations --num-samples 200
```

## Contamination Strategies

### Currently Implemented
- **inverted**: Rubrics that score opposite to intended criteria (complete implementation)
- **noise**: Random Gaussian noise injection into judge scoring (framework ready)
- **bias**: Systematic additive or multiplicative scoring bias (framework ready)

### Future Extensions (Framework Ready)
- **drift**: Gradual evolution of rubric interpretation over time
- **adversarial**: Targeted contamination designed to fool specific aggregation methods
- **pattern-based**: Contamination that follows specific behavioral patterns

### Contamination Parameters
- **Strength**: Complete, partial, or random contamination intensity
- **Selection**: All judges, random subset, or specific judge targeting
- **Temporal**: Static or time-varying contamination patterns

## Success Criteria

### Technical Success
- ✅ Successfully create contaminated judges via Martian API
- ✅ Advanced statistical detection of contamination effects
- ✅ Comprehensive robustness analysis across contamination rates
- ✅ Publication-quality visualizations and reporting
- ✅ Integration with existing pipeline architecture

### Research Success
- **Detection Accuracy**: >90% contamination detection rate for strong effects
- **Statistical Power**: >80% power for detecting medium effect sizes
- **Robustness Quantification**: Precise breakdown thresholds and degradation curves
- **Reproducibility**: Fully documented and replicable experimental protocol

## Output Structure

```
results/enhanced_contamination_YYYYMMDD_HHMMSS/
├── data/                              # Raw and processed data
│   ├── combined_results.csv           # All scores and metadata
│   ├── baseline_scores.csv            # Clean judge scores
│   ├── inverted_scores.csv            # Contaminated scores by type
│   └── ...
├── analysis/                          # Statistical analysis results
│   ├── contamination_analysis.json    # Detailed statistical results
│   ├── robustness_analysis.json       # Performance degradation analysis
│   ├── processed_results.json         # Comprehensive processed results
│   └── processed_results.yaml         # Human-readable results
├── plots/                             # Visualizations
│   ├── score_distributions.png        # Before/after contamination
│   ├── correlation_heatmap.png        # Judge correlation analysis
│   ├── robustness_curves.png          # Performance vs contamination rate
│   ├── contamination_dashboard.png    # Comprehensive summary
│   └── publication_figure.png         # Publication-ready figure
├── reports/                           # Generated reports
│   ├── EXPERIMENT_SUMMARY.md          # Executive summary
│   └── ...
├── exports/                           # Publication exports
│   ├── contamination_metrics.csv      # Tabular results
│   └── latex_tables.tex               # LaTeX formatted tables
├── metrics/                           # Performance tracking
│   └── metrics.db                     # SQLite database for trends
└── complete_results.json              # Full experiment archive
```

## Key Metrics Generated

### Contamination Detection
- **Average Correlation**: Judge-to-judge correlation coefficients
- **Contamination Success Rate**: Percentage of successfully contaminated judges
- **Statistical Significance**: p-values and confidence intervals
- **Effect Sizes**: Cohen's d and practical significance measures

### Robustness Analysis
- **Performance Degradation**: R² loss across contamination rates
- **Breakdown Thresholds**: Contamination rate where performance fails
- **Recovery Analysis**: System resilience and adaptation capacity

### Quality Assessment
- **Statistical Power**: Ability to detect true contamination effects
- **Experimental Validity**: Overall reliability of results
- **Data Quality Score**: Sample size adequacy and missing data rates

## Integration with Research Pipeline

This enhanced experiment integrates seamlessly with the broader Multi-Judge Interpretability framework:

- **Shared Data Formats**: Compatible with existing pipeline data structures
- **Consistent APIs**: Uses established judge creation and evaluation interfaces
- **Unified Reporting**: Results format compatible with other experiments
- **Extensible Architecture**: Easy to add new contamination strategies
- **Publication Pipeline**: Direct export to paper-ready formats

## Advanced Features

### Configuration Management
- **YAML Configuration**: Comprehensive parameter control via `config.yaml`
- **Environment Profiles**: Development, testing, and production configurations
- **Command-Line Overrides**: Runtime parameter adjustment

### Quality Assurance
- **Automated Validation**: Input data quality checks and constraint verification
- **Statistical Assumptions**: Automatic testing of analysis prerequisites  
- **Reproducibility**: Complete environment and parameter logging

### Performance Optimization
- **Batch Processing**: Efficient handling of large sample sizes
- **Parallel Execution**: Multi-threaded judge evaluation
- **Memory Management**: Optimized for large-scale experiments

### Extensibility
- **Plugin Architecture**: Easy addition of new analysis modules
- **Custom Visualizations**: Framework for domain-specific plots
- **Export Formats**: Multiple output formats for different use cases

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Run integration test to diagnose
   python test_integration.py
   ```

2. **Missing Data Files**
   - Check data source paths in `config.yaml`
   - Ensure required columns exist in dataset
   - Verify pickle file compatibility

3. **API Connection Issues**
   - Verify Martian API credentials
   - Check network connectivity
   - Review rate limiting settings

4. **Memory Issues**
   - Reduce `num_samples` parameter
   - Use `--quick` mode for testing
   - Increase system memory or use batch processing

### Performance Tips

- Use `--quick` mode during development
- Enable robustness analysis only when needed
- Optimize sample sizes based on available compute
- Use configuration profiles for different scenarios

### Getting Help

- Review `EXPERIMENT_SUMMARY.md` in results for insights
- Check `experiment.log` for detailed execution logs
- Run integration tests to validate setup
- Examine configuration options with `--help`

## Citation

If you use this enhanced judge contamination framework in your research, please cite:

```bibtex
@software{enhanced_judge_contamination,
  title={Enhanced Judge Contamination Analysis Framework},
  author={Multi-Judge Interpretability Research Team},
  year={2024},
  url={https://github.com/your-repo/multi-judge-interpretability},
  note={Experiment 1A: Enhanced Judge Contamination Analysis}
}
```