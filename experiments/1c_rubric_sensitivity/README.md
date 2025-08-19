# Experiment 1C: Rubric Sensitivity Analysis

**Status**: ✅ **COMPLETE** - Ready for NeurIPS paper integration  
**Last Updated**: August 19, 2025  

## Quick Start

**Main Analysis**: Run `python rubric_robustness_analysis.py`  
**Complete Documentation**: See `EXPERIMENT_1C_COMPLETE_PIPELINE.md`  
**Results**: `results_full_20250818_215910/rubric_robustness_analysis.png`  

## Key Finding

Learned aggregation models (GAM/MLP) achieve only **4% better performance** than simple judge averaging but are **3.29x less robust** to rubric variations. This suggests effort is better invested in judge quality than aggregation complexity.

**Performance Results** (Individual Persona Predictions):
- **MLP**: R² = 0.650 ± 0.016 (Best performance, least robust)
- **GAM**: R² = 0.640 ± 0.013  
- **Judge Mean**: R² = 0.636 ± 0.009 (Simple baseline, most robust)
- **Naive Mean**: R² = -0.021 ± 0.000 (Sanity check)

## Directory Structure

```
1c_rubric_sensitivity/
├── rubric_robustness_analysis.py          # Final corrected analysis ✅
├── EXPERIMENT_1C_COMPLETE_PIPELINE.md     # Complete documentation ✅  
├── EXPERIMENT_1C_STATUS.md                # Historical status record
├── run_full_experiment.sh                 # Data collection automation
├── results_full_20250818_215910/          # All data and results ✅
│   ├── restructured_scores_fixed.pkl      # Judge scores (1000×50)
│   ├── rubric_robustness_results.pkl      # Analysis results
│   └── plots_corrected/
│       └── rubric_robustness_analysis.png # Publication figure ✅
├── src/                                    # Utility modules
└── archived_files/                        # Deprecated scripts ⚠️
    ├── correct_aggregator_analysis.py     # Wrong methodology
    ├── rerun_analysis.py                  # Incorrect approach  
    └── [other deprecated files]
```

## Research Contribution

**Novel Finding**: First systematic study showing that simple judge averaging is surprisingly robust compared to learned aggregation models, with practical implications for multi-judge system deployment.

**For NeurIPS Paper**: Demonstrates performance-robustness tradeoff in judge aggregation with realistic individual human preference prediction (R² ~0.65 ceiling due to persona variation).

## Methodology Correction

**Critical Fix Applied**: Changed from using averaged persona scores (artificially smooth) to randomly sampling individual persona scores (realistic noise), making results much more credible and aligned with baseline experiments.

---
**Contact**: Experiment 1C Team | **Paper Deadline**: August 22, 2025