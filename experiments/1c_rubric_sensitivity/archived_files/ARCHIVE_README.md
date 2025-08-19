# Archived Files - Experiment 1C

**Archive Date**: August 19, 2025  
**Reason**: Cleanup after methodology correction and completion of analysis

## Archived Files

### Deprecated Analysis Scripts ❌
- `correct_aggregator_analysis.py` - Used wrong methodology (single pre-trained model testing)
- `rerun_analysis.py` - Based on incorrect approach  
- `fixed_analysis.py` - Intermediate attempt with flawed assumptions
- `generate_aggregator_comparison.py` - Incorrect aggregation methodology
- `fix_data_restructuring.py` - Data processing utility, superseded

### Superseded Documentation 📝
- `README.md` - Original documentation, replaced by complete pipeline docs
- `README_V2.md` - Second iteration, superseded  
- `EFFICIENT_APPROACH.md` - Planning document, no longer needed

### Test Results 🧪  
- `results_test_fixes/` - Experimental test outputs, superseded by final results

## What Was Wrong

**Critical Methodology Error**: Original scripts used single pre-trained model tested on different input combinations, which measures input sensitivity rather than training robustness.

**Correct Methodology**: Train separate models for each rubric combination and compare R² stability across training variations.

## Current Active Files

- `rubric_robustness_analysis.py` - Corrected analysis with proper methodology ✅
- `EXPERIMENT_1C_COMPLETE_PIPELINE.md` - Complete documentation ✅
- `results_full_20250818_215910/` - All data and final results ✅

These archived files are kept for reference but should not be used for research or publication.