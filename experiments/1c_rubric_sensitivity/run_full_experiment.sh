#!/bin/bash

# Run Full Rubric Sensitivity Experiment with Real API Calls
# This script runs the complete experiment with 100 examples and all judge variants

echo "============================================================"
echo "RUBRIC SENSITIVITY EXPERIMENT - FULL SCALE"
echo "============================================================"

# Configuration
EXAMPLES=1000  # Increased for better training data
WORKERS=10  # Parallel workers for API calls
OUTPUT_DIR="../results_full_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Configuration:"
echo "  Examples: $EXAMPLES"
echo "  Parallel Workers: $WORKERS"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Estimate API calls (CORRECTED)
# 4 variants × 10 judges × N examples = total API calls
# Combinations are created by REUSING these scores, no additional calls!
API_CALLS=$((4 * 10 * EXAMPLES))
echo "Estimated API Calls: $API_CALLS"
echo "  (4 variants × 10 judges × $EXAMPLES examples)"
echo "  Note: All combinations reuse these scores (no additional calls)"
echo ""

# Confirm before proceeding
read -p "This will make $API_CALLS API calls. Proceed? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Experiment cancelled."
    exit 1
fi

# Run the experiment
echo ""
echo "Starting experiment..."
echo ""

python3 src/experiment_runner_v2.py \
    --examples $EXAMPLES \
    --workers $WORKERS \
    --output "$OUTPUT_DIR" \
    2>&1 | tee experiment_full.log

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "EXPERIMENT COMPLETE!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log saved to: experiment_full.log"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "1. Review results in $OUTPUT_DIR/SUMMARY.txt"
    echo "2. Check visualizations in $OUTPUT_DIR/plots/"
    echo "3. Analyze detailed report in $OUTPUT_DIR/robustness_report.pkl"
else
    echo ""
    echo "============================================================"
    echo "EXPERIMENT FAILED!"
    echo "Check experiment_full.log for details"
    echo "============================================================"
    exit 1
fi