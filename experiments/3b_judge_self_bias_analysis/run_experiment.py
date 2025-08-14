#!/usr/bin/env python3
"""
Judge Self-Bias Analysis Experiment

This experiment investigates whether LLM judges show bias towards responses
from the same model family (e.g., GPT judges favoring GPT responses).

Research Questions:
1. Do judges from a specific model family systematically favor responses from the same family?
2. How does this bias compare across different model families (GPT, Claude, etc.)?
3. What are the implications for judge selection in multi-judge systems?

Usage:
    python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl --quick
    python run_experiment.py --data ../../dataset/data_with_judge_scores.pkl
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add pipeline to path
sys.path.append(str(Path(__file__).parent.parent.parent / "pipeline"))

from src.experiment_runner import JudgeSelfBiasRunner
from src.bias_analyzer import SelfBiasAnalyzer


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Judge Self-Bias Analysis Experiment")
    parser.add_argument("--data", required=True, help="Path to dataset with judge scores")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Config file path")
    parser.add_argument("--quick", action="store_true", help="Run in quick test mode")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Judge Self-Bias Analysis Experiment")
    logger.info(f"Data path: {args.data}")
    logger.info(f"Config path: {args.config}")
    logger.info(f"Quick mode: {args.quick}")
    
    try:
        # Initialize and run experiment
        runner = JudgeSelfBiasRunner(
            data_path=args.data,
            config_path=args.config,
            quick_mode=args.quick
        )
        
        # Run the experiment
        results = runner.run()
        
        # Analyze results for self-bias patterns
        analyzer = SelfBiasAnalyzer(results)
        bias_analysis = analyzer.analyze_self_bias()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{timestamp}_self_bias_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "experiment": "judge_self_bias_analysis",
                "timestamp": timestamp,
                "config": runner.config,
                "results": results,
                "bias_analysis": bias_analysis
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate and save report
        report = analyzer.generate_report()
        report_file = results_dir / f"{timestamp}_self_bias_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
