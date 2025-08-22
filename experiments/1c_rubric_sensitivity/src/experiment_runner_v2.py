"""
Rubric Sensitivity Experiment Runner V2 - With Real API Calls

This version uses real Martian API calls to create and evaluate judge variants
with modified rubrics, implementing proper parallelization.
"""

import asyncio
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import experiment components
from variant_judge_pipeline import VariantJudgePipeline, run_variant_experiment
from scoring_criteria_variations import generate_judge_combinations
from robustness_metrics import RobustnessAnalyzer

# Setup logging
def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / "experiment.log"
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class RubricSensitivityExperimentV2:
    """Enhanced rubric sensitivity experiment with real API calls."""
    
    def __init__(
        self,
        data_path: str = "../../dataset/data_with_judge_scores.pkl",
        output_dir: str = "../results_v2",
        n_examples: int = 100,
        max_workers: int = 10,  # Increased parallelization
        quick_mode: bool = False,
        use_real_api: bool = True  # Toggle for real API vs simulation
    ):
        """
        Initialize the experiment.
        
        Args:
            data_path: Path to dataset with judge scores and human feedback
            output_dir: Directory for output files
            n_examples: Number of examples to evaluate
            max_workers: Number of parallel workers for API calls
            quick_mode: Whether to run in quick mode with fewer combinations
            use_real_api: Whether to use real API calls or fall back to simulation
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.n_examples = n_examples
        self.max_workers = max_workers
        self.quick_mode = quick_mode
        self.use_real_api = use_real_api
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.output_dir)
        
        # Log configuration
        self.config = {
            'data_path': str(self.data_path),
            'output_dir': str(self.output_dir),
            'n_examples': n_examples,
            'max_workers': max_workers,
            'quick_mode': quick_mode,
            'use_real_api': use_real_api,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def generate_judge_combinations(self) -> List[Dict]:
        """Generate judge variant combinations for testing."""
        self.logger.info("Generating judge variant combinations")
        
        # Import optimized combinations
        from optimized_combinations import (
            generate_optimized_combinations,
            generate_minimal_combinations,
            calculate_api_calls
        )
        
        if self.quick_mode:
            # Quick mode: minimal 3 combinations for testing
            combinations = generate_minimal_combinations()
            self.logger.info(f"Quick mode: using {len(combinations)} minimal combinations")
        else:
            # Full mode: optimized 5-7 combinations
            combinations = generate_optimized_combinations()
            self.logger.info(f"Full mode: using {len(combinations)} optimized combinations")
        
        # Log combination details
        for combo in combinations:
            self.logger.info(f"  - {combo['name']}: {combo['description']}")
        
        # Calculate and log API call estimates
        stats = calculate_api_calls(len(combinations), self.n_examples)
        self.logger.info(f"Estimated API calls: {stats['total_api_calls']:,}")
        self.logger.info(f"Estimated time: {stats['estimated_time_hours']:.1f} hours")
        
        return combinations
    
    async def run_scoring_phase(self, combinations: List[Dict]) -> pd.DataFrame:
        """
        Run the scoring phase with EFFICIENT score reuse.
        
        Makes API calls only for unique judge-variant pairs (4 × 10 × N),
        then reuses those scores to create all combinations.
        
        Args:
            combinations: List of judge variant combinations
            
        Returns:
            DataFrame with scores for all combinations
        """
        self.logger.info(f"Starting EFFICIENT scoring phase")
        
        # Import the efficient pipeline
        from efficient_scoring_pipeline import EfficientScoringPipeline
        
        # Initialize efficient pipeline
        pipeline = EfficientScoringPipeline(
            data_path=str(self.data_path),
            max_workers=self.max_workers,
            use_real_api=self.use_real_api
        )
        
        # Step 1: Collect all unique variant scores (ONLY place with API calls)
        cache_path = self.output_dir / "variant_scores_cache.pkl"
        
        self.logger.info(f"Step 1: Collecting unique variant scores...")
        self.logger.info(f"API calls needed: 4 variants × 10 judges × {self.n_examples} samples = {4 * 10 * self.n_examples:,}")
        
        variant_scores = await pipeline.collect_all_variant_scores(
            n_examples=self.n_examples,
            cache_path=cache_path
        )
        
        self.logger.info(f"Collected {len(variant_scores)} unique judge-variant score arrays")
        
        # Step 2: Create all combinations by reusing scores (NO API calls)
        self.logger.info(f"Step 2: Creating {len(combinations)} combinations from cached scores (no API calls)...")
        
        results_list = []
        for combo_info in combinations:
            combo_name = combo_info['name']
            combination = combo_info['combination']
            
            self.logger.info(f"  Creating combination: {combo_name}")
            combo_scores = pipeline.create_combination_scores(combination)
            
            # Add metadata columns
            n_rows = len(combo_scores)
            combo_scores.insert(0, 'combination', combo_name)
            combo_scores.insert(0, 'example_idx', range(n_rows))
            
            results_list.append(combo_scores)
        
        # Combine all results
        results = pd.concat(results_list, ignore_index=True)
        
        # Save results
        results_path = self.output_dir / "raw_scores.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Scoring complete, results shape: {results.shape}")
        self.logger.info(f"Total API calls made: {4 * 10 * self.n_examples:,} (not {len(combinations) * 10 * self.n_examples:,}!)")
        
        return results
    
    def analyze_robustness(self, scores_df: pd.DataFrame) -> Dict:
        """
        Analyze robustness of aggregation methods to rubric variations.
        
        Args:
            scores_df: DataFrame with scores for all combinations
            
        Returns:
            Dictionary with robustness analysis results
        """
        self.logger.info("Analyzing robustness to rubric variations")
        
        # Load ground truth data
        ground_truth_df = None
        try:
            with open(self.data_path, 'rb') as f:
                gt_data = pickle.load(f)
            if hasattr(gt_data, 'to_dict'):
                ground_truth_df = gt_data.head(self.n_examples)
        except Exception as e:
            self.logger.warning(f"Could not load ground truth: {e}")
        
        # Initialize analyzer
        analyzer = RobustnessAnalyzer(
            scores_df=scores_df,
            ground_truth_df=ground_truth_df,
            model_type='mlp'
        )
        
        # Generate comprehensive report
        report = analyzer.generate_summary_report()
        
        # Save report
        report_path = str(self.output_dir / "robustness_report.pkl")
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        self.logger.info("Robustness analysis complete")
        
        # Store analyzer for visualization step
        self.analyzer = analyzer
        return report
    
    def generate_summary(self, robustness_report: Dict):
        """Generate and save experiment summary."""
        self.logger.info("Generating experiment summary")
        
        # Extract key metrics from the report structure
        summary = robustness_report.get('summary', {})
        aggregator_data = robustness_report.get('aggregator_robustness', {})
        
        # Get variance metrics
        learned_variance = aggregator_data.get('learned', {}).get('mean_variance', 0.0)
        mean_variance = aggregator_data.get('mean', {}).get('mean_variance', 0.0)
        
        # Get correlation metrics from summary
        overall_mean_correlation = summary.get('overall_mean_correlation', 0.0)
        learned_correlation = overall_mean_correlation  # Use overall as proxy
        mean_correlation = overall_mean_correlation
        
        # Determine success
        success = (learned_variance < mean_variance * 1.5 and 
                  learned_correlation > 0.85)
        
        # Create summary
        summary = f"""RUBRIC SENSITIVITY EXPERIMENT V2 SUMMARY
========================================

CONFIGURATION:
- Examples: {self.n_examples}
- Combinations: {len(robustness_report.get('aggregator_robustness', {}))}
- Real API Calls: {self.use_real_api}
- Parallelization: {self.max_workers} workers

RESULTS:
- Learned Aggregator Variance: {learned_variance:.4f}
- Mean Baseline Variance: {mean_variance:.4f}
- Learned Correlation: {learned_correlation:.4f}
- Mean Correlation: {mean_correlation:.4f}

VERDICT: {'SUCCESS ✅' if success else 'NEEDS IMPROVEMENT ❌'}

KEY FINDINGS:
- Variance Ratio (Learned/Mean): {learned_variance/mean_variance:.2f}x
- Correlation Drop: {(1 - learned_correlation) * 100:.1f}%
- API Calls Made: {4 * 10 * self.n_examples:,} (efficient score reuse)
"""
        
        # Save summary
        with open(self.output_dir / "SUMMARY.txt", 'w') as f:
            f.write(summary)
        
        self.logger.info(summary)
        
        return summary
    
    async def run_experiment(self):
        """Run the complete rubric sensitivity experiment."""
        self.logger.info("="*60)
        self.logger.info("STARTING RUBRIC SENSITIVITY EXPERIMENT V2")
        self.logger.info("="*60)
        
        try:
            # Step 1: Generate judge combinations
            self.logger.info("Step 1: Generating judge variant combinations...")
            combinations = self.generate_judge_combinations()
            
            # Save combinations for reference
            with open(self.output_dir / "combinations.json", 'w') as f:
                json.dump(combinations, f, indent=2)
            
            # Step 2: Score examples with variants
            self.logger.info("Step 2: Scoring examples through judge variants...")
            scores_df = await self.run_scoring_phase(combinations)
            
            # Step 3: Analyze robustness
            self.logger.info("Step 3: Analyzing robustness...")
            robustness_report = self.analyze_robustness(scores_df)
            
            # Step 4: Generate visualizations
            self.logger.info("Step 4: Generating visualizations...")
            try:
                self.analyzer.create_robustness_plots(
                    output_dir=str(self.output_dir / "plots"),
                    report=robustness_report
                )
            except Exception as e:
                self.logger.warning(f"Visualization generation failed: {e}, continuing without plots")
            
            # Step 5: Generate summary
            self.logger.info("Step 5: Generating summary...")
            summary = self.generate_summary(robustness_report)
            
            self.logger.info("="*60)
            self.logger.info("EXPERIMENT COMPLETE")
            self.logger.info("="*60)
            
            return {
                'scores': scores_df,
                'robustness_report': robustness_report,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rubric Sensitivity Experiment V2")
    parser.add_argument('--examples', type=int, default=100,
                       help='Number of examples to evaluate (default: 100)')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel workers (default: 10)')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode with fewer combinations')
    parser.add_argument('--simulate', action='store_true',
                       help='Use simulation instead of real API calls')
    parser.add_argument('--output', default='../results_v2',
                       help='Output directory (default: ../results_v2)')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = RubricSensitivityExperimentV2(
        n_examples=args.examples,
        max_workers=args.workers,
        quick_mode=args.quick,
        use_real_api=not args.simulate,
        output_dir=args.output
    )
    
    # Run experiment
    results = await experiment.run_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())