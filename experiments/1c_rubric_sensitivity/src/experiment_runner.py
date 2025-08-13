"""
Rubric Sensitivity Experiment Runner

Main orchestrator for the rubric sensitivity experiment.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .judge_variant_creator import JudgeVariantCreator
from .scoring_pipeline import MultiRubricScoringPipeline
from .robustness_metrics import RobustnessAnalyzer
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RubricSensitivityExperiment:
    """Main experiment runner for rubric sensitivity testing."""
    
    def __init__(
        self,
        data_path: str,
        output_dir: Path,
        model_path: Optional[str] = None,
        n_variations: int = 3,
        quick_mode: bool = False,
        config_path: Optional[str] = None
    ):
        """
        Initialize the experiment.
        
        Args:
            data_path: Path to dataset with judge scores
            output_dir: Directory for output files
            model_path: Path to trained aggregation model
            n_variations: Number of rubric variations per judge
            quick_mode: Whether to run in quick test mode
            config_path: Optional config file path
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.n_variations = n_variations
        self.quick_mode = quick_mode
        self.config_path = config_path
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.judge_creator = JudgeVariantCreator(config_path)
        self.scoring_pipeline = None
        self.analyzer = None
        
        # Results storage
        self.results = {}
        
        # Set up logging to file
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging to file."""
        log_file = self.output_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    def run(self) -> Dict:
        """
        Run the complete rubric sensitivity experiment.
        
        Returns:
            Dictionary with experiment results
        """
        logger.info("="*60)
        logger.info("STARTING RUBRIC SENSITIVITY EXPERIMENT")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Create judge variants
            logger.info("Step 1: Creating judge variants...")
            judge_variants = self._create_judge_variants()
            
            # Step 2: Score examples through all variants
            logger.info("Step 2: Scoring examples through judge variants...")
            scores_df = self._score_examples(judge_variants)
            
            # Step 3: Analyze robustness
            logger.info("Step 3: Analyzing robustness...")
            robustness_report = self._analyze_robustness(scores_df)
            
            # Step 4: Generate final results
            logger.info("Step 4: Generating final results...")
            final_results = self._generate_final_results(robustness_report)
            
            # Step 5: Save all outputs
            logger.info("Step 5: Saving results...")
            self._save_results(scores_df, robustness_report, final_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Experiment completed successfully in {duration}")
            logger.info("="*60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
    
    def _create_judge_variants(self) -> Dict[str, object]:
        """Create all judge variants."""
        # Determine which judges to use
        if self.quick_mode:
            # Test with just 2 judges in quick mode
            judge_ids = ['harmlessness-judge', 'factual-accuracy-judge']
            variation_types = ['original', 'formal', 'casual']
        else:
            # Use all judges
            judge_ids = None  # Will use all available
            variation_types = ['original', 'formal', 'casual', 'restructured']
        
        logger.info(f"Creating variants for {len(judge_ids) if judge_ids else 'all'} judges")
        logger.info(f"Variation types: {variation_types}")
        
        # Create variants
        variants = self.judge_creator.create_all_judge_variants(
            judge_ids=judge_ids,
            variation_types=variation_types
        )
        
        # Save variant info
        variant_info = {
            'judge_ids': judge_ids or list(JUDGE_RUBRICS.keys()),
            'variation_types': variation_types,
            'total_variants': len(variants),
            'variant_list': list(variants.keys())
        }
        
        with open(self.output_dir / "judge_variants.json", 'w') as f:
            json.dump(variant_info, f, indent=2)
        
        logger.info(f"Created {len(variants)} judge variants")
        return variants
    
    def _score_examples(self, judge_variants: Dict[str, object]) -> pd.DataFrame:
        """Score examples through all judge variants."""
        # Initialize scoring pipeline
        self.scoring_pipeline = MultiRubricScoringPipeline(
            data_path=self.data_path,
            config_path=self.config_path,
            batch_size=5 if self.quick_mode else 10
        )
        
        # Determine number of examples
        if self.quick_mode:
            n_examples = 100
        else:
            n_examples = 1000  # Default for full experiment
        
        logger.info(f"Scoring {n_examples} examples through {len(judge_variants)} variants")
        
        # Score examples
        scores_df = self.scoring_pipeline.score_examples(
            judge_variants=judge_variants,
            n_examples=n_examples,
            save_checkpoint=True,
            checkpoint_interval=50 if self.quick_mode else 100
        )
        
        # Save raw scores
        scores_path = self.output_dir / "raw_scores.pkl"
        with open(scores_path, 'wb') as f:
            pickle.dump(scores_df, f)
        
        # Also save as CSV for easy inspection
        csv_path = self.output_dir / "raw_scores.csv"
        scores_df.to_csv(csv_path, index=False)
        
        logger.info(f"Scored {len(scores_df)} examples, saved to {scores_path}")
        return scores_df
    
    def _analyze_robustness(self, scores_df: pd.DataFrame) -> Dict:
        """Analyze robustness of aggregation methods."""
        # Initialize analyzer
        self.analyzer = RobustnessAnalyzer(
            scores_df=scores_df,
            model_path=self.model_path
        )
        
        # Generate comprehensive report
        robustness_report = self.analyzer.generate_summary_report()
        
        # Save detailed report
        report_path = self.output_dir / "robustness_report.pkl"
        self.analyzer.save_report(report_path, robustness_report)
        
        logger.info(f"Generated robustness report, saved to {report_path}")
        return robustness_report
    
    def _generate_final_results(self, robustness_report: Dict) -> Dict:
        """Generate final experiment results."""
        summary = robustness_report.get('summary', {})
        aggregator_metrics = robustness_report.get('aggregator_robustness', {})
        
        # Extract key metrics
        learned_variance = np.nan
        baseline_variance = np.nan
        learned_correlation = np.nan
        baseline_correlation = np.nan
        
        if 'learned' in aggregator_metrics:
            learned_variance = aggregator_metrics['learned'].get('mean_variance', np.nan)
            if 'correlations_with_original' in aggregator_metrics['learned']:
                corr_vals = [c['r'] for c in aggregator_metrics['learned']['correlations_with_original'].values()]
                learned_correlation = np.mean(corr_vals) if corr_vals else np.nan
        
        if 'mean' in aggregator_metrics:
            baseline_variance = aggregator_metrics['mean'].get('mean_variance', np.nan)
            if 'correlations_with_original' in aggregator_metrics['mean']:
                corr_vals = [c['r'] for c in aggregator_metrics['mean']['correlations_with_original'].values()]
                baseline_correlation = np.mean(corr_vals) if corr_vals else np.nan
        
        # Calculate improvement factor
        improvement_factor = np.nan
        if not np.isnan(baseline_variance) and baseline_variance > 0 and not np.isnan(learned_variance):
            improvement_factor = baseline_variance / learned_variance
        
        # Determine success criteria
        variance_success = learned_variance < 0.05 if not np.isnan(learned_variance) else False
        correlation_success = learned_correlation > 0.95 if not np.isnan(learned_correlation) else False
        
        final_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'quick_mode': self.quick_mode,
                'n_examples': summary.get('n_examples', 0),
                'n_judges': summary.get('n_judges_tested', 0),
                'model_used': self.model_path is not None
            },
            'key_metrics': {
                'learned_variance': learned_variance,
                'baseline_variance': baseline_variance,
                'improvement_factor': improvement_factor,
                'learned_correlation': learned_correlation,
                'baseline_correlation': baseline_correlation
            },
            'success_criteria': {
                'variance_below_5_percent': variance_success,
                'correlation_above_95_percent': correlation_success,
                'overall_success': variance_success and correlation_success
            },
            'detailed_metrics': {
                'overall_mean_variance': summary.get('overall_mean_variance', np.nan),
                'overall_max_variance': summary.get('overall_max_variance', np.nan),
                'variance_below_5_percent_rate': summary.get('variance_below_5_percent', 0),
                'overall_mean_correlation': summary.get('overall_mean_correlation', np.nan),
                'correlation_above_95_percent_rate': summary.get('correlation_above_95_percent', 0)
            }
        }
        
        return final_results
    
    def _save_results(
        self,
        scores_df: pd.DataFrame,
        robustness_report: Dict,
        final_results: Dict
    ):
        """Save all experiment results."""
        # Save final results as JSON
        results_path = self.output_dir / "experiment_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            results_json = self._convert_for_json(final_results)
            json.dump(results_json, f, indent=2)
        
        # Generate human-readable report
        self._generate_markdown_report(final_results, robustness_report)
        
        # Create summary file
        self._create_summary_file(final_results)
        
        logger.info("All results saved successfully")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _generate_markdown_report(self, final_results: Dict, robustness_report: Dict):
        """Generate human-readable markdown report."""
        report_path = self.output_dir / "experiment_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Rubric Sensitivity Experiment Report\n\n")
            
            # Experiment info
            info = final_results['experiment_info']
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Timestamp**: {info['timestamp']}\n")
            f.write(f"- **Quick Mode**: {info['quick_mode']}\n")
            f.write(f"- **Examples Tested**: {info['n_examples']}\n")
            f.write(f"- **Judges Tested**: {info['n_judges']}\n")
            f.write(f"- **Model Used**: {info['model_used']}\n\n")
            
            # Key results
            metrics = final_results['key_metrics']
            success = final_results['success_criteria']
            
            f.write("## Key Results\n\n")
            f.write("### Success Criteria\n\n")
            variance_icon = "✅" if success['variance_below_5_percent'] else "❌"
            correlation_icon = "✅" if success['correlation_above_95_percent'] else "❌"
            overall_icon = "✅" if success['overall_success'] else "❌"
            
            f.write(f"{variance_icon} **Variance <5%**: {success['variance_below_5_percent']}\n")
            f.write(f"{correlation_icon} **Correlation >95%**: {success['correlation_above_95_percent']}\n")
            f.write(f"{overall_icon} **Overall Success**: {success['overall_success']}\n\n")
            
            f.write("### Quantitative Results\n\n")
            f.write(f"- **Learned Aggregator Variance**: {metrics['learned_variance']:.4f}\n")
            f.write(f"- **Baseline Variance**: {metrics['baseline_variance']:.4f}\n")
            f.write(f"- **Improvement Factor**: {metrics['improvement_factor']:.2f}x\n")
            f.write(f"- **Learned Correlation**: {metrics['learned_correlation']:.4f}\n")
            f.write(f"- **Baseline Correlation**: {metrics['baseline_correlation']:.4f}\n\n")
            
            # Detailed metrics
            detailed = final_results['detailed_metrics']
            f.write("## Detailed Analysis\n\n")
            f.write(f"- **Overall Mean Variance**: {detailed['overall_mean_variance']:.4f}\n")
            f.write(f"- **Overall Max Variance**: {detailed['overall_max_variance']:.4f}\n")
            f.write(f"- **Rate with <5% Variance**: {detailed['variance_below_5_percent_rate']*100:.1f}%\n")
            f.write(f"- **Overall Mean Correlation**: {detailed['overall_mean_correlation']:.4f}\n")
            f.write(f"- **Rate with >95% Correlation**: {detailed['correlation_above_95_percent_rate']*100:.1f}%\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            if success['overall_success']:
                f.write("✅ **SUCCESS**: The learned aggregator demonstrates robust performance across rubric variations.\n\n")
            else:
                f.write("❌ **NEEDS IMPROVEMENT**: The aggregator shows sensitivity to rubric phrasing.\n\n")
            
            if success['variance_below_5_percent']:
                f.write("- Low variance indicates stable scoring across equivalent rubrics\n")
            else:
                f.write("- High variance suggests sensitivity to rubric phrasing\n")
            
            if success['correlation_above_95_percent']:
                f.write("- High correlation shows consistent relative rankings\n")
            else:
                f.write("- Lower correlation indicates ranking instability\n")
        
        logger.info(f"Generated markdown report: {report_path}")
    
    def _create_summary_file(self, final_results: Dict):
        """Create a simple summary file for quick reference."""
        summary_path = self.output_dir / "SUMMARY.txt"
        
        success = final_results['success_criteria']
        metrics = final_results['key_metrics']
        
        with open(summary_path, 'w') as f:
            f.write("RUBRIC SENSITIVITY EXPERIMENT SUMMARY\n")
            f.write("="*40 + "\n\n")
            
            if success['overall_success']:
                f.write("RESULT: SUCCESS ✅\n\n")
            else:
                f.write("RESULT: NEEDS IMPROVEMENT ❌\n\n")
            
            f.write(f"Learned Aggregator Variance: {metrics['learned_variance']:.4f}\n")
            f.write(f"Baseline Variance: {metrics['baseline_variance']:.4f}\n")
            f.write(f"Improvement: {metrics['improvement_factor']:.2f}x better\n")
            f.write(f"Correlation: {metrics['learned_correlation']:.4f}\n")
        
        logger.info(f"Created summary file: {summary_path}")


def main():
    """Main entry point for the experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run rubric sensitivity experiment")
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--model', help='Path to trained aggregation model')
    parser.add_argument('--quick', action='store_true', help='Run in quick test mode')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = RubricSensitivityExperiment(
        data_path=args.data,
        output_dir=Path(args.output_dir),
        model_path=args.model,
        quick_mode=args.quick,
        config_path=args.config
    )
    
    results = experiment.run()
    
    # Print quick summary
    success = results['success_criteria']
    metrics = results['key_metrics']
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    
    if success['overall_success']:
        print("✅ SUCCESS: Aggregator is robust to rubric variations")
    else:
        print("❌ NEEDS IMPROVEMENT: Aggregator shows sensitivity")
    
    print(f"\nKey Metrics:")
    print(f"  Variance: {metrics['learned_variance']:.4f} ({'✅' if success['variance_below_5_percent'] else '❌'} <5%)")
    print(f"  Correlation: {metrics['learned_correlation']:.4f} ({'✅' if success['correlation_above_95_percent'] else '❌'} >95%)")
    print(f"  Improvement: {metrics['improvement_factor']:.2f}x over baseline")
    
    print(f"\nDetailed results: {Path(args.output_dir).absolute()}")


if __name__ == "__main__":
    main()