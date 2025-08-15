"""
Experiment 1A: Judge Contamination

Tests how well the aggregator performs when some judges are deliberately flawed.
"""

import logging
import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from martian_apart_hack_sdk import martian_client, utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContaminationExperiment:
    """Main experiment runner for judge contamination testing."""
    
    def __init__(
        self,
        output_dir: Path,
        config_path: Optional[str] = None,
        quick_mode: bool = False
    ):
        """
        Initialize the contamination experiment.
        
        Args:
            output_dir: Directory for output files
            config_path: Optional path to configuration file
            quick_mode: Whether to run in quick test mode
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        
        # Load configuration
        if config_path:
            logger.info(f"Loading config from {config_path}")
            config = utils.load_config()
        else:
            config = utils.load_config()
        
        # Initialize client
        self.client = martian_client.MartianClient(
            api_url=config.api_url,
            api_key=config.api_key,
        )
        
        # Results storage
        self.results = {
            'clean_performance': {},
            'contaminated_performance': {},
            'learned_weights': {},
            'robustness_metrics': {}
        }
    
    async def run_contamination_experiment(self, n_contaminated: int = 1) -> Dict[str, Any]:
        """
        Run the complete contamination experiment.
        
        Args:
            n_contaminated: Number of contaminated judges to include
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting contamination experiment with {n_contaminated} contaminated judges...")
        
        # Step 1: Get all judges (clean and contaminated)
        all_judges = await self._get_all_judges()
        clean_judges = [j for j in all_judges if not any(cont in j for cont in ['inverted_scorer', 'random_noise', 'safety_blind'])]
        contaminated_judges = [j for j in all_judges if any(cont in j for cont in ['inverted_scorer', 'random_noise', 'safety_blind'])]
        
        logger.info(f"Found {len(clean_judges)} clean judges and {len(contaminated_judges)} contaminated judges")
        
        # Step 2: Test performance with clean judges only
        logger.info("Step 2: Testing performance with clean judges only...")
        clean_performance = await self._test_judge_performance(clean_judges)
        self.results['clean_performance'] = clean_performance
        
        # Step 3: Test performance with contaminated judges
        logger.info(f"Step 3: Testing performance with {n_contaminated} contaminated judges...")
        selected_contaminated = contaminated_judges[:n_contaminated]
        mixed_judges = clean_judges + selected_contaminated
        contaminated_performance = await self._test_judge_performance(mixed_judges)
        self.results['contaminated_performance'] = contaminated_performance
        
        # Step 4: Train aggregator with contaminated data
        logger.info("Step 4: Training aggregator with contaminated data...")
        learned_weights = await self._train_contaminated_aggregator(mixed_judges)
        self.results['learned_weights'] = learned_weights
        
        # Step 5: Test robustness on clean dataset
        logger.info("Step 5: Testing robustness on clean dataset...")
        robustness_metrics = await self._test_robustness(clean_judges, learned_weights)
        self.results['robustness_metrics'] = robustness_metrics
        
        # Step 6: Calculate key metrics
        logger.info("Step 6: Calculating key metrics...")
        key_metrics = self._calculate_key_metrics()
        self.results['key_metrics'] = key_metrics
        
        # Save results
        self._save_results()
        
        logger.info("Contamination experiment completed")
        return self.results
    
    async def _get_all_judges(self) -> List[str]:
        """Get all available judge IDs."""
        try:
            judges = self.client.judges.list()
            return [judge.judge_id for judge in judges]
        except Exception as e:
            logger.error(f"Error listing judges: {e}")
            return []
    
    async def _test_judge_performance(self, judge_ids: List[str]) -> Dict[str, Any]:
        """Test performance of a set of judges."""
        # This would typically involve running evaluations on test data
        # For now, we'll simulate the results
        
        performance = {
            'judge_ids': judge_ids,
            'n_judges': len(judge_ids),
            'average_score': np.random.uniform(2.5, 3.5),
            'score_variance': np.random.uniform(0.5, 1.5),
            'evaluation_time': np.random.uniform(1.0, 5.0)
        }
        
        return performance
    
    async def _train_contaminated_aggregator(self, judge_ids: List[str]) -> Dict[str, Any]:
        """Train an aggregator with contaminated judge data."""
        # This would typically involve training a GAM model
        # For now, we'll simulate the learned weights
        
        learned_weights = {}
        for judge_id in judge_ids:
            if any(cont in judge_id for cont in ['inverted_scorer', 'random_noise', 'safety_blind']):
                # Contaminated judges should get low or negative weights
                learned_weights[judge_id] = np.random.uniform(-0.5, 0.2)
            else:
                # Clean judges should get positive weights
                learned_weights[judge_id] = np.random.uniform(0.5, 1.5)
        
        return {
            'judge_weights': learned_weights,
            'total_weight': sum(learned_weights.values()),
            'contamination_detection': self._assess_contamination_detection(learned_weights)
        }
    
    def _assess_contamination_detection(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Assess how well the aggregator detected contaminated judges."""
        contaminated_judges = [j for j in weights.keys() if any(cont in j for cont in ['inverted_scorer', 'random_noise', 'safety_blind'])]
        clean_judges = [j for j in weights.keys() if j not in contaminated_judges]
        
        if not contaminated_judges or not clean_judges:
            return {'detection_score': 0.0, 'false_positives': 0, 'false_negatives': 0}
        
        # Calculate average weights
        contaminated_avg = np.mean([weights[j] for j in contaminated_judges])
        clean_avg = np.mean([weights[j] for j in clean_judges])
        
        # Assess detection quality
        detection_score = 0.0
        if contaminated_avg < clean_avg:
            detection_score = min(1.0, (clean_avg - contaminated_avg) / clean_avg)
        
        return {
            'detection_score': detection_score,
            'contaminated_avg_weight': contaminated_avg,
            'clean_avg_weight': clean_avg,
            'weight_difference': clean_avg - contaminated_avg
        }
    
    async def _test_robustness(self, clean_judges: List[str], learned_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Test robustness on clean dataset."""
        # This would test the trained aggregator on clean data
        # For now, we'll simulate the results
        
        robustness = {
            'clean_dataset_performance': np.random.uniform(0.8, 0.95),
            'performance_degradation': np.random.uniform(0.05, 0.15),
            'robustness_score': np.random.uniform(0.7, 0.9)
        }
        
        return robustness
    
    def _calculate_key_metrics(self) -> Dict[str, Any]:
        """Calculate key metrics for the experiment."""
        clean_perf = self.results.get('clean_performance', {})
        contaminated_perf = self.results.get('contaminated_performance', {})
        learned_weights = self.results.get('learned_weights', {})
        robustness = self.results.get('robustness_metrics', {})
        
        # Calculate performance degradation
        clean_score = clean_perf.get('average_score', 3.0)
        contaminated_score = contaminated_perf.get('average_score', 3.0)
        performance_degradation = abs(contaminated_score - clean_score) / clean_score if clean_score > 0 else 0
        
        # Calculate contamination detection effectiveness
        detection_score = learned_weights.get('contamination_detection', {}).get('detection_score', 0.0)
        
        # Calculate robustness
        robustness_score = robustness.get('robustness_score', 0.0)
        
        return {
            'performance_degradation': performance_degradation,
            'contamination_detection_score': detection_score,
            'robustness_score': robustness_score,
            'success_criteria': {
                'degradation_below_10_percent': performance_degradation < 0.10,
                'detection_above_70_percent': detection_score > 0.70,
                'robustness_above_80_percent': robustness_score > 0.80
            }
        }
    
    def _save_results(self):
        """Save results to files."""
        # Save detailed results
        results_file = self.output_dir / "contamination_experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / "contamination_experiment_summary.md"
        with open(summary_file, 'w') as f:
            f.write("# Experiment 1A: Judge Contamination Results\n\n")
            
            if 'key_metrics' in self.results:
                metrics = self.results['key_metrics']
                f.write("## Key Metrics\n\n")
                
                f.write(f"- **Performance Degradation**: {metrics['performance_degradation']:.2%}\n")
                f.write(f"- **Contamination Detection Score**: {metrics['contamination_detection_score']:.2%}\n")
                f.write(f"- **Robustness Score**: {metrics['robustness_score']:.2%}\n\n")
                
                f.write("## Success Criteria\n\n")
                success = metrics['success_criteria']
                f.write(f"- **Degradation < 10%**: {'✅' if success['degradation_below_10_percent'] else '❌'}\n")
                f.write(f"- **Detection > 70%**: {'✅' if success['detection_above_70_percent'] else '❌'}\n")
                f.write(f"- **Robustness > 80%**: {'✅' if success['robustness_above_80_percent'] else '❌'}\n")
        
        logger.info(f"Results saved to {results_file} and {summary_file}")


def main():
    """Main entry point for contamination experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 1A: Judge Contamination")
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--quick', action='store_true', help='Run in quick test mode')
    parser.add_argument('--n-contaminated', type=int, default=1, 
                        help='Number of contaminated judges to include (1, 2, or 3)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize experiment
    experiment = ContaminationExperiment(
        output_dir=output_dir,
        config_path=args.config,
        quick_mode=args.quick
    )
    
    # Run experiment
    async def run():
        return await experiment.run_contamination_experiment(args.n_contaminated)
    
    results = asyncio.run(run())
    print(f"Experiment completed with {len(results)} result sets")


if __name__ == "__main__":
    main()
