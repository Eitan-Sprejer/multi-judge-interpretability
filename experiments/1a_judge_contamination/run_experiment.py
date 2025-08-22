#!/usr/bin/env python3
"""
Experiment 1A: Judge Contamination

Tests robustness of aggregation models to contaminated judges with inverted rubrics.
Creates poisoned judges and evaluates their impact on aggregation performance.

Usage:
    python run_experiment.py [--quick] [--num-samples 200]
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Pipeline imports
from pipeline.core.judge_creation import create_or_update_judge, JUDGE_MODEL, MIN_SCORE, MAX_SCORE
from pipeline.utils.judge_rubrics import INVERTED_JUDGE_RUBRICS
from pipeline.utils.create_martian_client import create_martian_client
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.dataset_loader import DatasetLoader
from martian_apart_hack_sdk import judge_specs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContaminationExperimentRunner:
    """Main experiment orchestrator for judge contamination analysis"""
    
    def __init__(self, args):
        """Initialize experiment runner"""
        self.args = args
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"contamination_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize clients
        self.client = create_martian_client()
        self.inverted_judge_ids = list(INVERTED_JUDGE_RUBRICS.keys())
        
        logger.info(f"🧪 Initialized Judge Contamination Experiment")
        logger.info(f"Results will be saved to: {self.experiment_dir}")
    
    def create_contaminated_judges(self):
        """Create judges with inverted rubrics"""
        logger.info("Creating contaminated judges with inverted rubrics...")
        
        created_judges = []
        
        for target_id in self.inverted_judge_ids:
            try:
                inverted_rubric = INVERTED_JUDGE_RUBRICS[target_id]()
                
                judge_spec = judge_specs.RubricJudgeSpec(
                    model_type="rubric_judge",
                    rubric=inverted_rubric,
                    model=JUDGE_MODEL,
                    min_score=MIN_SCORE,
                    max_score=MAX_SCORE,
                )
                
                contaminated_judge_id = f'contaminated_{target_id}'
                
                create_or_update_judge(
                    client=self.client,
                    judge_id=contaminated_judge_id,
                    judge_spec=judge_spec,
                    description=f'Contaminated judge with inverted rubric for {target_id}',
                )
                
                created_judges.append(contaminated_judge_id)
                logger.info(f"✅ Created contaminated judge: {contaminated_judge_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create contaminated judge for {target_id}: {e}")
        
        return created_judges
    
    def load_and_sample_data(self):
        """Load dataset and create sample for evaluation"""
        logger.info("Loading dataset...")
        
        dataset_loader = DatasetLoader()
        data = dataset_loader.load_existing_personas('data/data_with_all_personas.pkl')
        
        # Sample data based on arguments
        num_samples = 50 if self.args.quick else self.args.num_samples
        
        if len(data) > num_samples:
            data_sample = data.sample(n=num_samples, random_state=42)
            logger.info(f"Sampled {len(data_sample)} examples from {len(data)} total")
        else:
            data_sample = data
            logger.info(f"Using all {len(data_sample)} examples")
        
        return data_sample
    
    def evaluate_judges(self, contaminated_judges, data_sample):
        """Evaluate contaminated judges on sample data"""
        logger.info(f"Evaluating {len(contaminated_judges)} contaminated judges...")
        
        # Initialize judge evaluator
        judge_evaluator = JudgeEvaluator(judge_ids=contaminated_judges)
        
        # Evaluate all samples
        scores = []
        for i, (_, row) in enumerate(data_sample.iterrows()):
            question = row['instruction']
            answer = row['answer']
            
            try:
                judge_scores = judge_evaluator.evaluate_parallel(
                    question=question, 
                    answer=answer
                )
                scores.append(judge_scores)
                
                if (i + 1) % 25 == 0:
                    logger.info(f"Processed {i + 1}/{len(data_sample)} samples")
                    
            except Exception as e:
                logger.error(f"Failed to evaluate sample {i}: {e}")
                scores.append([np.nan] * len(contaminated_judges))
        
        # Create scores DataFrame
        scores_df = pd.DataFrame(scores, columns=contaminated_judges)
        
        return scores_df
    
    def extract_baseline_scores(self, data_sample):
        """Extract baseline judge scores if available"""
        logger.info("Extracting baseline scores...")
        
        if 'judge_scores' not in data_sample.columns:
            logger.warning("No baseline judge scores found in dataset")
            return None
        
        baseline_scores = []
        for _, row in data_sample.iterrows():
            if row['judge_scores'] and len(row['judge_scores']) >= len(self.inverted_judge_ids):
                baseline_scores.append(row['judge_scores'][:len(self.inverted_judge_ids)])
            else:
                baseline_scores.append([np.nan] * len(self.inverted_judge_ids))
        
        baseline_df = pd.DataFrame(baseline_scores, columns=self.inverted_judge_ids)
        logger.info(f"Extracted baseline scores for {len(baseline_df)} samples")
        
        return baseline_df
    
    def analyze_contamination_effects(self, baseline_df, contaminated_df):
        """Analyze the effects of judge contamination"""
        logger.info("Analyzing contamination effects...")
        
        analysis_results = {}
        
        if baseline_df is not None:
            # Score shift analysis
            score_shifts = {}
            correlations = {}
            
            for i, judge_id in enumerate(self.inverted_judge_ids):
                baseline_scores = baseline_df[judge_id]
                contaminated_scores = contaminated_df[f'contaminated_{judge_id}']
                
                # Calculate shift
                baseline_mean = baseline_scores.mean()
                contaminated_mean = contaminated_scores.mean()
                shift = contaminated_mean - baseline_mean
                
                # Calculate correlation (should be negative for inversion)
                correlation = baseline_scores.corr(contaminated_scores)
                
                score_shifts[judge_id] = {
                    'baseline_mean': float(baseline_mean),
                    'contaminated_mean': float(contaminated_mean),
                    'shift': float(shift)
                }
                
                correlations[judge_id] = float(correlation)
            
            analysis_results = {
                'score_shifts': score_shifts,
                'correlations': correlations,
                'avg_correlation': float(np.mean(list(correlations.values()))),
                'inversion_detected': np.mean(list(correlations.values())) < 0
            }
        else:
            # Basic contaminated score analysis
            contaminated_stats = {}
            for col in contaminated_df.columns:
                stats = contaminated_df[col].describe()
                contaminated_stats[col] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                }
            
            analysis_results = {
                'contaminated_stats': contaminated_stats,
                'baseline_available': False
            }
        
        return analysis_results
    
    def save_results(self, data_sample, baseline_df, contaminated_df, analysis_results, created_judges):
        """Save all experiment results"""
        logger.info(f"Saving results to {self.experiment_dir}")
        
        # Create combined results DataFrame
        results_df = data_sample[['instruction', 'answer']].copy().reset_index(drop=True)
        
        # Add baseline scores if available
        if baseline_df is not None:
            for judge_id in self.inverted_judge_ids:
                results_df[f'baseline_{judge_id}'] = baseline_df[judge_id].values
        
        # Add contaminated scores
        for col in contaminated_df.columns:
            results_df[col] = contaminated_df[col].values
        
        # Save datasets
        results_df.to_csv(self.experiment_dir / "contamination_results.csv", index=False)
        contaminated_df.to_csv(self.experiment_dir / "contaminated_scores.csv", index=False)
        
        if baseline_df is not None:
            baseline_df.to_csv(self.experiment_dir / "baseline_scores.csv", index=False)
        
        # Save analysis results
        with open(self.experiment_dir / "analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        # Save experiment metadata
        metadata = {
            'experiment_type': 'judge_contamination',
            'timestamp': self.timestamp,
            'args': vars(self.args),
            'created_judges': created_judges,
            'samples_evaluated': len(data_sample),
            'baseline_available': baseline_df is not None,
            'inverted_judge_ids': self.inverted_judge_ids
        }
        
        with open(self.experiment_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Results saved to: {self.experiment_dir}")
        
        return results_df
    
    def run_full_experiment(self):
        """Run the complete contamination experiment"""
        logger.info("🚀 Starting Judge Contamination Experiment")
        
        try:
            # Step 1: Create contaminated judges
            created_judges = self.create_contaminated_judges()
            
            if not created_judges:
                raise Exception("No contaminated judges were created successfully")
            
            # Step 2: Load and sample data
            data_sample = self.load_and_sample_data()
            
            # Step 3: Extract baseline scores
            baseline_df = self.extract_baseline_scores(data_sample)
            
            # Step 4: Evaluate contaminated judges
            contaminated_df = self.evaluate_judges(created_judges, data_sample)
            
            # Step 5: Analyze contamination effects
            analysis_results = self.analyze_contamination_effects(baseline_df, contaminated_df)
            
            # Step 6: Save results
            final_results = self.save_results(
                data_sample, baseline_df, contaminated_df, 
                analysis_results, created_judges
            )
            
            # Print summary
            self.print_summary(analysis_results, created_judges, len(data_sample))
            
            return {
                'success': True,
                'results_dir': str(self.experiment_dir),
                'analysis': analysis_results,
                'judges_created': len(created_judges),
                'samples_processed': len(data_sample)
            }
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def print_summary(self, analysis_results, created_judges, num_samples):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("EXPERIMENT 1A: JUDGE CONTAMINATION RESULTS")
        print("="*60)
        
        print(f"Judges Created: {len(created_judges)}")
        print(f"Samples Processed: {num_samples}")
        
        if 'correlations' in analysis_results:
            avg_correlation = analysis_results['avg_correlation']
            inversion_detected = analysis_results['inversion_detected']
            
            print(f"Average Baseline-Contaminated Correlation: {avg_correlation:.3f}")
            print(f"Inversion Detected: {'✅ Yes' if inversion_detected else '❌ No'}")
            
            print(f"\nIndividual Judge Correlations:")
            for judge_id, corr in analysis_results['correlations'].items():
                status = "✅ Inverted" if corr < -0.5 else "⚠️ Partial" if corr < 0 else "❌ Not inverted"
                print(f"  {judge_id}: {corr:.3f} {status}")
        
        print(f"\nResults saved to: {self.experiment_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run Judge Contamination Experiment")
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test with 50 samples'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
        help='Number of samples to evaluate (default: 200)'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    try:
        runner = ContaminationExperimentRunner(args)
        final_report = runner.run_full_experiment()
        
        if final_report['success']:
            print("\n" + "="*60)
            print("EXPERIMENT 1A COMPLETED SUCCESSFULLY!")
            print(f"Check results in: experiments/1a_judge_contamination/results/")
            print("="*60)
            return 0
        else:
            print(f"Experiment failed: {final_report['error']}")
            return 1
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
