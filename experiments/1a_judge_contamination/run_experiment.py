#!/usr/bin/env python3
"""
Experiment 1A: Enhanced Judge Contamination Analysis

Comprehensive testing of aggregation model robustness to contaminated judges.
Features advanced statistical analysis, visualization suite, and robustness metrics.

Usage:
    python run_experiment.py [options]
    
Examples:
    # Quick test with basic analysis
    python run_experiment.py --quick
    
    # Full analysis with robustness testing
    python run_experiment.py --num-samples 500 --enable-robustness
    
    # Multi-contamination comparison study
    python run_experiment.py --contamination-types all --generate-publication-figures
"""

#%%
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Add project root to path, for importing pipeline utils and such
PROJECT_ROOT = os.path.join(os.getcwd(), '..', '..')
sys.path.append(str(PROJECT_ROOT))

# Pipeline imports
from pipeline.core.judge_creation import create_or_update_judge, JUDGE_MODEL, MIN_SCORE, MAX_SCORE
from pipeline.utils.judge_rubrics import INVERTED_JUDGE_RUBRICS
from pipeline.utils.create_martian_client import create_martian_client
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.dataset_loader import DatasetLoader
from martian_apart_hack_sdk import judge_specs

#%%

# Enhanced analysis imports
from src.contamination_analysis import ContaminationAnalyzer
from src.visualizations import ContaminationVisualizer
from src.results_framework import ResultsProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#%%
class EnhancedContaminationExperimentRunner:
    """Enhanced experiment orchestrator for comprehensive judge contamination analysis"""
    
    def __init__(self, args):
        """Initialize enhanced experiment runner"""
        self.args = args
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamped results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"enhanced_contamination_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Load configuration first
        self.config = self._load_configuration()
        
        # Initialize components
        self.client = create_martian_client()
        self.inverted_judge_ids = list(INVERTED_JUDGE_RUBRICS.keys())
        self.analyzer = ContaminationAnalyzer(significance_level=args.significance_level)
        self.visualizer = ContaminationVisualizer(self.experiment_dir / "plots")
        self.results_processor = ResultsProcessor(self.experiment_dir)
        
        # Results storage
        self.results = {
            'experiment_metadata': self._get_experiment_metadata(),
            'contamination_analysis': {},
            'robustness_analysis': {},
            'visualizations': {},
            'summary_report': {}
        }
        
        logger.info(f"🧪 Initialized Enhanced Judge Contamination Experiment")
        logger.info(f"Results will be saved to: {self.experiment_dir}")
        logger.info(f"Configuration: {args.contamination_types} contamination types")
        logger.info(f"Sample size: {args.num_samples} ({'quick mode' if args.quick else 'full mode'})")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load experiment configuration from config file"""
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'contamination': {
                    'types': ['inverted'],
                    'inverted': {'enabled': True, 'description': 'Rubrics that score opposite to intended criteria'},
                    'noise': {'enabled': False, 'std_dev': 0.5},
                    'bias': {'enabled': False, 'bias_amount': 0.3}
                },
                'execution': {
                    'quick_mode_samples': 50,
                    'full_mode_samples': 1000,
                    'max_retries': 3
                }
            }
        return config
    
    def _get_experiment_metadata(self) -> Dict[str, Any]:
        """Generate experiment metadata"""
        return {
            'experiment_name': 'enhanced_judge_contamination',
            'timestamp': self.timestamp,
            'version': '2.0.0',
            'arguments': vars(self.args),
            'configuration': self.config,
            'target_judges': self.inverted_judge_ids,
            'analysis_modules': ['contamination_analysis', 'visualizations', 'robustness_testing']
        }
    
    def create_contaminated_judges(self) -> Dict[str, List[str]]:
        """Create judges with various contamination strategies"""
        logger.info("Creating contaminated judges with multiple strategies...")
        
        created_judges = {
            'inverted': [],
            'noise': [],
            'bias': []
        }
        
        # Create inverted judges if enabled
        if self._should_create_contamination_type('inverted'):
            created_judges['inverted'] = self._create_inverted_judges()
        
        # Create noise-based judges if enabled  
        if self._should_create_contamination_type('noise'):
            created_judges['noise'] = self._create_noise_judges()
            
        # Create bias-based judges if enabled
        if self._should_create_contamination_type('bias'):
            created_judges['bias'] = self._create_bias_judges()
        
        total_created = sum(len(judges) for judges in created_judges.values())
        logger.info(f"✅ Created {total_created} contaminated judges across {len([k for k, v in created_judges.items() if v])} strategies")
        
        return created_judges
    
    def _should_create_contamination_type(self, contamination_type: str) -> bool:
        """Check if contamination type should be created based on config and args"""
        if self.args.contamination_types == 'all':
            return self.config['contamination'][contamination_type]['enabled']
        elif self.args.contamination_types == contamination_type:
            return True
        elif isinstance(self.args.contamination_types, list) and contamination_type in self.args.contamination_types:
            return True
        return False
    
    def _create_inverted_judges(self) -> List[str]:
        """Create judges with inverted rubrics"""
        logger.info("Creating inverted rubric judges...")
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
                
                contaminated_judge_id = f'inverted_{target_id}'
                
                create_or_update_judge(
                    client=self.client,
                    judge_id=contaminated_judge_id,
                    judge_spec=judge_spec,
                    description=f'Judge with inverted rubric for {target_id}',
                )
                
                created_judges.append(contaminated_judge_id)
                logger.info(f"✅ Created inverted judge: {contaminated_judge_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create inverted judge for {target_id}: {e}")
        
        return created_judges
    
    def _create_noise_judges(self) -> List[str]:
        """Create judges with noise injection (placeholder implementation)"""
        logger.info("Noise-based contamination not yet implemented")
        return []
    
    def _create_bias_judges(self) -> List[str]:
        """Create judges with systematic bias (placeholder implementation)"""
        logger.info("Bias-based contamination not yet implemented")
        return []
    
    def load_and_sample_data(self) -> pd.DataFrame:
        """Load dataset and create sample for evaluation"""
        logger.info("Loading dataset for contamination analysis...")
        
        dataset_loader = DatasetLoader()
        
        # Try multiple data sources
        data_sources = [
            'data/data_with_all_personas.pkl',
            'data/data_with_judge_scores.pkl', 
        ]
        
        data = None
        for source in data_sources:
            full_source = os.path.join(PROJECT_ROOT, source)
            try:
                data = dataset_loader.load_existing_personas(full_source)
                logger.info(f"Successfully loaded data from: {full_source}")
                break
            except Exception as e:
                logger.debug(f"Failed to load from {full_source}: {e}")
        
        if data is None:
            raise FileNotFoundError("Could not load dataset from any source")
        
        # Sample data based on arguments and configuration
        if self.args.quick:
            num_samples = self.config['execution']['quick_mode_samples']
        else:
            num_samples = min(self.args.num_samples, self.config['execution']['full_mode_samples'])
        
        if len(data) > num_samples:
            # Stratified sampling if human feedback available
            if 'human_feedback' in data.columns:
                data_sample = self._stratified_sample(data, num_samples)
                logger.info(f"Stratified sampled {len(data_sample)} examples from {len(data)} total")
            else:
                data_sample = data.sample(n=num_samples, random_state=42)
                logger.info(f"Random sampled {len(data_sample)} examples from {len(data)} total")
        else:
            data_sample = data
            logger.info(f"Using all {len(data_sample)} examples")
        
        return data_sample
    
    def _stratified_sample(self, data: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """Perform stratified sampling based on human feedback distribution"""
        if 'human_feedback' not in data.columns:
            return data.sample(n=num_samples, random_state=42)
        
        # Create bins for stratification
        data['feedback_bin'] = pd.cut(data['human_feedback'], bins=5, labels=False)
        
        # Sample proportionally from each bin
        sampled_data = data.groupby('feedback_bin', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(num_samples * len(x) / len(data)))), random_state=42)
        )
        
        # If we need more samples, add random samples
        if len(sampled_data) < num_samples:
            remaining = num_samples - len(sampled_data)
            remaining_data = data.drop(sampled_data.index)
            additional = remaining_data.sample(min(remaining, len(remaining_data)), random_state=42)
            sampled_data = pd.concat([sampled_data, additional])
        
        return sampled_data.drop('feedback_bin', axis=1).reset_index(drop=True)
    
    def evaluate_judges(self, contaminated_judges: Dict[str, List[str]], data_sample: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Evaluate all contaminated judges on sample data"""
        all_judge_scores = {}
        
        for contamination_type, judge_list in contaminated_judges.items():
            if not judge_list:
                continue
                
            logger.info(f"Evaluating {len(judge_list)} {contamination_type} judges...")
            
            # Initialize judge evaluator for this contamination type
            judge_evaluator = JudgeEvaluator(judge_ids=judge_list)
        
            # Evaluate all samples for this contamination type
            scores = []
            batch_size = 10  # Process in batches for better progress tracking
            
            for i in range(0, len(data_sample), batch_size):
                batch = data_sample.iloc[i:i+batch_size]
                
                for j, (_, row) in enumerate(batch.iterrows()):
                    global_idx = i + j
                    question = row['instruction']
                    answer = row['answer']
                    
                    try:
                        judge_scores = judge_evaluator.evaluate_parallel(
                            question=question, 
                            answer=answer
                        )
                        scores.append(judge_scores)
                        
                    except Exception as e:
                        logger.warning(f"Failed to evaluate sample {global_idx}: {e}")
                        scores.append([np.nan] * len(judge_list))
                
                # Progress update
                processed = min(i + batch_size, len(data_sample))
                logger.info(f"Processed {processed}/{len(data_sample)} samples for {contamination_type} judges")
            
            # Create scores DataFrame for this contamination type
            scores_df = pd.DataFrame(scores, columns=judge_list)
            all_judge_scores[contamination_type] = scores_df
        
        return all_judge_scores
    
    def extract_baseline_scores(self, data_sample: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract baseline judge scores if available"""
        logger.info("Extracting baseline judge scores...")
        
        # Check multiple possible columns for baseline scores
        score_columns = ['judge_scores', 'baseline_scores', 'clean_judge_scores']
        baseline_column = None
        
        for col in score_columns:
            if col in data_sample.columns:
                baseline_column = col
                logger.info(f"Found baseline scores in column: {col}")
                break
        
        if baseline_column is None:
            logger.warning("No baseline judge scores found in dataset")
            return None
        
        baseline_scores = []
        valid_samples = 0
        
        for _, row in data_sample.iterrows():
            scores_data = row[baseline_column]
            
            if scores_data is not None and len(scores_data) >= len(self.inverted_judge_ids):
                baseline_scores.append(scores_data[:len(self.inverted_judge_ids)])
                valid_samples += 1
            else:
                baseline_scores.append([np.nan] * len(self.inverted_judge_ids))
        
        baseline_df = pd.DataFrame(baseline_scores, columns=self.inverted_judge_ids)
        logger.info(f"Extracted baseline scores for {valid_samples}/{len(baseline_df)} samples")
        
        return baseline_df
    
    def run_comprehensive_analysis(self, 
                                  baseline_df: Optional[pd.DataFrame], 
                                  contaminated_scores: Dict[str, pd.DataFrame],
                                  data_sample: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive contamination analysis using advanced statistical methods"""
        logger.info("Running comprehensive contamination analysis...")
        
        comprehensive_results = {}
        
        # Analyze each contamination type
        for contamination_type, contaminated_df in contaminated_scores.items():
            logger.info(f"Analyzing {contamination_type} contamination effects...")
            
            if baseline_df is not None:
                # Create judge mapping for analysis
                judge_mapping = {}
                for judge_id in self.inverted_judge_ids:
                    if contamination_type == 'inverted':
                        contaminated_judge_id = f'inverted_{judge_id}'
                    else:
                        contaminated_judge_id = f'{contamination_type}_{judge_id}'
                    
                    if contaminated_judge_id in contaminated_df.columns:
                        judge_mapping[judge_id] = contaminated_judge_id
                
                # Run advanced analysis
                analysis_results = self.analyzer.analyze_judge_inversion(
                    baseline_scores=baseline_df,
                    contaminated_scores=contaminated_df,
                    judge_mapping=judge_mapping
                )
                
                comprehensive_results[contamination_type] = analysis_results
                
            else:
                logger.warning(f"No baseline data available for {contamination_type} analysis")
                comprehensive_results[contamination_type] = {
                    'error': 'No baseline data available',
                    'contaminated_stats': self._compute_basic_stats(contaminated_df)
                }
        
        return comprehensive_results
    
    def _compute_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics when baseline data is not available"""
        stats = {}
        for col in df.columns:
            col_stats = df[col].describe()
            stats[col] = {
                'mean': float(col_stats['mean']),
                'std': float(col_stats['std']),
                'min': float(col_stats['min']),
                'max': float(col_stats['max']),
                'median': float(col_stats['50%'])
            }
        return stats
    
    def run_robustness_analysis(self, 
                               baseline_df: Optional[pd.DataFrame],
                               contaminated_scores: Dict[str, pd.DataFrame],
                               data_sample: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run robustness analysis if enabled and data is available"""
        if not self.args.enable_robustness or baseline_df is None:
            logger.info("Skipping robustness analysis (disabled or no baseline data)")
            return None
            
        logger.info("Running aggregator robustness analysis...")
        
        # Extract human feedback if available
        if 'human_feedback' not in data_sample.columns:
            logger.warning("No human feedback available for robustness analysis")
            return None
        
        human_feedback = data_sample['human_feedback']
        
        # Use inverted contamination for robustness testing
        if 'inverted' in contaminated_scores:
            contaminated_data = contaminated_scores['inverted']
            
            robustness_results = self.analyzer.analyze_aggregator_robustness(
                clean_data=baseline_df,
                contaminated_data=contaminated_data,
                human_feedback=human_feedback,
                contamination_rates=self.args.contamination_rates
            )
            
            return robustness_results
        
        return None
    
    def generate_visualizations(self, 
                               analysis_results: Dict[str, Any],
                               baseline_df: Optional[pd.DataFrame],
                               contaminated_scores: Dict[str, pd.DataFrame],
                               robustness_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        if not self.args.enable_visualizations:
            logger.info("Skipping visualization generation (disabled)")
            return {}
            
        logger.info("Generating comprehensive visualizations...")
        
        visualization_results = {}
        
        # Generate visualizations for each contamination type with baseline data
        for contamination_type, contaminated_df in contaminated_scores.items():
            if baseline_df is None:
                continue
                
            logger.info(f"Creating visualizations for {contamination_type} contamination...")
            
            # Create judge mapping
            judge_mapping = {}
            for judge_id in self.inverted_judge_ids:
                if contamination_type == 'inverted':
                    contaminated_judge_id = f'inverted_{judge_id}'
                else:
                    contaminated_judge_id = f'{contamination_type}_{judge_id}'
                
                if contaminated_judge_id in contaminated_df.columns:
                    judge_mapping[judge_id] = contaminated_judge_id
            
            if contamination_type in analysis_results:
                # Generate all visualizations for this contamination type
                figures = self.visualizer.generate_all_visualizations(
                    analysis_results=analysis_results[contamination_type],
                    baseline_scores=baseline_df,
                    contaminated_scores=contaminated_df,
                    judge_mapping=judge_mapping,
                    robustness_results=robustness_results
                )
                
                visualization_results[contamination_type] = list(figures.keys())
        
        return visualization_results
    
    def save_comprehensive_results(self, 
                                  data_sample: pd.DataFrame,
                                  baseline_df: Optional[pd.DataFrame],
                                  contaminated_scores: Dict[str, pd.DataFrame],
                                  analysis_results: Dict[str, Any],
                                  robustness_results: Optional[Dict[str, Any]],
                                  created_judges: Dict[str, List[str]],
                                  visualization_results: Dict[str, Any],
                                  processed_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save comprehensive experiment results with enhanced organization"""
        logger.info(f"Saving comprehensive results to {self.experiment_dir}")
        
        # Update results dictionary
        self.results.update({
            'contamination_analysis': analysis_results,
            'robustness_analysis': robustness_results,
            'visualizations': visualization_results,
            'created_judges': created_judges
        })
        
        # Create data directory for raw data
        data_dir = self.experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save raw datasets
        base_results_df = data_sample[['instruction', 'answer']].copy().reset_index(drop=True)
        
        if baseline_df is not None:
            for judge_id in self.inverted_judge_ids:
                if judge_id in baseline_df.columns:
                    base_results_df[f'baseline_{judge_id}'] = baseline_df[judge_id].values
            baseline_df.to_csv(data_dir / "baseline_scores.csv", index=False)
        
        # Save contaminated scores by type
        for contamination_type, scores_df in contaminated_scores.items():
            scores_df.to_csv(data_dir / f"{contamination_type}_scores.csv", index=False)
            
            # Add to combined results
            for col in scores_df.columns:
                base_results_df[col] = scores_df[col].values
        
        base_results_df.to_csv(data_dir / "combined_results.csv", index=False)
        
        # Save analysis results
        analysis_dir = self.experiment_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        with open(analysis_dir / "contamination_analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        if robustness_results:
            with open(analysis_dir / "robustness_analysis.json", "w") as f:
                json.dump(robustness_results, f, indent=2, default=str)
        
        # Generate comprehensive report
        self._generate_summary_report(analysis_results, robustness_results)
        
        # Save complete experiment results
        with open(self.experiment_dir / "complete_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"✅ Comprehensive results saved to: {self.experiment_dir}")
        
        return base_results_df
    
    def _generate_summary_report(self, 
                                analysis_results: Dict[str, Any],
                                robustness_results: Optional[Dict[str, Any]]) -> None:
        """Generate a comprehensive summary report"""
        report_path = self.experiment_dir / "EXPERIMENT_SUMMARY.md"
        
        with open(report_path, "w") as f:
            f.write("# Enhanced Judge Contamination Experiment Summary\n\n")
            f.write(f"**Experiment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Experiment ID**: {self.timestamp}\n")
            f.write(f"**Configuration**: {self.args.contamination_types} contamination\n")
            f.write(f"**Sample Size**: {self.args.num_samples}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            contamination_detected = False
            avg_correlations = {}
            
            for contamination_type, results in analysis_results.items():
                if 'judge_inversion' in results and 'aggregate_metrics' in results['judge_inversion']:
                    agg_metrics = results['judge_inversion']['aggregate_metrics']
                    avg_corr = agg_metrics['average_correlation']
                    avg_correlations[contamination_type] = avg_corr
                    
                    if agg_metrics['system_inversion_detected']:
                        contamination_detected = True
                        f.write(f"- **{contamination_type.title()} Contamination**: Successfully detected (avg correlation: {avg_corr:.3f})\n")
                    else:
                        f.write(f"- **{contamination_type.title()} Contamination**: Limited effect detected (avg correlation: {avg_corr:.3f})\n")
            
            f.write(f"\n**Overall Contamination Detection**: {'✅ Successful' if contamination_detected else '❌ Limited'}\n\n")
            
            # Detailed Results
            f.write("## Detailed Analysis Results\n\n")
            
            for contamination_type, results in analysis_results.items():
                f.write(f"### {contamination_type.title()} Contamination\n\n")
                
                if 'judge_inversion' in results:
                    judge_results = results['judge_inversion']
                    
                    if 'aggregate_metrics' in judge_results:
                        metrics = judge_results['aggregate_metrics']
                        f.write(f"- Average Judge Correlation: {metrics['average_correlation']:.3f}\n")
                        f.write(f"- Contamination Success Rate: {metrics['contamination_rate']:.1%}\n")
                        f.write(f"- System Inversion Detected: {metrics['system_inversion_detected']}\n\n")
                    
                    if 'inversion_detection' in judge_results:
                        patterns = judge_results['inversion_detection']['patterns']
                        f.write(f"**Contamination Patterns:**\n")
                        f.write(f"- Complete Inversion: {len(patterns['complete_inversion'])} judges\n")
                        f.write(f"- Partial Inversion: {len(patterns['partial_inversion'])} judges\n")
                        f.write(f"- No Inversion: {len(patterns['no_inversion'])} judges\n\n")
            
            # Robustness Analysis
            if robustness_results:
                f.write("## Robustness Analysis\n\n")
                robustness_metrics = robustness_results.get('robustness_metrics', {})
                
                if 'clean_performance' in robustness_metrics:
                    f.write(f"- Clean Performance (R²): {robustness_metrics['clean_performance']:.3f}\n")
                    f.write(f"- Final Performance (R²): {robustness_metrics['final_performance']:.3f}\n")
                    f.write(f"- Relative Degradation: {robustness_metrics['relative_degradation']:.1%}\n")
                    
                    if robustness_metrics.get('breakdown_rate'):
                        f.write(f"- Performance Breakdown at: {robustness_metrics['breakdown_rate']:.1%} contamination\n")
            
            # Files Generated
            f.write("\n## Generated Files\n\n")
            f.write("### Data Files\n")
            f.write("- `data/combined_results.csv`: All scores and results\n")
            f.write("- `data/baseline_scores.csv`: Clean judge scores\n")
            
            for contamination_type in analysis_results.keys():
                f.write(f"- `data/{contamination_type}_scores.csv`: {contamination_type.title()} contaminated scores\n")
            
            f.write("\n### Analysis Files\n")
            f.write("- `analysis/contamination_analysis.json`: Detailed statistical analysis\n")
            
            if robustness_results:
                f.write("- `analysis/robustness_analysis.json`: Aggregator robustness results\n")
            
            f.write("\n### Visualizations\n")
            f.write("- `plots/`: All generated visualization files\n")
            f.write("- `plots/publication_figure.png`: Publication-ready summary figure\n")
            f.write("- `plots/contamination_dashboard.png`: Comprehensive analysis dashboard\n")
        
        logger.info(f"Summary report generated: {report_path}")
    
    def run_enhanced_experiment(self) -> Dict[str, Any]:
        """Run the complete enhanced contamination experiment"""
        logger.info("🚀 Starting Enhanced Judge Contamination Experiment")
        
        try:
            # Step 1: Create contaminated judges
            logger.info("Step 1: Creating contaminated judges...")
            created_judges = self.create_contaminated_judges()
            
            total_judges = sum(len(judges) for judges in created_judges.values())
            if total_judges == 0:
                raise Exception("No contaminated judges were created successfully")
            
            # Step 2: Load and sample data
            logger.info("Step 2: Loading and sampling data...")
            data_sample = self.load_and_sample_data()
            
            # Step 3: Extract baseline scores
            logger.info("Step 3: Extracting baseline scores...")
            baseline_df = self.extract_baseline_scores(data_sample)
            
            # Step 4: Evaluate contaminated judges
            logger.info("Step 4: Evaluating contaminated judges...")
            contaminated_scores = self.evaluate_judges(created_judges, data_sample)
            
            # Step 5: Run comprehensive analysis
            logger.info("Step 5: Running comprehensive contamination analysis...")
            analysis_results = self.run_comprehensive_analysis(
                baseline_df, contaminated_scores, data_sample
            )
            
            # Step 6: Run robustness analysis (if enabled)
            logger.info("Step 6: Running robustness analysis...")
            robustness_results = self.run_robustness_analysis(
                baseline_df, contaminated_scores, data_sample
            )
            
            # Step 7: Generate visualizations
            logger.info("Step 7: Generating visualizations...")
            visualization_results = self.generate_visualizations(
                analysis_results, baseline_df, contaminated_scores, robustness_results
            )
            
            # Step 8: Process results with advanced framework
            logger.info("Step 8: Processing results with advanced framework...")
            execution_metadata = {
                'timestamp': self.timestamp,
                'samples_processed': len(data_sample),
                'judges_created': sum(len(judges) for judges in created_judges.values()),
                'execution_time': 0.0,  # Would be computed in real implementation
                'contamination_types': list(created_judges.keys())
            }
            
            processed_results = self.results_processor.process_contamination_results(
                analysis_results, robustness_results, execution_metadata
            )
            
            # Step 9: Save comprehensive results
            logger.info("Step 9: Saving comprehensive results...")
            final_results = self.save_comprehensive_results(
                data_sample, baseline_df, contaminated_scores,
                analysis_results, robustness_results, created_judges, 
                visualization_results, processed_results
            )
            
            # Print enhanced summary
            self.print_enhanced_summary(analysis_results, robustness_results, 
                                      created_judges, len(data_sample))
            
            return {
                'success': True,
                'results_dir': str(self.experiment_dir),
                'analysis_results': analysis_results,
                'robustness_results': robustness_results,
                'visualization_results': visualization_results,
                'judges_created': total_judges,
                'samples_processed': len(data_sample),
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'timestamp': self.timestamp
            }
    
    def print_enhanced_summary(self, 
                              analysis_results: Dict[str, Any],
                              robustness_results: Optional[Dict[str, Any]],
                              created_judges: Dict[str, List[str]], 
                              num_samples: int) -> None:
        """Print enhanced experiment summary"""
        print("\n" + "="*80)
        print("ENHANCED JUDGE CONTAMINATION EXPERIMENT RESULTS")
        print("="*80)
        
        total_judges = sum(len(judges) for judges in created_judges.values())
        print(f"📊 Experiment Overview:")
        print(f"   • Total Judges Created: {total_judges}")
        print(f"   • Contamination Types: {', '.join([k for k, v in created_judges.items() if v])}")
        print(f"   • Samples Processed: {num_samples}")
        print(f"   • Experiment ID: {self.timestamp}")
        
        print(f"\n🔬 Contamination Analysis Results:")
        
        overall_detection = False
        for contamination_type, results in analysis_results.items():
            if 'judge_inversion' in results and 'aggregate_metrics' in results['judge_inversion']:
                metrics = results['judge_inversion']['aggregate_metrics']
                avg_corr = metrics['average_correlation']
                success_rate = metrics['contamination_rate']
                detected = metrics['system_inversion_detected']
                
                if detected:
                    overall_detection = True
                
                print(f"   • {contamination_type.title()} Contamination:")
                print(f"     - Average Correlation: {avg_corr:.3f}")
                print(f"     - Success Rate: {success_rate:.1%}")
                print(f"     - Detection Status: {'✅ Detected' if detected else '❌ Not detected'}")
                
                # Show pattern breakdown
                if 'inversion_detection' in results['judge_inversion']:
                    patterns = results['judge_inversion']['inversion_detection']['patterns']
                    print(f"     - Patterns: {len(patterns['complete_inversion'])} complete, "
                          f"{len(patterns['partial_inversion'])} partial, "
                          f"{len(patterns['no_inversion'])} none")
        
        print(f"\n🛡️ Overall System Status: {'✅ Contamination Detected' if overall_detection else '❌ Contamination Not Detected'}")
        
        # Robustness results
        if robustness_results:
            print(f"\n📈 Robustness Analysis:")
            metrics = robustness_results.get('robustness_metrics', {})
            if 'clean_performance' in metrics:
                clean_r2 = metrics['clean_performance']
                final_r2 = metrics['final_performance']
                degradation = metrics['relative_degradation']
                
                print(f"   • Clean Performance (R²): {clean_r2:.3f}")
                print(f"   • Final Performance (R²): {final_r2:.3f}")
                print(f"   • Performance Degradation: {degradation:.1%}")
                
                if metrics.get('breakdown_rate'):
                    print(f"   • Breakdown Threshold: {metrics['breakdown_rate']:.1%} contamination")
        
        print(f"\n📁 Results Location: {self.experiment_dir}")
        print(f"📊 Visualizations: {self.experiment_dir}/plots/")
        print(f"📋 Summary Report: {self.experiment_dir}/EXPERIMENT_SUMMARY.md")
        print("="*80)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="Enhanced Judge Contamination Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                              # Quick test with 50 samples
  %(prog)s --num-samples 500                    # Full test with 500 samples
  %(prog)s --contamination-types all            # Test all contamination types
  %(prog)s --enable-robustness --enable-viz     # Full analysis with visualizations
  %(prog)s --generate-publication-figures       # Generate publication-ready figures
        """
    )
    
    # Basic execution options
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test mode (uses config quick_mode_samples)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
        help='Number of samples to evaluate (default: 200)'
    )
    
    # Contamination options
    parser.add_argument(
        '--contamination-types',
        nargs='*',
        default=['inverted'],
        help='Contamination types to test: inverted, noise, bias, or "all" (default: inverted)'
    )
    
    # Analysis options
    parser.add_argument(
        '--enable-robustness',
        action='store_true',
        help='Enable robustness analysis (requires human feedback data)'
    )
    
    parser.add_argument(
        '--contamination-rates',
        nargs='*',
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help='Contamination rates for robustness testing (default: 0.0 to 0.5)'
    )
    
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help='Statistical significance level for tests (default: 0.05)'
    )
    
    # Visualization options
    parser.add_argument(
        '--enable-visualizations',
        action='store_true',
        default=True,
        help='Enable visualization generation (default: True)'
    )
    
    parser.add_argument(
        '--disable-visualizations',
        dest='enable_visualizations',
        action='store_false',
        help='Disable visualization generation'
    )
    
    parser.add_argument(
        '--generate-publication-figures',
        action='store_true',
        help='Generate high-quality publication-ready figures'
    )
    
    # Output options
    parser.add_argument(
        '--output-format',
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format for results (default: both)'
    )
    
    return parser


def main() -> int:
    """Main entry point for enhanced contamination experiment"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle special argument processing
    if args.contamination_types == ['all']:
        args.contamination_types = 'all'
    elif len(args.contamination_types) == 1:
        args.contamination_types = args.contamination_types[0]
    
    # Enable robustness analysis for publication figures
    if args.generate_publication_figures:
        args.enable_robustness = True
        args.enable_visualizations = True
    
    # Run enhanced experiment
    try:
        runner = EnhancedContaminationExperimentRunner(args)
        final_report = runner.run_enhanced_experiment()
        
        if final_report['success']:
            print("\n" + "="*80)
            print("🎉 ENHANCED JUDGE CONTAMINATION EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"\n📊 Key Results:")
            print(f"   • {final_report['judges_created']} judges created")
            print(f"   • {final_report['samples_processed']} samples processed")
            print(f"   • Experiment ID: {final_report['timestamp']}")
            print(f"\n📁 Results Directory: {final_report['results_dir']}")
            print(f"📋 Summary Report: {final_report['results_dir']}/EXPERIMENT_SUMMARY.md")
            if final_report.get('visualization_results'):
                print(f"📊 Visualizations: {final_report['results_dir']}/plots/")
            print("="*80)
            return 0
        else:
            print(f"\n❌ Experiment failed: {final_report['error']}")
            return 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Experiment failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
