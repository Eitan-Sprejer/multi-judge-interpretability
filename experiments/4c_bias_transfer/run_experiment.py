"""
Experiment 4C: Framing Effects and Bias Transfer in Aggregated Models

Main runner script for testing whether learned judge aggregators inherit or mitigate
cognitive biases present in individual reward models, following Christian et al. (2024).

Usage:
    python run_experiment.py [--quick] [--use-real-judges] [--min-tokens 200]
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

# Add experiment source to path
experiment_src = Path(__file__).parent / "src"
sys.path.append(str(experiment_src))

# Local imports
from data_preparation import BiasDataPreparator
from judge_scoring import BiasJudgeScorer
from bias_analysis import BiasAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment orchestrator for bias transfer analysis"""
    
    def __init__(self, args):
        """
        Initialize experiment runner
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.data_preparator = BiasDataPreparator()
        self.judge_scorer = BiasJudgeScorer(
            judges_data_path=str(project_root / "dataset" / "data_with_judge_scores.pkl"),
            mlp_model_path=str(project_root / "models" / "agg_model_mlp.pt"),
            gam_model_path=str(project_root / "models" / "agg_model_gam.pt"),
            judge_subset=args.judge_subset
        )
        self.bias_analyzer = BiasAnalyzer(normalize_scores=args.normalize_scores)
        
        # Load vocabulary filter if provided
        self.vocabulary_filter = None
        if args.vocabulary_file and Path(args.vocabulary_file).exists():
            logger.info(f"Loading vocabulary filter from {args.vocabulary_file}")
            with open(args.vocabulary_file, 'r') as f:
                self.vocabulary_filter = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded vocabulary filter: {len(self.vocabulary_filter)} tokens")
        
        logger.info(f"Experiment 4C initialized - Run ID: {self.timestamp}")
        if args.normalize_scores:
            logger.info("Score normalization enabled")
    
    def run_full_experiment(self):
        """Run the complete bias transfer experiment"""
        logger.info("="*60)
        logger.info("EXPERIMENT 4C: FRAMING EFFECTS AND BIAS TRANSFER")
        logger.info("="*60)
        
        # Step 1: Data Preparation
        logger.info("Step 1: Preparing token dataset...")
        token_dataset = self._prepare_token_data()
        
        # Step 2: Score Collection
        logger.info("Step 2: Collecting judge scores...")
        scores_dataset = self._collect_judge_scores(token_dataset)
        
        # Step 3: Bias Analysis
        logger.info("Step 3: Analyzing bias patterns...")
        analysis_results = self._analyze_biases(scores_dataset)
        
        # Step 4: Generate Report
        logger.info("Step 4: Generating final report...")
        final_report = self._generate_final_report(
            token_dataset, scores_dataset, analysis_results
        )
        
        logger.info("="*60)
        logger.info("EXPERIMENT 4C COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved in: {self.results_dir}")
        logger.info("="*60)
        
        return final_report
    
    def _prepare_token_data(self):
        """Prepare the token dataset for bias analysis"""
        logger.info("Preparing AFINN-111 and neutral token dataset...")
        
        # Determine minimum tokens based on quick mode
        min_tokens = 50 if self.args.quick else self.args.min_tokens
        
        # Prepare dataset with vocabulary filtering
        token_dataset = self.data_preparator.prepare_bias_dataset(
            min_tokens=min_tokens,
            vocabulary_filter=self.vocabulary_filter,
            save_path=str(self.results_dir / f"{self.timestamp}_token_dataset.pkl")
        )
        
        # Validate dataset
        validation = self.data_preparator.validate_dataset(token_dataset)
        
        if not validation['overall_valid']:
            logger.warning("Dataset validation failed - proceeding with caution")
            
        logger.info(f"Token dataset prepared: {len(token_dataset)} tokens")
        logger.info(f"Positive sentiment: {token_dataset['is_positive'].sum()}")
        logger.info(f"Negative sentiment: {token_dataset['is_negative'].sum()}")
        logger.info(f"Neutral control: {token_dataset['is_neutral_control'].sum()}")
        
        return token_dataset
    
    def _collect_judge_scores(self, token_dataset):
        """Collect scores from all judges and aggregators"""
        logger.info("Collecting scores with framing prompts...")
        
        # Subsample for quick mode
        if self.args.quick:
            sample_size = min(100, len(token_dataset))
            token_sample = token_dataset.sample(n=sample_size, random_state=42)
            logger.info(f"Quick mode: using {sample_size} tokens")
        else:
            token_sample = token_dataset
        
        # Score tokens with judges
        scores_dataset = self.judge_scorer.score_tokens_with_judges(
            token_sample, 
            use_mock_scoring=not self.args.use_real_judges
        )
        
        # Save scores
        scores_path = self.results_dir / f"{self.timestamp}_bias_scores.pkl"
        self.judge_scorer.save_scores(scores_dataset, str(scores_path))
        
        logger.info(f"Score collection completed: {len(scores_dataset)} score records")
        logger.info(f"Prompts used: {scores_dataset['prompt_type'].nunique()}")
        logger.info(f"Tokens scored: {scores_dataset['token'].nunique()}")
        
        return scores_dataset
    
    def _analyze_biases(self, scores_dataset):
        """Run comprehensive bias analysis"""
        logger.info("Running bias analysis...")
        
        # Identify model columns
        model_columns = self._identify_model_columns(scores_dataset)
        logger.info(f"Analyzing {len(model_columns)} models: {model_columns}")
        
        # Run framing effects analysis
        framing_results = self.bias_analyzer.analyze_framing_effects(
            scores_dataset, model_columns
        )
        
        # Run frequency bias analysis
        frequency_results = self.bias_analyzer.analyze_frequency_bias(
            scores_dataset, model_columns
        )
        
        # Run significance tests
        significance_results = self.bias_analyzer.run_significance_tests(
            scores_dataset, model_columns
        )
        
        # Compare bias reduction
        individual_results, aggregator_results = self._separate_model_results(
            framing_results, frequency_results, model_columns
        )
        
        comparisons = self.bias_analyzer.compare_bias_reduction(
            individual_results, aggregator_results
        )
        
        # Save analysis results
        analysis_path = self.results_dir / f"{self.timestamp}_bias_analysis.json"
        self.bias_analyzer.save_results(str(analysis_path))
        
        logger.info("Bias analysis completed")
        logger.info(f"Individual judges analyzed: {len(individual_results)}")
        logger.info(f"Aggregators analyzed: {len(aggregator_results)}")
        
        return {
            'framing_results': framing_results,
            'frequency_results': frequency_results,
            'significance_results': significance_results,
            'comparisons': comparisons
        }
    
    def _identify_model_columns(self, scores_df):
        """Identify which columns contain model scores"""
        model_columns = []
        
        # Individual judges
        judge_cols = [col for col in scores_df.columns if col.startswith('judge_')]
        model_columns.extend(judge_cols)
        
        # Aggregators
        if 'naive_average' in scores_df.columns:
            model_columns.append('naive_average')
        if 'mlp_aggregator' in scores_df.columns:
            model_columns.append('mlp_aggregator')
        if 'gam_aggregator' in scores_df.columns:
            model_columns.append('gam_aggregator')
        
        return model_columns
    
    def _separate_model_results(self, framing_results, frequency_results, model_columns):
        """Separate individual judge results from aggregator results"""
        individual_results = {}
        aggregator_results = {}
        
        for model_col in model_columns:
            combined_results = {}
            
            if model_col in framing_results:
                combined_results.update(framing_results[model_col])
            if model_col in frequency_results:
                combined_results.update(frequency_results[model_col])
            
            if model_col.startswith('judge_'):
                individual_results[model_col] = combined_results
            else:
                aggregator_results[model_col] = combined_results
        
        return individual_results, aggregator_results
    
    def _generate_final_report(self, token_dataset, scores_dataset, analysis_results):
        """Generate comprehensive final report"""
        logger.info("Generating final experiment report...")
        
        # Generate summary from analyzer
        summary = self.bias_analyzer.generate_summary_report()
        
        # Add experiment metadata
        report = {
            'experiment_info': {
                'name': 'Experiment 4C: Framing Effects and Bias Transfer',
                'timestamp': self.timestamp,
                'run_date': datetime.now().isoformat(),
                'quick_mode': self.args.quick,
                'use_real_judges': self.args.use_real_judges,
                'min_tokens': self.args.min_tokens
            },
            'dataset_info': {
                'total_tokens': len(token_dataset),
                'positive_tokens': int(token_dataset['is_positive'].sum()),
                'negative_tokens': int(token_dataset['is_negative'].sum()),
                'neutral_control_tokens': int(token_dataset['is_neutral_control'].sum()),
                'total_score_records': len(scores_dataset),
                'unique_tokens_scored': int(scores_dataset['token'].nunique())
            },
            'analysis_summary': summary,
            'detailed_results': analysis_results
        }
        
        # Save full report
        report_path = self.results_dir / f"{self.timestamp}_FINAL_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        markdown_summary = self._generate_markdown_summary(report)
        summary_path = self.results_dir / f"{self.timestamp}_EXPERIMENT_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(markdown_summary)
        
        logger.info(f"Final report saved: {report_path}")
        logger.info(f"Summary saved: {summary_path}")
        
        return report
    
    def _generate_markdown_summary(self, report):
        """Generate human-readable markdown summary"""
        summary = report['analysis_summary']
        
        md = f"""# Experiment 4C: Framing Effects and Bias Transfer
        
## Run Information
- **Date**: {report['experiment_info']['run_date'][:19]}
- **Mode**: {'Quick' if report['experiment_info']['quick_mode'] else 'Full'}
- **Judge Scoring**: {'Real' if report['experiment_info']['use_real_judges'] else 'Mock'}

## Dataset Summary
- **Total Tokens**: {report['dataset_info']['total_tokens']}
- **Positive Sentiment**: {report['dataset_info']['positive_tokens']}
- **Negative Sentiment**: {report['dataset_info']['negative_tokens']}  
- **Neutral Control**: {report['dataset_info']['neutral_control_tokens']}
- **Score Records**: {report['dataset_info']['total_score_records']}

## Key Findings

"""
        
        # Add conclusions
        if 'conclusions' in summary:
            for conclusion in summary['conclusions']:
                md += f"- {conclusion}\\n"
        
        # Add framing effects summary
        if 'framing_effects_summary' in summary:
            framing = summary['framing_effects_summary']
            md += f"""
## Framing Effects Analysis
- **Models Analyzed**: {framing.get('n_models_analyzed', 'N/A')}
- **Mean Framing Flip**: {framing.get('mean_framing_flip', 'N/A'):.3f}
- **Models with Strong Bias**: {framing.get('models_with_strong_bias', 'N/A')}
"""
        
        # Add frequency bias summary
        if 'frequency_bias_summary' in summary:
            frequency = summary['frequency_bias_summary']
            md += f"""
## Frequency Bias Analysis
- **Models Analyzed**: {frequency.get('n_models_analyzed', 'N/A')}
- **Mean Frequency Bias**: {frequency.get('mean_frequency_bias', 'N/A'):.3f}
- **Models with Strong Bias**: {frequency.get('models_with_strong_bias', 'N/A')}
"""
        
        # Add bias reduction summary
        if 'comparisons' in report['detailed_results']:
            comparisons = report['detailed_results']['comparisons']
            framing_reductions = [v for k, v in comparisons.items() if 'framing_reduction' in k]
            frequency_reductions = [v for k, v in comparisons.items() if 'frequency_reduction' in k]
            
            if framing_reductions or frequency_reductions:
                md += "\\n## Bias Reduction Results\\n"
                
                if framing_reductions:
                    avg_framing = np.mean(framing_reductions)
                    md += f"- **Framing Bias Reduction**: {avg_framing:.1f}% average\\n"
                
                if frequency_reductions:
                    avg_frequency = np.mean(frequency_reductions)
                    md += f"- **Frequency Bias Reduction**: {avg_frequency:.1f}% average\\n"
        
        md += f"""
## Files Generated
- Full results: `{report['experiment_info']['timestamp']}_bias_analysis.json`
- Score data: `{report['experiment_info']['timestamp']}_bias_scores.pkl`
- Token data: `{report['experiment_info']['timestamp']}_token_dataset.pkl`

## Next Steps
1. Run visualization script: `python analyze_results.py`
2. Review detailed analysis in JSON files
3. Compare with other experiments in Track 4
"""
        
        return md


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Experiment 4C: Framing Effects and Bias Transfer Analysis"
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick test with reduced dataset size'
    )
    
    parser.add_argument(
        '--use-real-judges',
        action='store_true',
        help='Use real judge API calls instead of mock scoring'
    )
    
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=200,
        help='Minimum number of AFINN tokens to include (default: 200)'
    )
    
    parser.add_argument(
        '--normalize-scores',
        action='store_true',
        help='Normalize all model scores to [0,1] range'
    )
    
    parser.add_argument(
        '--vocabulary-file',
        type=str,
        help='Path to file containing allowed vocabulary tokens (one per line)'
    )
    
    parser.add_argument(
        '--judge-subset',
        type=str, 
        nargs='+',
        help='Specific judge IDs to use (defaults to all judges in judge_rubrics.py)'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    try:
        runner = ExperimentRunner(args)
        final_report = runner.run_full_experiment()
        
        print("\\n" + "="*60)
        print("EXPERIMENT 4C COMPLETED SUCCESSFULLY!")
        print(f"Check results in: experiments/4c_bias_transfer/results/")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())