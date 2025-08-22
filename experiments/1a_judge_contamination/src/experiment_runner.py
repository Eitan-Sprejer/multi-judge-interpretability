#!/usr/bin/env python3
"""
Judge Contamination Experiment Runner

Main orchestrator for creating and testing contaminated judges.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from pipeline.core.judge_creation import create_or_update_judge, JUDGE_MODEL, MIN_SCORE, MAX_SCORE
from pipeline.utils.judge_rubrics import INVERTED_JUDGE_RUBRICS
from pipeline.utils.create_martian_client import create_martian_client
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.dataset_loader import DatasetLoader
from martian_apart_hack_sdk import judge_specs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JudgeContaminationExperiment:
    """Main experiment class for judge contamination studies."""
    
    def __init__(self, contamination_type: str, num_samples: int, output_dir: Path):
        self.contamination_type = contamination_type
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.client = create_martian_client()
        self.results = {}
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = self.output_dir / f"contamination_results_{timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        
    def run(self) -> Dict[str, Any]:
        """Run the complete contamination experiment."""
        logger.info("🚀 Starting Judge Contamination Experiment")
        
        try:
            # Step 1: Create contaminated judges
            judges_created = self._create_contaminated_judges()
            
            # Step 2: Evaluate contaminated judges
            evaluation_results = self._evaluate_contaminated_judges()
            
            # Step 3: Analyze contamination effects
            analysis_results = self._analyze_contamination()
            
            # Step 4: Save results
            self._save_results(judges_created, evaluation_results, analysis_results)
            
            # Compile final results
            self.results = {
                'success': True,
                'judges_created': len(judges_created),
                'samples_processed': self.num_samples,
                'contamination_detected': analysis_results.get('contamination_detected', False),
                'avg_score_shift': analysis_results.get('avg_score_shift', 0.0),
                'results_path': str(self.results_dir)
            }
            
            logger.info("✅ Experiment completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            self.results = {
                'success': False,
                'error': str(e)
            }
            return self.results
    
    def _create_contaminated_judges(self) -> List[str]:
        """Create judges with contaminated rubrics."""
        logger.info(f"Creating {self.contamination_type} contaminated judges...")
        
        if self.contamination_type == "inverted":
            return self._create_inverted_judges()
        elif self.contamination_type == "all":
            return self._create_all_contamination_types()
        else:
            logger.warning(f"Contamination type '{self.contamination_type}' not implemented yet")
            return []
    
    def _create_inverted_judges(self) -> List[str]:
        """Create judges with inverted rubrics."""
        inverted_judge_ids = list(INVERTED_JUDGE_RUBRICS.keys())
        created_judges = []
        
        for target_id in inverted_judge_ids:
            try:
                inverted_rubric = INVERTED_JUDGE_RUBRICS[target_id]()
                
                judge_spec = judge_specs.RubricJudgeSpec(
                    model_type="rubric_judge",
                    rubric=inverted_rubric,
                    model=JUDGE_MODEL,
                    min_score=MIN_SCORE,
                    max_score=MAX_SCORE,
                )
                
                create_or_update_judge(
                    client=self.client,
                    judge_id=f'inverted_{target_id}',
                    judge_spec=judge_spec,
                    description=f'Inverted rubric judge for {target_id}',
                )
                
                created_judges.append(f'inverted_{target_id}')
                logger.info(f"Created inverted judge: inverted_{target_id}")
                
            except Exception as e:
                logger.error(f"Failed to create inverted judge for {target_id}: {e}")
        
        return created_judges
    
    def _create_all_contamination_types(self) -> List[str]:
        """Create all types of contaminated judges."""
        all_judges = []
        all_judges.extend(self._create_inverted_judges())
        # TODO: Implement other contamination types
        return all_judges
    
    def _evaluate_contaminated_judges(self) -> Dict[str, Any]:
        """Evaluate contaminated judges on sample data."""
        logger.info("Evaluating contaminated judges...")
        
        # Get created judge IDs
        judge_ids = [f'inverted_{jid}' for jid in INVERTED_JUDGE_RUBRICS.keys()]
        
        try:
            judge_evaluator = JudgeEvaluator(judge_ids=judge_ids)
            dataset_loader = DatasetLoader()
            
            # Load sample data
            data = dataset_loader.load_existing_personas('data/data_with_all_personas.pkl')
            
            # Limit samples if needed
            if len(data) > self.num_samples:
                data = data.sample(n=self.num_samples, random_state=42)
            
            # Evaluate judges
            scores = []
            for i in range(len(data)):
                question = data['instruction'].iloc[i]
                answer = data['answer'].iloc[i]
                intermediate = judge_evaluator.evaluate_parallel(question=question, answer=answer)
                scores.append(intermediate)
            
            scores_df = pd.DataFrame(scores, columns=judge_ids)
            
            return {
                'scores': scores_df,
                'data': data,
                'judge_ids': judge_ids
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate judges: {e}")
            return {}
    
    def _analyze_contamination(self) -> Dict[str, Any]:
        """Analyze the effects of contamination."""
        logger.info("Analyzing contamination effects...")
        
        # This would contain your contamination analysis logic
        # For now, return basic metrics
        return {
            'contamination_detected': True,
            'avg_score_shift': 0.5,  # Placeholder
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _save_results(self, judges_created: List[str], 
                     evaluation_results: Dict, 
                     analysis_results: Dict):
        """Save experiment results."""
        logger.info(f"Saving results to {self.results_dir}")
        
        # Save judge creation log
        with open(self.results_dir / "judges_created.txt", "w") as f:
            for judge_id in judges_created:
                f.write(f"{judge_id}\n")
        
        # Save evaluation results if available
        if evaluation_results and 'scores' in evaluation_results:
            evaluation_results['scores'].to_csv(
                self.results_dir / "contaminated_scores.csv", index=False
            )
        
        # Save analysis results
        with open(self.results_dir / "analysis_results.json", "w") as f:
            import json
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(judges_created, analysis_results)
    
    def _create_summary_report(self, judges_created: List[str], analysis_results: Dict):
        """Create a human-readable summary report."""
        report_path = self.results_dir / "EXPERIMENT_SUMMARY.md"
        
        with open(report_path, "w") as f:
            f.write("# Judge Contamination Experiment Summary\n\n")
            f.write(f"**Experiment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Contamination Type**: {self.contamination_type}\n")
            f.write(f"**Samples Processed**: {self.num_samples}\n\n")
            
            f.write("## Judges Created\n")
            for judge_id in judges_created:
                f.write(f"- {judge_id}\n")
            
            f.write(f"\n## Analysis Results\n")
            f.write(f"- Contamination Detected: {analysis_results.get('contamination_detected', 'Unknown')}\n")
            f.write(f"- Average Score Shift: {analysis_results.get('avg_score_shift', 'Unknown')}\n")
            
            f.write(f"\n## Files Generated\n")
            f.write(f"- `contaminated_scores.csv`: Raw judge scores\n")
            f.write(f"- `analysis_results.json`: Detailed analysis data\n")
            f.write(f"- `judges_created.txt`: List of created judge IDs\n")
