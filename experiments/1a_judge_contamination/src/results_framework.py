#!/usr/bin/env python3
"""
Results Framework for Judge Contamination Analysis

Simple result processing and metric computation for contamination experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ContaminationMetrics:
    """Basic metrics for contamination analysis."""
    
    average_correlation: float
    contamination_success_rate: float
    system_inversion_detected: bool
    statistical_significance: bool
    p_value: float

class ResultsProcessor:
    """Simple processor for contamination experiment results."""
    
    def __init__(self, output_dir: Path):
        """Initialize the results processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def process_contamination_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process contamination analysis results and compute basic metrics."""
        
        logger.info("Processing contamination analysis results...")
        
        processed_results = {
            'timestamp': datetime.now().isoformat(),
            'contamination_metrics': {},
            'summary': {}
        }
        
        # Process each contamination type
        for contamination_type, results in analysis_results.items():
            logger.info(f"Processing {contamination_type} results...")
            
            metrics = self._compute_basic_metrics(results)
            processed_results['contamination_metrics'][contamination_type] = {
                'average_correlation': metrics.average_correlation,
                'success_rate': metrics.contamination_success_rate,
                'system_detected': metrics.system_inversion_detected,
                'significant': metrics.statistical_significance,
                'p_value': metrics.p_value
            }
        
        # Generate summary
        processed_results['summary'] = self._generate_summary(processed_results['contamination_metrics'])
        
        # Save results
        self._save_results(processed_results)
        
        return processed_results
    
    def _compute_basic_metrics(self, results: Dict[str, Any]) -> ContaminationMetrics:
        """Compute basic metrics for contamination analysis."""
        
        if 'judge_inversion' not in results:
            return ContaminationMetrics(
                average_correlation=0.0,
                contamination_success_rate=0.0,
                system_inversion_detected=False,
                statistical_significance=False,
                p_value=1.0
            )
        
        judge_results = results['judge_inversion']
        agg_metrics = judge_results.get('aggregate_metrics', {})
        stats_tests = judge_results.get('statistical_tests', {})
        system_stats = stats_tests.get('system_level', {})
        dist_shift = system_stats.get('distribution_shift', {})
        
        return ContaminationMetrics(
            average_correlation=float(agg_metrics.get('average_correlation', 0.0)),
            contamination_success_rate=float(agg_metrics.get('contamination_rate', 0.0)),
            system_inversion_detected=bool(agg_metrics.get('system_inversion_detected', False)),
            statistical_significance=bool(dist_shift.get('significant', False)),
            p_value=float(dist_shift.get('ks_p_value', 1.0))
        )
    
    def _generate_summary(self, contamination_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple summary of the results."""
        
        total_types = len(contamination_metrics)
        detected = sum(1 for metrics in contamination_metrics.values() 
                      if metrics.get('system_detected', False))
        significant = sum(1 for metrics in contamination_metrics.values() 
                         if metrics.get('significant', False))
        
        avg_correlation = np.mean([metrics.get('average_correlation', 0.0) 
                                  for metrics in contamination_metrics.values()])
        
        return {
            'total_contamination_types': total_types,
            'detected_contaminations': detected,
            'significant_results': significant,
            'average_correlation': float(avg_correlation),
            'detection_rate': detected / total_types if total_types > 0 else 0.0
        }
    
    def _save_results(self, processed_results: Dict[str, Any]) -> None:
        """Save results as JSON."""
        
        results_path = self.output_dir / "contamination_results.json"
        with open(results_path, 'w') as f:
            json.dump(processed_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def export_csv(self, processed_results: Dict[str, Any]) -> str:
        """Export results as CSV."""
        
        contamination_metrics = processed_results.get('contamination_metrics', {})
        
        data = []
        for ctype, metrics in contamination_metrics.items():
            data.append({
                'contamination_type': ctype,
                'average_correlation': metrics.get('average_correlation', 0.0),
                'success_rate': metrics.get('success_rate', 0.0),
                'system_detected': metrics.get('system_detected', False),
                'significant': metrics.get('significant', False),
                'p_value': metrics.get('p_value', 1.0)
            })
        
        df = pd.DataFrame(data)
        csv_path = self.output_dir / "contamination_results.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)