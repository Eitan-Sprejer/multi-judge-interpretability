#!/usr/bin/env python3
"""
Cross-Correlation Analysis Module

Analyzes cross-correlations between judges and personas to understand
their relationship patterns. Creates heatmaps and detailed analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json

from pipeline.core.judge_evaluation import JUDGE_IDS
from pipeline.core.persona_simulation import PERSONAS


class CorrelationAnalyzer:
    """Analyzes cross-correlations between judges and personas."""
    
    def __init__(self, results_dir: Path):
        """
        Initialize correlation analyzer.
        
        Args:
            results_dir: Path to experiment results directory
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def load_experiment_data(self) -> pd.DataFrame:
        """Load experiment data with judge scores and persona feedback."""
        data_path = self.results_dir / "data" / "data_with_judge_scores.pkl"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Experiment data not found at {data_path}")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def extract_judge_scores_matrix(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Extract judge scores as a matrix.
        
        Returns:
            judge_matrix: Shape (n_samples, n_judges)
            judge_names: List of judge names
            valid_indices: Indices of samples with valid judge scores
        """
        judge_scores_list = []
        valid_indices = []
        
        for idx, row in data.iterrows():
            if 'judge_scores' in row and isinstance(row['judge_scores'], list):
                if len(row['judge_scores']) == len(JUDGE_IDS):
                    judge_scores_list.append(row['judge_scores'])
                    valid_indices.append(idx)
        
        judge_matrix = np.array(judge_scores_list)
        judge_names = [judge_id.replace('-judge', '').replace('-', ' ').title() for judge_id in JUDGE_IDS]
        
        return judge_matrix, judge_names, valid_indices
    
    def extract_persona_scores_matrix(self, data: pd.DataFrame, valid_indices: List[int]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract persona scores as a matrix for valid samples.
        
        Args:
            data: Experiment dataframe
            valid_indices: Indices of samples with valid judge scores
            
        Returns:
            persona_matrix: Shape (n_samples, n_personas)
            persona_names: List of persona names
        """
        persona_names = list(PERSONAS.keys())
        persona_scores_list = []
        
        for idx in valid_indices:
            row = data.iloc[idx]
            
            if ('human_feedback' in row and 'personas' in row['human_feedback']):
                personas_feedback = row['human_feedback']['personas']
                
                persona_scores = []
                for persona_name in persona_names:
                    if (persona_name in personas_feedback and 
                        'score' in personas_feedback[persona_name] and
                        personas_feedback[persona_name]['score'] is not None):
                        persona_scores.append(personas_feedback[persona_name]['score'])
                    else:
                        # Use NaN for missing scores
                        persona_scores.append(np.nan)
                
                persona_scores_list.append(persona_scores)
            else:
                # All NaN if no persona feedback
                persona_scores_list.append([np.nan] * len(persona_names))
        
        persona_matrix = np.array(persona_scores_list)
        
        return persona_matrix, persona_names
    
    def compute_judge_correlation_matrix(self, judge_matrix: np.ndarray, judge_names: List[str]) -> np.ndarray:
        """Compute correlation matrix between judges."""
        # Remove samples with any NaN values for judge correlations
        valid_mask = ~np.isnan(judge_matrix).any(axis=1)
        clean_judge_matrix = judge_matrix[valid_mask]
        
        if len(clean_judge_matrix) < 2:
            return np.full((len(judge_names), len(judge_names)), np.nan)
        
        return np.corrcoef(clean_judge_matrix.T)
    
    def compute_persona_correlation_matrix(self, persona_matrix: np.ndarray, persona_names: List[str]) -> np.ndarray:
        """Compute correlation matrix between personas."""
        # Handle NaN values in persona matrix
        n_personas = len(persona_names)
        correlation_matrix = np.full((n_personas, n_personas), np.nan)
        
        for i in range(n_personas):
            for j in range(n_personas):
                # Get scores for personas i and j
                scores_i = persona_matrix[:, i]
                scores_j = persona_matrix[:, j]
                
                # Find samples where both personas have valid scores
                valid_mask = ~(np.isnan(scores_i) | np.isnan(scores_j))
                
                if valid_mask.sum() >= 2:  # Need at least 2 samples for correlation
                    correlation_matrix[i, j] = np.corrcoef(scores_i[valid_mask], scores_j[valid_mask])[0, 1]
        
        return correlation_matrix
    
    def compute_judge_persona_correlation_matrix(self, judge_matrix: np.ndarray, persona_matrix: np.ndarray, 
                                               judge_names: List[str], persona_names: List[str]) -> np.ndarray:
        """Compute correlation matrix between judges and personas."""
        n_judges = len(judge_names)
        n_personas = len(persona_names)
        correlation_matrix = np.full((n_judges, n_personas), np.nan)
        
        for i in range(n_judges):
            for j in range(n_personas):
                # Get scores for judge i and persona j
                judge_scores = judge_matrix[:, i]
                persona_scores = persona_matrix[:, j]
                
                # Find samples where both have valid scores
                valid_mask = ~(np.isnan(judge_scores) | np.isnan(persona_scores))
                
                if valid_mask.sum() >= 2:  # Need at least 2 samples for correlation
                    correlation_matrix[i, j] = np.corrcoef(judge_scores[valid_mask], persona_scores[valid_mask])[0, 1]
        
        return correlation_matrix
    
    def create_correlation_heatmaps(self, judge_corr_matrix: np.ndarray, persona_corr_matrix: np.ndarray,
                                  judge_persona_corr_matrix: np.ndarray, judge_names: List[str], 
                                  persona_names: List[str]) -> None:
        """Create comprehensive correlation heatmaps."""
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Cross-Correlation Analysis: Judges and Personas', fontsize=16, fontweight='bold')
        
        # 1. Judge-Judge Correlations
        ax1 = axes[0, 0]
        mask1 = np.isnan(judge_corr_matrix)
        sns.heatmap(judge_corr_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, vmax=1,
                   xticklabels=judge_names,
                   yticklabels=judge_names,
                   mask=mask1,
                   ax=ax1,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax1.set_title('Judge-Judge Correlations', fontweight='bold', pad=20)
        ax1.set_xlabel('Judges')
        ax1.set_ylabel('Judges')
        
        # 2. Persona-Persona Correlations
        ax2 = axes[0, 1]
        mask2 = np.isnan(persona_corr_matrix)
        sns.heatmap(persona_corr_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, vmax=1,
                   xticklabels=persona_names,
                   yticklabels=persona_names,
                   mask=mask2,
                   ax=ax2,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax2.set_title('Persona-Persona Correlations', fontweight='bold', pad=20)
        ax2.set_xlabel('Personas')
        ax2.set_ylabel('Personas')
        
        # 3. Judge-Persona Correlations
        ax3 = axes[1, 0]
        mask3 = np.isnan(judge_persona_corr_matrix)
        sns.heatmap(judge_persona_corr_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, vmax=1,
                   xticklabels=persona_names,
                   yticklabels=judge_names,
                   mask=mask3,
                   ax=ax3,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax3.set_title('Judge-Persona Cross-Correlations', fontweight='bold', pad=20)
        ax3.set_xlabel('Personas')
        ax3.set_ylabel('Judges')
        
        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary statistics
        judge_stats = self._calculate_summary_stats(judge_corr_matrix, "Judge-Judge")
        persona_stats = self._calculate_summary_stats(persona_corr_matrix, "Persona-Persona")
        cross_stats = self._calculate_summary_stats(judge_persona_corr_matrix, "Judge-Persona")
        
        summary_text = f"""
Summary Statistics

Judge-Judge Correlations:
• Mean: {judge_stats['mean']:.3f}
• Median: {judge_stats['median']:.3f}
• Std: {judge_stats['std']:.3f}
• Min: {judge_stats['min']:.3f}
• Max: {judge_stats['max']:.3f}
• Strong (|r| > 0.7): {judge_stats['strong_count']}
• Moderate (0.3 < |r| ≤ 0.7): {judge_stats['moderate_count']}
• Weak (|r| ≤ 0.3): {judge_stats['weak_count']}

Persona-Persona Correlations:
• Mean: {persona_stats['mean']:.3f}
• Median: {persona_stats['median']:.3f}
• Std: {persona_stats['std']:.3f}
• Min: {persona_stats['min']:.3f}
• Max: {persona_stats['max']:.3f}
• Strong (|r| > 0.7): {persona_stats['strong_count']}
• Moderate (0.3 < |r| ≤ 0.7): {persona_stats['moderate_count']}
• Weak (|r| ≤ 0.3): {persona_stats['weak_count']}

Judge-Persona Cross-Correlations:
• Mean: {cross_stats['mean']:.3f}
• Median: {cross_stats['median']:.3f}
• Std: {cross_stats['std']:.3f}
• Min: {cross_stats['min']:.3f}
• Max: {cross_stats['max']:.3f}
• Strong (|r| > 0.7): {cross_stats['strong_count']}
• Moderate (0.3 < |r| ≤ 0.7): {cross_stats['moderate_count']}
• Weak (|r| ≤ 0.3): {cross_stats['weak_count']}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the plot
        save_path = self.plots_dir / "cross_correlation_heatmaps.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cross-correlation heatmaps saved to: {save_path}")
    
    def _calculate_summary_stats(self, correlation_matrix: np.ndarray, matrix_type: str) -> Dict[str, Any]:
        """Calculate summary statistics for a correlation matrix."""
        # Flatten and remove NaN values and diagonal (self-correlations)
        if matrix_type in ["Judge-Judge", "Persona-Persona"]:
            # Remove diagonal elements for self-correlation matrices
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            values = correlation_matrix[mask]
        else:
            # Keep all values for cross-correlation matrix
            values = correlation_matrix.flatten()
        
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            return {
                'mean': float('nan'), 'median': float('nan'), 'std': float('nan'),
                'min': float('nan'), 'max': float('nan'),
                'strong_count': 0, 'moderate_count': 0, 'weak_count': 0
            }
        
        abs_values = np.abs(values)
        strong_count = np.sum(abs_values > 0.7)
        moderate_count = np.sum((abs_values > 0.3) & (abs_values <= 0.7))
        weak_count = np.sum(abs_values <= 0.3)
        
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'strong_count': int(strong_count),
            'moderate_count': int(moderate_count),
            'weak_count': int(weak_count)
        }
    
    def save_correlation_analysis(self, judge_corr_matrix: np.ndarray, persona_corr_matrix: np.ndarray,
                                judge_persona_corr_matrix: np.ndarray, judge_names: List[str], 
                                persona_names: List[str]) -> None:
        """Save detailed correlation analysis to JSON."""
        
        # Convert matrices to lists for JSON serialization
        def matrix_to_dict(matrix, row_names, col_names):
            result = {}
            for i, row_name in enumerate(row_names):
                result[row_name] = {}
                for j, col_name in enumerate(col_names):
                    value = matrix[i, j]
                    result[row_name][col_name] = float(value) if not np.isnan(value) else None
            return result
        
        analysis = {
            'judge_judge_correlations': matrix_to_dict(judge_corr_matrix, judge_names, judge_names),
            'persona_persona_correlations': matrix_to_dict(persona_corr_matrix, persona_names, persona_names),
            'judge_persona_correlations': matrix_to_dict(judge_persona_corr_matrix, judge_names, persona_names),
            'summary_statistics': {
                'judge_judge': self._calculate_summary_stats(judge_corr_matrix, "Judge-Judge"),
                'persona_persona': self._calculate_summary_stats(persona_corr_matrix, "Persona-Persona"),
                'judge_persona': self._calculate_summary_stats(judge_persona_corr_matrix, "Judge-Persona")
            }
        }
        
        # Save to JSON
        save_path = self.results_dir / "results" / "cross_correlation_analysis.json"
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Detailed correlation analysis saved to: {save_path}")
    
    def run_correlation_analysis(self) -> Dict[str, Any]:
        """Run complete correlation analysis."""
        print("Loading experiment data...")
        data = self.load_experiment_data()
        
        print("Extracting judge scores...")
        judge_matrix, judge_names, valid_indices = self.extract_judge_scores_matrix(data)
        
        print("Extracting persona scores...")
        persona_matrix, persona_names = self.extract_persona_scores_matrix(data, valid_indices)
        
        print("Computing correlation matrices...")
        judge_corr_matrix = self.compute_judge_correlation_matrix(judge_matrix, judge_names)
        persona_corr_matrix = self.compute_persona_correlation_matrix(persona_matrix, persona_names)
        judge_persona_corr_matrix = self.compute_judge_persona_correlation_matrix(
            judge_matrix, persona_matrix, judge_names, persona_names
        )
        
        print("Creating heatmaps...")
        self.create_correlation_heatmaps(
            judge_corr_matrix, persona_corr_matrix, judge_persona_corr_matrix,
            judge_names, persona_names
        )
        
        print("Saving detailed analysis...")
        self.save_correlation_analysis(
            judge_corr_matrix, persona_corr_matrix, judge_persona_corr_matrix,
            judge_names, persona_names
        )
        
        return {
            'judge_matrix': judge_matrix,
            'persona_matrix': persona_matrix,
            'judge_names': judge_names,
            'persona_names': persona_names,
            'judge_corr_matrix': judge_corr_matrix,
            'persona_corr_matrix': persona_corr_matrix,
            'judge_persona_corr_matrix': judge_persona_corr_matrix
        }


def main():
    """Main entry point for standalone correlation analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-correlation analysis for judges and personas")
    parser.add_argument('--results-dir', required=True, help='Path to experiment results directory')
    
    args = parser.parse_args()
    
    analyzer = CorrelationAnalyzer(args.results_dir)
    results = analyzer.run_correlation_analysis()
    
    print("\n" + "="*60)
    print("CROSS-CORRELATION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(results['judge_names'])} judges and {len(results['persona_names'])} personas")
    print(f"Judge matrix shape: {results['judge_matrix'].shape}")
    print(f"Persona matrix shape: {results['persona_matrix'].shape}")
    print("Check the plots/ directory for visualization results")
    

if __name__ == "__main__":
    main()