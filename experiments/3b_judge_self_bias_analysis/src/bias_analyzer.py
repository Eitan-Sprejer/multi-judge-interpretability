"""
Judge Self-Bias Analyzer

This module analyzes the results of the judge self-bias experiment
and generates comprehensive reports and visualizations.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Optional imports for visualization (only if available)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SelfBiasAnalyzer:
    """
    Analyzes judge self-bias experiment results and generates insights.
    
    This class processes the raw experiment results to identify patterns,
    generate visualizations, and create human-readable reports.
    """
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize the analyzer with experiment results.
        
        Args:
            results: Dictionary containing experiment results
        """
        self.results = results
        self.family_bias = results.get("family_bias_analysis", {})
        self.model_families = results.get("model_families", {})
        self.summary = results.get("summary", {})
        
        logger.info("Initialized SelfBiasAnalyzer")
    
    def analyze_self_bias(self) -> Dict[str, Any]:
        """
        Perform comprehensive self-bias analysis.
        
        Returns:
            Dictionary containing detailed bias analysis
        """
        logger.info("Performing comprehensive self-bias analysis")
        
        analysis = {
            "self_bias_patterns": self._analyze_self_bias_patterns(),
            "cross_family_bias": self._analyze_cross_family_bias(),
            "statistical_significance": self._analyze_statistical_significance(),
            "bias_magnitude": self._analyze_bias_magnitude(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_self_bias_patterns(self) -> Dict[str, Any]:
        """Analyze patterns of judges favoring their own model family."""
        self_bias_patterns = {}
        
        for judge_family, response_families in self.family_bias.items():
            if judge_family in response_families:
                self_bias = response_families[judge_family]
                cross_bias_scores = []
                
                # Calculate average score for cross-family responses
                for response_family, results in response_families.items():
                    if response_family != judge_family and results.get("mean_score") is not None:
                        cross_bias_scores.append(results["mean_score"])
                
                if cross_bias_scores:
                    avg_cross_bias = np.mean(cross_bias_scores)
                    self_bias_advantage = self_bias["mean_score"] - avg_cross_bias
                    
                    self_bias_patterns[judge_family] = {
                        "self_family_score": self_bias["mean_score"],
                        "avg_cross_family_score": avg_cross_bias,
                        "self_bias_advantage": self_bias_advantage,
                        "bias_detected": self_bias.get("bias_detected", False),
                        "sample_size": self_bias.get("sample_size", 0)
                    }
        
        return self_bias_patterns
    
    def _analyze_cross_family_bias(self) -> Dict[str, Any]:
        """Analyze bias patterns across different model families."""
        cross_family_bias = {}
        
        for judge_family, response_families in self.family_bias.items():
            cross_family_bias[judge_family] = {}
            
            for response_family, results in response_families.items():
                if response_family != judge_family:
                    cross_family_bias[judge_family][response_family] = {
                        "mean_score": results.get("mean_score"),
                        "bias_detected": results.get("bias_detected", False),
                        "sample_size": results.get("sample_size", 0),
                        "confidence_interval": results.get("confidence_interval")
                    }
        
        return cross_family_bias
    
    def _analyze_statistical_significance(self) -> Dict[str, Any]:
        """Analyze statistical significance of bias patterns."""
        significance_analysis = {}
        
        for judge_family, response_families in self.family_bias.items():
            significance_analysis[judge_family] = {}
            
            for response_family, results in response_families.items():
                if results.get("mean_score") is not None and results.get("standard_error") is not None:
                    # Calculate t-statistic for bias detection
                    t_stat = abs(results["mean_score"]) / results["standard_error"]
                    
                    # Determine significance level (simplified)
                    if t_stat > 2.58:
                        significance = "high"
                    elif t_stat > 1.96:
                        significance = "medium"
                    elif t_stat > 1.65:
                        significance = "low"
                    else:
                        significance = "none"
                    
                    significance_analysis[judge_family][response_family] = {
                        "t_statistic": t_stat,
                        "significance_level": significance,
                        "p_value_approx": self._estimate_p_value(t_stat)
                    }
        
        return significance_analysis
    
    def _analyze_bias_magnitude(self) -> Dict[str, Any]:
        """Analyze the magnitude of detected biases."""
        bias_magnitudes = {}
        
        for judge_family, response_families in self.family_bias.items():
            bias_magnitudes[judge_family] = {}
            
            for response_family, results in response_families.items():
                if results.get("mean_score") is not None:
                    score = results["mean_score"]
                    
                    if abs(score) < 0.1:
                        magnitude = "negligible"
                    elif abs(score) < 0.3:
                        magnitude = "small"
                    elif abs(score) < 0.5:
                        magnitude = "medium"
                    elif abs(score) < 0.7:
                        magnitude = "large"
                    else:
                        magnitude = "very_large"
                    
                    bias_magnitudes[judge_family][response_family] = {
                        "magnitude": magnitude,
                        "absolute_score": abs(score),
                        "direction": "positive" if score > 0 else "negative"
                    }
        
        return bias_magnitudes
    
    def _estimate_p_value(self, t_stat: float) -> str:
        """Estimate p-value based on t-statistic (simplified)."""
        if t_stat > 3.29:
            return "< 0.001"
        elif t_stat > 2.58:
            return "< 0.01"
        elif t_stat > 1.96:
            return "< 0.05"
        elif t_stat > 1.65:
            return "< 0.10"
        else:
            return "> 0.10"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        
        # Analyze self-bias patterns
        self_bias_count = self.summary.get("self_bias_detected", 0)
        total_families = self.summary.get("total_judge_families", 0)
        
        if self_bias_count > 0:
            recommendations.append(
                f"Self-bias detected in {self_bias_count}/{total_families} model families. "
                "Consider diversifying judge selection across model families."
            )
        
        # Analyze bias magnitude
        high_bias_families = []
        for judge_family, response_families in self.family_bias.items():
            for response_family, results in response_families.items():
                if results.get("bias_detected", False):
                    score = abs(results.get("mean_score", 0))
                    if score > 0.5:  # Large bias threshold
                        high_bias_families.append(f"{judge_family}→{response_family}")
        
        if high_bias_families:
            recommendations.append(
                f"Large biases detected in: {', '.join(high_bias_families)}. "
                "These combinations should be avoided or carefully monitored."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "No significant biases detected. Current judge selection appears balanced."
            )
        
        recommendations.extend([
            "Consider implementing bias monitoring in production systems.",
            "Regular bias audits recommended when updating judge models.",
            "Document bias patterns for future research and system design."
        ])
        
        return recommendations
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive markdown report.
        
        Returns:
            Markdown-formatted report string
        """
        logger.info("Generating comprehensive report")
        
        # Get analysis results
        analysis = self.analyze_self_bias()
        
        report = f"""# Judge Self-Bias Analysis Report

## Experiment Overview
- **Experiment**: Judge Self-Bias Analysis
- **Timestamp**: {self.results.get("experiment_info", {}).get("timestamp", "Unknown")}
- **Data Shape**: {self.results.get("experiment_info", {}).get("data_shape", "Unknown")}

## Executive Summary
{self._generate_executive_summary(analysis)}

## Model Families Analyzed
{self._format_model_families()}

## Self-Bias Analysis
{self._format_self_bias_analysis(analysis)}

## Cross-Family Bias Analysis
{self._format_cross_family_bias(analysis)}

## Statistical Significance
{self._format_statistical_significance(analysis)}

## Bias Magnitude Analysis
{self._format_bias_magnitude(analysis)}

## Key Findings
{self._format_key_findings(analysis)}

## Recommendations
{self._format_recommendations(analysis)}

## Methodology
This analysis examines whether judges from specific model families systematically favor responses from the same family. Bias is detected when the mean score difference exceeds the configured threshold ({self.results.get("summary", {}).get("bias_threshold", 0.1)}).

## Data Quality
- **Total Judge Families**: {self.summary.get("total_judge_families", 0)}
- **Total Response Families**: {self.summary.get("total_response_families", 0)}
- **Bias Detected**: {self.summary.get("bias_detected_count", 0)} combinations
- **Self-Bias Detected**: {self.summary.get("self_bias_detected", 0)} families
- **Cross-Bias Detected**: {self.summary.get("cross_bias_detected", 0)} combinations

---
*Report generated automatically by Judge Self-Bias Analyzer*
"""
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        self_bias_count = self.summary.get("self_bias_detected", 0)
        total_families = self.summary.get("total_judge_families", 0)
        
        if self_bias_count == 0:
            return "No significant self-bias detected across model families. Judges appear to evaluate responses fairly regardless of model family."
        elif self_bias_count <= total_families // 2:
            return f"Moderate self-bias detected in {self_bias_count}/{total_families} model families. Some judges show preference for responses from their own family."
        else:
            return f"Significant self-bias detected in {self_bias_count}/{total_families} model families. This suggests systematic bias that requires attention in judge selection."
    
    def _format_model_families(self) -> str:
        """Format model families section."""
        if not self.model_families:
            return "No model families identified in the data."
        
        lines = []
        for family, models in self.model_families.items():
            lines.append(f"- **{family}**: {', '.join(models)}")
        
        return "\n".join(lines)
    
    def _format_self_bias_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format self-bias analysis section."""
        self_bias_patterns = analysis.get("self_bias_patterns", {})
        
        if not self_bias_patterns:
            return "No self-bias patterns detected."
        
        lines = []
        for family, pattern in self_bias_patterns.items():
            lines.append(f"### {family} Family")
            lines.append(f"- **Self-Family Score**: {pattern['self_family_score']:.3f}")
            lines.append(f"- **Cross-Family Average**: {pattern['avg_cross_family_score']:.3f}")
            lines.append(f"- **Bias Advantage**: {pattern['self_bias_advantage']:.3f}")
            lines.append(f"- **Bias Detected**: {'Yes' if pattern['bias_detected'] else 'No'}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_cross_family_bias(self, analysis: Dict[str, Any]) -> str:
        """Format cross-family bias section."""
        cross_family_bias = analysis.get("cross_family_bias", {})
        
        if not cross_family_bias:
            return "No cross-family bias data available."
        
        lines = []
        for judge_family, response_families in cross_family_bias.items():
            lines.append(f"### {judge_family} Judges")
            for response_family, results in response_families.items():
                score = results.get("mean_score", "N/A")
                bias_detected = "Yes" if results.get("bias_detected", False) else "No"
                lines.append(f"- **vs {response_family}**: Score={score}, Bias={bias_detected}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_statistical_significance(self, analysis: Dict[str, Any]) -> str:
        """Format statistical significance section."""
        significance = analysis.get("statistical_significance", {})
        
        if not significance:
            return "Statistical significance analysis not available."
        
        lines = []
        for judge_family, response_families in significance.items():
            lines.append(f"### {judge_family} Judges")
            for response_family, results in response_families.items():
                t_stat = results.get("t_statistic", "N/A")
                sig_level = results.get("significance_level", "N/A")
                p_value = results.get("p_value_approx", "N/A")
                lines.append(f"- **vs {response_family}**: t={t_stat:.2f}, Sig={sig_level}, p{p_value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_bias_magnitude(self, analysis: Dict[str, Any]) -> str:
        """Format bias magnitude section."""
        magnitudes = analysis.get("bias_magnitude", {})
        
        if not magnitudes:
            return "Bias magnitude analysis not available."
        
        lines = []
        for judge_family, response_families in magnitudes.items():
            lines.append(f"### {judge_family} Judges")
            for response_family, results in response_families.items():
                magnitude = results.get("magnitude", "N/A")
                abs_score = results.get("absolute_score", "N/A")
                direction = results.get("direction", "N/A")
                lines.append(f"- **vs {response_family}**: {magnitude} ({direction}, |{abs_score}|)")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_key_findings(self, analysis: Dict[str, Any]) -> str:
        """Format key findings section."""
        findings = []
        
        # Self-bias findings
        self_bias_patterns = analysis.get("self_bias_patterns", {})
        if self_bias_patterns:
            high_bias_families = [
                family for family, pattern in self_bias_patterns.items()
                if pattern.get("self_bias_advantage", 0) > 0.3
            ]
            if high_bias_families:
                findings.append(f"**High self-bias families**: {', '.join(high_bias_families)}")
        
        # Cross-family findings
        cross_bias = analysis.get("cross_family_bias", {})
        if cross_bias:
            negative_bias_combinations = []
            for judge_family, response_families in cross_bias.items():
                for response_family, results in response_families.items():
                    if results.get("mean_score", 0) < -0.3:
                        negative_bias_combinations.append(f"{judge_family}→{response_family}")
            
            if negative_bias_combinations:
                findings.append(f"**Negative bias combinations**: {', '.join(negative_bias_combinations)}")
        
        if not findings:
            findings.append("No significant bias patterns detected in the current analysis.")
        
        return "\n".join(f"- {finding}" for finding in findings)
    
    def _format_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Format recommendations section."""
        recommendations = analysis.get("recommendations", [])
        
        if not recommendations:
            return "No specific recommendations at this time."
        
        return "\n".join(f"- {rec}" for rec in recommendations)
