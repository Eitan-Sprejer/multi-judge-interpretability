"""
Judge Self-Bias Analysis Experiment Source Package

This package contains the core implementation for analyzing whether
LLM judges show bias towards responses from the same model family.
"""

from .experiment_runner import JudgeSelfBiasRunner
from .bias_analyzer import SelfBiasAnalyzer

__all__ = [
    "JudgeSelfBiasRunner",
    "SelfBiasAnalyzer"
]

__version__ = "1.0.0"
