"""
Judge Self-Bias Analysis Experiment Runner

This module implements the main experiment logic for analyzing whether
LLM judges show bias towards responses from the same model family.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyYAML not available, using default configuration")

logger = logging.getLogger(__name__)


class JudgeSelfBiasRunner:
    """
    Main experiment runner for judge self-bias analysis.
    
    This class orchestrates the experiment to test whether judges from
    a specific model family systematically favor responses from the same family.
    """
    
    def __init__(self, data_path: str, config_path: str, quick_mode: bool = False):
        """
        Initialize the experiment runner.
        
        Args:
            data_path: Path to the dataset with judge scores
            config_path: Path to configuration file
            quick_mode: If True, run with reduced data for quick testing
        """
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.quick_mode = quick_mode
        self.config = self._load_config()
        self.data = None
        self.results = {}
        
        logger.info(f"Initialized JudgeSelfBiasRunner with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, using default configuration")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file fails to load."""
        return {
            "experiment": {
                "name": "judge_self_bias_analysis",
                "description": "Analyze judge bias towards same model family responses"
            },
            "data": {
                "sample_size": 1000 if self.quick_mode else 10000,
                "random_seed": 42
            },
            "analysis": {
                "confidence_level": 0.95,
                "min_sample_size": 50,
                "bias_threshold": 0.1
            },
            "output": {
                "save_intermediate": True,
                "generate_plots": True
            }
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.pkl':
                data = pd.read_pickle(self.data_path)
            elif self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            elif self.data_path.suffix == '.json':
                data = pd.read_json(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            logger.info(f"Loaded data with shape: {data.shape}")
            
            # Apply quick mode sampling if needed
            if self.quick_mode:
                np.random.seed(self.config["data"]["random_seed"])
                sample_size = min(self.config["data"]["sample_size"], len(data))
                data = data.sample(n=sample_size, random_state=self.config["data"]["random_seed"])
                logger.info(f"Quick mode: sampled {len(data)} rows")
            
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def extract_model_families(self) -> Dict[str, List[str]]:
        """
        Extract model families from the data.
        
        Returns:
            Dictionary mapping model family names to lists of model identifiers
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Extracting model families from data")
        
        # This is a placeholder - actual implementation depends on data structure
        # You'll need to adapt this based on your actual data columns
        model_families = {}
        
        # Example implementation - modify based on your data structure
        if 'judge_model' in self.data.columns:
            judge_models = self.data['judge_model'].unique()
            for model in judge_models:
                if 'gpt' in model.lower():
                    family = 'GPT'
                elif 'claude' in model.lower():
                    family = 'Claude'
                elif 'llama' in model.lower():
                    family = 'Llama'
                elif 'gemini' in model.lower():
                    family = 'Gemini'
                else:
                    family = 'Other'
                
                if family not in model_families:
                    model_families[family] = []
                model_families[family].append(model)
        
        if 'response_model' in self.data.columns:
            response_models = self.data['response_model'].unique()
            for model in response_models:
                if 'gpt' in model.lower():
                    family = 'GPT'
                elif 'claude' in model.lower():
                    family = 'Claude'
                elif 'llama' in model.lower():
                    family = 'Llama'
                elif 'gemini' in model.lower():
                    family = 'Gemini'
                else:
                    family = 'Other'
                
                if family not in model_families:
                    model_families[family] = []
                if model not in model_families[family]:
                    model_families[family].append(model)
        
        logger.info(f"Identified model families: {list(model_families.keys())}")
        return model_families
    
    def analyze_family_bias(self) -> Dict[str, Any]:
        """
        Analyze bias patterns within and across model families.
        
        Returns:
            Dictionary containing bias analysis results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Analyzing family bias patterns")
        
        model_families = self.extract_model_families()
        bias_results = {}
        
        for judge_family, judge_models in model_families.items():
            bias_results[judge_family] = {}
            
            for response_family, response_models in model_families.items():
                # Filter data for this judge-family vs response-family combination
                judge_mask = self.data['judge_model'].isin(judge_models)
                response_mask = self.data['response_model'].isin(response_models)
                combined_mask = judge_mask & response_mask
                
                if combined_mask.sum() < self.config["analysis"]["min_sample_size"]:
                    logger.warning(f"Insufficient data for {judge_family} judges vs {response_family} responses")
                    bias_results[judge_family][response_family] = {
                        "sample_size": combined_mask.sum(),
                        "mean_score": None,
                        "bias_detected": False,
                        "confidence_interval": None
                    }
                    continue
                
                # Calculate scores for this combination
                subset = self.data[combined_mask]
                scores = subset['judge_score'].values  # Adjust column name as needed
                
                # Calculate bias metrics
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                se_score = std_score / np.sqrt(len(scores))
                
                # Calculate confidence interval
                try:
                    from scipy import stats
                    ci = stats.t.interval(
                        self.config["analysis"]["confidence_level"],
                        len(scores) - 1,
                        loc=mean_score,
                        scale=se_score
                    )
                except ImportError:
                    logger.warning("scipy not available, skipping confidence interval calculation")
                    ci = None
                
                # Determine if bias is detected
                bias_detected = abs(mean_score) > self.config["analysis"]["bias_threshold"]
                
                bias_results[judge_family][response_family] = {
                    "sample_size": len(scores),
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "bias_detected": bias_detected,
                    "confidence_interval": ci,
                    "standard_error": se_score
                }
        
        return bias_results
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Returns:
            Dictionary containing all experiment results
        """
        logger.info("Starting Judge Self-Bias Analysis Experiment")
        
        # Load data
        self.load_data()
        
        # Analyze family bias
        family_bias = self.analyze_family_bias()
        
        # Compile results
        self.results = {
            "experiment_info": {
                "name": "judge_self_bias_analysis",
                "timestamp": pd.Timestamp.now().isoformat(),
                "config": self.config,
                "data_shape": self.data.shape if self.data is not None else None
            },
            "model_families": self.extract_model_families(),
            "family_bias_analysis": family_bias,
            "summary": self._generate_summary(family_bias)
        }
        
        logger.info("Experiment completed successfully")
        return self.results
    
    def _generate_summary(self, family_bias: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from bias analysis."""
        summary = {
            "total_judge_families": len(family_bias),
            "total_response_families": len(family_bias),
            "bias_detected_count": 0,
            "self_bias_detected": 0,
            "cross_bias_detected": 0
        }
        
        for judge_family, response_families in family_bias.items():
            for response_family, results in response_families.items():
                if results.get("bias_detected", False):
                    summary["bias_detected_count"] += 1
                    
                    if judge_family == response_family:
                        summary["self_bias_detected"] += 1
                    else:
                        summary["cross_bias_detected"] += 1
        
        return summary
