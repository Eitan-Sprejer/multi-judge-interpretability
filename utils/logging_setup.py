#!/usr/bin/env python3
"""
Universal Logging Configuration for Multi-Judge Interpretability Project
Sets up comprehensive logging to both console and files for any experiment
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def setup_universal_logging(
    experiment_name: str = "baseline_experiment",
    log_dir: str = "logs", 
    log_level: int = logging.INFO,
    console_level: int = logging.INFO
) -> Dict[str, Any]:
    """
    Set up comprehensive logging for any experiment in the project
    
    Args:
        experiment_name: Name of the experiment (used in log filenames)
        log_dir: Directory to store log files
        log_level: File logging level (INFO, DEBUG, etc.)
        console_level: Console logging level
        
    Returns:
        Dictionary with logger info and paths
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_timestamp = f"{experiment_name}_{timestamp}"
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # 1. Console handler (for immediate feedback)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. Main experiment log file (all INFO and above)
    main_log_file = log_path / f"{experiment_timestamp}.log"
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # 3. Debug log file (everything including DEBUG)
    debug_log_file = log_path / f"debug_{experiment_timestamp}.log"
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_handler)
    
    # 4. Progress/Results log (important events only)
    progress_log_file = log_path / f"progress_{experiment_timestamp}.log"
    progress_handler = logging.FileHandler(progress_log_file)
    progress_handler.setLevel(logging.WARNING)  # Only warnings and above
    progress_handler.setFormatter(simple_formatter)
    
    # Create a custom logger for progress tracking
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.addHandler(progress_handler)
    progress_logger.setLevel(logging.INFO)
    
    # Log the setup
    logging.info("=" * 80)
    logging.info(f"LOGGING INITIALIZED: {experiment_name.upper()}")
    logging.info("=" * 80)
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Main log: {main_log_file}")
    logging.info(f"Debug log: {debug_log_file}")
    logging.info(f"Progress log: {progress_log_file}")
    logging.info(f"Log level: {logging.getLevelName(log_level)}")
    logging.info("=" * 80)
    
    return {
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'main_log': main_log_file,
        'debug_log': debug_log_file,
        'progress_log': progress_log_file,
        'progress_logger': progress_logger,
        'log_dir': log_path
    }

def log_experiment_start(config: Dict[str, Any]):
    """Log experiment configuration and parameters"""
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning("🚀 EXPERIMENT START")
    logging.info("Experiment Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")

def log_experiment_progress(current: int, total: int, item_type: str = "items", stage: str = "Processing"):
    """Log progress updates"""
    progress_pct = (current / total) * 100
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning(f"📊 {stage}: {current}/{total} {item_type} ({progress_pct:.1f}%)")
    
    # Also log to main logger at INFO level
    logging.info(f"📊 {stage}: {current}/{total} {item_type} ({progress_pct:.1f}%)")

def log_experiment_milestone(milestone: str, details: Optional[Dict[str, Any]] = None):
    """Log experiment milestones"""
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning(f"🎯 MILESTONE: {milestone}")
    logging.info(f"🎯 MILESTONE: {milestone}")
    
    if details:
        logging.info("Milestone Details:")
        for key, value in details.items():
            logging.info(f"  {key}: {value}")

def log_experiment_complete(results_summary: Dict[str, Any]):
    """Log experiment completion and results"""
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning("✅ EXPERIMENT COMPLETE")
    logging.info("Results Summary:")
    for key, value in results_summary.items():
        logging.info(f"  {key}: {value}")

def log_model_results(model_name: str, train_metrics: Dict[str, float], test_metrics: Dict[str, float]):
    """Log model training results with special formatting"""
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning(f"🔍 MODEL RESULTS: {model_name}")
    
    logging.info(f"{model_name} Results:")
    logging.info(f"  Train - R²: {train_metrics.get('r2', 'N/A'):.4f}, MAE: {train_metrics.get('mae', 'N/A'):.4f}, MSE: {train_metrics.get('mse', 'N/A'):.4f}")
    logging.info(f"  Test  - R²: {test_metrics.get('r2', 'N/A'):.4f}, MAE: {test_metrics.get('mae', 'N/A'):.4f}, MSE: {test_metrics.get('mse', 'N/A'):.4f}")

def log_data_validation(stage: str, samples: int, valid_samples: int, validation_details: Optional[Dict[str, Any]] = None):
    """Log data validation results"""
    success_rate = (valid_samples / samples) * 100 if samples > 0 else 0
    
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning(f"✅ DATA VALIDATION: {stage}")
    
    logging.info(f"Data Validation - {stage}:")
    logging.info(f"  Total samples: {samples}")
    logging.info(f"  Valid samples: {valid_samples}")
    logging.info(f"  Success rate: {success_rate:.1f}%")
    
    if validation_details:
        logging.info("  Validation Details:")
        for key, value in validation_details.items():
            logging.info(f"    {key}: {value}")

if __name__ == "__main__":
    # Test the logging setup
    log_info = setup_universal_logging("test_experiment")
    
    # Test different logging functions
    log_experiment_start({"test_param": "test_value", "data_size": 100})
    log_experiment_progress(25, 100, "samples", "Processing")
    log_experiment_milestone("Data Loading Complete", {"samples_loaded": 100})
    log_model_results("Test Model", {"r2": 0.85, "mae": 1.2}, {"r2": 0.78, "mae": 1.4})
    log_data_validation("Persona Simulation", 100, 95, {"failed_personas": 5})
    log_experiment_complete({"final_r2": 0.85, "best_model": "GAM"})
    
    print(f"✅ Logging test complete - check {log_info['log_dir']} directory")