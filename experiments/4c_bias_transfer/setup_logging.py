#!/usr/bin/env python3
"""
Logging Configuration for Experiment 4C
Sets up comprehensive logging to both console and files
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_experiment_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """
    Set up comprehensive logging for experiment runs
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (INFO, DEBUG, etc.)
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. Main experiment log file (all INFO and above)
    main_log_file = log_path / f"experiment_{timestamp}.log"
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # 3. Debug log file (everything including DEBUG)
    debug_log_file = log_path / f"debug_{timestamp}.log"
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(debug_handler)
    
    # 4. Progress/Results log (important events only)
    progress_log_file = log_path / f"progress_{timestamp}.log"
    progress_handler = logging.FileHandler(progress_log_file)
    progress_handler.setLevel(logging.WARNING)  # Only warnings and above
    progress_handler.setFormatter(simple_formatter)
    
    # Create a custom logger for progress tracking
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.addHandler(progress_handler)
    progress_logger.setLevel(logging.INFO)
    
    # Log the setup
    logging.info("="*60)
    logging.info("EXPERIMENT 4C LOGGING INITIALIZED")
    logging.info("="*60)
    logging.info(f"Timestamp: {timestamp}")
    logging.info(f"Main log: {main_log_file}")
    logging.info(f"Debug log: {debug_log_file}")
    logging.info(f"Progress log: {progress_log_file}")
    logging.info(f"Log level: {logging.getLevelName(log_level)}")
    logging.info("="*60)
    
    return {
        'timestamp': timestamp,
        'main_log': main_log_file,
        'debug_log': debug_log_file,
        'progress_log': progress_log_file,
        'progress_logger': progress_logger
    }

def log_experiment_start(config: dict):
    """Log experiment configuration and parameters"""
    logging.warning("🚀 EXPERIMENT START")
    logging.info("Experiment Configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")

def log_experiment_progress(current: int, total: int, item_type: str = "items"):
    """Log progress updates"""
    progress_pct = (current / total) * 100
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning(f"📊 Progress: {current}/{total} {item_type} ({progress_pct:.1f}%)")
    
    # Also log to main logger at INFO level
    logging.info(f"📊 Progress: {current}/{total} {item_type} ({progress_pct:.1f}%)")

def log_experiment_complete(results_summary: dict):
    """Log experiment completion and results"""
    logging.warning("✅ EXPERIMENT COMPLETE")
    logging.info("Results Summary:")
    for key, value in results_summary.items():
        logging.info(f"  {key}: {value}")

if __name__ == "__main__":
    # Test the logging setup
    setup_experiment_logging()
    
    logging.info("Testing INFO level")
    logging.warning("Testing WARNING level")
    logging.error("Testing ERROR level")
    logging.debug("Testing DEBUG level")
    
    progress_logger = logging.getLogger('experiment.progress')
    progress_logger.warning("Testing progress logger")
    
    print("✅ Logging test complete - check logs/ directory")