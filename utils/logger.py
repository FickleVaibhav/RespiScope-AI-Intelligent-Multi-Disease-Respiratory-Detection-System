"""
Logging utilities for RespiScope-AI
Provides structured logging for training, evaluation, and inference
"""

import logging
import os
from datetime import datetime
from pathlib import Path
import json
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logger(name: str = 'RespiScope-AI',
                log_dir: str = 'logs',
                log_file: str = None,
                level: int = logging.INFO,
                console: bool = True) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_file: Log file name (default: timestamped)
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler
    if log_file is None:
        log_file = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log file: {file_path}")
    
    return logger


class ExperimentLogger:
    """Logger for tracking experiments with metrics and hyperparameters"""
    
    def __init__(self, experiment_name: str, log_dir: str = 'experiments'):
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = setup_logger(
            name=f'Experiment-{experiment_name}',
            log_dir=self.log_dir,
            log_file='experiment.log'
        )
        
        self.metrics = {}
        self.hyperparameters = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'log_dir': self.log_dir
        }
    
    def log_hyperparameters(self, params: dict):
        """Log hyperparameters"""
        self.hyperparameters.update(params)
        self.logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        # Save to file
        with open(os.path.join(self.log_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics for a training step"""
        if step is not None:
            self.logger.info(f"Step {step}: {json.dumps(metrics)}")
            if step not in self.metrics:
                self.metrics[step] = {}
            self.metrics[step].update(metrics)
        else:
            self.logger.info(f"Metrics: {json.dumps(metrics)}")
            self.metrics.update(metrics)
        
        # Save to file
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_model_summary(self, model_summary: str):
        """Log model architecture summary"""
        summary_file = os.path.join(self.log_dir, 'model_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(model_summary)
        self.logger.info(f"Model summary saved to {summary_file}")
    
    def log_artifact(self, filepath: str, artifact_type: str = 'file'):
        """Log an artifact (model, plot, etc.)"""
        self.logger.info(f"Artifact saved: {filepath} (type: {artifact_type})")
        
        if 'artifacts' not in self.metadata:
            self.metadata['artifacts'] = []
        
        self.metadata['artifacts'].append({
            'path': filepath,
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        })
        
        self.save_metadata()
    
    def save_metadata(self):
        """Save experiment metadata"""
        self.metadata['end_time'] = datetime.now().isoformat()
        metadata_file = os.path.join(self.log_dir, 'metadata.json')
        
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def finish(self):
        """Finish experiment and save all data"""
        self.save_metadata()
        self.logger.info(f"Experiment {self.experiment_name} finished")
        self.logger.info(f"Results saved to {self.log_dir}")


def get_git_commit():
    """Get current git commit hash for reproducibility"""
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit
    except:
        return None


def log_environment_info(logger: logging.Logger):
    """Log environment information"""
    import platform
    import torch
    
    logger.info("="*80)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("="*80)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    git_commit = get_git_commit()
    if git_commit:
        logger.info(f"Git commit: {git_commit}")
    
    logger.info("="*80)


if __name__ == '__main__':
    # Test logger
    logger = setup_logger('TestLogger', 'test_logs')
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test experiment logger
    exp_logger = ExperimentLogger('test_experiment', 'test_experiments')
    exp_logger.log_hyperparameters({'lr': 0.001, 'batch_size': 32})
    exp_logger.log_metrics({'loss': 0.5, 'accuracy': 0.85}, step=1)
    exp_logger.finish()
    
    print("✅ Logger test complete")
