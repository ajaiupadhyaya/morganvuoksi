"""
Logging configuration for the quantitative finance system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger
from ..utils.config import get_config

def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with JSON formatting and file/console handlers.
    
    Args:
        name: Logger name
        level: Logging level (defaults to config)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    config = get_config()
    logger = logging.getLogger(name)
    
    # Set level from config or parameter
    level = level or config['LOG_LEVEL']
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    if config['LOG_FORMAT'] == 'json':
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name) 