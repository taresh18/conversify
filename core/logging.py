"""
Logging configuration for the Conversify system.
"""

import logging
import os
import sys
from typing import Optional, List
from .config import config

def setup_logging(log_file: Optional[str] = None, log_level: Optional[str] = None):
    """Configure logging for the application.
    
    Args:
        log_file: Path to the log file. If None, uses value from config.
        log_level: Logging level. If None, uses value from config.
    """
    # Get configuration from config if not provided
    log_file = log_file or config.get('logging.file', 'app.log')
    log_level_str = log_level or config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Convert string log level to numeric value
    log_level_num = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Configure root logger - this affects all loggers that don't override settings
    logging.basicConfig(
        level=log_level_num,
        format=log_format,
        force=True,  # Force reconfiguration
        handlers=[
            # Log to both file and console
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # List of specific loggers to ensure are properly configured
    loggers_to_configure = [
        'conversify',
        'livekit.agents.pipeline',
        'livekit.agents',
        'voice-agent',
        'asyncio',
        'livekit'
    ]
    
    # Configure each logger to ensure it logs to the file
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level_num)
        
        # Check if this logger already has handlers to avoid duplicates
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        
        if not has_file_handler:
            # Add a file handler if one doesn't exist
            file_handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter(log_format)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    # Create or ensure the directory for the log file exists
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Log startup information
    logger = logging.getLogger('conversify')
    logger.info(f"Logging initialized at level {log_level_str}")
    
    return logger 