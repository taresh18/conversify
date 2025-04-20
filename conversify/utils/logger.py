import logging
import os
import sys
from typing import Dict, Any

# Sentinel to prevent multiple configurations
_logging_configured = False

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = "INFO"

def setup_logging(config: Dict[str, Any], project_root: str):
    """Configures the root logger based on settings from the config dictionary.

    Args:
        config: The loaded configuration dictionary.
        project_root: The absolute path to the project's root directory.
    """
    global _logging_configured
    if _logging_configured:
        return

    # --- Extract Settings from Config --- 
    log_cfg = config.get('logging', {})
    log_level_str = log_cfg.get('level', DEFAULT_LOG_LEVEL)
    log_file_rel = log_cfg.get('file') # Path relative to project root
    
    log_file_abs = None
    if log_file_rel:
        # Resolve relative path using the provided project_root
        log_file_abs = os.path.abspath(os.path.join(project_root, log_file_rel))

    # --- Get Log Level --- 
    level = logging.getLevelName(log_level_str.upper())
    if not isinstance(level, int):
        print(f"Warning: Invalid log level '{log_level_str}' in config. Defaulting to {DEFAULT_LOG_LEVEL}.", file=sys.stderr)
        level = logging.INFO

    # --- Create Formatter --- 
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # --- Get Root Logger --- 
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # --- Clear Existing Handlers --- 
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # --- Setup Console Handler --- 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level) 
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # --- Setup File Handler (If specified) --- 
    if log_file_abs:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_abs)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}", file=sys.stderr)
            
            file_handler = logging.FileHandler(log_file_abs, mode='a') 
            file_handler.setLevel(level) 
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"Logging configured. Level: {log_level_str.upper()}, File: {log_file_abs}", file=sys.stderr)
        except Exception as e:
            print(f"Error setting up file logging to {log_file_abs}: {e}", file=sys.stderr)
            # Continue with console logging only
    else:
         print(f"Logging configured. Level: {log_level_str.upper()}, Console only.", file=sys.stderr)

    _logging_configured = True
