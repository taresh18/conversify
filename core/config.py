"""
Configuration management for the Conversify system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv

class ConfigManager:
    """Manages configuration settings from YAML and environment variables."""
    
    def __init__(self, config_path: str = None, env_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
            env_path: Path to the .env file for environment variables
        """
        # Get base directory path
        self.base_dir = self._get_base_dir()
        
        # Set default paths relative to package
        if config_path is None:
            config_path = os.path.join(self.base_dir, "config.yaml")
        
        if env_path is None:
            # Try .env.local first, then fall back to .env
            env_path = os.path.join(self.base_dir, ".env.local")
            if not os.path.exists(env_path):
                env_path = os.path.join(self.base_dir, ".env")
        
        # Load environment variables
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            logging.info(f"Loaded environment from {env_path}")
        else:
            logging.warning(f"Environment file not found: {env_path}")
            
        # Load configuration file
        self.config_path = config_path
        self.config = self._load_config()
        
    def _get_base_dir(self) -> str:
        """Get the base directory of the package."""
        # Find the directory where the config.py file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the conversify package
        return os.path.dirname(current_dir)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                logging.info(f"Loaded configuration from {self.config_path}")
                return config_data
        except Exception as e:
            logging.error(f"Failed to load configuration from {self.config_path}: {e}")
            return {}
            
    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            path: Configuration path in dot notation (e.g., 'logging.level')
            default: Default value if path doesn't exist
            
        Returns:
            The configuration value
        """
        keys = path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
            
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get an environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if environment variable doesn't exist
            
        Returns:
            The environment variable value
        """
        return os.environ.get(key, default)


# Create a singleton instance
config = ConfigManager() 