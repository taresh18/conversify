import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration loading, parsing, path resolution,
    and prompt loading for the Conversify application.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the ConfigManager with a path to the config file."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.project_root = self._get_project_root()
    
    def _get_project_root(self) -> str:
        """Get the absolute path to the project root directory."""
        # Assuming this file is in 'utils' subdirectory of the project root
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def _resolve_path(self, relative_path: str) -> str:
        """Convert a relative path to an absolute path based on project root."""
        return os.path.abspath(os.path.join(self.project_root, relative_path))
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        abs_config_path = self._resolve_path(self.config_path)
        logger.info(f"Loading configuration from: {abs_config_path}")
        
        try:
            with open(abs_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    raise ValueError("Configuration file does not contain a valid YAML dictionary")
                logger.info(f"Configuration loaded successfully from {abs_config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading YAML configuration {abs_config_path}: {e}")
            raise
    
    def _load_prompt(self, prompt_path: str) -> str:
        """Load prompt content from a file."""
        abs_prompt_path = self._resolve_path(prompt_path)
        logger.info(f"Loading prompt from: {abs_prompt_path}")
        
        try:
            with open(abs_prompt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info(f"Prompt loaded successfully from {abs_prompt_path}")
                return content
        except Exception as e:
            logger.error(f"Error loading prompt from {abs_prompt_path}: {e}")
            raise
    
    def _resolve_paths_in_config(self) -> None:
        """
        Resolve all relative paths in the configuration to absolute paths.
        Also loads any file content that needs to be loaded (e.g., prompts).
        """
        # Load agent instructions
        agent_cfg = self.config['agent']
        prompt_file = agent_cfg['instructions_file']
        agent_cfg['instructions'] = self._load_prompt(prompt_file)
        
        # Resolve memory directory
        memory_cfg = self.config['memory']
        if memory_cfg.get('use', False):
            memory_dir_rel = memory_cfg['dir']
            memory_dir_abs = self._resolve_path(memory_dir_rel)
            memory_cfg['dir_abs'] = memory_dir_abs
            logger.info(f"Memory enabled. Directory path: {memory_dir_abs}")
        else:
            logger.info("Memory usage is disabled in config.")
        
        # Handle STT paths - check if they should be absolute or need resolution
        stt_cfg = self.config.get('stt', {})
        whisper_cfg = stt_cfg['whisper']
        
        # Check if model_cache_directory is relative and needs resolution
        if 'model_cache_directory' in whisper_cfg and not os.path.isabs(whisper_cfg['model_cache_directory']):
            whisper_cfg['model_cache_directory'] = self._resolve_path(whisper_cfg['model_cache_directory'])
        
        # Check if warmup_audio is relative and needs resolution
        if 'warmup_audio' in whisper_cfg and not os.path.isabs(whisper_cfg['warmup_audio']):
            whisper_cfg['warmup_audio'] = self._resolve_path(whisper_cfg['warmup_audio'])
        
        # Add logging path resolution
        logging_cfg = self.config['logging']
        log_file_rel = logging_cfg['file']
        if not os.path.isabs(log_file_rel):
            logging_cfg['file_abs'] = self._resolve_path(log_file_rel)
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load and process the configuration file.
        Returns the processed configuration dictionary.
        """
        self.config = self._load_yaml_config()
        self._resolve_paths_in_config()
        logger.info("Configuration processed successfully.")
        return self.config