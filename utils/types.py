"""
Common type definitions for the Conversify system.
"""

from typing import Literal, Any, Dict, Union

# Speech-to-Text model types
WhisperModels = Literal[
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
]

# Text-to-Speech model types
TTSModels = Literal[
    "tts-1", 
]

# Text-to-Speech voice options
TTSVoices = Literal[
    "af_heart",
    "af_bella"
]

# LLM model types
ChatModels = Literal[
    'gpt',
    'gemini',
    'qwen',
]

# Configuration type
ConfigDict = Dict[str, Any] 