"""
Speech-to-Text implementation for the Conversify system.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.utils import AudioBuffer

from core.config import config
from utils import WhisperModels, find_time

logger = logging.getLogger(__name__)

@dataclass
class WhisperOptions:
    """Configuration options for WhisperSTT."""
    language: str
    model: WhisperModels | str
    device: str | None = None
    compute_type: str | None = None
    cache_dir: str | None = None
    model_cache_directory: str | None = None


class WhisperSTT(stt.STT):
    """STT implementation using Whisper model."""
    
    def __init__(
        self,
        *,
        language: str = None,
        model: WhisperModels | str = None,
        device: str | None = None,
        compute_type: str | None = None,
        cache_dir: str | None = None,
        model_cache_directory: str | None = None,
    ):
        """Initialize the WhisperSTT instance.
        
        Args:
            language: Language to detect
            model: Whisper model to use
            device: Compute device (cuda/cpu)
            compute_type: Compute type (float16/int8)
            cache_dir: Directory to cache models
            model_cache_directory: Directory to store downloaded models from Hugging Face
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        
        # Load from config if not provided
        language = language or config.get('stt.whisper.language', 'en')
        model = model or config.get('stt.whisper.model', 'large-v3')
        device = device or config.get('stt.whisper.device')
        compute_type = compute_type or config.get('stt.whisper.compute_type')
        cache_dir = cache_dir or config.get('stt.whisper.cache_dir', 'models')
        model_cache_directory = model_cache_directory or config.get('stt.whisper.model_cache_directory', './model_cache')

        self._opts = WhisperOptions(
            language=language,
            model=model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
            model_cache_directory=model_cache_directory,
        )
        
        self._model = None
        self._initialize_model()
        
        # Warmup the model with a sample audio if available
        warmup_path = config.get('stt.whisper.warmup_audio', '/Workspace/tr/repos/s2s/input/warmup_audio.wav')
        if os.path.exists(warmup_path):
            self._warmup(warmup_path)

    def _initialize_model(self):
        """Initialize the Whisper model."""
        device = self._opts.device
        compute_type = self._opts.compute_type
        
        # Auto-detect device and compute type if not specified
        if device is None or compute_type is None:
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        
        logger.info(f"Using device: {device}, with compute: {compute_type}")
        
        # Ensure cache directories exist
        cache_dir = self._opts.cache_dir
        model_cache_dir = self._opts.model_cache_directory
        
        os.makedirs(cache_dir, exist_ok=True)
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            logger.info(f"Using model cache directory: {model_cache_dir}")
        
        logger.info(f"Loading model from directory: {cache_dir}")
        self._model = WhisperModel(
            model_size_or_path=str(self._opts.model),
            device=device,
            compute_type=compute_type,
            download_root=model_cache_dir or cache_dir  # Use model_cache_directory if provided, otherwise fall back to cache_dir
        )
        logger.info("Whisper model loaded successfully")

    def _warmup(self, warmup_audio_path: str) -> None:
        """Performs a warmup transcription.
        
        Args:
            warmup_audio_path: Path to audio file for warmup
        """
        logger.info(f"Starting STT engine warmup using {warmup_audio_path}...")
        try:
            with find_time('STT_warmup'):
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = self._model.transcribe(warmup_audio_data, 
                                                        language=self._opts.language, 
                                                        beam_size=1)
                model_warmup_transcription = " ".join(segment.text for segment in segments)
            logger.info(f"STT engine warmed up. Text: {model_warmup_transcription}")
        except Exception as e:
            logger.error(f"Failed to warm up STT engine: {e}")

    def update_options(
        self,
        *,
        model: Optional[WhisperModels | str] = None,
        language: Optional[str] = None,
        model_cache_directory: Optional[str] = None,
    ) -> None:
        """Update STT options.
        
        Args:
            model: Whisper model to use
            language: Language to detect
            model_cache_directory: Directory to store downloaded models
        """
        reinitialize = False
        
        if model:
            self._opts.model = model
            reinitialize = True
            
        if model_cache_directory:
            self._opts.model_cache_directory = model_cache_directory
            reinitialize = True
            
        if language:
            self._opts.language = language
            
        if reinitialize:
            self._initialize_model()

    def _sanitize_options(self, *, language: Optional[str] = None) -> WhisperOptions:
        """Create a copy of options with optional overrides.
        
        Args:
            language: Language override
            
        Returns:
            Copy of options with overrides applied
        """
        options = dataclasses.replace(self._opts)
        if language:
            options.language = language
        return options

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement speech recognition.
        
        Args:
            buffer: Audio buffer
            language: Language to detect
            conn_options: Connection options
            
        Returns:
            Speech recognition event
        """
        try:
            logger.info(f"Received audio, transcribing to text")
            options = self._sanitize_options(language=language)
            audio_data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            
            # Convert WAV to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            with find_time('STT_inference'):
                segments, info = self._model.transcribe(
                    audio_array,
                    language=options.language,
                    beam_size=1,
                    best_of=1,
                    condition_on_previous_text=True,
                    vad_filter=False,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )

            segments_list = list(segments)
            full_text = " ".join(segment.text.strip() for segment in segments_list)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text or "",
                        language=options.language,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Error in speech recognition: {e}", exc_info=True)
            raise APIConnectionError() from e 