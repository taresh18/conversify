
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
import torch
import numpy as np
from faster_whisper import WhisperModel

import soundfile as sf

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)
from livekit.agents.utils import AudioBuffer

from .utils import WhisperModels, find_time
import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = "/Workspace/tr/repos/s2s/models"


@dataclass
class _STTOptions:
    language: str
    model: WhisperModels | str
    device: str | None = None
    compute_type: str | None = None
    cache_dir: str | None = None

class WhisperSTT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "en",
        model: WhisperModels | str = "large-v3",
        device: str | None = None,
        compute_type: str | None = None,
        cache_dir: str | None = None,
    ):
        """
        Создать новый экземпляр Whisper STT.
        
        Args:
            language: Language to detect
            model: Whisper model to use
            device: Compute device (cuda/cpu)
            compute_type: Compute type (float16/int8)
            cache_dir: Directory to cache models
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        self._opts = _STTOptions(
            language=language,
            model=model,
            device=device,
            compute_type=compute_type,
            cache_dir=cache_dir,
        )
        
        self._model = None
        self._initialize_model()
        self._warmup('/Workspace/tr/repos/s2s/input/warmup_audio.wav')

    def _initialize_model(self):
        
        device = self._opts.device
        compute_type = self._opts.compute_type
        
        if device is None or compute_type is None:
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        
        logger.info(f"using device: {device}, with compute: {compute_type}")
        
        cache_dir = self._opts.cache_dir or DEFAULT_MODELS_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"loading from directory: {cache_dir}")
        self._model = WhisperModel(
            model_size_or_path=str(self._opts.model),
            device=device,
            compute_type=compute_type,
            download_root=cache_dir
        )
        logger.info("Whisper model loaded successfully")

    def _warmup(self, warmup_audio_path) -> None:
        """Performs a warmup transcription. Raises error on failure."""
        logger.info(f"Starting STT engine warmup using {warmup_audio_path}...")
        with find_time('STT_warmup'):
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            segments, info = self._model.transcribe(warmup_audio_data, 
                                                    language=self._opts.language, 
                                                    beam_size=1)
            model_warmup_transcription = " ".join(segment.text for segment in segments)
        logger.info(f"STT engine warmed up. text: {model_warmup_transcription}")

    def update_options(
        self,
        *,
        model: WhisperModels | None = None,
        language: str | None = None,
    ) -> None:
        """Update settings STT."""
        if model:
            self._opts.model = model
            self._initialize_model()
        if language:
            self._opts.language = language

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            logger.info(f"received audio, transcribing to text")
            config = self._sanitize_options(language=language)
            audio_data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            
            # convert WAV to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            with find_time('STT_inference'):
                segments, info = self._model.transcribe(
                    audio_array,
                    language=config.language,
                    beam_size=1,
                    best_of=1,
                    condition_on_previous_text=True,
                    # no_speech_threshold=0.3,
                    # compression_ratio_threshold=2.0,
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
                        language=config.language,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"error: {e}", exc_info=True)
            raise APIConnectionError() from e