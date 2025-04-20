import logging
from dataclasses import dataclass
from typing import Any, Dict

import httpx
import openai

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .utils import TTSModels, TTSVoices, find_time

logger = logging.getLogger(__name__)

TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1

@dataclass
class KokoroTTSOptions:
    """Configuration options for KokoroTTS."""
    model: TTSModels | str
    voice: TTSVoices | str
    speed: float


class KokoroTTS(tts.TTS):
    """TTS implementation using Kokoro API."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        client: openai.AsyncClient | None = None,
    ) -> None:
        """Initialize the KokoroTTS instance.
        
        Args:
            client: Optional pre-configured OpenAI AsyncClient
            config: Configuration dictionary (from config.yaml)
        """
        tts_config = config['tts']['kokoro']
        
        model = tts_config['model']
        voice = tts_config['voice']
        speed = tts_config['speed']
        api_key = tts_config['api_key']
        base_url = tts_config['base_url']
        
        logger.info(f"Using TTS API URL: {base_url}")

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )

        self._opts = KokoroTTSOptions(
            model=model,
            voice=voice,
            speed=speed,
        )

        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update TTS options.
        
        Args:
            model: TTS model to use
            voice: Voice to use
            speed: Speech speed multiplier
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KokoroTTSStream":
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            conn_options: Connection options
            
        Returns:
            Stream of audio chunks
        """
        return KokoroTTSStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._client,
        )


class KokoroTTSStream(tts.ChunkedStream):
    """Stream implementation for KokoroTTS."""
    
    def __init__(
        self,
        *,
        tts: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: KokoroTTSOptions,
        client: openai.AsyncClient,
    ) -> None:
        """Initialize the stream.
        
        Args:
            tts: TTS instance
            input_text: Text to synthesize
            conn_options: Connection options
            opts: TTS options
            client: OpenAI AsyncClient
        """
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._opts = opts

    async def _run(self):
        """Run the TTS synthesis."""
        oai_stream = self._client.audio.speech.with_streaming_response.create(
            input=self.input_text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format="pcm",  # raw pcm buffers
            speed=self._opts.speed,
            timeout=httpx.Timeout(30, connect=self._conn_options.timeout),
        )

        request_id = utils.shortuuid()

        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )

        logger.info(f"Kokoro -> converting text to audio")

        try:
            with find_time('TTS_inferencing'):
                async with oai_stream as stream:
                    async for data in stream.iter_bytes():
                        for frame in audio_bstream.write(data):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    frame=frame,
                                    request_id=request_id,
                                )
                            )
                    # Flush any remaining data in the buffer
                    for frame in audio_bstream.flush():
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                frame=frame,
                                request_id=request_id,
                            )
                        )

        except openai.APITimeoutError:
            raise APITimeoutError()
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError() from e