"""
Main agent orchestration for the Conversify system.
"""

import asyncio
import logging
import os
from typing import Optional, AsyncIterable, Union, Any

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, turn_detector
from livekit.agents.metrics import LLMMetrics, TTSMetrics, PipelineEOUMetrics

from core.config import config
from models.tts import KokoroTTS, clean_text_for_tts
from models.stt import WhisperSTT
from models.llm import OpenaiLLM
from utils import find_time

logger = logging.getLogger(__name__)

class ConversifyAgent:
    """Main agent class that orchestrates the voice pipeline."""
    
    def __init__(self):
        """Initialize the agent."""
        self.agent = None
        self.latest_image: Optional[rtc.VideoFrame] = None
        self.end_of_utterance_delay = 0
        self.llm_ttft = 0
        self.tts_ttfb = 0
        
    def prewarm(self, proc: JobProcess):
        """Prewarm the models to reduce cold start latency.
        
        Args:
            proc: The job process to store prewarmed models
        """
        # Load silero weights and store to process userdata
        try:
            proc.userdata["vad"] = silero.VAD.load(
                min_speech_duration=config.get('vad.min_speech_duration', 0.20),
                min_silence_duration=config.get('vad.min_silence_duration', 0.50),
                prefix_padding_duration=config.get('vad.prefix_padding_duration', 0.5),
                max_buffered_speech=config.get('vad.max_buffered_speech', 60.0),
                activation_threshold=config.get('vad.activation_threshold', 0.5),
                force_cpu=config.get('vad.force_cpu', False),
                sample_rate=config.get('vad.sample_rate', 16000)
            )
            logger.info("VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VAD model: {e}")
            raise
        
    async def entrypoint(self, ctx: JobContext):
        """Main entrypoint for the agent.
        
        Args:
            ctx: The job context
        """
        # Load system prompts
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        llm_system_prompt_path = os.path.join(base_dir, 'prompts', 'llm_system_prompt.txt')
        llm_system_prompt = self._load_system_prompt(llm_system_prompt_path)
        
        # Create the chat context
        chat_context = llm.ChatContext()
        chat_context.messages.append(
            llm.ChatMessage(
                role="system",
                content=llm_system_prompt
            ),
        )
        
        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        
        # Initialize the models
        try:
            stt_obj = self._create_stt()
            llm_obj = self._create_llm()
            tts_obj = self._create_tts()
            
            # Initialize the end-of-utterance detector if enabled
            eou = None
            if config.get('eou.enabled', True):
                eou = turn_detector.EOUModel(unlikely_threshold=config.get('eou.unlikely_threshold', 0.0289))
                logger.info("End-of-Utterance (EOG) detection enabled")
            else:
                logger.info("End-of-Utterance (EOG) detection disabled")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
        
        # Start the video processing loop if enabled
        video_task = None
        if config.get('video.enabled', True):
            logger.info("Video input is enabled, starting video processing")
            video_task = asyncio.create_task(self._video_processing_loop(ctx))
            # Ensure the task is cancelled on shutdown
            ctx.add_shutdown_callback(lambda: self._cancel_video_task(video_task))
            
        # Create the agent
        self.agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=stt_obj,
            llm=llm_obj,
            tts=tts_obj,
            turn_detector=eou, 
            chat_ctx=chat_context,
            allow_interruptions=config.get('pipeline.allow_interruptions', True),
            interrupt_speech_duration=config.get('pipeline.interrupt_speech_duration', 0.5),
            interrupt_min_words=config.get('pipeline.interrupt_min_words', 0),
            min_endpointing_delay=config.get('pipeline.min_endpointing_delay', 0.3),
            preemptive_synthesis=config.get('pipeline.preemptive_synthesis', True),
            before_tts_cb=clean_text_for_tts,
            before_llm_cb=self.before_llm_vision_callback,
        )
        
        # Set up metrics collection
        usage_collector = metrics.UsageCollector()
        self._setup_metrics_collection(usage_collector)
        
        # Wait for the first participant to connect
        participant = await ctx.wait_for_participant()
        logger.info(f"Connected to room {ctx.room.name} with participant {participant.identity}")
        
        # Start the agent
        self.agent.start(ctx.room, participant)
        
        # Greet the user
        await self.agent.say("Hey, how can I help you today?", allow_interruptions=True)
        
    def _create_stt(self) -> WhisperSTT:
        """Create the STT model.
        
        Returns:
            Configured WhisperSTT instance
        """
        return WhisperSTT(
            model=config.get('stt.whisper.model'),
            device=config.get('stt.whisper.device'),
            compute_type=config.get('stt.whisper.compute_type'),
            language=config.get('stt.whisper.language', 'en'),
            cache_dir=config.get('stt.whisper.cache_dir'),
            model_cache_directory=config.get('stt.whisper.model_cache_directory')
        )
        
    def _create_llm(self) -> OpenaiLLM:
        """Create the LLM model.
        
        Returns:
            Configured OpenaiLLM instance
        """
        return OpenaiLLM(
            model=config.get('llm.openai.model'),
            temperature=config.get('llm.openai.temperature'),
            parallel_tool_calls=config.get('llm.openai.parallel_tool_calls'),
            max_tokens=config.get('llm.openai.max_tokens')
        )
        
    def _create_tts(self) -> KokoroTTS:
        """Create the TTS model.
        
        Returns:
            Configured KokoroTTS instance
        """
        return KokoroTTS(
            model=config.get('tts.kokoro.model'),
            voice=config.get('tts.kokoro.voice'),
            speed=config.get('tts.kokoro.speed', 1.0)
        )
        
    def _setup_metrics_collection(self, usage_collector: metrics.UsageCollector):
        """Set up metrics collection for the agent.
        
        Args:
            usage_collector: Metrics usage collector
        """
        @self.agent.on("metrics_collected")
        def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
            metrics.log_metrics(agent_metrics)
            usage_collector.collect(agent_metrics)
            
            if isinstance(agent_metrics, PipelineEOUMetrics):
                self.end_of_utterance_delay = agent_metrics.end_of_utterance_delay
            elif isinstance(agent_metrics, LLMMetrics):
                self.llm_ttft = agent_metrics.ttft
            elif isinstance(agent_metrics, TTSMetrics):
                self.tts_ttfb = agent_metrics.ttfb
                e2e_latency = self.end_of_utterance_delay + self.llm_ttft + self.tts_ttfb
                logger.info(f"TOTAL E2E LATENCY: {e2e_latency}, eou: {self.end_of_utterance_delay}, ttft: {self.llm_ttft}, ttfb: {self.tts_ttfb}")
                
    def _load_system_prompt(self, prompt_path: str) -> str:
        """Load a system prompt from a file.
        
        Args:
            prompt_path: Path to the prompt file
            
        Returns:
            The prompt text
        """
        try:
            logger.info(f"Loading system prompt from {prompt_path}")
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
                logger.info(f"System prompt loaded: {len(prompt)} characters")
                return prompt
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {prompt_path}")
            # Fallback prompt
            fallback = ("You are Conversify, a friendly assistant. Keep responses concise "
                    "and focused on the query. Provide helpful, correct, and satisfying answers.")
            logger.info(f"Using fallback prompt: {fallback}")
            return fallback
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return ("You are Conversify, a friendly and efficient assistant.")
    
    def _cancel_video_task(self, video_task: asyncio.Task):
        """Cancel a video processing task.
        
        Args:
            video_task: The task to cancel
        """
        if video_task and not video_task.done():
            video_task.cancel()
            logger.info("Video processing task cancelled.")
        
    async def _video_processing_loop(self, ctx: JobContext):
        """Process video frames from a video track.
        
        Args:
            ctx: The job context
        """
        logger.info("Looking for video track...")
        # Wait for a remote participant with a video track
        video_track = None
        while video_track is None:
            for participant in ctx.room.remote_participants.values():
                for track_pub in participant.track_publications.values():
                    if track_pub.kind == rtc.TrackKind.KIND_VIDEO and track_pub.track:
                        # Found a video track, subscribe if not already
                        if not track_pub.subscribed:
                            track_pub.set_subscribed(True)
                        # Wait a moment for subscription to complete if needed
                        await asyncio.sleep(0.5)
                        if isinstance(track_pub.track, rtc.RemoteVideoTrack):
                            video_track = track_pub.track
                            logger.info(f"Using video track: {video_track.sid} from {participant.identity}")
                            break
                if video_track:
                    break
            if video_track is None:
                await asyncio.sleep(1)  # Wait and retry if no track found yet

        # Process the video stream
        video_stream = rtc.VideoStream(video_track)
        interval_ms = config.get('video.process_interval_ms', 200) / 1000.0  # Convert to seconds
        
        try:
            async for event in video_stream:
                self.latest_image = event.frame
                # Sleep to avoid processing every frame
                await asyncio.sleep(interval_ms)
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")
        finally:
            logger.info("Video stream ended.")
            await video_stream.aclose()  # Ensure stream is closed
            
    async def before_llm_vision_callback(
        self,
        agent: VoicePipelineAgent,
        chat_ctx: llm.ChatContext
    ):
        """Callback that runs before LLM is called, to add vision to the context.
        
        Args:
            agent: The voice pipeline agent
            chat_ctx: The chat context
        """
        if not config.get('video.enabled', True):
            logger.debug("Video input is disabled, skipping vision callback")
            return
        logger.debug("Executing before_llm_vision_callback")

        if not chat_ctx.messages:
            logger.debug("No messages in context, skipping vision check.")
            return  # Nothing to process

        # Get the last message (should be the user's latest input)
        last_message = chat_ctx.messages[-1]

        # Ensure it's a user message and content is a string (initially)
        if last_message.role != "user" or not isinstance(last_message.content, str):
            logger.debug("Last message is not a user string, skipping vision check.")
            return

        user_text = last_message.content
        logger.debug(f"Checking user text for vision trigger: '{user_text}'")

        # Decision Logic: When to add the image?
        # Simple keyword-based check (adjust keywords as needed)
        vision_keywords = ["see", "look", "picture", "image", "visual", "color", "this", 
                          "object", "view", "frame", "screen", "desk", "holding"]
        should_add_image = any(keyword in user_text.lower() for keyword in vision_keywords)

        if should_add_image and self.latest_image:
            logger.info(f"Vision keyword detected. Adding image to context for text: '{user_text}'")
            try:
                # IMPORTANT: Modify the content of the *existing* last message
                # Change content from str to list[Union[str, llm.ChatImage]]
                last_message.content = [
                    user_text,  # Keep the original text
                    llm.ChatImage(image=self.latest_image)  # Add the image object
                ]
                logger.debug("Successfully added ChatImage to the last message content.")
            except Exception as e:
                logger.error(f"Error adding ChatImage to context: {e}")
                # Revert to text only on error
                last_message.content = user_text
        elif should_add_image:
            logger.warning("Vision keyword detected, but no image available.")
        else:
            logger.debug("No vision keyword detected.") 