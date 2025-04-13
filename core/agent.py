"""
Main agent orchestration for the Conversify system.
"""

import asyncio
import logging
import os
import json
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
from livekit.agents.llm import ChatMessage
from core.config import config
from core.memory import ChatCompletionsModel, VLLMEmbeddingModel
from models.tts import KokoroTTS, clean_text_for_tts
from models.stt import WhisperSTT
from models.llm import OpenaiLLM
from utils import find_time
from memoripy import MemoryManager, JSONStorage
from datetime import datetime
logger = logging.getLogger(__name__)

MEMORY_DIR = config.get('memory.directory', "./conversation_memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

class ConversifyAgent:
    """Main agent class that orchestrates the voice pipeline."""
    
    def __init__(self):
        """Initialize the agent."""
        self.agent = None
        self.latest_image: Optional[rtc.VideoFrame] = None
        self.end_of_utterance_delay = 0
        self.llm_ttft = 0
        self.tts_ttfb = 0
        # video processing
        self.video_task: Optional[asyncio.Task] = None
        # metrics summary
        self.usage_collector: Optional[metrics.UsageCollector] = None
        # conversation history
        self.conversation_history = []
        # initialize memory manager
        self.memory_manager: Optional[MemoryManager] = None
        self.participant_identity: Optional[str] = None 
        
        
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
        
    def _setup_adding_user_message(self):
        """Set up the adding user message callback."""
        @self.agent.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            if isinstance(msg.content, list):
                content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content)
            else:
                content = msg.content
                
            logger.info(f"User speech committed: {content[:50]}...")
            try:
                self.conversation_history.append({
                    "role": "user",
                    "text": content,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"Added user message to conversation history: {content[:50]}...")
            except Exception as e:
                logger.error(f"Error adding user message to history: {e}")
                
    def _setup_adding_agent_message(self):
        """Set up the adding agent message callback."""
        @self.agent.on("agent_speech_committed")
        def on_agent_speech_committed(msg: llm.ChatMessage):
            content = msg.content
            logger.info(f"Agent speech committed: {content[:50]}...")
            try:
                text_content = content if isinstance(content, str) else json.dumps(content) # Basic serialization for non-string
                self.conversation_history.append({
                    "role": "assistant",
                    "text": text_content,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"Added agent message to conversation history: {content[:50]}...")
            except Exception as e:
                logger.error(f"Error adding agent message to history: {e}")
    
    async def _save_conversation_history(self):
        # Check if memory manager was initialized and user ID is known
        if not self.memory_manager or not self.participant_identity:
            logger.warning("Memory manager or participant identity not available. Skipping history save.")
            return
            
        logger.info(f"Saving conversation history via Memoripy for user: {self.participant_identity}")
        logger.info(f"Conversation history messages count: {len(self.conversation_history)}")
        try:
            # Process messages in pairs (User Prompt -> Assistant Response)
            i = 0
            processed_count = 0
            while i < len(self.conversation_history):
                user_msg = None
                assistant_msg = None

                # Find the next user message
                if self.conversation_history[i]["role"] == "user":
                    user_msg = self.conversation_history[i]
                    # Find the corresponding assistant message (if it exists)
                    if i + 1 < len(self.conversation_history) and self.conversation_history[i+1]["role"] == "assistant":
                        assistant_msg = self.conversation_history[i+1]
                        i += 2 # Move past both
                    else:
                        i += 1 # Move past only user msg (no response?)
                elif self.conversation_history[i]["role"] == "assistant":
                     # Handle case where session starts/ends with assistant msg? Maybe log/skip.
                     logger.warning(f"Skipping assistant message without preceding user message at index {i}")
                     i += 1
                     continue
                else: # Skip system messages etc.
                    i += 1
                    continue

                # Process the interaction pair
                if user_msg:
                    user_prompt = user_msg.get("text", "")
                    # Handle potential non-string content stored in history
                    if not isinstance(user_prompt, str): user_prompt = str(user_prompt)

                    assistant_response = ""
                    if assistant_msg:
                        assistant_response = assistant_msg.get("text", "")
                        if not isinstance(assistant_response, str): assistant_response = str(assistant_response)

                    combined_text = f"{user_prompt} {assistant_response}".strip()

                    if not combined_text:
                        logger.debug("Skipping empty interaction.")
                        continue
                    
                    try:
                        concepts = await asyncio.to_thread(self.memory_manager.extract_concepts, combined_text)
                        embedding = await asyncio.to_thread(self.memory_manager.get_embedding, combined_text)
                        await asyncio.to_thread(
                            self.memory_manager.add_interaction,
                            prompt=user_prompt,
                            response=assistant_response,
                            embedding=embedding,
                            concepts=concepts
                        )
                        processed_count += 1
                        logger.debug(f"Added interaction to Memoripy: User: '{user_prompt[:50]}...' Assistant: '{assistant_response[:50]}...'")
                    except Exception as e:
                        logger.error(f"Error processing/adding interaction via Memoripy: {e}")

            logger.info(f"Successfully added {processed_count} interactions into conversational memory for {self.participant_identity}")

        except Exception as e:
            logger.error(f"Failed during Memoripy history saving for {self.participant_identity}: {e}")
        
                
    async def application_shutdown(self):
        """Shutdown the application."""
        logger.info("Application shutdown initiated")
        # cancel video task
        if self.video_task and not self.video_task.done():
            try:
                self.video_task.cancel()
                await self.video_task
                logger.info("Video processing task cancelled.")
            except Exception as e:
                 logger.error(f"Error cancelling video task: {e}")

        # log usage
        try:
            summary = self.usage_collector.get_summary()
            logger.info(f"Usage: ${summary}")
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")

        # save conversation history
        await self._save_conversation_history()
        
    
    def initialize_memory(self, initial_chat_context: llm.ChatContext):
        user_memory_file = os.path.join(MEMORY_DIR, f"{self.participant_identity}.json")
        self.memory_manager = MemoryManager(
            chat_model=ChatCompletionsModel() ,
            embedding_model=VLLMEmbeddingModel() ,
            storage=JSONStorage(user_memory_file)
        )
        logger.info(f"Initialized Memory for user {self.participant_identity} with storage {user_memory_file}")

        # Load conversational memory for this user from file
        initial_messages_from_memory = []
        try:
            short_term_history, _ = self.memory_manager.load_history()
            num_interactions_to_load = config.get('memory.load_last_n', 6)
            memory_interactions = short_term_history[-num_interactions_to_load:]

            for interaction in memory_interactions:
                if interaction.get('prompt'):
                        initial_messages_from_memory.append(ChatMessage(role="user", content=interaction['prompt']))
                if interaction.get('output'):
                        initial_messages_from_memory.append(ChatMessage(role="assistant", content=interaction['output']))
            # Prepend loaded history to the initial context
            initial_chat_context.messages.extend(initial_messages_from_memory)
            logger.info(f"Prepended {len(initial_messages_from_memory)} interactions to the initial context ({len(initial_chat_context.messages)} total messages).")

        except FileNotFoundError:
                logger.info(f"No previous history file found for {self.participant_identity}. Starting fresh.")
        except Exception as e:
            logger.error(f"Failed to load history via Memoripy for {self.participant_identity}: {e}")

            
    def initialize_models(self):
        # Initialize models
        stt_obj = WhisperSTT(
            model=config.get('stt.whisper.model'),
            device=config.get('stt.whisper.device'),
            compute_type=config.get('stt.whisper.compute_type'),
            language=config.get('stt.whisper.language', 'en'),
            cache_dir=config.get('stt.whisper.cache_dir'),
            model_cache_directory=config.get('stt.whisper.model_cache_directory'),
        )
        llm_obj = OpenaiLLM(
            model=config.get('llm.openai.model'),
            temperature=config.get('llm.openai.temperature'),
            max_tokens=config.get('llm.openai.max_tokens'),
            base_url=config.get('llm.openai.api_endpoint'),
            api_key=config.get('llm.openai.api_key')
        )
        tts_obj = KokoroTTS(
            model=config.get('tts.kokoro.model'),
            voice=config.get('tts.kokoro.voice'),
            speed=config.get('tts.kokoro.speed', 1.0),
            base_url=config.get('tts.kokoro.api_endpoint')
        )
        # Initialize EOU detector
        eou = None
        if config.get('eou.enabled', True):
            eou = turn_detector.EOUModel(unlikely_threshold=config.get('eou.unlikely_threshold', 0.0289))
            logger.info("End-of-Utterance (EOU) detection enabled")
        else:
            logger.info("End-of-Utterance (EOU) detection disabled")
            
        return stt_obj, llm_obj, tts_obj, eou
            
            
        
    async def entrypoint(self, ctx: JobContext):
        """Main entrypoint for the agent."""
        llm_system_prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'llm_system_prompt.txt')
        llm_system_prompt = self._load_system_prompt(llm_system_prompt_path)

        initial_chat_context = llm.ChatContext()
        initial_chat_context.messages.append(
            llm.ChatMessage(role="system", content=llm_system_prompt)
        )

        logger.info(f"Connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        
        # Wait for the first participant 
        participant = await ctx.wait_for_participant()
        # self.participant_identity = participant.identity
        # hardcode participant identity for dev
        self.participant_identity = "identity-qfXx"
        logger.info(f"Participant connected: {self.participant_identity}")

        # Initialize models
        stt_obj, llm_obj, tts_obj, eou = self.initialize_models()
        
        # Initialize MemoryManager 
        self.initialize_memory(initial_chat_context)

        # Start video processing if enabled
        if config.get('video.enabled', True):
            logger.info("Video input is enabled, starting video processing")
            self.video_task = asyncio.create_task(self._video_processing_loop(ctx))

        # Create the agent
        self.agent = VoicePipelineAgent(
            vad=ctx.proc.userdata["vad"],
            stt=stt_obj,
            llm=llm_obj,
            tts=tts_obj,
            turn_detector=eou,
            chat_ctx=initial_chat_context,
            allow_interruptions=config.get('pipeline.allow_interruptions', True),
            interrupt_speech_duration=config.get('pipeline.interrupt_speech_duration', 0.5),
            interrupt_min_words=config.get('pipeline.interrupt_min_words', 0),
            min_endpointing_delay=config.get('pipeline.min_endpointing_delay', 0.3),
            preemptive_synthesis=config.get('pipeline.preemptive_synthesis', True),
            before_tts_cb=clean_text_for_tts,
            before_llm_cb=self.before_llm_vision_callback,
        )

        # Set up metrics collection
        self.usage_collector = metrics.UsageCollector()
        self._setup_metrics_collection()
        # Set up callback to add user and agent messages to conversation history
        self._setup_adding_user_message()
        self._setup_adding_agent_message()

        # Add shutdown callback
        ctx.add_shutdown_callback(lambda: self.application_shutdown())

        # Start the agent
        self.agent.start(ctx.room, participant)

        # Greet the user
        await self.agent.say("Hey, how can I help you today?", allow_interruptions=True)
   
        
    def _setup_metrics_collection(self):
        """Set up metrics collection for the agent.
        
        Args:
            usage_collector: Metrics usage collector
        """
        @self.agent.on("metrics_collected")
        def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
            metrics.log_metrics(agent_metrics)
            self.usage_collector.collect(agent_metrics)
            
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
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return ("You are Conversify, a friendly and efficient assistant.")
        
        
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