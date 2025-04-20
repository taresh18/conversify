import logging
import re
import asyncio 
from typing import AsyncIterable, Dict, Any

from livekit.agents import (
    Agent,
    llm, 
    FunctionTool, 
    ChatContext 
)
from livekit.agents.voice import ModelSettings 
from livekit import rtc 
from livekit.agents.llm.chat_context import ImageContent 

from .memory import AgentMemoryManager

logger = logging.getLogger(__name__)

class ConversifyAgent(Agent):
    """ 
    Agent class that handles interaction logic, including optional
    memory management and image processing.
    Depends on shared_state for inter-task communication (e.g., latest_image)
    and an AgentMemoryManager instance for persistence.
    """
    def __init__(self, 
                 participant_identity: str,
                 shared_state: Dict[str, Any],
                 config: Dict[str, Any]) -> None: 
        
        agent_config = config['agent']
        memory_config = config['memory']
        
        # Get instructions from the config for the Agent constructor
        super().__init__(
            instructions=agent_config['instructions'], 
        )
        
        # Use default participant identity if it's provided and not empty
        self.participant_identity = agent_config['default_participant_identity'] or participant_identity
        
        self.config = config 
        self.shared_state = shared_state
        self.vision_keywords = ['see', 'look', 'picture', 'image', 'visual', 'color', 'this', 'object', 'view', 'frame', 'screen', 'desk', 'holding']
        
        # Initialize memory handler using config if enabled
        self.memory_handler = None
        if memory_config['use']:
            self.memory_handler = AgentMemoryManager(
                participant_identity=self.participant_identity,
                config=config 
            )
        
        logger.info(f"ConversifyAgent initialized for identity: {self.participant_identity}. Memory: {'Enabled' if self.memory_handler else 'Disabled'}")

    async def on_enter(self):
        """Called when the agent joins. Loads memory (if enabled) and greets."""
        logger.info(f"Agent '{self.participant_identity}' entering session.")
        if self.memory_handler:
            logger.info("Loading agent memory...")
            await self.memory_handler.load_memory(self.update_chat_ctx)
            logger.info("Agent memory loaded.")
        
        await self.session.say(self.config['agent']['greeting'])
        
    async def on_exit(self):
        """Called when the agent leaves. Says goodbye."""
        logger.info(f"Agent '{self.participant_identity}' exiting session.")
        await self.session.say(self.config['agent']['goodbye'])

    def process_image(self, chat_ctx: llm.ChatContext):
        """Checks for vision keywords and adds latest image from shared_state if applicable."""
        # Check if latest_image exists in shared_state
        if 'latest_image' not in self.shared_state:
            logger.warning("No 'latest_image' key found in shared_state")
            return
            
        latest_image = self.shared_state['latest_image']
        if not latest_image:
            logger.debug("Latest image is None or empty")
            return

        if not chat_ctx.items:
            return

        last_message = chat_ctx.items[-1]

        if last_message.role != "user" or not last_message.content or not isinstance(last_message.content[0], str):
            return

        user_text = last_message.content[0]
        
        should_add_image = any(keyword in user_text.lower() for keyword in self.vision_keywords)

        if should_add_image:
            logger.info(f"Vision keyword found in '{user_text[:50]}...'. Adding image to context.")
            if not isinstance(last_message.content, list):
                 last_message.content = [last_message.content] 
            last_message.content.append(ImageContent(image=latest_image))
            logger.debug("Successfully added ImageContent to the last message.")
        
    @staticmethod
    def clean_text(text_chunk: str) -> str:
        """Cleans text by removing special tags, code blocks, markdown, and emojis."""
        # Remove special tags
        cleaned = text_chunk.replace("<think>", "").replace("</think>", "")
        # Remove code blocks enclosed in triple backticks
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
        # Remove code blocks enclosed in triple single quotes
        cleaned = re.sub(r"'''(.*?)'''", r'\1', cleaned, flags=re.DOTALL)
        # Remove markdown bold/italic markers
        cleaned = re.sub(r'(\*\*|__)(.*?)\1', r'\2', cleaned)
        cleaned = re.sub(r'(\*|_)(.*?)\1', r'\2', cleaned)
        # Remove inline code markers (backticks)
        cleaned = re.sub(r'`([^`]*)`', r'\1', cleaned)
        # Remove LaTeX inline delimiters: remove one or more backslashes preceding "(" or ")"
        cleaned = re.sub(r'\\+\(', '', cleaned)
        cleaned = re.sub(r'\\+\)', '', cleaned)
        # Remove emojis (Unicode emoji ranges)
        cleaned = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', cleaned)
        return cleaned

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """Processes context via LLM, potentially adding image first. Delegates to default."""
        logger.debug(f"LLM node received context with {len(chat_ctx.items)} items.")
        
        # Only process image if vision is enabled in config
        if self.config['vision']['use']:
            self.process_image(chat_ctx)

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Cleans text stream and delegates to default TTS node."""
        logger.debug("TTS node received text stream.")
        
        cleaned_text_chunks = []
        
        async for chunk in text:
            # Process each chunk with the clean_text method
            cleaned_chunk = self.clean_text(chunk)
            if cleaned_chunk:
                cleaned_text_chunks.append(cleaned_chunk)

        if cleaned_text_chunks:
            logger.debug(f"Sending {len(cleaned_text_chunks)} cleaned chunks to default TTS.")
            async def text_stream():
                for cleaned_chunk in cleaned_text_chunks:
                    yield cleaned_chunk
            
            # Pass self as the first parameter to the default.tts_node method
            async for frame in self.default.tts_node(self, text_stream(), model_settings):
                yield frame
            logger.debug("TTS node finished streaming audio frames.")
        else:
            logger.info("No text content left after cleaning for TTS.")

    