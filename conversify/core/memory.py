import logging
import os
from typing import List, Any, Dict, Optional, Callable

import numpy as np
from vllm import LLM
from pydantic import BaseModel, Field

from memoripy import MemoryManager, JSONStorage, ChatModel, EmbeddingModel

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from livekit.agents import ChatMessage, ChatContext

logger = logging.getLogger(__name__)


class ConceptExtractionResponse(BaseModel):
    """Model for structured response from concept extraction."""
    concepts: List[str] = Field(description="List of key concepts extracted from the text.")


class ChatCompletionsModel(ChatModel):
    """Implementation of ChatModel for concept extraction using LLM."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize the ChatCompletionsModel with configuration.
        
        Args:
            llm_config: Dictionary containing LLM configuration (base_url, api_key, model)
        """
        api_endpoint = llm_config['base_url']
        api_key = llm_config['api_key']
        model_name = llm_config['model']

        logger.info(f"Initializing ChatCompletionsModel with endpoint: {api_endpoint}, model: {model_name}")
        try:
            self.llm = ChatOpenAI(
                openai_api_base=api_endpoint, 
                openai_api_key=api_key, 
                model_name=model_name,
                request_timeout=30.0,  
                max_retries=2         
            )
            self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
            self.prompt_template = PromptTemplate(
                template=(
                    "Extract key concepts from the following text in a concise, context-specific manner. "
                    "Include only the most highly relevant and specific core concepts that best capture the text's meaning. "
                    "Return nothing but the JSON string.\n"
                    "{format_instructions}\n{text}"
                ),
                input_variables=["text"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )
            logger.info("ChatCompletionsModel initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsModel components: {e}", exc_info=True)
            raise

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        """
        Invoke the LLM with a list of messages.
        
        Args:
            messages: List of message dictionaries to send to the LLM
        
        Returns:
            Response content as a string
        """
        if not messages:
            logger.warning("Empty messages list provided to ChatCompletionsModel.invoke()")
            return ""
        
        try:
            response = self.llm.invoke(messages)
            return str(response.content) if response and hasattr(response, 'content') else ""
        except Exception as e:
            logger.error(f"Error during ChatCompletionsModel invocation: {e}", exc_info=True)
            return "Error processing request."

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from the input text.
        
        Args:
            text: The text to extract concepts from
            
        Returns:
            List of extracted concept strings
        """
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or whitespace-only text provided to extract_concepts()")
            return []
        
        try:
            chain = self.prompt_template | self.llm | self.parser
            response = chain.invoke({"text": text})
            concepts = response.get("concepts", [])
            
            # Validate concepts
            valid_concepts = []
            for concept in concepts:
                if isinstance(concept, str) and concept.strip():
                    valid_concepts.append(concept.strip())
                    
            logger.debug(f"Concepts extracted: {valid_concepts}")
            return valid_concepts
        except Exception as e:
            logger.error(f"Error during concept extraction: {e}", exc_info=True)
            return []


class VLLMEmbeddingModel(EmbeddingModel):
    """Implementation of EmbeddingModel using VLLM."""
    
    def __init__(self, embedding_config: Dict[str, Any]):
        """
        Initialize the VLLMEmbeddingModel with configuration.
        
        Args:
            embedding_config: Dictionary containing embedding model configuration
        """
        model_name = embedding_config['vllm_model_name']
        
        self.model = LLM(
            model=model_name,
            enforce_eager=True,
        )
        logger.info(f"VLLMEmbeddingModel initialized successfully: {model_name}")

    def initialize_embedding_dimension(self) -> int:
        """
        Determine the embedding dimension by encoding a test string.
        
        Returns:
            Integer dimension of the embedding
        """
        try:
            test_text = "dimension_check"
            outputs = self.model.encode([test_text])
            
            embedding = self._extract_embedding_from_output(outputs)
            
            if embedding is not None:
                dimension = len(np.array(embedding))
                logger.info(f"Determined embedding dimension: {dimension}")
                return dimension
            else:
                logger.error(f"Failed to determine embedding dimension: Unexpected output structure from VLLM model")
                raise RuntimeError("Failed to determine embedding dimension due to unexpected model output.")
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension during initialization: {e}", exc_info=True)
            # Fallback dimension
            logger.warning("Falling back to default embedding dimension 768.")
            return 768
    
    def _extract_embedding_from_output(self, outputs) -> Optional[np.ndarray]:
        """
        Extract embedding from different possible VLLM output structures.
        
        Args:
            outputs: The output from VLLM's encode method
            
        Returns:
            Extracted embedding as numpy array or None if extraction fails
        """
        if not outputs:
            return None
            
        # Try different output structures that VLLM might return
        # Structure 1: outputs[0].outputs.embedding
        if hasattr(outputs[0], 'outputs') and hasattr(outputs[0].outputs, 'embedding'):
            return outputs[0].outputs.embedding
        # Structure 2: outputs[0].embedding
        elif hasattr(outputs[0], 'embedding'):
            return outputs[0].embedding
        # Structure 3: outputs[0] is the embedding directly
        elif isinstance(outputs[0], (list, np.ndarray)) and len(outputs[0]) > 0:
            return outputs[0]
        # Structure 4: outputs itself is the embedding
        elif isinstance(outputs, (list, np.ndarray)) and len(outputs) > 0:
            return outputs
        
        return None

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for the input text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        # Keep try-except for external model inference
        try:
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning("Empty text provided for embedding, returning zero vector")
                return np.zeros(self.dimension or 768)
                
            outputs = self.model.encode([text])
            embedding = self._extract_embedding_from_output(outputs)
            
            if embedding is not None:
                return np.array(embedding)
            else:
                logger.error(f"Could not extract embedding from VLLM output")
                return np.zeros(self.dimension or 768) 
        except Exception as e:
            logger.error(f"Error getting VLLM embedding for text '{text[:50] if text else ''}...': {e}", exc_info=True)
            # Return a zero vector as fallback
            return np.zeros(self.dimension or 768)


class AgentMemoryManager:
    """Manages agent memory using the Memoripy library."""
    
    def __init__(self, participant_identity: str, config: Dict[str, Any]): 
        """
        Initialize the AgentMemoryManager.
        
        Args:
            participant_identity: Identifier for the participant
            config: Application configuration
        """
        self.participant_identity = participant_identity
        self.config = config
        self.memory_config = config['memory']
        self.memory_manager = None
        self._initialize_memory_manager()

    def _initialize_memory_manager(self) -> None:
        """Initialize the Memoripy MemoryManager with model instances."""
        if not self.memory_config['use']:
            logger.info(f"Memory is disabled in config for {self.participant_identity}. Skipping initialization.")
            return
            
        memory_dir_abs = self.memory_config['dir_abs']
        
        # Ensure the directory exists
        os.makedirs(memory_dir_abs, exist_ok=True)
        logger.info(f"Ensuring memory directory exists: {memory_dir_abs}")
             
        user_memory_file = os.path.join(memory_dir_abs, f"{self.participant_identity}.json")
        
        llm_cfg = self.config['llm']
        embedding_cfg = self.config['embedding']
            
        try:
            chat_model_instance = ChatCompletionsModel(llm_config=llm_cfg)
            embedding_model_instance = VLLMEmbeddingModel(embedding_config=embedding_cfg)
            
            self.memory_manager = MemoryManager(
                chat_model=chat_model_instance,
                embedding_model=embedding_model_instance,
                storage=JSONStorage(user_memory_file)
            )
            logger.info(f"Initialized MemoryManager for user {self.participant_identity} with storage {user_memory_file}")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager components for {self.participant_identity}: {e}", exc_info=True)
            self.memory_manager = None

    async def load_memory(self, update_chat_ctx_func: Callable) -> None:
        """
        Load conversation history from storage and update the agent's chat context.
        
        Args:
            update_chat_ctx_func: Function to update chat context with loaded memory
        """
        if not self.memory_config.get('use', False):
            logger.info(f"Memory is disabled in config for {self.participant_identity}. Skipping load.")
            return
            
        if not self.memory_manager:
            logger.warning(f"MemoryManager not initialized for {self.participant_identity}. Cannot load history.")
            return

        initial_messages_from_memory = []
        
        try:
            short_term_history, _ = self.memory_manager.load_history()
            # Use config value for number of interactions
            num_interactions_to_load = self.memory_config.get('load_last_n', 5)
            memory_interactions = short_term_history[-num_interactions_to_load:] if short_term_history else []

            for interaction in memory_interactions:
                if interaction.get('prompt'):
                    initial_messages_from_memory.append(ChatMessage(role="user", content=[interaction['prompt']]))
                if interaction.get('output'):
                    initial_messages_from_memory.append(ChatMessage(role="assistant", content=[interaction['output']]))

            if initial_messages_from_memory:
                await update_chat_ctx_func(ChatContext(initial_messages_from_memory))
                logger.info(f"Prepended {len(initial_messages_from_memory)} interactions to the initial context for {self.participant_identity}.")
            else:
                logger.info(f"No interactions loaded from memory for {self.participant_identity}.")

        except FileNotFoundError:
            logger.info(f"No previous history file found for {self.participant_identity}. Starting fresh.")
        except Exception as e:
            logger.error(f"Failed to load history via Memoripy for {self.participant_identity}: {e}", exc_info=True)

    def _extract_message_content(self, message: ChatMessage) -> str:
        """
        Extract text content from a ChatMessage.
        
        Args:
            message: The ChatMessage to extract content from
            
        Returns:
            Extracted text content as a string
        """
        if not message or not message.content:
            return ""
            
        # Handle different content structures
        if isinstance(message.content, list):
            if not message.content:
                return ""
            content_item = message.content[0]
            if isinstance(content_item, str):
                return content_item
            elif hasattr(content_item, 'text'):
                return content_item.text
            else:
                return str(content_item)
        else:
            return str(message.content)

    async def save_memory(self, chat_ctx: ChatContext) -> None:
        """
        Save the current conversation history to storage.
        
        Args:
            chat_ctx: ChatContext containing the conversation messages
        """
        if not self.memory_config.get('use', False):
            logger.info(f"Memory is disabled in config for {self.participant_identity}. Skipping save.")
            return
            
        if self.memory_manager is None:
            logger.warning(f"Memory manager not available for {self.participant_identity}. Skipping history save.")
            return

        if not chat_ctx or not chat_ctx.items:
            logger.info(f"No conversation items to save for {self.participant_identity}.")
            return

        logger.info(f"Saving conversation history via Memoripy for user: {self.participant_identity}")
        logger.info(f"Conversation history messages count: {len(chat_ctx.items)}")
        
        i = 0
        processed_count = 0
        items = chat_ctx.items
        
        while i < len(items):
            user_msg = None
            assistant_msg = None

            # Find the next user message
            if items[i].role == "user":
                user_msg = items[i]
                # Find the corresponding assistant message (if it exists)
                if i + 1 < len(items) and items[i+1].role == "assistant":
                    assistant_msg = items[i+1]
                    i += 2 # Move past both
                else:
                    i += 1 # Move past only user msg
            elif items[i].role == "assistant":
                # Skip assistant message without preceding user message
                logger.warning(f"Skipping assistant message without preceding user message at index {i}")
                i += 1
                continue
            else: # Skip system messages etc.
                i += 1
                continue

            # Process the interaction pair
            if user_msg:
                # Extract content using helper method
                user_prompt = self._extract_message_content(user_msg)
                assistant_response = self._extract_message_content(assistant_msg) if assistant_msg else ""

                combined_text = f"{user_prompt} {assistant_response}".strip()

                if not combined_text:
                    logger.debug("Skipping empty interaction.")
                    continue

                try:
                    concepts = self.memory_manager.extract_concepts(combined_text)
                    embedding = self.memory_manager.get_embedding(combined_text)
                    self.memory_manager.add_interaction(
                        prompt=user_prompt,
                        output=assistant_response,
                        embedding=embedding,
                        concepts=concepts
                    )
                    processed_count += 1
                    logger.debug(f"Added interaction to Memoripy: User: '{user_prompt[:50]}...' Assistant: '{assistant_response[:50]}...'")
                except Exception as e:
                    logger.error(f"Error processing interaction via Memoripy: {e} for interaction: User='{user_prompt[:50]}...', Assistant='{assistant_response[:50]}...'", exc_info=True)
        
        if processed_count > 0:
            logger.info(f"Successfully added {processed_count} interactions into conversational memory for {self.participant_identity}")
        else:
            logger.warning(f"No interactions were added to memory for {self.participant_identity}") 