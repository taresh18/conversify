from abc import ABC, abstractmethod
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from vllm import LLM
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ChatModel(ABC):
    @abstractmethod
    def invoke(self, messages: list) -> str:
        """
        Generate a response from the chat model given a list of messages.
        """
        pass

    @abstractmethod
    def extract_concepts(self, text: str) -> list[str]:
        """
        Extract key concepts from the provided text.
        """
        pass
    
class EmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        """
        pass

    @abstractmethod
    def initialize_embedding_dimension(self) -> int:
        """
        Determine the dimension of the embeddings.
        """
        pass


class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")


class ChatCompletionsModel(ChatModel):
    def __init__(self, api_endpoint: str="http://127.0.0.1:30000/v1", api_key: str="None", model_name: str="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatOpenAI(openai_api_base = api_endpoint, openai_api_key = api_key, model_name = model_name)
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

    def invoke(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return str(response.content)
    
    def extract_concepts(self, text: str) -> list[str]:
        chain = self.prompt_template | self.llm | self.parser
        response = chain.invoke({"text": text})
        concepts = response.get("concepts", [])
        logger.info(f"Concepts extracted: {concepts}")
        return concepts
        
class VLLMEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1"):
        self.model_name = model_name
        self.model = LLM(
            model=self.model_name,
            task="embed",
            enforce_eager=True,
        )
        self.dimension = self.initialize_embedding_dimension()

    def get_embedding(self, text: str) -> np.ndarray:
        # Generate embedding. The output is a list of EmbeddingRequestOutputs.
        outputs = self.model.embed([text])
        embeds = outputs[0].outputs.embedding
        if embeds is None:
            raise ValueError("Failed to generate embedding.")
        return np.array(embeds)

    def initialize_embedding_dimension(self) -> int:
        test_text = "Test to determine embedding dimension"
        outputs = self.get_embedding(test_text)
        return len(outputs)
    
        
    
    