from abc import ABC, abstractmethod
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from memoripy import ChatModel, EmbeddingModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from core.config import config

from vllm import LLM
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")


class ChatCompletionsModel(ChatModel):
    def __init__(self):
        api_endpoint = config.get('llm.openai.api_endpoint')
        api_key = config.get('llm.openai.api_key', 'None') 
        model_name = config.get('llm.openai.model')

        logger.info(f"Initializing ChatCompletionsModel with endpoint: {api_endpoint}, model: {model_name}")
        self.llm = ChatOpenAI(openai_api_base=api_endpoint, openai_api_key=api_key, model_name=model_name)

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
    def __init__(self):
        model_name = config.get('memory.embedding.vllm.model') 
        self.model = LLM(
            model=model_name,
            task="embed",
            enforce_eager=True,
        )
        self.dimension = self.initialize_embedding_dimension()
        logger.info(f"VLLMEmbeddingModel initialized successfully. Model: {model_name}, Dimension: {self.dimension}")

    def get_embedding(self, text: str) -> np.ndarray:
        outputs = self.model.embed([text])
        embeds = outputs[0].outputs.embedding
        return np.array(embeds)

    def initialize_embedding_dimension(self) -> int:
        test_text = "Test to determine embedding dimension"
        outputs = self.get_embedding(test_text)
        dimension = len(outputs)
        logger.info(f"Determined embedding dimension: {dimension}")
        return dimension
    
        
    
    