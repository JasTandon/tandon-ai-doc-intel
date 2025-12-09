from abc import ABC, abstractmethod
from typing import List
import os
from openai import OpenAI

class EmbeddingsProvider(ABC):
    @abstractmethod
    def embed(self, text_chunks: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbeddings(EmbeddingsProvider):
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model

    def embed(self, text_chunks: List[str]) -> List[List[float]]:
        if not self.client or not text_chunks:
            return []
        
        # OpenAI batch limit is often 2048, but usually we send smaller batches.
        # Here we assume text_chunks is reasonable size.
        try:
            # Replace newlines in text as recommended by OpenAI for some models
            cleaned_chunks = [t.replace("\n", " ") for t in text_chunks]
            response = self.client.embeddings.create(input=cleaned_chunks, model=self.model)
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

