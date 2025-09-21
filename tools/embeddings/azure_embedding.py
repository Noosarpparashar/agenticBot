from .base_embedding import BaseEmbedding
from langchain_openai import AzureOpenAIEmbeddings
import os

class AzureEmbedding(BaseEmbedding):
    def __init__(self, endpoint=None, api_key=None, deployment=None, api_version=None):
        self.embedder = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["EMBEDDING_ENDPOINT_URL"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            model=os.environ["EMBEDDING_DEPLOYMENT"],  # your embedding deployment
        )

    def embed(self, text: str):
        return self.embedder.embed_query(text)

    def embed_batch(self, texts: list):
        return [self.embedder.embed_query(t) for t in texts]
