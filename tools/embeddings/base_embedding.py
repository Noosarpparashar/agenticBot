from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, text: str):
        """Return a vector for the text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list):
        """Return a list of vectors for multiple texts."""
        pass
