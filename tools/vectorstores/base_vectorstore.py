from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def build_index(self, passages: list, embeddings: list, metadatas: list):
        pass

    @abstractmethod
    def retrieve(self, query_vector, top_k: int):
        pass
