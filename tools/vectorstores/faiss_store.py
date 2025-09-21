import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base_vectorstore import BaseVectorStore
from tqdm import tqdm  # for progress display

class FAISSStore(BaseVectorStore):
    def __init__(self, top_k=4):
        self.top_k = top_k

    def build_index(self, passages, embeddings, metadatas):
        self.passages = passages
        self.metadatas = metadatas
        self.embeddings = np.array(embeddings, dtype="float32")

        # Normalize
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        print(f"Building FAISS index for {len(passages)} passages...")
        self.nn = NearestNeighbors(n_neighbors=min(self.top_k, len(passages)), metric="cosine")
        self.nn.fit(self.embeddings)
        print("FAISS index ready ✅")

    def retrieve(self, query_vector, top_k=None):
        k = top_k or self.top_k
        dists, idxs = self.nn.kneighbors(
            np.array(query_vector, dtype="float32").reshape(1, -1),
            n_neighbors=min(k, len(self.passages))
        )
        return [(self.passages[i], self.metadatas[i], float(d)) for i, d in zip(idxs[0], dists[0])]
