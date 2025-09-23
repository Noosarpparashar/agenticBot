import faiss
import numpy as np
import os
import json
import logging

class FAISSStore:
    def __init__(self, top_k=4):
        self.index = None
        self.passages = []
        self.metadatas = []
        self.top_k = top_k

    def build_index(self, passages, embeddings, metadatas):
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)

        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)

        self.passages = passages
        self.metadatas = metadatas

        # ✅ Validation: embeddings count must match passages
        assert len(self.passages) == self.index.ntotal, (
            f"Mismatch: {len(self.passages)} passages but {self.index.ntotal} vectors in FAISS"
        )

        logging.debug(f"[BUILD] Built index with dim={dim}, "
                      f"{self.index.ntotal} vectors, "
                      f"{len(self.passages)} passages")

    def search(self, query_embedding):
        query_embedding = np.array([query_embedding]).astype("float32")
        logging.debug(f"[SEARCH] Query embedding shape: {query_embedding.shape}")

        distances, indices = self.index.search(query_embedding, self.top_k)
        logging.debug(f"[SEARCH] Raw FAISS indices: {indices}")
        logging.debug(f"[SEARCH] Raw FAISS distances: {distances}")

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if dist == np.finfo("float32").max:  # filter invalid results
                logging.warning(f"[SEARCH] Skipping invalid result (dist=max_float) at idx={idx}")
                continue
            if idx < len(self.passages):
                result = {
                    "text": self.passages[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(dist),
                }
                results.append(result)
                logging.debug(f"[SEARCH] Retrieved -> idx={idx}, dist={dist}, "
                              f"source={result['metadata'].get('source', 'N/A')}")
        return results

    # ----------------------------
    # Persistence Methods
    # ----------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        # Save metadata + passages
        with open(os.path.join(path, "store.json"), "w", encoding="utf-8") as f:
            json.dump({
                "passages": self.passages,
                "metadatas": self.metadatas
            }, f, ensure_ascii=False, indent=2)

        # ✅ Validation after save
        logging.info(f"[SAVE] Saved FAISS with {self.index.ntotal} vectors, "
                     f"{len(self.passages)} passages, {len(self.metadatas)} metadatas")

    def load(self, path: str):
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))

        # Load metadata + passages
        with open(os.path.join(path, "store.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            self.passages = data["passages"]
            self.metadatas = data["metadatas"]

        # ✅ Validation after load
        if len(self.passages) != self.index.ntotal:
            raise ValueError(
                f"[LOAD] Corruption detected: {len(self.passages)} passages "
                f"but {self.index.ntotal} vectors in FAISS index"
            )
        logging.info(f"[LOAD] Loaded FAISS with {self.index.ntotal} vectors "
                     f"and {len(self.passages)} passages")

    def retrieve(self, query_embedding):
        logging.debug("[RETRIEVE] Starting retrieval...")
        results = self.search(query_embedding)
        logging.debug(f"[RETRIEVE] Results: {results}")
        print("results********", results)  # keep your print
        return [r["text"] for r in results]
