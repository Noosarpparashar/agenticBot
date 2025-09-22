import os
import logging
from agents.registry import AgentRegistry
from agents.rag_agent import RAGAgent
from tools.embeddings.azure_embedding import AzureEmbedding
from tools.vectorstores.faiss_store import FAISSStore
from tools.llms.azure_llm import AzureLLM
from utils.document_loader import load_documents, split_into_passages
from tqdm import tqdm

# ---------------- CONFIG ----------------
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
TOP_K = int(os.getenv("TOP_K", 4))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    try:
        logging.info("Initializing embedder...")
        azure_embedder = AzureEmbedding()

        logging.info("Loading documents...")
        docs = load_documents(DOCS_DIR)
        if not docs:
            logging.warning(f"No documents found in {DOCS_DIR}. Exiting.")
            return

        passages, metadatas, embeddings = [], [], []
        for path, text in docs:
            chunks = split_into_passages(text)
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {os.path.basename(path)}")):
                passages.append(chunk)
                metadatas.append({"source": path, "chunk_index": i, "char_len": len(chunk)})

                try:
                    vec = azure_embedder.embed(chunk)
                    embeddings.append(vec)
                except Exception as e:
                    logging.error(f"Embedding failed for {path} [chunk {i}]: {e}")

        logging.info("Building FAISS vectorstore...")
        vectorstore = FAISSStore(top_k=TOP_K)
        vectorstore.build_index(passages, embeddings, metadatas)

        logging.info("Registering agents...")
        registry = AgentRegistry()
        rag_agent = RAGAgent(embedder=azure_embedder, vectorstore=vectorstore, llm=AzureLLM())
        registry.register("rag", rag_agent)

        logging.info("Starting interactive loop. Type 'exit' to quit.")
        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]:
                break

            agent = registry.get("rag")
            if not agent:
                logging.error("No agent found for 'rag'")
                continue

            try:
                answer = agent.run(query)
                print("\nAssistant:\n", answer)
                print("=" * 60)
            except Exception as e:
                logging.error(f"Error while running agent: {e}")

    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
