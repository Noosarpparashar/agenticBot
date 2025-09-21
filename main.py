from agents.registry import AgentRegistry
from tools.embeddings.azure_embedding import AzureEmbedding
from tools.vectorstores.faiss_store import FAISSStore
from utils.document_loader import load_documents, split_into_passages
from agents.rag_agent import RAGAgent
from tools.llms.azure_llm import AzureLLM
import os
import numpy as np
from tqdm import tqdm

DOCS_DIR = "docs"
TOP_K = 4


def main():
    # 1️⃣ Setup embedders
    azure_embedder = AzureEmbedding()

    # 2️⃣ Load & process docs
    docs = load_documents(DOCS_DIR)
    passages, metadatas, embeddings = [], [], []

    for path, text in docs:
        chunks = split_into_passages(text)
        for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {os.path.basename(path)}")):
            passages.append(chunk)
            metadatas.append({"source": path, "chunk_index": i, "char_len": len(chunk)})

            # For a single chunk, use embed()
            vec = azure_embedder.embed(chunk)
            embeddings.append(vec)

    # 3️⃣ Build vectorstore
    vectorstore = FAISSStore(top_k=TOP_K)
    vectorstore.build_index(passages, embeddings, metadatas)

    # 4️⃣ Register multiple agents
    registry = AgentRegistry()


    rag_agent = RAGAgent(embedder=azure_embedder, vectorstore=vectorstore, llm=AzureLLM())
    registry.register("rag", rag_agent)

    # Example: you could add summarizer, graph-agent, etc.
    # registry.register("summarizer", SummarizerAgent(...))

    # 5️⃣ Interactive / API loop
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # Decide which agent to use
        agent_name = "rag"  # could be dynamic
        agent = registry.get(agent_name)
        answer = agent.run(query)
        print("\nAssistant:\n", answer)
        print("="*60)

if __name__ == "__main__":
    main()
