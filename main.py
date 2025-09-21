import os
import glob
import textwrap
from typing import List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_openai import AzureOpenAIEmbeddings
from tqdm import tqdm
import openai


# ---------------- CONFIG ----------------
DOCS_DIR = "docs"
TOP_K = 4
PASSAGE_CHARS = 800  # size of chunks

# Azure OpenAI settings
openai.api_type = "azure"
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]  # e.g. https://<your-resource>.openai.azure.com
openai.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]


# ---------------- DOCUMENT LOADING ----------------
def load_documents(path: str) -> List[Tuple[str, str]]:
    """Load .txt files. Returns list of (source_path, text)."""
    files = glob.glob(os.path.join(path, "*.txt"))
    docs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
            if text:
                docs.append((f, text))
    return docs


def split_into_passages(text: str, size: int = PASSAGE_CHARS, overlap: int = 200) -> List[str]:
    """Simple sliding-window text splitter by characters."""
    passages = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        passages.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return passages


# ---------------- EMBEDDING & INDEX ----------------
def build_index(docs: List[Tuple[str, str]], embedder: AzureOpenAIEmbeddings):
    """Returns (passages, metadatas, embeddings, nn_index)."""
    passages = []
    metadatas = []
    for path, text in docs:
        chunks = split_into_passages(text)
        for i, c in enumerate(chunks):
            passages.append(c)
            metadatas.append({"source": path, "chunk_index": i, "char_len": len(c)})

    print(f"[index] {len(passages)} passages to embed (this may take a moment).")

    # 🚀 Show progress while embedding
    embeddings = []
    for passage in tqdm(passages, desc="Embedding passages"):
        vec = embedder.embed_query(passage)   # embedding one passage
        embeddings.append(vec)

    embeddings = np.array(embeddings, dtype="float32")

    # Normalise for cosine distance
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    embeddings = embeddings / norm

    nn = NearestNeighbors(n_neighbors=min(TOP_K, len(passages)), metric="cosine")
    nn.fit(embeddings)

    return passages, metadatas, embeddings, nn


def retrieve(query: str, embedder: AzureOpenAIEmbeddings, passages, metadatas, embeddings, nn, k=TOP_K):
    """Retrieve top passages for a query using embeddings."""
    q_emb = embedder.embed_query(query)
    q_emb = np.array(q_emb, dtype="float32").reshape(1, -1)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

    dists, idxs = nn.kneighbors(q_emb, n_neighbors=min(k, len(passages)))
    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        results.append((passages[idx], metadatas[idx], float(dist)))
    return results


# ---------------- CHAT COMPLETION ----------------
def call_azure_openai_chat(query: str, context_passages: list) -> str:
    """Call Azure OpenAI ChatCompletion with retrieved passages."""
    if not context_passages:
        return "No context provided."

    context_text = "\n\n---\n\n".join(context_passages)

    system_prompt = (
        "You are an assistant that answers user questions using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer concisely."

    response = openai.chat.completions.create(
        model=os.environ["DEPLOYMENT_NAME"],  # your Azure deployment name
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=512,
    )

    return response.choices[0].message.content.strip()


# ---------------- INTERACTIVE LOOP ----------------
def interactive_loop(passages, metadatas, embeddings, nn, embedder: AzureOpenAIEmbeddings):
    print("---- Minimal RAG Chatbot (Azure) ----")
    print("Type 'exit' to quit. Type 'sources' to see loaded files.")
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Bye 👋")
            break
        if query.lower() == "sources":
            srcs = sorted({m["source"] for m in metadatas})
            print("Loaded sources:")
            for s in srcs:
                print(" -", s)
            continue

        results = retrieve(query, embedder, passages, metadatas, embeddings, nn, k=TOP_K)
        print("\n[retrieved top passages:]\n")
        for i, (p, meta, d) in enumerate(results, 1):
            snippet = textwrap.shorten(p.replace("\n", " "), width=240)
            print(f"[{i}] dist={d:.4f} src={meta['source']} len={meta['char_len']} -> {snippet}\n")

        context_passages = [r[0] for r in results]
        answer = call_azure_openai_chat(query, context_passages)
        print("\nAssistant:\n")
        print(answer)
        print("\n" + "="*60)


# ---------------- MAIN ----------------
def main():
    # Setup LangChain Azure embeddings
    embedder = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["EMBEDDING_ENDPOINT_URL"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        model=os.environ["EMBEDDING_DEPLOYMENT"],  # your embedding deployment
    )

    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"No documents found in {DOCS_DIR}. Create the folder and add .txt files. Exiting.")
        return

    passages, metadatas, embeddings, nn = build_index(docs, embedder)
    interactive_loop(passages, metadatas, embeddings, nn, embedder)


if __name__ == "__main__":
    main()
