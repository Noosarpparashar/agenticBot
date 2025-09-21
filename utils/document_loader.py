import glob
from typing import List, Tuple

def load_documents(path: str) -> List[Tuple[str, str]]:
    files = glob.glob(f"{path}/*.txt")
    docs = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
            if text:
                docs.append((f, text))
    return docs

def split_into_passages(text: str, size=800, overlap=200):
    passages = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        passages.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return passages
