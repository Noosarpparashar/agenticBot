from .base_agent import BaseAgent

class RAGAgent(BaseAgent):
    def __init__(self, embedder, vectorstore, llm):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm

    def run(self, query: str):
        query_vector = self.embedder.embed(query)
        results = self.vectorstore.retrieve(query_vector)
        context_passages = [r for r in results]
        return self.llm.generate(query, context_passages)
