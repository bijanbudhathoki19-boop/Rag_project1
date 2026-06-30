
import os
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.text_chunks = []
        self.index = None

    def get_embeddings(self, texts):
        return np.array(
            self.embedder.encode(texts, show_progress_bar=False)
        ).astype("float32")

    def add_chunks(self, chunks: List[str]):
        if not chunks:
            return

        embeddings = self.get_embeddings(chunks)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            return []

        query_embedding = self.get_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(dist)))

        return results
