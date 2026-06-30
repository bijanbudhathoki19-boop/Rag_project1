
import os
import pickle
from typing import List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class RAGEngine:
    """
    Simple RAG Engine using FAISS + SentenceTransformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "faiss.index"):
        self.model_name = model_name
        self.index_path = index_path
        self.text_chunks: List[str] = []

        if SentenceTransformer:
            self.embedder = SentenceTransformer(model_name)
        else:
            self.embedder = None

        self.index = None
        self.dimension = 384

        if faiss and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.embedder:
            raise ImportError("SentenceTransformer not installed")

        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype("float32")

    def build_index(self, texts: List[str]):
        self.text_chunks = texts
        embeddings = self.get_embeddings(texts)

        if faiss is None:
            raise ImportError("FAISS not installed")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)

    def add_document(self, document: str):
        chunks = self.chunk_text(document)
        embeddings = self.get_embeddings(chunks)

        if self.index is None:
            if faiss is None:
                raise ImportError("FAISS not installed")
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.text_chunks = []

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

        faiss.write_index(self.index, self.index_path)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            return []

        query_vec = self.get_embeddings([query])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(dist)))

        return results


def load_engine(path: str = "faiss.index") -> RAGEngine:
    return RAGEngine(index_path=path)
