import os
import re
from typing import List

# PDF reader
from PyPDF2 import PdfReader

# Optional embeddings (recommended)
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _model = None


# -----------------------------
# PDF LOADING
# -----------------------------
def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return clean_text(text)


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text: str) -> str:
    """
    Basic text cleaning for RAG pipelines.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap  # overlap for context continuity

    return chunks


# -----------------------------
# EMBEDDINGS
# -----------------------------
def get_embeddings(texts: List[str]):
    """
    Generate embeddings using SentenceTransformers (offline) by default.
    Replace with OpenAI/Groq embeddings if needed.
    """
    if _model:
        return _model.encode(texts)

    raise ImportError(
        "SentenceTransformer model not loaded. Install: pip install sentence-transformers"
    )


# -----------------------------
# OPTIONAL: SINGLE TEXT EMBEDDING
# -----------------------------
def embed_text(text: str):
    """
    Embed a single text string.
    """
    return get_embeddings([text])[0]


# -----------------------------
# SIMPLE SEARCH HELPER (optional for FAISS use)
# -----------------------------
def normalize(text: str) -> str:
    """
    Normalize text for consistency.
    """
    return text.lower().strip()