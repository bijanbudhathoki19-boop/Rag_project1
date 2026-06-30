import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# API KEYS
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -----------------------------
# LLM SETTINGS
# -----------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | openai | gemini

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Temperature controls randomness
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

# Max tokens for response
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

# -----------------------------
# EMBEDDING SETTINGS
# -----------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# -----------------------------
# RAG SETTINGS
# -----------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

TOP_K = int(os.getenv("TOP_K", 5))  # number of retrieved chunks

# -----------------------------
# FAISS SETTINGS
# -----------------------------
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vectorstore/faiss.index")
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "vectorstore/docstore.pkl")

# -----------------------------
# FILE STORAGE
# -----------------------------
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# -----------------------------
# STREAMLIT SETTINGS
# -----------------------------
APP_NAME = "RAG Chatbot"
APP_ICON = "🤖"

# -----------------------------
# DEBUG SETTINGS
# -----------------------------
DEBUG = os.getenv("DEBUG", "False").lower() == "true"