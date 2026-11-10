
"""Central configuration. Every value can be overridden with an environment variable."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# Ordered access levels.
# A user may read documents whose rank <= their own rank.
ACCESS_LEVELS = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3
}


def _env(name, default, cast=str):
    raw = os.getenv(name)
    return default if raw in (None, "") else cast(raw)


@dataclass(frozen=True)
class Settings:

    
   
    # LLM configuration
    llm_provider: str = _env("LLM_PROVIDER", "ollama").lower()

    ollama_model: str = _env("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_base_url: str = _env(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    groq_model: str = _env(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    temperature: float = _env("TEMPERATURE", 0.0, float)


    # Embedding configuration
    embedding_model: str = _env(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = _env(
        "EMBEDDING_DEVICE",
        "cpu"
    )

    # Vector database
    chroma_dir: str = _env(
        "CHROMA_DIR",
        str(BASE_DIR / "chroma_db")
    )
    collection_name: str = _env(
        "COLLECTION_NAME",
        "insurance_kb"
    )

    # Documents and text splitting
    documents_dir: str = _env(
        "DOCUMENTS_DIR",
        str(BASE_DIR / "data" / "documents")
    )
    chunk_size: int = _env("CHUNK_SIZE", 800, int)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 120, int)

    # Retrieval configuration
    vector_k: int = _env("VECTOR_K", 8, int)
    bm25_k: int = _env("BM25_K", 8, int)
    final_k: int = _env("FINAL_K", 4, int)
    rrf_k: int = _env("RRF_K", 60, int)

    # Relevance thresholds
    vector_min_sim: float = _env("VECTOR_MIN_SIM", 0.35, float)
    bm25_min_score: float = _env("BM25_MIN_SCORE", 1.0, float)


settings = Settings()
