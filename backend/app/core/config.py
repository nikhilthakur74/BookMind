"""Application configuration managed via Pydantic Settings."""

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: 4 levels up from this file (backend/app/core/config.py -> c:/bookreader)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """BookMind application settings loaded from environment and .env file."""

    # Application
    APP_NAME: str = "BookMind API"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api"

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Support comma-separated strings or JSON arrays for CORS_ORIGINS."""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    # PDF Processing & Ingestion Settings (Phase 2)
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf"]
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    OCR_THRESHOLD_MIN_CHARS_PER_PAGE: int = 50

    # LLM Provider Configuration
    LLM_PROVIDER: str = "ollama"  # "ollama", "groq", or "mock"
    LLM_MODEL: Optional[str] = None  # Optional override for active provider model
    PROMPT_VERSION: str = "v1.0"
    MAX_CONCURRENT_LLM_CALLS: int = 1  # Concurrency limit for LLM generation
    LLM_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT_SECONDS: float = 120.0
    LONG_CHAPTER_CHUNK_THRESHOLD: int = 8

    # Ollama Local LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SECONDS: float = 120.0
    OLLAMA_NUM_CTX: int = 4096  # Bounded context window for RTX 3050 (4GB VRAM)
    OLLAMA_NUM_PREDICT: int = 2048  # Output token generation limit
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_KEEP_ALIVE: str = "5m"
    OLLAMA_MAX_CONCURRENT: int = 1  # Sequential local generation to protect GPU VRAM

    # Groq LLM API (Preserved for cloud inference)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Conversational RAG & Reranker (Phase 5)
    CHAT_PROMPT_VERSION: str = "v1.0"
    CHAT_MAX_HISTORY_TURNS: int = 6
    CHAT_TOP_K: int = 5
    ENABLE_RERANKER: bool = False
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-6-v2"

    # Qdrant Vector Database
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "bookmind_chunks"
    QDRANT_STORAGE_DIR: str = "data/qdrant_storage"

    # Embeddings (Phase 3: BGE-M3)
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"  # "cpu", "cuda", "mps", "auto"
    EMBEDDING_PROVIDER: str = "auto"  # "auto", "onnx", "mock"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_DIMENSION: int = 1024

    # Lexical BM25 & Hybrid Fusion (Phase 3)
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    DEFAULT_RETRIEVAL_TOP_K: int = 5

    # Storage Paths
    UPLOAD_DIR: str = "data/uploads"

    @property
    def resolved_upload_dir(self) -> Path:
        """Resolve upload directory relative to project root."""
        p = Path(self.UPLOAD_DIR)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    @property
    def books_dir(self) -> Path:
        """Directory where processed books, PDFs, and metadata are stored."""
        target = self.resolved_upload_dir / "books"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def resolved_qdrant_storage_dir(self) -> Path:
        """Directory where local embedded Qdrant database persists collections and vectors."""
        p = Path(self.QDRANT_STORAGE_DIR)
        target = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()


settings = get_settings()
