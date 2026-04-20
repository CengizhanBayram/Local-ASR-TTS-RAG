"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # Application Settings
    app_name: str = "Voice AI RAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # Google Gemini API
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", env="GEMINI_MODEL")

    # Faster Whisper (local STT)
    whisper_model_size: str = Field(default="small", env="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", env="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", env="WHISPER_COMPUTE_TYPE")
    speech_language: str = Field(default="tr", env="SPEECH_LANGUAGE")

    # Piper TTS (local)
    piper_model_path: str = Field(
        default="models/tr_TR-dfki-medium.onnx",
        env="PIPER_MODEL_PATH"
    )

    # RAG Settings
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")

    # Storage Paths
    documents_path: str = Field(default="data/documents", env="DOCUMENTS_PATH")
    chroma_path: str = Field(default="data/chroma_db", env="CHROMA_PATH")

    # CORS Settings
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")

    # RAG Pipeline - Advanced
    score_threshold: float = Field(default=0.25, env="SCORE_THRESHOLD")
    enable_query_rewriting: bool = Field(default=False, env="ENABLE_QUERY_REWRITING")

    # Conversation Memory
    max_conversation_history: int = Field(default=10, env="MAX_CONVERSATION_HISTORY")
    session_expiry_minutes: int = Field(default=60, env="SESSION_EXPIRY_MINUTES")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
