"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "Voice AI RAG"
    app_version: str = "2.0.0"
    debug: bool = False

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # gemini | ollama | openai_compatible
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")

    # Gemini
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", env="GEMINI_MODEL")

    # Ollama  (local GPU LLM)
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:3b", env="OLLAMA_MODEL")

    # OpenAI-compatible  (vLLM, LM Studio, LocalAI …)
    openai_compat_base_url: str = Field(default="http://localhost:8080/v1", env="OPENAI_COMPAT_BASE_URL")
    openai_compat_model: str = Field(default="meta-llama/Llama-3.2-3B-Instruct", env="OPENAI_COMPAT_MODEL")
    openai_compat_api_key: str = Field(default="dummy", env="OPENAI_COMPAT_API_KEY")

    # ── STT — Faster Whisper ─────────────────────────────────────────────────
    whisper_model_size: str = Field(default="small", env="WHISPER_MODEL_SIZE")
    # cpu | cuda   — set cuda on GPU server
    whisper_device: str = Field(default="cpu", env="WHISPER_DEVICE")
    # int8 (CPU) | float16 (GPU) | float32
    whisper_compute_type: str = Field(default="int8", env="WHISPER_COMPUTE_TYPE")
    # CTranslate2 thread count; 0=auto (all cores). On 128-core EPYC, 16 avoids contention.
    whisper_cpu_threads: int = Field(default=16, env="WHISPER_CPU_THREADS")
    speech_language: str = Field(default="tr", env="SPEECH_LANGUAGE")

    # ── TTS ──────────────────────────────────────────────────────────────────
    # piper | edge_tts
    tts_backend: str = Field(default="edge_tts", env="TTS_BACKEND")
    # Piper (lokal, ONNX CPU)
    piper_model_path: str = Field(
        default="models/tr_TR-dfki-medium.onnx",
        env="PIPER_MODEL_PATH"
    )
    # Edge-TTS (Microsoft Neural, online, en iyi Türkçe kalitesi)
    edge_tts_voice: str = Field(default="tr-TR-EmelNeural", env="EDGE_TTS_VOICE")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    # cpu | cuda
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")

    # ── Reranker ─────────────────────────────────────────────────────────────
    enable_reranking: bool = Field(default=True, env="ENABLE_RERANKING")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RERANKER_MODEL"
    )
    reranker_device: str = Field(default="cpu", env="RERANKER_DEVICE")

    # ── RAG Core ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    score_threshold: float = Field(default=0.25, env="SCORE_THRESHOLD")

    # ── Hybrid Search (BM25 + Vector) ────────────────────────────────────────
    enable_hybrid_search: bool = Field(default=True, env="ENABLE_HYBRID_SEARCH")
    hybrid_alpha: float = Field(default=0.6, env="HYBRID_ALPHA")  # vector weight

    # ── Multi-Query RAG ──────────────────────────────────────────────────────
    enable_multi_query: bool = Field(default=False, env="ENABLE_MULTI_QUERY")
    multi_query_count: int = Field(default=3, env="MULTI_QUERY_COUNT")

    # ── Query Rewriting ──────────────────────────────────────────────────────
    enable_query_rewriting: bool = Field(default=False, env="ENABLE_QUERY_REWRITING")

    # ── Parent-Child Chunking ────────────────────────────────────────────────
    enable_parent_child: bool = Field(default=True, env="ENABLE_PARENT_CHILD")
    child_chunk_size: int = Field(default=250, env="CHILD_CHUNK_SIZE")
    parent_chunk_size: int = Field(default=1200, env="PARENT_CHUNK_SIZE")

    # ── Inline Citations ─────────────────────────────────────────────────────
    enable_citations: bool = Field(default=True, env="ENABLE_CITATIONS")

    # ── OCR (optional — needs pdf2image + pytesseract installed) ─────────────
    enable_ocr: bool = Field(default=False, env="ENABLE_OCR")
    ocr_language: str = Field(default="tur+eng", env="OCR_LANGUAGE")

    # ── Storage ──────────────────────────────────────────────────────────────
    documents_path: str = Field(default="data/documents", env="DOCUMENTS_PATH")
    chroma_path: str = Field(default="data/chroma_db", env="CHROMA_PATH")

    # ── Conversation Memory ──────────────────────────────────────────────────
    max_conversation_history: int = Field(default=10, env="MAX_CONVERSATION_HISTORY")
    session_expiry_minutes: int = Field(default=60, env="SESSION_EXPIRY_MINUTES")

    # ── VAD (sent to frontend on connect) ────────────────────────────────────
    vad_silence_threshold: float = Field(default=0.008, env="VAD_SILENCE_THRESHOLD")
    vad_silence_duration_ms: int = Field(default=700, env="VAD_SILENCE_DURATION_MS")

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
