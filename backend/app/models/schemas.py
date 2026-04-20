"""
Pydantic schemas for request/response models
API için veri modelleri
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class DocumentMetadata(BaseModel):
    filename: str
    file_type: DocumentType
    file_size: int
    page_count: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    chunk_count: int = 0


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    message: str
    success: bool = True


class DocumentListItem(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total_count: int


class SourceDocument(BaseModel):
    filename: str
    content: str
    score: float
    page: Optional[int] = None


class PipelineMetrics(BaseModel):
    total_ms: float
    stt_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    rewrite_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    tts_ms: Optional[float] = None
    docs_retrieved: Optional[int] = None
    docs_after_threshold: Optional[int] = None


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument] = []
    processing_time_ms: float
    metrics: Optional[PipelineMetrics] = None


class VoiceQueryResponse(BaseModel):
    transcribed_text: str
    answer: str
    sources: List[SourceDocument] = []
    audio_base64: str
    processing_time_ms: float
    metrics: Optional[PipelineMetrics] = None


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    include_audio: bool = Field(default=False)


class TextQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceDocument] = []
    audio_base64: Optional[str] = None
    processing_time_ms: float
    metrics: Optional[PipelineMetrics] = None


# ── Conversation / Chat ──────────────────────────────────────────────────────

class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    role: ConversationRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    include_audio: bool = Field(default=False)
    mode: str = Field(default="rag")  # "rag" or "free"


class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    session_id: str
    sources: List[SourceDocument] = []
    audio_base64: Optional[str] = None
    processing_time_ms: float
    conversation_turn: int
    metrics: Optional[PipelineMetrics] = None
    rewritten_query: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[ConversationMessage]
    turn_count: int


# ── System ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    services: dict


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
