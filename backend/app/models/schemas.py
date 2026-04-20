"""
Pydantic schemas for request/response models
API için veri modelleri
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Desteklenen belge türleri"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class DocumentMetadata(BaseModel):
    """Belge metadata bilgileri"""
    filename: str
    file_type: DocumentType
    file_size: int
    page_count: Optional[int] = None
    uploaded_at: datetime = Field(default_factory=datetime.now)
    chunk_count: int = 0


class DocumentResponse(BaseModel):
    """Belge yükleme yanıtı"""
    id: str
    filename: str
    file_type: str
    chunk_count: int
    message: str
    success: bool = True


class DocumentListItem(BaseModel):
    """Belge listesi item"""
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    """Belge listesi yanıtı"""
    documents: List[DocumentListItem]
    total_count: int


class RAGQueryRequest(BaseModel):
    """RAG sorgu isteği"""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)


class SourceDocument(BaseModel):
    """Kaynak belge bilgisi"""
    filename: str
    content: str
    score: float
    page: Optional[int] = None


class RAGQueryResponse(BaseModel):
    """RAG sorgu yanıtı"""
    query: str
    answer: str
    sources: List[SourceDocument] = []
    processing_time_ms: float


class VoiceQueryResponse(BaseModel):
    """Sesli sorgu yanıtı"""
    transcribed_text: str
    answer: str
    sources: List[SourceDocument] = []
    audio_base64: str  # Base64 encoded audio response
    processing_time_ms: float


class TextQueryRequest(BaseModel):
    """Text tabanlı sorgu isteği"""
    query: str = Field(..., min_length=1, max_length=2000)
    include_audio: bool = Field(default=False)


class TextQueryResponse(BaseModel):
    """Text tabanlı sorgu yanıtı"""
    query: str
    answer: str
    sources: List[SourceDocument] = []
    audio_base64: Optional[str] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check yanıtı"""
    status: str = "healthy"
    version: str
    services: dict


class ErrorResponse(BaseModel):
    """Hata yanıtı"""
    error: str
    detail: Optional[str] = None
    status_code: int
