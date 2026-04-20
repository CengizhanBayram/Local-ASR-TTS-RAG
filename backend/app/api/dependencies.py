"""
FastAPI Dependencies
Dependency Injection ile servis yönetimi
"""

from functools import lru_cache
from typing import Generator

from ..services.speech_service import SpeechService
from ..services.document_service import DocumentService
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService


# Singleton service instances
_speech_service: SpeechService = None
_document_service: DocumentService = None
_rag_service: RAGService = None
_llm_service: LLMService = None


def get_speech_service() -> SpeechService:
    """Speech service dependency"""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service


def get_document_service() -> DocumentService:
    """Document service dependency"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


def get_rag_service() -> RAGService:
    """RAG service dependency"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_llm_service() -> LLMService:
    """LLM service dependency"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_services() -> None:
    """Reset all services (for testing)"""
    global _speech_service, _document_service, _rag_service, _llm_service
    _speech_service = None
    _document_service = None
    _rag_service = None
    _llm_service = None
