"""
FastAPI Dependencies — singleton service registry
"""

from ..services.speech_service import SpeechService
from ..services.document_service import DocumentService
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService
from ..services.reranker_service import RerankerService
from ..services.conversation_service import ConversationService
from ..services.cache_service import SemanticCacheService

_speech_service:       SpeechService       = None
_document_service:     DocumentService     = None
_rag_service:          RAGService          = None
_llm_service:          LLMService          = None
_reranker_service:     RerankerService     = None
_conversation_service: ConversationService = None
_cache_service:        SemanticCacheService = None


def get_speech_service() -> SpeechService:
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service


def get_conversation_service() -> ConversationService:
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service


def get_cache_service() -> SemanticCacheService:
    global _cache_service
    if _cache_service is None:
        from ..config import get_settings
        settings = get_settings()
        rag = get_rag_service()
        _cache_service = SemanticCacheService(
            embed_fn=rag._embedding_model.encode,
            max_size=settings.cache_max_size,
            similarity_threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
            enabled=settings.cache_enabled,
        )
    return _cache_service


def reset_services() -> None:
    global _speech_service, _document_service, _rag_service, _llm_service
    global _reranker_service, _conversation_service, _cache_service
    _speech_service = _document_service = _rag_service = _llm_service = None
    _reranker_service = _conversation_service = _cache_service = None
