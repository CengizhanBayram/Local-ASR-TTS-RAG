"""
FastAPI Dependencies
Dependency Injection ile servis yönetimi
"""

from ..services.speech_service import SpeechService
from ..services.document_service import DocumentService
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService
from ..services.conversation_service import ConversationService


_speech_service: SpeechService = None
_document_service: DocumentService = None
_rag_service: RAGService = None
_llm_service: LLMService = None
_conversation_service: ConversationService = None


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


def get_conversation_service() -> ConversationService:
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service


def reset_services() -> None:
    global _speech_service, _document_service, _rag_service, _llm_service, _conversation_service
    _speech_service = None
    _document_service = None
    _rag_service = None
    _llm_service = None
    _conversation_service = None
