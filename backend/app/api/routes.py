"""
API Routes - Tüm REST endpoint'leri
"""

import time
import base64
import logging
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..models.schemas import (
    DocumentResponse,
    DocumentListResponse,
    DocumentListItem,
    RAGQueryRequest,
    RAGQueryResponse,
    VoiceQueryResponse,
    TextQueryRequest,
    TextQueryResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    SessionHistoryResponse,
    ConversationMessage,
    ConversationRole,
    SourceDocument,
    HealthResponse,
    ErrorResponse,
    PipelineMetrics,
)
from ..models.exceptions import VoiceAIException, NoDocumentsError
from ..services.speech_service import SpeechService
from ..services.document_service import DocumentService
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService
from ..services.conversation_service import ConversationService
from .dependencies import (
    get_speech_service,
    get_document_service,
    get_rag_service,
    get_llm_service,
    get_conversation_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ══════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    speech_service: SpeechService = Depends(get_speech_service),
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    conv_service: ConversationService = Depends(get_conversation_service),
):
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "speech": speech_service.is_healthy(),
            "document": document_service.is_healthy(),
            "rag": rag_service.is_healthy(),
            "llm": llm_service.is_healthy(),
            "conversation": conv_service.is_healthy(),
            "document_count": rag_service.get_document_count(),
        }
    )


# ══════════════════════════════════════════════════════════
# Documents
# ══════════════════════════════════════════════════════════

@router.post("/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        document_id, chunks = await document_service.process_document(file)
        chunk_count = await rag_service.add_documents(chunks)

        return DocumentResponse(
            id=document_id,
            filename=file.filename,
            file_type=file.filename.split('.')[-1],
            chunk_count=chunk_count,
            message=f"Belge başarıyla yüklendi. {chunk_count} parça oluşturuldu."
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
):
    documents = document_service.get_documents()
    items = [
        DocumentListItem(
            id=doc_id,
            filename=meta.filename,
            file_type=meta.file_type.value,
            file_size=meta.file_size,
            chunk_count=meta.chunk_count,
            uploaded_at=meta.uploaded_at.isoformat()
        )
        for doc_id, meta in documents.items()
    ]
    return DocumentListResponse(documents=items, total_count=len(items))


@router.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        await rag_service.delete_document(document_id)
        deleted = await document_service.delete_document(document_id)
        if deleted:
            return {"message": "Belge silindi", "success": True}
        raise HTTPException(status_code=404, detail="Belge bulunamadı")
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/documents", tags=["Documents"])
async def clear_all_documents(
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
):
    try:
        count = await rag_service.clear_all()
        for doc_id in list(document_service.get_documents().keys()):
            await document_service.delete_document(doc_id)
        return {"message": f"{count} parça silindi", "success": True}
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ══════════════════════════════════════════════════════════
# RAG Query (stateless)
# ══════════════════════════════════════════════════════════

@router.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    t0 = time.time()

    try:
        t_ret = time.time()
        context, sources, total_retrieved = await rag_service.get_context(
            request.query, request.top_k
        )
        retrieval_ms = (time.time() - t_ret) * 1000

        t_llm = time.time()
        answer = await llm_service.generate_response(request.query, context)
        llm_ms = (time.time() - t_llm) * 1000

        total_ms = (time.time() - t0) * 1000

        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources if request.include_sources else [],
            processing_time_ms=round(total_ms, 2),
            metrics=PipelineMetrics(
                total_ms=round(total_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                llm_ms=round(llm_ms, 2),
                docs_retrieved=total_retrieved,
                docs_after_threshold=len(sources),
            )
        )

    except NoDocumentsError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ══════════════════════════════════════════════════════════
# Text Query (stateless, opsiyonel ses)
# ══════════════════════════════════════════════════════════

@router.post("/text/query", response_model=TextQueryResponse, tags=["Query"])
async def text_query(
    request: TextQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    speech_service: SpeechService = Depends(get_speech_service),
):
    t0 = time.time()

    try:
        t_ret = time.time()
        context, sources, total_retrieved = await rag_service.get_context(request.query)
        retrieval_ms = (time.time() - t_ret) * 1000

        t_llm = time.time()
        answer = await llm_service.generate_response(request.query, context)
        llm_ms = (time.time() - t_llm) * 1000

        audio_base64 = None
        tts_ms = None
        if request.include_audio:
            t_tts = time.time()
            audio_base64 = await speech_service.synthesize_to_base64(answer)
            tts_ms = round((time.time() - t_tts) * 1000, 2)

        total_ms = (time.time() - t0) * 1000

        return TextQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            audio_base64=audio_base64,
            processing_time_ms=round(total_ms, 2),
            metrics=PipelineMetrics(
                total_ms=round(total_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                llm_ms=round(llm_ms, 2),
                tts_ms=tts_ms,
                docs_retrieved=total_retrieved,
                docs_after_threshold=len(sources),
            )
        )

    except NoDocumentsError:
        answer = "Henüz belge yüklenmemiş. Lütfen önce bir belge yükleyin."
        audio_base64 = None
        if request.include_audio:
            audio_base64 = await speech_service.synthesize_to_base64(answer)
        return TextQueryResponse(
            query=request.query,
            answer=answer,
            sources=[],
            audio_base64=audio_base64,
            processing_time_ms=round((time.time() - t0) * 1000, 2),
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ══════════════════════════════════════════════════════════
# Chat Query (konuşma hafızalı, sesli/sessiz)
# ══════════════════════════════════════════════════════════

@router.post("/chat/query", response_model=ChatQueryResponse, tags=["Chat"])
async def chat_query(
    request: ChatQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    speech_service: SpeechService = Depends(get_speech_service),
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """
    Konuşma hafızalı sorgu endpoint'i.
    - session_id ile oturum sürekliliği sağlar
    - RAG veya serbest mod destekler
    - Query rewriting (ENABLE_QUERY_REWRITING=true ile)
    """
    t0 = time.time()
    settings = get_settings()

    # Session yönet
    session_id = conv_service.get_or_create_session(request.session_id)
    history_text = conv_service.get_history_as_text(session_id, max_turns=5)

    rewritten_query = None
    rewrite_ms = None
    context = ""
    sources: List[SourceDocument] = []
    total_retrieved = 0

    try:
        # ── Query Rewriting ──────────────────────────
        search_query = request.query
        if settings.enable_query_rewriting and request.mode == "rag":
            t_rw = time.time()
            search_query = await llm_service.rewrite_query(request.query)
            rewrite_ms = round((time.time() - t_rw) * 1000, 2)
            if search_query != request.query:
                rewritten_query = search_query

        # ── Retrieval ────────────────────────────────
        retrieval_ms = None
        if request.mode == "rag":
            try:
                t_ret = time.time()
                context, sources, total_retrieved = await rag_service.get_context(search_query)
                retrieval_ms = round((time.time() - t_ret) * 1000, 2)
            except NoDocumentsError:
                context = ""
                sources = []

        # ── LLM ──────────────────────────────────────
        t_llm = time.time()
        if request.mode == "free":
            answer = await llm_service.generate_free_response(
                request.query, conversation_history=history_text
            )
        else:
            answer = await llm_service.generate_response(
                request.query,
                context,
                conversation_history=history_text
            )
        llm_ms = round((time.time() - t_llm) * 1000, 2)

        # ── TTS ──────────────────────────────────────
        audio_base64 = None
        tts_ms = None
        if request.include_audio:
            t_tts = time.time()
            audio_base64 = await speech_service.synthesize_to_base64(answer)
            tts_ms = round((time.time() - t_tts) * 1000, 2)

        # ── Konuşma geçmişine kaydet ─────────────────
        conv_service.add_user_message(session_id, request.query)
        conv_service.add_assistant_message(session_id, answer)
        turn_count = conv_service.get_turn_count(session_id)

        total_ms = round((time.time() - t0) * 1000, 2)

        return ChatQueryResponse(
            query=request.query,
            answer=answer,
            session_id=session_id,
            sources=sources,
            audio_base64=audio_base64,
            processing_time_ms=total_ms,
            conversation_turn=turn_count,
            rewritten_query=rewritten_query,
            metrics=PipelineMetrics(
                total_ms=total_ms,
                retrieval_ms=retrieval_ms,
                rewrite_ms=rewrite_ms,
                llm_ms=llm_ms,
                tts_ms=tts_ms,
                docs_retrieved=total_retrieved,
                docs_after_threshold=len(sources),
            )
        )

    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Chat query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions/{session_id}", response_model=SessionHistoryResponse, tags=["Chat"])
async def get_session(
    session_id: str,
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """Session konuşma geçmişini getir"""
    messages = conv_service.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[
            ConversationMessage(role=ConversationRole(m.role), content=m.content)
            for m in messages
        ],
        turn_count=conv_service.get_turn_count(session_id),
    )


@router.delete("/chat/sessions/{session_id}", tags=["Chat"])
async def delete_session(
    session_id: str,
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """Session geçmişini sil"""
    deleted = conv_service.delete_session(session_id)
    return {"success": deleted, "message": "Session silindi" if deleted else "Session bulunamadı"}


# ══════════════════════════════════════════════════════════
# Voice Query (REST)
# ══════════════════════════════════════════════════════════

@router.post("/voice/query", response_model=VoiceQueryResponse, tags=["Voice"])
async def voice_query(
    audio: UploadFile = File(...),
    speech_service: SpeechService = Depends(get_speech_service),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Sesli sorgu — WAV → STT → RAG → LLM → TTS"""
    t0 = time.time()

    try:
        audio_bytes = await audio.read()

        t_stt = time.time()
        transcribed_text = await speech_service.transcribe_audio(audio_bytes)
        stt_ms = round((time.time() - t_stt) * 1000, 2)

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Ses tanınamadı")

        context = ""
        sources = []
        total_retrieved = 0
        try:
            t_ret = time.time()
            context, sources, total_retrieved = await rag_service.get_context(transcribed_text)
            retrieval_ms = round((time.time() - t_ret) * 1000, 2)
        except NoDocumentsError:
            retrieval_ms = None

        t_llm = time.time()
        answer = await llm_service.generate_response(transcribed_text, context)
        llm_ms = round((time.time() - t_llm) * 1000, 2)

        t_tts = time.time()
        audio_base64 = await speech_service.synthesize_to_base64(answer)
        tts_ms = round((time.time() - t_tts) * 1000, 2)

        total_ms = round((time.time() - t0) * 1000, 2)

        return VoiceQueryResponse(
            transcribed_text=transcribed_text,
            answer=answer,
            sources=sources,
            audio_base64=audio_base64,
            processing_time_ms=total_ms,
            metrics=PipelineMetrics(
                total_ms=total_ms,
                stt_ms=stt_ms,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                tts_ms=tts_ms,
                docs_retrieved=total_retrieved,
                docs_after_threshold=len(sources),
            )
        )

    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/voices", tags=["Voice"])
async def get_available_voices(
    speech_service: SpeechService = Depends(get_speech_service),
):
    return {
        "voices": speech_service.get_available_voices(),
        "current": get_settings().piper_model_path
    }
