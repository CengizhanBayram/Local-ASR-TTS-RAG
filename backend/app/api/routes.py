"""
API Routes — REST + SSE endpoints
"""

import json
import time
import base64
import logging
from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

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
from ..services.reranker_service import RerankerService
from ..services.conversation_service import ConversationService
from .dependencies import (
    get_speech_service,
    get_document_service,
    get_rag_service,
    get_llm_service,
    get_reranker_service,
    get_conversation_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    speech_service:   SpeechService   = Depends(get_speech_service),
    document_service: DocumentService = Depends(get_document_service),
    rag_service:      RAGService      = Depends(get_rag_service),
    llm_service:      LLMService      = Depends(get_llm_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
    conv_service:     ConversationService = Depends(get_conversation_service),
):
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "speech":         speech_service.is_healthy(),
            "document":       document_service.is_healthy(),
            "rag":            rag_service.is_healthy(),
            "llm":            llm_service.is_healthy(),
            "reranker":       reranker_service.is_healthy(),
            "conversation":   conv_service.is_healthy(),
            "document_count": rag_service.get_document_count(),
        },
    )


# ── Documents ─────────────────────────────────────────────────────────────────

@router.post("/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    file:             UploadFile    = File(...),
    document_service: DocumentService = Depends(get_document_service),
    rag_service:      RAGService      = Depends(get_rag_service),
):
    try:
        document_id, child_chunks, parent_chunks = await document_service.process_document(file)
        chunk_count = await rag_service.add_documents(child_chunks, parent_chunks)
        return DocumentResponse(
            id=document_id,
            filename=file.filename,
            file_type=file.filename.split('.')[-1],
            chunk_count=chunk_count,
            message=f"Document uploaded. {chunk_count} chunks indexed.",
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(document_service: DocumentService = Depends(get_document_service)):
    documents = document_service.get_documents()
    items = [
        DocumentListItem(
            id=doc_id,
            filename=meta.filename,
            file_type=meta.file_type.value,
            file_size=meta.file_size,
            chunk_count=meta.chunk_count,
            uploaded_at=meta.uploaded_at.isoformat(),
        )
        for doc_id, meta in documents.items()
    ]
    return DocumentListResponse(documents=items, total_count=len(items))


@router.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id:      str,
    document_service: DocumentService = Depends(get_document_service),
    rag_service:      RAGService      = Depends(get_rag_service),
):
    try:
        await rag_service.delete_document(document_id)
        deleted = await document_service.delete_document(document_id)
        if deleted:
            return {"message": "Document deleted", "success": True}
        raise HTTPException(status_code=404, detail="Document not found")
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/documents", tags=["Documents"])
async def clear_all_documents(
    document_service: DocumentService = Depends(get_document_service),
    rag_service:      RAGService      = Depends(get_rag_service),
):
    try:
        count = await rag_service.clear_all()
        for doc_id in list(document_service.get_documents().keys()):
            await document_service.delete_document(doc_id)
        return {"message": f"{count} chunks deleted", "success": True}
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ── Internal helper: RAG pipeline ────────────────────────────────────────────

async def _run_rag_pipeline(
    query: str,
    rag_service: RAGService,
    llm_service: LLMService,
    reranker_service: RerankerService,
    settings,
    extra_queries: Optional[List[str]] = None,
) -> tuple:
    """Returns (context, sources, total_retrieved, retrieval_ms)"""
    t_ret = time.time()
    context, sources, total_retrieved = await rag_service.get_context(
        query, extra_queries=extra_queries
    )
    retrieval_ms = round((time.time() - t_ret) * 1000, 2)

    if settings.enable_reranking and sources:
        sources = await reranker_service.rerank(query, sources)
        # Rebuild context from reranked sources
        parts = []
        for i, s in enumerate(sources, 1):
            pct = int(s.score * 100)
            parts.append(f"[Source {i}: {s.filename} | Match: {pct}%]\n{s.content}")
        context = "\n\n---\n\n".join(parts)

    return context, sources, total_retrieved, retrieval_ms


# ── RAG Query (stateless) ─────────────────────────────────────────────────────

from typing import Optional

@router.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(
    request:          RAGQueryRequest,
    rag_service:      RAGService      = Depends(get_rag_service),
    llm_service:      LLMService      = Depends(get_llm_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
):
    t0 = time.time()
    settings = get_settings()
    try:
        context, sources, total_retrieved, retrieval_ms = await _run_rag_pipeline(
            request.query, rag_service, llm_service, reranker_service, settings
        )
        t_llm = time.time()
        answer = await llm_service.generate_response(request.query, context)
        llm_ms = round((time.time() - t_llm) * 1000, 2)
        total_ms = round((time.time() - t0) * 1000, 2)
        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources if request.include_sources else [],
            processing_time_ms=total_ms,
            metrics=PipelineMetrics(
                total_ms=total_ms, retrieval_ms=retrieval_ms, llm_ms=llm_ms,
                docs_retrieved=total_retrieved, docs_after_threshold=len(sources),
            ),
        )
    except NoDocumentsError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ── Text Query ────────────────────────────────────────────────────────────────

@router.post("/text/query", response_model=TextQueryResponse, tags=["Query"])
async def text_query(
    request:          TextQueryRequest,
    rag_service:      RAGService      = Depends(get_rag_service),
    llm_service:      LLMService      = Depends(get_llm_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
    speech_service:   SpeechService   = Depends(get_speech_service),
):
    t0 = time.time()
    settings = get_settings()
    try:
        context, sources, total_retrieved, retrieval_ms = await _run_rag_pipeline(
            request.query, rag_service, llm_service, reranker_service, settings
        )
        t_llm = time.time()
        answer = await llm_service.generate_response(request.query, context)
        llm_ms = round((time.time() - t_llm) * 1000, 2)

        audio_base64 = tts_ms = None
        if request.include_audio:
            t_tts = time.time()
            audio_base64 = await speech_service.synthesize_to_base64(answer)
            tts_ms = round((time.time() - t_tts) * 1000, 2)

        total_ms = round((time.time() - t0) * 1000, 2)
        return TextQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            audio_base64=audio_base64,
            processing_time_ms=total_ms,
            metrics=PipelineMetrics(
                total_ms=total_ms, retrieval_ms=retrieval_ms, llm_ms=llm_ms, tts_ms=tts_ms,
                docs_retrieved=total_retrieved, docs_after_threshold=len(sources),
            ),
        )
    except NoDocumentsError:
        answer = "No documents loaded yet. Please upload a document first."
        audio_base64 = None
        if request.include_audio:
            audio_base64 = await speech_service.synthesize_to_base64(answer)
        return TextQueryResponse(
            query=request.query, answer=answer, sources=[], audio_base64=audio_base64,
            processing_time_ms=round((time.time() - t0) * 1000, 2),
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ── Chat Query (session-aware, blocking) ─────────────────────────────────────

@router.post("/chat/query", response_model=ChatQueryResponse, tags=["Chat"])
async def chat_query(
    request:          ChatQueryRequest,
    rag_service:      RAGService      = Depends(get_rag_service),
    llm_service:      LLMService      = Depends(get_llm_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
    speech_service:   SpeechService   = Depends(get_speech_service),
    conv_service:     ConversationService = Depends(get_conversation_service),
):
    t0 = time.time()
    settings = get_settings()
    session_id   = conv_service.get_or_create_session(request.session_id)
    history_text = conv_service.get_history_as_text(session_id, max_turns=5)

    rewritten_query = rewrite_ms = None
    context = ""
    sources: List[SourceDocument] = []
    total_retrieved = retrieval_ms = 0

    try:
        search_query = request.query
        if settings.enable_query_rewriting and request.mode == "rag":
            t_rw = time.time()
            search_query = await llm_service.rewrite_query(request.query)
            rewrite_ms = round((time.time() - t_rw) * 1000, 2)
            if search_query != request.query:
                rewritten_query = search_query

        if request.mode == "rag":
            try:
                extra_queries = None
                if settings.enable_multi_query:
                    extra_queries = await llm_service.generate_query_variations(
                        search_query, settings.multi_query_count
                    )
                context, sources, total_retrieved, retrieval_ms = await _run_rag_pipeline(
                    search_query, rag_service, llm_service, reranker_service, settings,
                    extra_queries=extra_queries,
                )
            except NoDocumentsError:
                context = ""

        t_llm = time.time()
        if request.mode == "free":
            answer = await llm_service.generate_free_response(request.query, conversation_history=history_text)
        else:
            answer = await llm_service.generate_response(request.query, context, conversation_history=history_text)
        llm_ms = round((time.time() - t_llm) * 1000, 2)

        audio_base64 = tts_ms = None
        if request.include_audio:
            t_tts = time.time()
            audio_base64 = await speech_service.synthesize_to_base64(answer)
            tts_ms = round((time.time() - t_tts) * 1000, 2)

        conv_service.add_user_message(session_id, request.query)
        conv_service.add_assistant_message(session_id, answer)
        turn_count = conv_service.get_turn_count(session_id)
        total_ms   = round((time.time() - t0) * 1000, 2)

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
                total_ms=total_ms, retrieval_ms=retrieval_ms, rewrite_ms=rewrite_ms,
                llm_ms=llm_ms, tts_ms=tts_ms,
                docs_retrieved=total_retrieved, docs_after_threshold=len(sources),
            ),
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"chat_query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat Stream (SSE, session-aware) ──────────────────────────────────────────

@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request:          ChatQueryRequest,
    rag_service:      RAGService      = Depends(get_rag_service),
    llm_service:      LLMService      = Depends(get_llm_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
    conv_service:     ConversationService = Depends(get_conversation_service),
):
    """
    Server-Sent Events streaming endpoint.
    Events: rewrite | sources | token | done | error
    """
    settings     = get_settings()
    session_id   = conv_service.get_or_create_session(request.session_id)
    history_text = conv_service.get_history_as_text(session_id, max_turns=5)

    async def event_generator():
        t0 = time.time()
        rewritten_query = None
        context = ""
        sources: List[SourceDocument] = []
        total_retrieved = 0

        try:
            # Query rewrite
            search_query = request.query
            if settings.enable_query_rewriting and request.mode == "rag":
                search_query = await llm_service.rewrite_query(request.query)
                if search_query != request.query:
                    rewritten_query = search_query
                    yield f"data: {json.dumps({'type': 'rewrite', 'query': rewritten_query})}\n\n"

            # Retrieval
            if request.mode == "rag":
                try:
                    extra_queries = None
                    if settings.enable_multi_query:
                        extra_queries = await llm_service.generate_query_variations(
                            search_query, settings.multi_query_count
                        )
                    context, sources, total_retrieved, _ = await _run_rag_pipeline(
                        search_query, rag_service, llm_service, reranker_service, settings,
                        extra_queries=extra_queries,
                    )
                    yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"
                except NoDocumentsError:
                    pass

            # Stream LLM tokens
            full_answer = ""
            if request.mode == "free":
                gen = llm_service.generate_free_stream(request.query, conversation_history=history_text)
            else:
                gen = llm_service.generate_response_stream(
                    request.query, context, conversation_history=history_text
                )

            async for chunk in gen:
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"

            # Save to conversation
            conv_service.add_user_message(session_id, request.query)
            conv_service.add_assistant_message(session_id, full_answer)
            turn_count = conv_service.get_turn_count(session_id)
            total_ms   = round((time.time() - t0) * 1000, 2)

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'turn': turn_count, 'rewritten_query': rewritten_query, 'metrics': {'total_ms': total_ms, 'docs_retrieved': total_retrieved, 'docs_after_threshold': len(sources)}})}\n\n"

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session management ────────────────────────────────────────────────────────

@router.get("/chat/sessions/{session_id}", response_model=SessionHistoryResponse, tags=["Chat"])
async def get_session(
    session_id:   str,
    conv_service: ConversationService = Depends(get_conversation_service),
):
    messages = conv_service.get_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[ConversationMessage(role=ConversationRole(m.role), content=m.content) for m in messages],
        turn_count=conv_service.get_turn_count(session_id),
    )


@router.delete("/chat/sessions/{session_id}", tags=["Chat"])
async def delete_session(
    session_id:   str,
    conv_service: ConversationService = Depends(get_conversation_service),
):
    deleted = conv_service.delete_session(session_id)
    return {"success": deleted, "message": "Session deleted" if deleted else "Session not found"}


# ── Voice Query ───────────────────────────────────────────────────────────────

@router.post("/voice/query", response_model=VoiceQueryResponse, tags=["Voice"])
async def voice_query(
    audio:            UploadFile  = File(...),
    speech_service:   SpeechService   = Depends(get_speech_service),
    rag_service:      RAGService      = Depends(get_rag_service),
    llm_service:      LLMService      = Depends(get_llm_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
):
    t0 = time.time()
    settings = get_settings()
    try:
        audio_bytes = await audio.read()

        t_stt = time.time()
        transcribed_text = await speech_service.transcribe_audio(audio_bytes)
        stt_ms = round((time.time() - t_stt) * 1000, 2)

        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Speech not recognized")

        context = ""
        sources = []
        total_retrieved = retrieval_ms = 0
        try:
            context, sources, total_retrieved, retrieval_ms = await _run_rag_pipeline(
                transcribed_text, rag_service, llm_service, reranker_service, settings
            )
        except NoDocumentsError:
            pass

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
                total_ms=total_ms, stt_ms=stt_ms, retrieval_ms=retrieval_ms,
                llm_ms=llm_ms, tts_ms=tts_ms,
                docs_retrieved=total_retrieved, docs_after_threshold=len(sources),
            ),
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"voice_query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/voices", tags=["Voice"])
async def get_available_voices(speech_service: SpeechService = Depends(get_speech_service)):
    return {"voices": speech_service.get_available_voices(), "current": get_settings().piper_model_path}
