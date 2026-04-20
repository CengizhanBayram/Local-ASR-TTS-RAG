"""
API Routes
Tüm API endpoint'leri
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
    SourceDocument,
    HealthResponse,
    ErrorResponse
)
from ..models.exceptions import VoiceAIException, NoDocumentsError
from ..services.speech_service import SpeechService
from ..services.document_service import DocumentService
from ..services.rag_service import RAGService
from ..services.llm_service import LLMService
from .dependencies import (
    get_speech_service,
    get_document_service,
    get_rag_service,
    get_llm_service
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Health Check ==============

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    speech_service: SpeechService = Depends(get_speech_service),
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Sistem sağlık kontrolü
    Tüm servislerin durumunu kontrol eder
    """
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "speech": speech_service.is_healthy(),
            "document": document_service.is_healthy(),
            "rag": rag_service.is_healthy(),
            "llm": llm_service.is_healthy(),
            "document_count": rag_service.get_document_count()
        }
    )


# ============== Document Endpoints ==============

@router.post("/documents/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Belge yükle
    PDF, DOCX, TXT ve MD dosyalarını destekler
    """
    try:
        # Belgeyi işle
        document_id, chunks = await document_service.process_document(file)
        
        # Vector DB'ye ekle
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
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Yüklenen belgeleri listele
    """
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
    
    return DocumentListResponse(
        documents=items,
        total_count=len(items)
    )


@router.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Belge sil
    """
    try:
        # Vector DB'den sil
        await rag_service.delete_document(document_id)
        
        # Dosyayı sil
        deleted = await document_service.delete_document(document_id)
        
        if deleted:
            return {"message": "Belge silindi", "success": True}
        else:
            raise HTTPException(status_code=404, detail="Belge bulunamadı")
            
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/documents", tags=["Documents"])
async def clear_all_documents(
    document_service: DocumentService = Depends(get_document_service),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Tüm belgeleri sil
    """
    try:
        # Vector DB'yi temizle
        count = await rag_service.clear_all()
        
        # Tüm dosyaları sil
        for doc_id in list(document_service.get_documents().keys()):
            await document_service.delete_document(doc_id)
        
        return {"message": f"{count} parça silindi", "success": True}
        
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ============== RAG Query Endpoints ==============

@router.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(
    request: RAGQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Text tabanlı RAG sorgusu
    Belgelerden cevap üretir
    """
    start_time = time.time()
    
    try:
        # Context al
        context, sources = await rag_service.get_context(request.query, request.top_k)
        
        # LLM ile cevap üret
        answer = await llm_service.generate_response(request.query, context)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources if request.include_sources else [],
            processing_time_ms=round(processing_time, 2)
        )
        
    except NoDocumentsError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/text/query", response_model=TextQueryResponse, tags=["Query"])
async def text_query(
    request: TextQueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
    speech_service: SpeechService = Depends(get_speech_service)
):
    """
    Text sorgusu - opsiyonel ses yanıtı ile
    """
    start_time = time.time()
    
    try:
        # Context al
        context, sources = await rag_service.get_context(request.query)
        
        # LLM ile cevap üret
        answer = await llm_service.generate_response(request.query, context)
        
        # Opsiyonel audio
        audio_base64 = None
        if request.include_audio:
            audio_base64 = await speech_service.synthesize_to_base64(answer)
        
        processing_time = (time.time() - start_time) * 1000
        
        return TextQueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            audio_base64=audio_base64,
            processing_time_ms=round(processing_time, 2)
        )
        
    except NoDocumentsError as e:
        # Belge yoksa bile cevap ver
        answer = "Henüz belge yüklenmemiş. Lütfen önce bir belge yükleyin."
        audio_base64 = None
        if request.include_audio:
            audio_base64 = await speech_service.synthesize_to_base64(answer)
        
        return TextQueryResponse(
            query=request.query,
            answer=answer,
            sources=[],
            audio_base64=audio_base64,
            processing_time_ms=round((time.time() - start_time) * 1000, 2)
        )
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# ============== Voice Endpoints ==============

@router.post("/voice/query", response_model=VoiceQueryResponse, tags=["Voice"])
async def voice_query(
    audio: UploadFile = File(...),
    speech_service: SpeechService = Depends(get_speech_service),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Sesli sorgu - Ses gönder, sesli cevap al
    WAV formatında ses dosyası bekler
    """
    start_time = time.time()
    
    try:
        # Ses dosyasını oku
        audio_bytes = await audio.read()
        
        # STT - Ses -> Metin
        transcribed_text = await speech_service.transcribe_audio(audio_bytes)
        
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Ses tanınamadı")
        
        # RAG Context
        try:
            context, sources = await rag_service.get_context(transcribed_text)
        except NoDocumentsError:
            context = ""
            sources = []
        
        # LLM Response
        answer = await llm_service.generate_response(transcribed_text, context)
        
        # TTS - Metin -> Ses
        audio_base64 = await speech_service.synthesize_to_base64(answer)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VoiceQueryResponse(
            transcribed_text=transcribed_text,
            answer=answer,
            sources=sources,
            audio_base64=audio_base64,
            processing_time_ms=round(processing_time, 2)
        )
        
    except VoiceAIException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/voices", tags=["Voice"])
async def get_available_voices(
    speech_service: SpeechService = Depends(get_speech_service)
):
    """
    Kullanılabilir Türkçe ses seçeneklerini listele
    """
    return {
        "voices": speech_service.get_available_voices(),
        "current": get_settings().piper_model_path
    }
