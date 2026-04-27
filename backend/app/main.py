"""
Voice AI RAG Application
FastAPI Entry Point
"""

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .api.routes import router
from .api.websocket_routes import router as ws_router
from .models.exceptions import VoiceAIException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def _warmup_whisper(settings) -> None:
    """Run a 0.5s silent WAV through Whisper to trigger CUDA JIT compilation."""
    import io, struct, wave
    from .services.realtime_service import get_whisper_model

    sample_rate = 16000
    n_samples = sample_rate // 2  # 0.5s of silence
    pcm = bytes(n_samples * 2)   # int16 zeros

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    wav_bytes = buf.getvalue()

    model = await get_whisper_model(settings)

    def _run():
        segs, _ = model.transcribe(io.BytesIO(wav_bytes), language=settings.speech_language)
        list(segs)  # exhaust generator

    loop = asyncio.get_event_loop()
    t = time.monotonic()
    await loop.run_in_executor(None, _run)
    logger.info(f"Whisper CUDA warm-up done in {time.monotonic()-t:.1f}s")


async def _preload_models(settings) -> None:
    from .services.realtime_service import get_whisper_model, get_piper_voice
    from .api.dependencies import get_rag_service, get_llm_service, get_reranker_service, get_conversation_service

    async def _load_whisper():
        t = time.monotonic()
        await get_whisper_model(settings)
        logger.info(f"Whisper ready in {time.monotonic()-t:.1f}s")

    async def _load_piper():
        t = time.monotonic()
        await get_piper_voice(settings)
        logger.info(f"Piper TTS ready in {time.monotonic()-t:.1f}s")

    def _load_rag():
        t = time.monotonic()
        get_rag_service()
        logger.info(f"RAG/embeddings ready in {time.monotonic()-t:.1f}s")

    def _load_llm():
        t = time.monotonic()
        get_llm_service()
        logger.info(f"LLM provider ready in {time.monotonic()-t:.1f}s")

    def _load_reranker():
        t = time.monotonic()
        get_reranker_service()
        logger.info(f"Reranker ready in {time.monotonic()-t:.1f}s")

    def _load_conv():
        get_conversation_service()

    loop = asyncio.get_event_loop()

    await asyncio.gather(
        _load_whisper(),
        _load_piper(),
        loop.run_in_executor(None, _load_rag),
        loop.run_in_executor(None, _load_llm),
        loop.run_in_executor(None, _load_reranker),
        loop.run_in_executor(None, _load_conv),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}  |  provider={settings.llm_provider}")
    logger.info(f"  whisper={settings.whisper_model_size} device={settings.whisper_device}/{settings.whisper_compute_type}")
    logger.info(f"  embedding_device={settings.embedding_device}  debug={settings.debug}")
    logger.info("=" * 60)

    # Expand the default thread pool so CPU-bound executor tasks don't queue behind each other
    loop = asyncio.get_event_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=32))

    # Cap PyTorch intra-op threads for sentence-transformers / reranker (avoid NUMA contention)
    try:
        import torch
        torch.set_num_threads(16)
        torch.set_num_interop_threads(8)
    except Exception:
        pass

    t_start = time.monotonic()
    try:
        await _preload_models(settings)
        logger.info(f"All models loaded in {time.monotonic()-t_start:.1f}s")
    except Exception as exc:
        logger.warning(f"Model preload partial failure: {exc}  (will retry on first request)")

    # Warm-up: run a silent dummy inference to trigger CUDA kernel JIT compilation.
    # Without this, the first real Whisper call on CUDA takes 10-30s → "stuck" bug.
    try:
        await _warmup_whisper(settings)
        logger.info(f"Startup complete in {time.monotonic()-t_start:.1f}s — ready to serve ✓")
    except Exception as exc:
        logger.warning(f"Whisper warm-up skipped: {exc}")

    yield

    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    """
    FastAPI application factory
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Voice AI RAG API

Azure servisleri ile gerçek zamanlı sesli soru-cevap sistemi.

### Özellikler:
- 🎤 **Sesli Soru-Cevap**: Ses kaydı gönder, sesli cevap al
- 📄 **Belge Yükleme**: PDF, DOCX, TXT, MD desteği
- 🔍 **RAG**: Belgelerden akıllı arama ve cevap üretme
- 🇹🇷 **Türkçe Destek**: Tam Türkçe dil desteği

### Kullanım:
1. `/api/documents/upload` ile belge yükleyin
2. `/api/voice/query` ile sesli soru sorun
3. Sesli cevabı dinleyin!
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handler
    @app.exception_handler(VoiceAIException)
    async def voice_ai_exception_handler(request: Request, exc: VoiceAIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "status_code": exc.status_code
            }
        )
    
    # Include routers
    app.include_router(router, prefix="/api")
    app.include_router(ws_router, prefix="/api")
    
    # Serve frontend static files
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.exists():
        # Mount static files (CSS, JS)
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")
        
        @app.get("/", tags=["Root"])
        async def root():
            return FileResponse(str(frontend_dir / "index.html"))
        
        @app.get("/styles.css")
        async def serve_css():
            return FileResponse(str(frontend_dir / "styles.css"), media_type="text/css")
        
        @app.get("/app.js")
        async def serve_js():
            return FileResponse(str(frontend_dir / "app.js"), media_type="application/javascript")
    else:
        @app.get("/", tags=["Root"])
        async def root():
            return {
                "message": "Voice AI RAG API",
                "version": settings.app_version,
                "docs": "/docs",
                "health": "/api/health"
            }
    
    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        loop="uvloop",
        http="httptools",
    )
