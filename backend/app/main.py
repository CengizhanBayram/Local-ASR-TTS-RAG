"""
Voice AI RAG Application
FastAPI Entry Point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .api.routes import router
from .api.websocket_routes import router as ws_router
from .models.exceptions import VoiceAIException

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    Startup ve shutdown işlemleri
    """
    # Startup
    logger.info("=" * 50)
    logger.info("🚀 Voice AI RAG Application Starting...")
    logger.info("=" * 50)
    
    settings = get_settings()
    logger.info(f"App: {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    
    # Services lazy-loaded on first request
    logger.info("Services will be initialized on first request")
    
    yield
    
    # Shutdown
    logger.info("=" * 50)
    logger.info("👋 Voice AI RAG Application Shutting Down...")
    logger.info("=" * 50)


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
    
    # Root endpoint
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
        reload=settings.debug
    )
