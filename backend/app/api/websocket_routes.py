"""
WebSocket Routes - Real-time Voice Streaming
Gerçek zamanlı ses akışı endpoint'leri
"""

import json
import logging
import base64
import asyncio
from typing import Optional, Dict
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.realtime_service import RealtimeVoicePipeline, RealtimeEvent
from ..api.dependencies import get_rag_service, get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """WebSocket bağlantı yöneticisi"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.pipelines: Dict[str, RealtimeVoicePipeline] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Yeni bağlantı kabul et"""
        try:
            await websocket.accept()
            self.connections[client_id] = websocket
            logger.info(f"Client connected: {client_id}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self, client_id: str):
        """Bağlantıyı kapat"""
        if client_id in self.connections:
            del self.connections[client_id]
        if client_id in self.pipelines:
            del self.pipelines[client_id]
        logger.info(f"Client disconnected: {client_id}")
    
    async def send_event(self, client_id: str, event: RealtimeEvent):
        """Event gönder"""
        if client_id in self.connections:
            try:
                await self.connections[client_id].send_text(event.to_json())
            except Exception as e:
                logger.error(f"Send error: {e}")
    
    def get_pipeline(self, client_id: str) -> Optional[RealtimeVoicePipeline]:
        """Pipeline al"""
        return self.pipelines.get(client_id)
    
    def set_pipeline(self, client_id: str, pipeline: RealtimeVoicePipeline):
        """Pipeline ayarla"""
        self.pipelines[client_id] = pipeline


manager = ConnectionManager()


@router.websocket("/ws/realtime/{client_id}")
async def realtime_voice_websocket(websocket: WebSocket, client_id: str):
    """
    Real-time Voice WebSocket Endpoint
    
    == Client -> Server Mesajları ==
    
    {"type": "start"}
        Dinlemeye başla
    
    {"type": "audio", "data": "<base64 PCM audio>"}
        Audio chunk gönder (16-bit PCM, 16kHz, mono)
    
    {"type": "stop"}
        Dinlemeyi durdur ve işlemeye başla
    
    {"type": "cancel"}
        İşlemi iptal et
    
    {"type": "ping"}
        Bağlantı kontrolü
    
    == Server -> Client Mesajları ==
    
    {"type": "connected", "data": {"client_id": "..."}}
        Bağlantı kuruldu
    
    {"type": "state", "data": {"state": "idle|listening|processing|speaking"}}
        Durum değişikliği
    
    {"type": "listening_started", "data": {}}
        Dinleme başladı
    
    {"type": "transcription", "data": {"text": "...", "is_final": bool}}
        Ses tanıma sonucu (partial veya final)
    
    {"type": "user_message", "data": {"text": "..."}}
        Kullanıcı mesajı (final transcription)
    
    {"type": "answer", "data": {"text": "...", "sources": [...]}}
        AI cevabı
    
    {"type": "audio_chunk", "data": {"data": "<base64>", "format": "mp3"}}
        TTS audio chunk
    
    {"type": "audio_complete", "data": {"full_audio": "<base64>"}}
        TTS tamamlandı (replay için tam audio)
    
    {"type": "error", "data": {"message": "..."}}
        Hata
    
    {"type": "pong", "data": {}}
        Ping yanıtı
    """
    
    # Bağlantıyı kabul et
    if not await manager.connect(websocket, client_id):
        return
    
    # Bağlantı event'i gönder
    await manager.send_event(client_id, RealtimeEvent("connected", {"client_id": client_id}))
    
    # Services
    rag_service = get_rag_service()
    llm_service = get_llm_service()
    
    try:
        while True:
            # Mesaj al
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")
            
            logger.debug(f"Received: {msg_type}")
            
            if msg_type == "ping":
                await manager.send_event(client_id, RealtimeEvent("pong"))
            
            elif msg_type == "start":
                # Get mode from message (rag or free)
                chat_mode = message.get("mode", "rag")
                
                # Yeni pipeline oluştur
                async def send_event_wrapper(event: RealtimeEvent):
                    await manager.send_event(client_id, event)
                
                # Sync wrapper for async send
                def sync_send(event: RealtimeEvent):
                    asyncio.create_task(send_event_wrapper(event))
                
                pipeline = RealtimeVoicePipeline(
                    rag_service=rag_service if chat_mode == "rag" else None,
                    llm_service=llm_service,
                    send_event=sync_send,
                    mode=chat_mode
                )
                manager.set_pipeline(client_id, pipeline)
                
                await pipeline.start_listening()
            
            elif msg_type == "audio":
                # Audio chunk işle
                pipeline = manager.get_pipeline(client_id)
                if pipeline:
                    audio_b64 = message.get("data", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        pipeline.push_audio(audio_bytes)
            
            elif msg_type == "stop":
                # Dinlemeyi durdur ve işle
                pipeline = manager.get_pipeline(client_id)
                if pipeline:
                    await pipeline.stop_listening()
            
            elif msg_type == "cancel":
                # İptal et
                pipeline = manager.get_pipeline(client_id)
                if pipeline:
                    await pipeline.cancel()
                manager.pipelines.pop(client_id, None)
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        await manager.send_event(client_id, RealtimeEvent("error", {"message": "Invalid JSON"}))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_event(client_id, RealtimeEvent("error", {"message": str(e)}))
    finally:
        # Cleanup
        pipeline = manager.get_pipeline(client_id)
        if pipeline:
            await pipeline.cancel()
        manager.disconnect(client_id)


@router.websocket("/ws/voice/{client_id}")
async def voice_websocket_legacy(websocket: WebSocket, client_id: str):
    """Legacy endpoint - redirects to realtime"""
    await realtime_voice_websocket(websocket, client_id)
