"""
WebSocket Routes - Gerçek zamanlı ses akışı
Session ID ile konuşma hafızası desteği
"""

import json
import logging
import base64
import asyncio
from typing import Optional, Dict
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.realtime_service import RealtimeVoicePipeline, RealtimeEvent
from ..api.dependencies import get_rag_service, get_llm_service, get_conversation_service

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.pipelines: Dict[str, RealtimeVoicePipeline] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        try:
            await websocket.accept()
            self.connections[client_id] = websocket
            logger.info(f"Client connected: {client_id}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self, client_id: str):
        self.connections.pop(client_id, None)
        self.pipelines.pop(client_id, None)
        logger.info(f"Client disconnected: {client_id}")

    async def send_event(self, client_id: str, event: RealtimeEvent):
        ws = self.connections.get(client_id)
        if ws:
            try:
                await ws.send_text(event.to_json())
            except Exception as e:
                logger.error(f"Send error: {e}")

    def get_pipeline(self, client_id: str) -> Optional[RealtimeVoicePipeline]:
        return self.pipelines.get(client_id)

    def set_pipeline(self, client_id: str, pipeline: RealtimeVoicePipeline):
        self.pipelines[client_id] = pipeline


manager = ConnectionManager()


@router.websocket("/ws/realtime/{client_id}")
async def realtime_voice_websocket(websocket: WebSocket, client_id: str):
    """
    Gerçek zamanlı ses pipeline WebSocket endpoint'i.

    Client → Server mesaj tipleri:
      {"type": "start", "mode": "rag"|"free", "session_id": "..."}
      {"type": "audio", "data": "<base64 PCM 16kHz mono>"}
      {"type": "stop"}
      {"type": "cancel"}
      {"type": "ping"}

    Server → Client mesaj tipleri:
      connected / state / listening_started / transcription /
      user_message / answer / audio_chunk / audio_complete / error / pong / canceled
    """
    if not await manager.connect(websocket, client_id):
        return

    from ..config import get_settings as _gs
    _s = _gs()
    await manager.send_event(client_id, RealtimeEvent("connected", {
        "client_id": client_id,
        "vad_silence_threshold":   _s.vad_silence_threshold,
        "vad_silence_duration_ms": _s.vad_silence_duration_ms,
    }))

    rag_service = get_rag_service()
    llm_service = get_llm_service()
    conv_service = get_conversation_service()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            logger.debug(f"WS [{client_id}] recv: {msg_type}")

            if msg_type == "ping":
                await manager.send_event(client_id, RealtimeEvent("pong"))

            elif msg_type == "start":
                chat_mode = message.get("mode", "rag")
                session_id = message.get("session_id") or str(uuid4())

                # Session kaydı
                session_id = conv_service.get_or_create_session(session_id)

                async def send_event_wrapper(event: RealtimeEvent):
                    await manager.send_event(client_id, event)

                def sync_send(event: RealtimeEvent):
                    asyncio.create_task(send_event_wrapper(event))

                pipeline = RealtimeVoicePipeline(
                    rag_service=rag_service if chat_mode == "rag" else None,
                    llm_service=llm_service,
                    conv_service=conv_service,
                    session_id=session_id,
                    send_event=sync_send,
                    mode=chat_mode,
                )
                manager.set_pipeline(client_id, pipeline)
                await pipeline.start_listening()

                # Client'a session_id gönder
                await manager.send_event(
                    client_id,
                    RealtimeEvent("session", {"session_id": session_id})
                )

            elif msg_type == "audio":
                pipeline = manager.get_pipeline(client_id)
                if pipeline:
                    audio_b64 = message.get("data", "")
                    if audio_b64:
                        pipeline.push_audio(base64.b64decode(audio_b64))

            elif msg_type == "stop":
                pipeline = manager.get_pipeline(client_id)
                if pipeline:
                    await pipeline.stop_listening()

            elif msg_type == "cancel":
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
        pipeline = manager.get_pipeline(client_id)
        if pipeline:
            await pipeline.cancel()
        manager.disconnect(client_id)


@router.websocket("/ws/voice/{client_id}")
async def voice_websocket_legacy(websocket: WebSocket, client_id: str):
    """Legacy endpoint - realtime'a yönlendirir"""
    await realtime_voice_websocket(websocket, client_id)
