"""
Speech Service - Local (Faster Whisper STT + Piper TTS)
Used by REST API routes (/voice/query, /text/query, /health)
"""

import io
import wave
import struct
import base64
import asyncio
import logging

from ..config import get_settings
from ..models.exceptions import TranscriptionError, SynthesisError
from .realtime_service import get_whisper_model, get_piper_voice

logger = logging.getLogger(__name__)


class SpeechService:
    def __init__(self):
        self.settings = get_settings()

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        try:
            model = await get_whisper_model(self.settings)

            def _run():
                audio_io = io.BytesIO(audio_bytes)
                segments, _ = model.transcribe(audio_io, language=self.settings.speech_language)
                return "".join(s.text for s in segments).strip()

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, _run)
            logger.info(f"Transcribed: {text[:80]}")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise TranscriptionError(str(e))

    async def synthesize_speech(self, text: str) -> bytes:
        try:
            voice = await get_piper_voice(self.settings)

            def _run():
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    voice.synthesize_wav(text, wf)
                return wav_io.getvalue()

            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(None, _run)
            logger.info(f"Synthesized: {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise SynthesisError(str(e))

    async def synthesize_to_base64(self, text: str) -> str:
        audio_bytes = await self.synthesize_speech(text)
        return base64.b64encode(audio_bytes).decode('utf-8')

    def get_available_voices(self) -> list:
        return [
            {
                "name": "tr_TR-dfki-medium",
                "gender": "Neutral",
                "description": "Türkçe - DFKI Medium (Piper TTS)",
            }
        ]

    def is_healthy(self) -> bool:
        import os
        return os.path.exists(self.settings.piper_model_path)
