"""
Speech Service - Local (Faster Whisper STT + Piper TTS / Edge TTS)
Used by REST API routes (/voice/query, /text/query, /health)
"""

import io
import wave
import base64
import asyncio
import logging
from collections import OrderedDict

# Module-level TTS audio cache (key: "backend|text" → audio bytes).
# Avoids re-synthesizing identical strings (error messages, greetings, etc.)
_TTS_CACHE: OrderedDict[str, bytes] = OrderedDict()
_TTS_CACHE_MAX = 128

from ..config import get_settings
from ..models.exceptions import TranscriptionError, SynthesisError
from .realtime_service import get_whisper_model, get_piper_voice

logger = logging.getLogger(__name__)


class SpeechService:
    def __init__(self):
        self.settings = get_settings()

    @property
    def tts_format(self) -> str:
        """Audio format produced by the active TTS backend ('mp3' or 'wav')."""
        return "mp3" if self.settings.tts_backend == "edge_tts" else "wav"

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        try:
            model = await get_whisper_model(self.settings)

            def _run():
                audio_io = io.BytesIO(audio_bytes)
                segments, _ = model.transcribe(
                    audio_io,
                    language=self.settings.speech_language,
                    beam_size=self.settings.whisper_beam_size,
                    vad_filter=self.settings.whisper_vad_filter,
                    condition_on_previous_text=self.settings.whisper_condition_on_previous,
                )
                return "".join(s.text for s in segments).strip()

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, _run)
            logger.info(f"Transcribed: {text[:80]}")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise TranscriptionError(str(e))

    async def synthesize_speech(self, text: str) -> bytes:
        cache_key = f"{self.settings.tts_backend}|{text}"
        if cache_key in _TTS_CACHE:
            logger.debug(f"TTS cache HIT: {len(text)} chars")
            _TTS_CACHE.move_to_end(cache_key)
            return _TTS_CACHE[cache_key]

        if self.settings.tts_backend == "edge_tts":
            audio = await self._synthesize_edge_tts(text)
        else:
            audio = await self._synthesize_piper(text)

        if len(_TTS_CACHE) >= _TTS_CACHE_MAX:
            _TTS_CACHE.popitem(last=False)
        _TTS_CACHE[cache_key] = audio
        return audio

    async def _synthesize_edge_tts(self, text: str) -> bytes:
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self.settings.edge_tts_voice)
            parts = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    parts.append(chunk["data"])
            audio_bytes = b"".join(parts)
            logger.info(f"Edge TTS synthesized: {len(audio_bytes)} bytes (mp3)")
            return audio_bytes
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            raise SynthesisError(str(e))

    async def _synthesize_piper(self, text: str) -> bytes:
        try:
            voice = await get_piper_voice(self.settings)

            def _run():
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    voice.synthesize_wav(text, wf)
                return wav_io.getvalue()

            loop = asyncio.get_running_loop()
            audio_bytes = await loop.run_in_executor(None, _run)
            logger.info(f"Piper TTS synthesized: {len(audio_bytes)} bytes (wav)")
            return audio_bytes
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            raise SynthesisError(str(e))

    async def synthesize_to_base64(self, text: str) -> str:
        audio_bytes = await self.synthesize_speech(text)
        return base64.b64encode(audio_bytes).decode('utf-8')

    @staticmethod
    def combine_audio(parts: list[bytes], fmt: str) -> bytes:
        """Combine multiple audio byte chunks into a single file."""
        if not parts:
            return b""
        if fmt == "mp3":
            return b"".join(parts)
        # WAV: strip headers, concatenate PCM, rebuild one WAV container
        pcm_chunks = []
        params = None
        for wav_bytes in parts:
            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                    if params is None:
                        params = wf.getparams()
                    pcm_chunks.append(wf.readframes(wf.getnframes()))
            except Exception:
                pcm_chunks.append(wav_bytes)
        if params is None:
            return b"".join(pcm_chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(b"".join(pcm_chunks))
        return buf.getvalue()

    def get_available_voices(self) -> list:
        if self.settings.tts_backend == "edge_tts":
            return [
                {
                    "name": self.settings.edge_tts_voice,
                    "gender": "Female",
                    "description": f"Edge TTS — {self.settings.edge_tts_voice}",
                }
            ]
        return [
            {
                "name": "tr_TR-dfki-medium",
                "gender": "Neutral",
                "description": "Türkçe - DFKI Medium (Piper TTS)",
            }
        ]

    def is_healthy(self) -> bool:
        if self.settings.tts_backend == "edge_tts":
            try:
                import edge_tts  # noqa: F401
                return True
            except ImportError:
                return False
        import os
        return os.path.exists(self.settings.piper_model_path)
