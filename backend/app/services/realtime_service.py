"""
Real-time Voice Streaming Service - Faster Whisper STT + Piper TTS
Conversation history destekli tam pipeline
"""

import asyncio
import io
import json
import logging
import base64
import struct
import wave
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from ..config import get_settings
from ..models.exceptions import SpeechServiceError

logger = logging.getLogger(__name__)

# Singleton model yükleme
_whisper_model = None
_piper_voice = None
_model_lock = asyncio.Lock()


def _load_whisper_sync(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    logger.info(f"Loading Faster Whisper: {model_size} ({device}/{compute_type})")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _load_piper_sync(model_path: str):
    from piper import PiperVoice
    logger.info(f"Loading Piper TTS: {model_path}")
    return PiperVoice.load(model_path)


async def get_whisper_model(settings):
    global _whisper_model
    if _whisper_model is None:
        async with _model_lock:
            if _whisper_model is None:
                loop = asyncio.get_event_loop()
                _whisper_model = await loop.run_in_executor(
                    None,
                    _load_whisper_sync,
                    settings.whisper_model_size,
                    settings.whisper_device,
                    settings.whisper_compute_type,
                )
    return _whisper_model


async def get_piper_voice(settings):
    global _piper_voice
    if _piper_voice is None:
        async with _model_lock:
            if _piper_voice is None:
                loop = asyncio.get_event_loop()
                _piper_voice = await loop.run_in_executor(
                    None, _load_piper_sync, settings.piper_model_path
                )
    return _piper_voice


class StreamState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class RealtimeEvent:
    type: str
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "data": self.data}, ensure_ascii=False)


class AudioBuffer:
    def __init__(self, max_size: int = 1024 * 1024):
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def write(self, data: bytes) -> None:
        async with self.lock:
            self.buffer.extend(data)

    async def read_all(self) -> bytes:
        async with self.lock:
            result = bytes(self.buffer)
            self.buffer.clear()
            return result

    def __len__(self) -> int:
        return len(self.buffer)


class RealtimeTranscriber:
    def __init__(self, settings):
        self.settings = settings
        self.audio_buffer = AudioBuffer()
        self.is_running = False
        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.final_text = ""

    def start(self) -> None:
        self.audio_buffer = AudioBuffer()
        self.is_running = True
        self.final_text = ""
        logger.info("Realtime transcription started")

    def push_audio(self, audio_data: bytes) -> None:
        if self.is_running:
            asyncio.create_task(self.audio_buffer.write(audio_data))

    async def stop(self) -> str:
        if not self.is_running:
            return self.final_text

        self.is_running = False
        audio_data = await self.audio_buffer.read_all()

        if not audio_data or len(audio_data) < 8000:
            logger.warning(f"Not enough audio: {len(audio_data) if audio_data else 0} bytes")
            return ""

        try:
            text = await self._transcribe(audio_data)
            self.final_text = text
            if self.on_final and text:
                self.on_final(text)
            logger.info(f"Transcription: {self.final_text}")
            return self.final_text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            if self.on_error:
                self.on_error(str(e))
            return ""

    async def _transcribe(self, audio_data: bytes) -> str:
        model = await get_whisper_model(self.settings)
        wav_data = self._create_wav(audio_data)

        def _run():
            audio_io = io.BytesIO(wav_data)
            segments, _ = model.transcribe(audio_io, language=self.settings.speech_language)
            return "".join(s.text for s in segments).strip()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    def _create_wav(self, pcm_data: bytes) -> bytes:
        sample_rate, bits, channels = 16000, 16, 1
        data_size = len(pcm_data)
        header = bytearray()
        header.extend(b'RIFF')
        header.extend(struct.pack('<I', data_size + 36))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend(struct.pack('<I', 16))
        header.extend(struct.pack('<H', 1))
        header.extend(struct.pack('<H', channels))
        header.extend(struct.pack('<I', sample_rate))
        header.extend(struct.pack('<I', sample_rate * channels * bits // 8))
        header.extend(struct.pack('<H', channels * bits // 8))
        header.extend(struct.pack('<H', bits))
        header.extend(b'data')
        header.extend(struct.pack('<I', data_size))
        return bytes(header) + pcm_data

    async def close(self):
        pass


class RealtimeSynthesizer:
    def __init__(self, settings):
        self.settings = settings

    async def synthesize_streaming(
        self,
        text: str,
        on_chunk: Callable[[bytes], None],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        try:
            sentences = self._split_sentences(text)
            if not sentences:
                if on_complete:
                    on_complete()
                return

            logger.info(f"TTS: synthesizing {len(sentences)} sentences")
            for sentence in sentences:
                if not sentence.strip():
                    continue
                audio_data = await self._synthesize(sentence)
                chunk_size = 8192
                for i in range(0, len(audio_data), chunk_size):
                    on_chunk(audio_data[i:i + chunk_size])

            logger.info("TTS completed")
            if on_complete:
                on_complete()

        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise SpeechServiceError(f"TTS failed: {str(e)}")

    async def synthesize_full(self, text: str) -> bytes:
        return await self._synthesize(text)

    async def _synthesize(self, text: str) -> bytes:
        voice = await get_piper_voice(self.settings)

        def _run():
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                voice.synthesize(text, wf)
            return wav_io.getvalue()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    def _split_sentences(self, text: str) -> list:
        import re
        parts = re.split(r'([.!?]+)', text)
        sentences = []
        current = ""
        for part in parts:
            current += part
            if re.search(r'[.!?]', part):
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        return sentences if sentences else [text.strip()]

    async def close(self):
        pass


class RealtimeVoicePipeline:
    """
    Tam ses pipeline: STT (Whisper) → RAG → LLM (Gemini) → TTS (Piper)
    Konuşma hafızası destekli.
    """

    def __init__(
        self,
        rag_service,
        llm_service,
        send_event: Callable[[RealtimeEvent], None],
        mode: str = "rag",
        conv_service=None,
        session_id: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.conv_service = conv_service
        self.session_id = session_id
        self.send_event = send_event
        self.mode = mode
        self.transcriber: Optional[RealtimeTranscriber] = None
        self.synthesizer = RealtimeSynthesizer(self.settings)
        self.state = StreamState.IDLE

    def _emit(self, event_type: str, data: dict = None):
        self.send_event(RealtimeEvent(event_type, data or {}))

    def _set_state(self, new_state: StreamState):
        self.state = new_state
        self._emit("state", {"state": new_state.value})

    async def start_listening(self) -> None:
        self._set_state(StreamState.LISTENING)
        self.transcriber = RealtimeTranscriber(self.settings)
        self.transcriber.on_partial = lambda text: self._emit(
            "transcription", {"text": text, "is_final": False}
        )
        self.transcriber.on_final = lambda text: self._emit(
            "transcription", {"text": text, "is_final": True}
        )
        self.transcriber.on_error = lambda err: self._emit("error", {"message": err})
        self.transcriber.start()
        self._emit("listening_started", {})

    def push_audio(self, audio_data: bytes) -> None:
        if self.transcriber and self.transcriber.is_running:
            self.transcriber.push_audio(audio_data)

    async def stop_listening(self) -> None:
        if not self.transcriber:
            return

        final_text = await self.transcriber.stop()
        if not final_text:
            logger.info("Audio too short or empty, ignoring.")
            self._set_state(StreamState.IDLE)
            return

        self._emit("user_message", {"text": final_text})
        await self._process_query(final_text)

    async def _process_query(self, query: str) -> None:
        import re as _re
        self._set_state(StreamState.PROCESSING)
        try:
            history_text = ""
            if self.conv_service and self.session_id:
                history_text = self.conv_service.get_history_as_text(self.session_id, max_turns=5)

            context = ""
            sources = []
            if self.mode == "rag" and self.rag_service:
                try:
                    context, source_docs, _ = await self.rag_service.get_context(query)
                    sources = [{"filename": s.filename, "score": s.score} for s in source_docs]
                except Exception as e:
                    logger.warning(f"RAG skipped: {e}")

            self._set_state(StreamState.SPEAKING)

            # ── Parallel LLM→TTS ─────────────────────────────────────────────
            # Producer puts complete sentences into sentence_q.
            # Consumer TTS-encodes each sentence and puts audio into audio_q in order.
            # Main loop drains audio_q and emits chunks as they arrive.

            sentence_q: asyncio.Queue = asyncio.Queue()
            audio_q:    asyncio.Queue = asyncio.Queue()  # (order, bytes)
            full_answer_parts = []

            async def produce_sentences():
                buf = ""
                order = 0
                if self.mode == "free":
                    gen = self.llm_service.generate_free_stream(query, conversation_history=history_text)
                else:
                    gen = self.llm_service.generate_response_stream(
                        query, context, conversation_history=history_text
                    )
                async for token in gen:
                    full_answer_parts.append(token)
                    buf += token
                    self._emit("answer_token", {"text": token})
                    if _re.search(r'[.!?。！？]\s*$', buf.rstrip()) or len(buf) > 250:
                        s = buf.strip()
                        buf = ""
                        if s:
                            await sentence_q.put((order, s))
                            order += 1
                if buf.strip():
                    await sentence_q.put((order, buf.strip()))
                    order += 1
                await sentence_q.put(None)  # sentinel

            async def consume_sentences():
                while True:
                    item = await sentence_q.get()
                    if item is None:
                        await audio_q.put(None)
                        break
                    order, sentence = item
                    try:
                        audio = await self.synthesizer.synthesize_full(sentence)
                        await audio_q.put((order, audio))
                    except Exception as e:
                        logger.error(f"TTS sentence error: {e}")

            # Run producer & consumer concurrently
            await asyncio.gather(produce_sentences(), consume_sentences())

            # Drain audio queue in order and emit
            full_answer = "".join(full_answer_parts)
            if self.conv_service and self.session_id:
                self.conv_service.add_user_message(self.session_id, query)
                self.conv_service.add_assistant_message(self.session_id, full_answer)

            self._emit("answer", {"text": full_answer, "sources": sources})

            audio_chunks_all = []
            while True:
                item = await audio_q.get()
                if item is None:
                    break
                _, audio_data = item
                audio_chunks_all.append(audio_data)
                for i in range(0, len(audio_data), 8192):
                    b64 = base64.b64encode(audio_data[i:i + 8192]).decode()
                    self._emit("audio_chunk", {"data": b64, "format": "wav"})

            if audio_chunks_all:
                full_b64 = base64.b64encode(b''.join(audio_chunks_all)).decode()
                self._emit("audio_complete", {"full_audio": full_b64})

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._emit("error", {"message": str(e)})
        finally:
            self._set_state(StreamState.IDLE)

    async def cancel(self) -> None:
        if self.transcriber and self.transcriber.is_running:
            await self.transcriber.stop()
        self._set_state(StreamState.IDLE)
        self._emit("canceled", {})

    async def close(self) -> None:
        if self.transcriber:
            await self.transcriber.close()
        await self.synthesizer.close()
