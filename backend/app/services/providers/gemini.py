"""
Gemini Provider — Google Generative AI
"""

import asyncio
import logging
import threading
from typing import AsyncGenerator

import google.generativeai as genai

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):

    def __init__(self, settings):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY gerekli (LLM_PROVIDER=gemini)")
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)
        self._gen_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1000,
        )
        logger.info(f"Gemini provider: {settings.gemini_model}")

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def generate(self, system_prompt: str, user_message: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_message}"

        def _run():
            return self._model.generate_content(
                full_prompt, generation_config=self._gen_config
            ).text

        return await asyncio.get_event_loop().run_in_executor(None, _run)

    async def generate_stream(
        self, system_prompt: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        full_prompt = f"{system_prompt}\n\n{user_message}"
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _stream():
            try:
                response = self._model.generate_content(
                    full_prompt,
                    stream=True,
                    generation_config=self._gen_config,
                )
                for chunk in response:
                    if chunk.text:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_stream, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def is_healthy(self) -> bool:
        return self._model is not None
