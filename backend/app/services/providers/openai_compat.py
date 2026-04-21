"""
OpenAI-Compatible Provider — vLLM, LM Studio, LocalAI, TGI …
Herhangi bir /v1/chat/completions endpoint'i
"""

import json
import logging
from typing import AsyncGenerator

import httpx

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseLLMProvider):

    def __init__(self, settings):
        self._base_url = settings.openai_compat_base_url.rstrip("/")
        self._model = settings.openai_compat_model
        self._headers = {
            "Authorization": f"Bearer {settings.openai_compat_api_key}",
            "Content-Type": "application/json",
        }
        logger.info(
            f"OpenAI-compat provider: {self._base_url}  model={self._model}"
        )

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    def _messages(self, system_prompt: str, user_message: str) -> list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    async def generate(self, system_prompt: str, user_message: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": self._model,
                    "messages": self._messages(system_prompt, user_message),
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def generate_stream(
        self, system_prompt: str, user_message: str
    ) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json={
                    "model": self._model,
                    "messages": self._messages(system_prompt, user_message),
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get(
                            "content", ""
                        )
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def is_healthy(self) -> bool:
        try:
            r = httpx.get(
                f"{self._base_url}/models",
                headers=self._headers,
                timeout=3.0,
            )
            return r.status_code == 200
        except Exception:
            return False
