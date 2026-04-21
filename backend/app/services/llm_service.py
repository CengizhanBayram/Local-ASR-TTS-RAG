"""
LLM Service — provider-agnostic katman
Gemini / Ollama / OpenAI-compat üzerinde çalışır
"""

import logging
from typing import AsyncGenerator, List, Optional

from ..config import get_settings
from ..models.exceptions import LLMError, ConfigurationError
from .providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMService:

    # ── System Prompts ───────────────────────────────────────────────────────
    DEFAULT_SYSTEM_PROMPT = """You are an expert knowledge assistant that answers questions in Turkish.
Your responses are grounded exclusively in the source documents provided in each query.

## Core Responsibilities
- Synthesize information from the provided sources to produce accurate, helpful answers
- Communicate clearly and naturally in Turkish
- Acknowledge the limits of your knowledge honestly

## Strict Constraints
- **Source fidelity**: Only use facts explicitly stated in the provided sources. Never introduce outside knowledge.
- **No hallucination**: If the answer cannot be found in the sources, respond with exactly: "Bu konuda kaynaklarda bilgi bulamadım."
- **No speculation**: Do not infer, extrapolate, or guess beyond what the sources state.
- **No fabricated citations**: Do not reference sources that are not provided.

## Response Style
- Be concise: answer the question directly without unnecessary preamble
- Use fluent, natural Turkish appropriate for the topic
- For complex topics, use brief bullet points or numbered lists where helpful
- Match the formality level of the user's question"""

    CITATION_SYSTEM_PROMPT = """You are an expert knowledge assistant that answers questions in Turkish with inline citations.
Your responses are grounded exclusively in the numbered source documents provided in each query.

## Core Responsibilities
- Synthesize information from the provided sources to produce accurate, cited answers
- Append a citation marker [N] immediately after every claim, where N is the source number
- Communicate clearly and naturally in Turkish

## Strict Constraints
- **Source fidelity**: Only use facts explicitly stated in the provided sources. Never introduce outside knowledge.
- **No hallucination**: If the answer cannot be found in the sources, respond with exactly: "Bu konuda kaynaklarda bilgi bulamadım."
- **Citation accuracy**: Every factual statement must have a [N] marker. Do not cite a source number that was not provided.
- **No speculation**: Do not infer, extrapolate, or guess beyond what the sources state.

## Response Style
- Be concise and direct
- Use fluent, natural Turkish
- Place citation markers [N] at the end of the sentence or clause they support, before punctuation
- Example: "Şirket 2023 yılında kurulmuştur [1] ve 500 çalışanı bulunmaktadır [2]." """

    FREE_SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant that converses in Turkish.
You are helpful, honest, and direct.

## Behavior
- Answer questions accurately using your general knowledge
- Be concise: get to the point without unnecessary filler
- Use natural, fluent Turkish appropriate to the conversational register
- If you are unsure about something, say so clearly rather than guessing
- Do not make up information or fabricate facts

## Tone
- Warm and approachable, but professional
- Adapt your formality to match how the user is speaking to you"""

    # ── Prompt Templates ────────────────────────────────────────────────────
    CONTEXT_TEMPLATE = """{history_section}<sources>
{context}
</sources>

<question>
{query}
</question>

Answer the question using only the information in <sources>. If the answer is not there, say so."""

    QUERY_REWRITE_PROMPT = """Your task is to rewrite a user's question into a more effective document retrieval query.

Guidelines:
- Preserve the full intent and meaning of the original question
- Make implicit concepts explicit (e.g., expand abbreviations, resolve pronouns)
- Use terminology likely to appear in formal documents on this topic
- Output ONLY the rewritten query — no explanation, no preamble, no punctuation changes

Original question: {query}
Rewritten query:"""

    MULTI_QUERY_PROMPT = """Your task is to generate {n} distinct reformulations of the following question to maximize recall in a document retrieval system.

Guidelines:
- Each reformulation must preserve the original intent
- Vary vocabulary, phrasing, and specificity across reformulations
- Include both broader and narrower phrasings where appropriate
- Output exactly {n} lines, one query per line, no numbering, no extra text

Original question: {query}"""

    # ────────────────────────────────────────────────────────────────────────

    def __init__(self):
        self.settings = get_settings()
        self._provider: BaseLLMProvider = self._init_provider()

    def _init_provider(self) -> BaseLLMProvider:
        p = self.settings.llm_provider.lower()
        try:
            if p == "gemini":
                from .providers.gemini import GeminiProvider
                return GeminiProvider(self.settings)
            elif p == "ollama":
                from .providers.ollama import OllamaProvider
                return OllamaProvider(self.settings)
            elif p in ("openai_compatible", "vllm", "lmstudio", "localai"):
                from .providers.openai_compat import OpenAICompatProvider
                return OpenAICompatProvider(self.settings)
            else:
                raise ConfigurationError(f"Bilinmeyen LLM_PROVIDER: '{p}'")
        except (ValueError, ImportError) as e:
            raise ConfigurationError(str(e))

    # ── Public helpers ───────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    def _build_user_message(
        self,
        query: str,
        context: str,
        conversation_history: str = "",
    ) -> str:
        history_section = (
            f"<conversation_history>\n{conversation_history}\n</conversation_history>\n\n"
            if conversation_history
            else ""
        )
        if context:
            return self.CONTEXT_TEMPLATE.format(
                history_section=history_section,
                context=context,
                query=query,
            )
        return (
            f"{history_section}<question>\n{query}\n</question>\n\n"
            "Note: No documents are currently loaded."
        )

    def _system_prompt(self, override: Optional[str] = None) -> str:
        if override:
            return override
        if self.settings.enable_citations:
            return self.CITATION_SYSTEM_PROMPT
        return self.DEFAULT_SYSTEM_PROMPT

    # ── Core generation ──────────────────────────────────────────────────────

    async def generate_response(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: str = "",
    ) -> str:
        try:
            return await self._provider.generate(
                self._system_prompt(system_prompt),
                self._build_user_message(query, context, conversation_history),
            )
        except Exception as e:
            logger.error(f"generate_response error: {e}")
            raise LLMError(f"Cevap üretilemedi: {e}")

    async def generate_response_stream(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: str = "",
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self._provider.generate_stream(
                self._system_prompt(system_prompt),
                self._build_user_message(query, context, conversation_history),
            ):
                yield chunk
        except Exception as e:
            logger.error(f"generate_stream error: {e}")
            raise LLMError(f"Streaming başarısız: {e}")

    async def generate_free_response(
        self,
        query: str,
        conversation_history: str = "",
    ) -> str:
        history = f"<conversation_history>\n{conversation_history}\n</conversation_history>\n\n" if conversation_history else ""
        user_msg = f"{history}User: {query}"
        try:
            return await self._provider.generate(self.FREE_SYSTEM_PROMPT, user_msg)
        except Exception as e:
            raise LLMError(f"Response generation failed: {e}")

    async def generate_free_stream(
        self,
        query: str,
        conversation_history: str = "",
    ) -> AsyncGenerator[str, None]:
        history = f"<conversation_history>\n{conversation_history}\n</conversation_history>\n\n" if conversation_history else ""
        user_msg = f"{history}User: {query}"
        async for chunk in self._provider.generate_stream(self.FREE_SYSTEM_PROMPT, user_msg):
            yield chunk

    # ── Query helpers ────────────────────────────────────────────────────────

    async def rewrite_query(self, query: str) -> str:
        try:
            result = await self._provider.generate(
                "Sen bir arama sorgusu optimize uzmanısın.",
                self.QUERY_REWRITE_PROMPT.format(query=query),
            )
            rewritten = result.strip()
            return rewritten if len(rewritten) > 5 else query
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query

    async def generate_query_variations(
        self, query: str, n: int = 3
    ) -> List[str]:
        try:
            result = await self._provider.generate(
                "Sen bir arama sorgusu uzmanısın.",
                self.MULTI_QUERY_PROMPT.format(query=query, n=n),
            )
            lines = [
                ln.strip().lstrip("0123456789.-) ")
                for ln in result.strip().split("\n")
                if ln.strip()
            ]
            variations = lines[:n]
            if query not in variations:
                variations.insert(0, query)
            return variations[: n + 1]
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return [query]

    # ── Health ───────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        return self._provider.is_healthy()
