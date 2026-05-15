"""
Semantic Query Cache — embedding tabanlı sorgu önbellekleme.
Anlamsal olarak benzer sorgular için LLM çağrısını atlar.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """
    In-memory semantic cache (LRU eviction).
    Stores (normalized_embedding, answer, timestamp) per query.

    similarity_threshold: cosine similarity above which a query is a cache hit.
    0.92 works well for paraphrase-multilingual-MiniLM-L12-v2.
    """

    def __init__(
        self,
        embed_fn: Callable[[list], np.ndarray],
        max_size: int = 512,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ):
        self._embed_fn = embed_fn
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._enabled = enabled
        # key=query_text, value=(normalized_embedding, answer, timestamp)
        self._cache: OrderedDict[str, Tuple[np.ndarray, str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _embed_normalize(self, text: str) -> np.ndarray:
        vec = self._embed_fn([text], show_progress_bar=False)[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    async def lookup(self, query: str) -> Optional[str]:
        """Return cached answer or None. Runs embedding in thread executor."""
        if not self._enabled or not self._cache:
            self._misses += 1
            return None

        loop = asyncio.get_running_loop()
        q_emb = await loop.run_in_executor(None, self._embed_normalize, query)

        now = time.monotonic()
        best_sim, best_key, best_answer = 0.0, None, None

        for key, (emb, answer, ts) in self._cache.items():
            if now - ts > self._ttl:
                continue
            sim = float(np.dot(q_emb, emb))
            if sim > best_sim:
                best_sim, best_key, best_answer = sim, key, answer

        if best_sim >= self._threshold and best_key is not None:
            self._cache.move_to_end(best_key)
            self._hits += 1
            logger.info(f"Cache HIT sim={best_sim:.3f}: '{query[:60]}'")
            return best_answer

        self._misses += 1
        return None

    async def store(self, query: str, answer: str) -> None:
        """Embed the query and cache (query, answer). LRU evicts oldest on overflow."""
        if not self._enabled:
            return
        loop = asyncio.get_running_loop()
        emb = await loop.run_in_executor(None, self._embed_normalize, query)

        if query in self._cache:
            self._cache.move_to_end(query)
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[query] = (emb, answer, time.monotonic())

    def invalidate(self, query: str) -> bool:
        """Remove a specific entry from the cache."""
        if query in self._cache:
            del self._cache[query]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._hits = self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "enabled": self._enabled,
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "threshold": self._threshold,
        }

    def is_healthy(self) -> bool:
        return self._enabled
