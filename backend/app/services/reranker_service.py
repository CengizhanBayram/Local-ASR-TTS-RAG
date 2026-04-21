"""
Reranker Service — cross-encoder reranking for retrieved documents.
Uses sentence-transformers CrossEncoder (e.g. ms-marco-MiniLM-L-6-v2).
"""

import asyncio
import logging
from typing import List

from ..config import get_settings
from ..models.schemas import SourceDocument

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        if self.settings.enable_reranking:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker: {self.settings.reranker_model}")
            self._model = CrossEncoder(
                self.settings.reranker_model,
                device=self.settings.reranker_device,
            )
            logger.info("Reranker ready")
        except Exception as e:
            logger.warning(f"Reranker load failed (reranking disabled): {e}")
            self._model = None

    async def rerank(
        self,
        query: str,
        docs: List[SourceDocument],
        top_k: int = 5,
    ) -> List[SourceDocument]:
        if not self._model or not docs:
            return docs

        pairs = [(query, d.content[:512]) for d in docs]

        def _run():
            return self._model.predict(pairs, show_progress_bar=False)

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, _run)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = []
        for score, doc in ranked[:top_k]:
            # Replace score with cross-encoder score (0–1 sigmoid approx)
            import math
            sigmoid = 1 / (1 + math.exp(-float(score)))
            result.append(SourceDocument(
                filename=doc.filename,
                content=doc.content,
                score=round(sigmoid, 4),
                page=doc.page,
            ))
        return result

    def is_healthy(self) -> bool:
        return self._model is not None or not self.settings.enable_reranking
