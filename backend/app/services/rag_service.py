"""
RAG Service
ChromaDB vector search + BM25 hybrid search + RRF fusion +
parent-child retrieval + multi-query deduplication.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..models.schemas import SourceDocument
from ..models.exceptions import RAGError, NoDocumentsError
from .document_service import DocumentChunk, ParentChunk

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


class RAGService:
    COLLECTION_NAME = "documents"

    def __init__(self):
        self.settings = get_settings()
        self._embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client = None
        self._collection = None

        # BM25
        self._bm25 = None
        self._bm25_ids: List[str] = []

        # parent_id → content (in-memory cache rebuilt from SQLite on startup)
        self._parent_content: Dict[str, str] = {}

        self._initialize()

    def _initialize(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._embedding_model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )

            chroma_path = Path(self.settings.chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB ready — {self._collection.count()} chunks")

            self._rebuild_bm25()
            self._rebuild_parent_map()

        except Exception as e:
            logger.error(f"RAGService init failed: {e}")
            raise RAGError(f"RAG init failed: {e}")

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        if not self.settings.enable_hybrid_search:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed — hybrid search disabled")
            return

        results = self._collection.get(include=["documents", "ids"])
        if not results or not results.get("documents"):
            self._bm25 = None
            self._bm25_ids = []
            return

        self._bm25_ids = results["ids"]
        tokenized = [doc.lower().split() for doc in results["documents"]]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug(f"BM25 index rebuilt: {len(self._bm25_ids)} docs")

    # ── Parent-child map ──────────────────────────────────────────────────────

    def _rebuild_parent_map(self) -> None:
        if not self.settings.enable_parent_child:
            return
        db_path = Path(self.settings.chroma_path).parent / "rag_store.db"
        if not db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, content FROM parent_chunks").fetchall()
            conn.close()
            self._parent_content = {r["id"]: r["content"] for r in rows}
            logger.info(f"Parent map loaded: {len(self._parent_content)} parent chunks")
        except Exception as e:
            logger.warning(f"Could not load parent map: {e}")

    # ── add_documents ─────────────────────────────────────────────────────────

    async def add_documents(
        self,
        chunks: List[DocumentChunk],
        parent_chunks: Optional[List[ParentChunk]] = None,
    ) -> int:
        if not chunks:
            return 0
        try:
            texts      = [c.content for c in chunks]
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False).tolist()
            ids        = [f"{c.document_id}_{c.chunk_index}" for c in chunks]
            metadatas  = [
                {
                    "document_id": c.document_id,
                    "filename":    c.filename,
                    "chunk_index": c.chunk_index,
                    "page":        c.page or -1,
                    "parent_id":   c.parent_id or "",
                }
                for c in chunks
            ]
            self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

            # Cache parent content
            if parent_chunks:
                for p in parent_chunks:
                    self._parent_content[p.id] = p.content

            self._rebuild_bm25()
            logger.info(f"Added {len(chunks)} chunks to ChromaDB")
            return len(chunks)
        except Exception as e:
            logger.error(f"add_documents error: {e}")
            raise RAGError(f"add_documents failed: {e}")

    # ── Query ─────────────────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        extra_queries: Optional[List[str]] = None,
    ) -> Tuple[List[SourceDocument], int]:
        """
        Hybrid vector + BM25 retrieval with RRF fusion.
        If extra_queries provided (multi-query), merges results across all queries.
        Returns (filtered_docs, total_before_filter).
        """
        if self._collection.count() == 0:
            raise NoDocumentsError()

        top_k     = top_k     or self.settings.retrieval_top_k
        threshold = score_threshold if score_threshold is not None else self.settings.score_threshold
        all_queries = [query] + (extra_queries or [])

        try:
            # ── Vector search ─────────────────────────────────────────────────
            query_embeddings = self._embedding_model.encode(all_queries, show_progress_bar=False).tolist()
            n_results = min(top_k * 2, self._collection.count())

            # RRF accumulator: id → accumulated score
            rrf_scores: Dict[str, float] = {}
            id_to_meta: Dict[str, dict]  = {}
            id_to_text: Dict[str, str]   = {}
            id_to_vscore: Dict[str, float] = {}

            for q_emb in query_embeddings:
                results = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                if not results or not results["documents"] or not results["documents"][0]:
                    continue
                for rank, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )):
                    cid = f"{meta['document_id']}_{meta['chunk_index']}"
                    rrf_scores[cid]   = rrf_scores.get(cid, 0) + _rrf_score(rank)
                    id_to_meta[cid]   = meta
                    id_to_text[cid]   = doc
                    id_to_vscore[cid] = max(id_to_vscore.get(cid, 0), round(1 - dist, 4))

            # ── BM25 search ───────────────────────────────────────────────────
            if self.settings.enable_hybrid_search and self._bm25 is not None:
                for q_text in all_queries:
                    scores = self._bm25.get_scores(q_text.lower().split())
                    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                    for rank, (idx, _) in enumerate(ranked[:n_results]):
                        cid = self._bm25_ids[idx]
                        rrf_scores[cid] = rrf_scores.get(cid, 0) + _rrf_score(rank)

            # ── Merge & sort ──────────────────────────────────────────────────
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

            all_docs: List[SourceDocument] = []
            for cid in sorted_ids[:top_k]:
                if cid not in id_to_meta:
                    continue
                meta = id_to_meta[cid]
                page = meta.get("page") if meta.get("page", -1) != -1 else None

                # Parent-child: use parent content for LLM context if available
                parent_id = meta.get("parent_id", "")
                if self.settings.enable_parent_child and parent_id and parent_id in self._parent_content:
                    ctx_content = self._parent_content[parent_id]
                else:
                    ctx_content = id_to_text.get(cid, "")

                all_docs.append(SourceDocument(
                    filename=meta.get("filename", "unknown"),
                    content=ctx_content,
                    score=id_to_vscore.get(cid, 0.0),
                    page=page,
                ))

            # De-duplicate by content (multi-query can produce identical parents)
            seen_contents: set = set()
            deduped: List[SourceDocument] = []
            for d in all_docs:
                key = d.content[:200]
                if key not in seen_contents:
                    seen_contents.add(key)
                    deduped.append(d)

            total_retrieved = len(deduped)

            # ── Score threshold filter ────────────────────────────────────────
            if threshold > 0:
                filtered = [d for d in deduped if d.score >= threshold]
                logger.info(
                    f"Query '{query[:50]}': {total_retrieved} retrieved, "
                    f"{len(filtered)} after threshold={threshold}"
                )
            else:
                filtered = deduped

            return filtered, total_retrieved

        except NoDocumentsError:
            raise
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise RAGError(f"Query failed: {e}")

    async def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        extra_queries: Optional[List[str]] = None,
    ) -> Tuple[str, List[SourceDocument], int]:
        """Returns (context_string, source_docs, total_before_filter)."""
        sources, total = await self.query(query, top_k, score_threshold, extra_queries)
        if not sources:
            return "", [], total

        context_parts = []
        for i, s in enumerate(sources, 1):
            pct = int(s.score * 100)
            header = f"[Source {i}: {s.filename} | Match: {pct}%]"
            context_parts.append(f"{header}\n{s.content}")
        return "\n\n---\n\n".join(context_parts), sources, total

    # ── Delete / Clear ────────────────────────────────────────────────────────

    async def delete_document(self, document_id: str) -> bool:
        try:
            results = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
            if results and results["ids"]:
                # Remove related parent content from cache
                for meta in results["metadatas"]:
                    pid = meta.get("parent_id", "")
                    self._parent_content.pop(pid, None)
                self._collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for {document_id}")
                self._rebuild_bm25()
                return True
            return False
        except Exception as e:
            raise RAGError(f"Delete failed: {e}")

    async def clear_all(self) -> int:
        try:
            count = self._collection.count()
            self._chroma_client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._bm25 = None
            self._bm25_ids = []
            self._parent_content = {}
            logger.info(f"Cleared all: {count} chunks deleted")
            return count
        except Exception as e:
            raise RAGError(f"Clear failed: {e}")

    def get_document_count(self) -> int:
        return self._collection.count()

    def is_healthy(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False
