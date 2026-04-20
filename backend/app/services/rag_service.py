"""
RAG (Retrieval Augmented Generation) Service
ChromaDB vektör veritabanı, similarity search, score threshold ve query rewriting
"""

import logging
from typing import List, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..models.schemas import SourceDocument
from ..models.exceptions import RAGError, NoDocumentsError
from .document_service import DocumentChunk

logger = logging.getLogger(__name__)


class RAGService:
    COLLECTION_NAME = "documents"

    def __init__(self):
        self.settings = get_settings()
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)

            chroma_path = Path(self.settings.chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)

            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"RAG Service initialized. Collection size: {self._collection.count()}")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise RAGError(f"RAG sistemi başlatılamadı: {str(e)}")

    async def add_documents(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0

        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self._embedding_model.encode(texts).tolist()

            ids = [f"{chunk.document_id}_{chunk.chunk_index}" for chunk in chunks]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page or -1
                }
                for chunk in chunks
            ]

            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(f"Added {len(chunks)} chunks to vector database")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise RAGError(f"Belgeler eklenemedi: {str(e)}")

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Tuple[List[SourceDocument], int]:
        """
        Sorguya en benzer belgeleri getir.

        Returns:
            (filtered_docs, total_retrieved) — total_retrieved threshold öncesi sayı
        """
        if self._collection.count() == 0:
            raise NoDocumentsError()

        top_k = top_k or self.settings.retrieval_top_k
        threshold = score_threshold if score_threshold is not None else self.settings.score_threshold

        try:
            query_embedding = self._embedding_model.encode([query]).tolist()

            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )

            all_docs: List[SourceDocument] = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = round(1 - distance, 4)

                    all_docs.append(SourceDocument(
                        filename=metadata.get('filename', 'unknown'),
                        content=doc,
                        score=score,
                        page=metadata.get('page') if metadata.get('page', -1) != -1 else None
                    ))

            total_retrieved = len(all_docs)

            # Score threshold filtresi uygula
            if threshold > 0:
                filtered = [d for d in all_docs if d.score >= threshold]
                logger.info(
                    f"Query '{query[:50]}': {total_retrieved} retrieved, "
                    f"{len(filtered)} after threshold={threshold}"
                )
            else:
                filtered = all_docs

            return filtered, total_retrieved

        except NoDocumentsError:
            raise
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise RAGError(f"Sorgu başarısız: {str(e)}")

    async def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Tuple[str, List[SourceDocument], int]:
        """
        Sorgu için RAG context oluştur.

        Returns:
            (context_string, source_documents, total_retrieved_before_filter)
        """
        sources, total_retrieved = await self.query(query, top_k, score_threshold)

        if not sources:
            return "", [], total_retrieved

        context_parts = []
        for i, source in enumerate(sources, 1):
            score_pct = int(source.score * 100)
            header = f"[Kaynak {i}: {source.filename} | Uyum: %{score_pct}]"
            context_parts.append(f"{header}\n{source.content}")

        context = "\n\n---\n\n".join(context_parts)
        return context, sources, total_retrieved

    async def delete_document(self, document_id: str) -> bool:
        try:
            results = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )

            if results and results['ids']:
                self._collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise RAGError(f"Belge silinemedi: {str(e)}")

    async def clear_all(self) -> int:
        try:
            count = self._collection.count()
            self._chroma_client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared all documents: {count} chunks deleted")
            return count

        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise RAGError(f"Belgeler silinemedi: {str(e)}")

    def get_document_count(self) -> int:
        return self._collection.count()

    def is_healthy(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False
