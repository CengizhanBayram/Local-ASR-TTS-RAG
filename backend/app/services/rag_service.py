"""
RAG (Retrieval Augmented Generation) Service
ChromaDB ile vektör veritabanı ve similarity search
"""

import logging
from typing import List, Optional
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
    """
    RAG (Retrieval Augmented Generation) servisi
    Belgeleri vektör veritabanına ekler ve benzer içerikleri getirir
    """
    
    COLLECTION_NAME = "documents"
    
    def __init__(self):
        self.settings = get_settings()
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None
        self._initialize()
    
    def _initialize(self) -> None:
        """RAG sistemini başlat"""
        try:
            # Embedding model yükle
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            
            # ChromaDB başlat
            chroma_path = Path(self.settings.chroma_path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            self._chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Collection oluştur veya al
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )
            
            logger.info(f"RAG Service initialized. Collection size: {self._collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise RAGError(f"RAG sistemi başlatılamadı: {str(e)}")
    
    async def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """
        Belge chunk'larını vektör veritabanına ekle
        
        Args:
            chunks: Belge parçaları listesi
            
        Returns:
            Eklenen chunk sayısı
        """
        if not chunks:
            return 0
        
        try:
            # Embedding oluştur
            texts = [chunk.content for chunk in chunks]
            embeddings = self._embedding_model.encode(texts).tolist()
            
            # ChromaDB'ye ekle
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
        top_k: Optional[int] = None
    ) -> List[SourceDocument]:
        """
        Sorguya en benzer belgeleri getir
        
        Args:
            query: Arama sorgusu
            top_k: Getirilecek sonuç sayısı
            
        Returns:
            Benzer belgeler listesi
        """
        if self._collection.count() == 0:
            raise NoDocumentsError()
        
        top_k = top_k or self.settings.retrieval_top_k
        
        try:
            # Query embedding oluştur
            query_embedding = self._embedding_model.encode([query]).tolist()
            
            # Similarity search
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            # Sonuçları SourceDocument'a çevir
            source_documents = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    # Distance'ı similarity score'a çevir (cosine distance)
                    score = 1 - distance
                    
                    source_documents.append(SourceDocument(
                        filename=metadata.get('filename', 'unknown'),
                        content=doc,
                        score=round(score, 4),
                        page=metadata.get('page') if metadata.get('page', -1) != -1 else None
                    ))
            
            logger.info(f"Query returned {len(source_documents)} results")
            return source_documents
            
        except NoDocumentsError:
            raise
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise RAGError(f"Sorgu başarısız: {str(e)}")
    
    async def get_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> tuple[str, List[SourceDocument]]:
        """
        Sorgu için RAG context oluştur
        
        Args:
            query: Arama sorgusu
            top_k: Kullanılacak belge sayısı
            
        Returns:
            (context_string, source_documents) tuple
        """
        sources = await self.query(query, top_k)
        
        if not sources:
            return "", []
        
        # Context oluştur
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[Kaynak {i}: {source.filename}]\n{source.content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, sources
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Belgenin tüm chunk'larını sil
        
        Args:
            document_id: Silinecek belge ID'si
            
        Returns:
            Başarılı ise True
        """
        try:
            # Belgeye ait tüm chunk'ları bul
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
        """
        Tüm belgeleri sil
        
        Returns:
            Silinen chunk sayısı
        """
        try:
            count = self._collection.count()
            
            # Collection'ı sil ve yeniden oluştur
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
        """Toplam chunk sayısını getir"""
        return self._collection.count()
    
    def is_healthy(self) -> bool:
        """Service health check"""
        try:
            self._collection.count()
            return True
        except Exception:
            return False
