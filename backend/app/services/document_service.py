"""
Document Processing Service
PDF, DOCX, TXT dosyalarını işleme ve chunk'lama
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from fastapi import UploadFile
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument

from ..config import get_settings
from ..models.schemas import DocumentType, DocumentMetadata
from ..models.exceptions import DocumentProcessingError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Belge parçası"""
    def __init__(
        self,
        content: str,
        document_id: str,
        filename: str,
        chunk_index: int,
        page: Optional[int] = None,
        metadata: Optional[dict] = None
    ):
        self.content = content
        self.document_id = document_id
        self.filename = filename
        self.chunk_index = chunk_index
        self.page = page
        self.metadata = metadata or {}


class DocumentService:
    """
    Belge işleme servisi
    Dosya yükleme, parse etme ve chunk'lama işlemleri
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    
    def __init__(self):
        self.settings = get_settings()
        self.documents_path = Path(self.settings.documents_path)
        self.documents_path.mkdir(parents=True, exist_ok=True)
        
        # Yüklenen belgelerin metadata'sı
        self._documents: dict[str, DocumentMetadata] = {}
    
    async def process_document(self, file: UploadFile) -> Tuple[str, List[DocumentChunk]]:
        """
        Belgeyi işle ve chunk'lara ayır
        
        Args:
            file: Yüklenen dosya
            
        Returns:
            (document_id, chunks) tuple
        """
        # Dosya uzantısını kontrol et
        filename = file.filename
        extension = Path(filename).suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(extension)
        
        # Dosyayı oku
        content = await file.read()
        file_size = len(content)
        
        # Unique ID oluştur
        document_id = str(uuid.uuid4())
        
        # Dosyayı kaydet
        file_path = self.documents_path / f"{document_id}{extension}"
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # İçeriği parse et
        text, page_count = self._parse_document(content, extension)
        
        # Chunk'lara ayır
        chunks = self._create_chunks(
            text=text,
            document_id=document_id,
            filename=filename
        )
        
        # Metadata kaydet
        self._documents[document_id] = DocumentMetadata(
            filename=filename,
            file_type=DocumentType(extension[1:]),  # .pdf -> pdf
            file_size=file_size,
            page_count=page_count,
            chunk_count=len(chunks)
        )
        
        logger.info(f"Document processed: {filename} -> {len(chunks)} chunks")
        
        return document_id, chunks
    
    def _parse_document(self, content: bytes, extension: str) -> Tuple[str, Optional[int]]:
        """
        Belge içeriğini parse et
        
        Returns:
            (text, page_count) tuple
        """
        try:
            if extension == '.pdf':
                return self._parse_pdf(content)
            elif extension == '.docx':
                return self._parse_docx(content)
            elif extension in ['.txt', '.md']:
                return self._parse_text(content)
            else:
                raise UnsupportedFileTypeError(extension)
        except UnsupportedFileTypeError:
            raise
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise DocumentProcessingError(f"Belge parse edilemedi: {str(e)}")
    
    def _parse_pdf(self, content: bytes) -> Tuple[str, int]:
        """PDF dosyasını parse et"""
        import io
        
        text_parts = []
        page_count = 0
        
        try:
            # pdfplumber ile dene (daha iyi sonuç verir)
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            # PyPDF2 ile dene
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                page_count = len(reader.pages)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            except Exception as e2:
                raise DocumentProcessingError(f"PDF okunamadı: {str(e2)}")
        
        return '\n\n'.join(text_parts), page_count
    
    def _parse_docx(self, content: bytes) -> Tuple[str, None]:
        """DOCX dosyasını parse et"""
        import io
        
        try:
            doc = DocxDocument(io.BytesIO(content))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n\n'.join(paragraphs), None
        except Exception as e:
            raise DocumentProcessingError(f"DOCX okunamadı: {str(e)}")
    
    def _parse_text(self, content: bytes) -> Tuple[str, None]:
        """TXT/MD dosyasını parse et"""
        try:
            # UTF-8 ile decode et
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Windows Türkçe encoding
                text = content.decode('cp1254')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
        
        return text, None
    
    def _create_chunks(
        self,
        text: str,
        document_id: str,
        filename: str
    ) -> List[DocumentChunk]:
        """
        Metni chunk'lara ayır
        Overlapping chunks ile context kaybını önle
        """
        chunks = []
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap
        
        # Metni temizle
        text = self._clean_text(text)
        
        if not text:
            return chunks
        
        # Paragraf bazlı bölme
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Chunk'a sığıyor mu?
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Mevcut chunk'ı kaydet
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                
                # Overlap için son kısmı al
                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + '\n\n' + paragraph
                else:
                    current_chunk = paragraph
                
                # Paragraf tek başına chunk_size'dan büyükse böl
                while len(current_chunk) > chunk_size:
                    chunks.append(DocumentChunk(
                        content=current_chunk[:chunk_size],
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = current_chunk[chunk_size - chunk_overlap:]
        
        # Son chunk'ı kaydet
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk,
                document_id=document_id,
                filename=filename,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Metni temizle"""
        # Çoklu boşlukları tek boşluğa çevir
        import re
        text = re.sub(r' +', ' ', text)
        # Çoklu satır sonlarını düzelt
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def get_documents(self) -> dict[str, DocumentMetadata]:
        """Yüklenen belgeleri listele"""
        return self._documents
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Belge metadata'sını getir"""
        return self._documents.get(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Belgeyi sil"""
        if document_id not in self._documents:
            return False
        
        # Dosyayı bul ve sil
        for ext in self.SUPPORTED_EXTENSIONS:
            file_path = self.documents_path / f"{document_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                break
        
        # Metadata'dan sil
        del self._documents[document_id]
        
        return True
    
    def is_healthy(self) -> bool:
        """Service health check"""
        return self.documents_path.exists()
