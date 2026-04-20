"""
Document Processing Service
PDF, DOCX, TXT dosyalarını işleme ve sentence-aware chunking
"""

import os
import re
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

# Türkçe dahil cümle sonu örüntüsü
_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+')
# Boşluk normalleştirme
_MULTI_SPACE = re.compile(r' +')
_MULTI_NEWLINE = re.compile(r'\n{3,}')


class DocumentChunk:
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
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}

    def __init__(self):
        self.settings = get_settings()
        self.documents_path = Path(self.settings.documents_path)
        self.documents_path.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, DocumentMetadata] = {}

    async def process_document(self, file: UploadFile) -> Tuple[str, List[DocumentChunk]]:
        filename = file.filename
        extension = Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(extension)

        content = await file.read()
        file_size = len(content)
        document_id = str(uuid.uuid4())

        file_path = self.documents_path / f"{document_id}{extension}"
        with open(file_path, 'wb') as f:
            f.write(content)

        # Parse: (text, page_count, page_map)
        # page_map: list of (page_num, text) tuples (PDF only)
        text, page_count, page_map = self._parse_document(content, extension)

        if page_map:
            chunks = self._create_chunks_with_pages(page_map, document_id, filename)
        else:
            chunks = self._create_chunks(text, document_id, filename)

        self._documents[document_id] = DocumentMetadata(
            filename=filename,
            file_type=DocumentType(extension[1:]),
            file_size=file_size,
            page_count=page_count,
            chunk_count=len(chunks)
        )

        logger.info(f"Document processed: {filename} -> {len(chunks)} chunks")
        return document_id, chunks

    def _parse_document(
        self,
        content: bytes,
        extension: str
    ) -> Tuple[str, Optional[int], Optional[List[Tuple[int, str]]]]:
        """
        Returns: (full_text, page_count, page_map_or_None)
        page_map: [(page_number, page_text), ...]  — only for PDF
        """
        try:
            if extension == '.pdf':
                return self._parse_pdf(content)
            elif extension == '.docx':
                text, pages = self._parse_docx(content)
                return text, pages, None
            elif extension in ['.txt', '.md']:
                text, pages = self._parse_text(content)
                return text, pages, None
            else:
                raise UnsupportedFileTypeError(extension)
        except UnsupportedFileTypeError:
            raise
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            raise DocumentProcessingError(f"Belge parse edilemedi: {str(e)}")

    def _parse_pdf(self, content: bytes) -> Tuple[str, int, List[Tuple[int, str]]]:
        import io
        page_map: List[Tuple[int, str]] = []

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        page_map.append((i, page_text))
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                for i, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        page_map.append((i, page_text))
            except Exception as e2:
                raise DocumentProcessingError(f"PDF okunamadı: {str(e2)}")

        full_text = "\n\n".join(t for _, t in page_map)
        return full_text, len(page_map), page_map

    def _parse_docx(self, content: bytes) -> Tuple[str, None]:
        import io
        try:
            doc = DocxDocument(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return '\n\n'.join(paragraphs), None
        except Exception as e:
            raise DocumentProcessingError(f"DOCX okunamadı: {str(e)}")

    def _parse_text(self, content: bytes) -> Tuple[str, None]:
        for encoding in ['utf-8', 'cp1254', 'latin-1']:
            try:
                return content.decode(encoding), None
            except UnicodeDecodeError:
                continue
        return content.decode('latin-1', errors='replace'), None

    # ── Chunking ─────────────────────────────────────────────────────────────

    def _create_chunks_with_pages(
        self,
        page_map: List[Tuple[int, str]],
        document_id: str,
        filename: str
    ) -> List[DocumentChunk]:
        """PDF sayfa başına chunk oluştur, büyük sayfaları böl."""
        chunks: List[DocumentChunk] = []
        chunk_index = 0

        for page_num, page_text in page_map:
            page_text = self._clean_text(page_text)
            if not page_text:
                continue

            page_chunks = self._split_text_into_chunks(page_text)
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(DocumentChunk(
                        content=chunk_text.strip(),
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_index,
                        page=page_num
                    ))
                    chunk_index += 1

        return chunks

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        filename: str
    ) -> List[DocumentChunk]:
        """Sayfa bilgisi olmayan belgeler için chunk oluştur."""
        text = self._clean_text(text)
        if not text:
            return []

        chunk_texts = self._split_text_into_chunks(text)
        return [
            DocumentChunk(
                content=t.strip(),
                document_id=document_id,
                filename=filename,
                chunk_index=i
            )
            for i, t in enumerate(chunk_texts) if t.strip()
        ]

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Cümle sınırlarına dikkat eden overlapping chunk bölme.
        1. Metni cümlelere ayır
        2. Cümleleri chunk_size'ı geçmeyecek şekilde birleştir
        3. chunk_overlap kadar örtüşme ekle
        """
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)

            # Tek cümle chunk_size'dan büyükse kırp
            if s_len > chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                # Büyük cümleyi kelime bazlı böl
                chunks.extend(self._hard_split(sentence, chunk_size, chunk_overlap))
                current_sentences = []
                current_len = 0
                continue

            if current_len + s_len + 1 > chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Overlap: son N karakter değil, son birkaç cümle
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev in reversed(current_sentences):
                    if overlap_len + len(prev) > chunk_overlap:
                        break
                    overlap_sentences.insert(0, prev)
                    overlap_len += len(prev) + 1
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += s_len + 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Metni cümlelere ayır (paragraf yapısını koru)."""
        paragraphs = text.split('\n\n')
        sentences: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Cümlelere böl
            parts = _SENTENCE_END.split(para)
            sentences.extend(p.strip() for p in parts if p.strip())
        return sentences

    def _hard_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Kelime sınırına dikkat ederek büyük metni böl."""
        words = text.split()
        chunks: List[str] = []
        i = 0
        while i < len(words):
            chunk_words = []
            length = 0
            j = i
            while j < len(words) and length + len(words[j]) + 1 <= chunk_size:
                chunk_words.append(words[j])
                length += len(words[j]) + 1
                j += 1
            if not chunk_words:
                chunk_words = [words[i]]
                j = i + 1
            chunks.append(" ".join(chunk_words))
            # Overlap hesapla (kelime bazlı)
            back = 0
            k = j - 1
            while k >= i and back < overlap:
                back += len(words[k]) + 1
                k -= 1
            i = k + 2 if k + 2 <= j else j
        return chunks

    def _clean_text(self, text: str) -> str:
        text = _MULTI_SPACE.sub(' ', text)
        text = _MULTI_NEWLINE.sub('\n\n', text)
        return text.strip()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get_documents(self) -> dict[str, DocumentMetadata]:
        return self._documents

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        return self._documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False

        for ext in self.SUPPORTED_EXTENSIONS:
            file_path = self.documents_path / f"{document_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                break

        del self._documents[document_id]
        return True

    def is_healthy(self) -> bool:
        return self.documents_path.exists()
