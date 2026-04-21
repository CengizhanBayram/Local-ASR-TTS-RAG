"""
Document Processing Service
PDF/DOCX/TXT — sentence-aware chunking, SQLite persistence, parent-child chunks,
PDF table extraction, optional OCR.
"""

import io
import re
import sqlite3
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import UploadFile
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument

from ..config import get_settings
from ..models.schemas import DocumentType, DocumentMetadata
from ..models.exceptions import DocumentProcessingError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+')
_MULTI_SPACE   = re.compile(r' +')
_MULTI_NEWLINE = re.compile(r'\n{3,}')

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    page_count  INTEGER,
    chunk_count INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parent_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    filename    TEXT NOT NULL,
    content     TEXT NOT NULL,
    page        INTEGER,
    chunk_index INTEGER DEFAULT 0
);
"""


@dataclass
class ParentChunk:
    """Large context chunk stored in SQLite for LLM input."""
    id: str
    content: str
    document_id: str
    filename: str
    page: Optional[int] = None
    chunk_index: int = 0


@dataclass
class DocumentChunk:
    """Small chunk for ChromaDB embedding. Has parent_id when parent-child is enabled."""
    content: str
    document_id: str
    filename: str
    chunk_index: int
    page: Optional[int] = None
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DocumentService:
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
    DB_NAME = "rag_store.db"

    def __init__(self):
        self.settings = get_settings()
        self.documents_path = Path(self.settings.documents_path)
        self.documents_path.mkdir(parents=True, exist_ok=True)

        db_dir = Path(self.settings.chroma_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_dir / self.DB_NAME)
        self._init_db()

        self._documents: dict[str, DocumentMetadata] = {}
        self._load_from_db()

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DB_SCHEMA)
        logger.info(f"DocumentService DB: {self._db_path}")

    def _load_from_db(self) -> None:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
        for row in rows:
            self._documents[row["id"]] = DocumentMetadata(
                filename=row["filename"],
                file_type=DocumentType(row["file_type"]),
                file_size=row["file_size"],
                page_count=row["page_count"],
                chunk_count=row["chunk_count"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            )
        logger.info(f"Loaded {len(self._documents)} documents from DB")

    def get_parent_chunk(self, parent_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM parent_chunks WHERE id=?", (parent_id,)
            ).fetchone()
        return row["content"] if row else None

    def get_all_parent_chunks(self) -> List[Tuple[str, str]]:
        """Returns [(parent_id, content), ...]"""
        with self._conn() as conn:
            rows = conn.execute("SELECT id, content FROM parent_chunks").fetchall()
        return [(r["id"], r["content"]) for r in rows]

    # ── Public API ────────────────────────────────────────────────────────────

    async def process_document(
        self, file: UploadFile
    ) -> Tuple[str, List[DocumentChunk], List[ParentChunk]]:
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

        text, page_count, page_map = self._parse_document(content, extension)

        if self.settings.enable_parent_child:
            effective_map = page_map or ([(1, text)] if text else [])
            child_chunks, parent_chunks = self._create_parent_child_chunks(
                effective_map, document_id, filename
            )
        else:
            if page_map:
                child_chunks = self._create_chunks_with_pages(page_map, document_id, filename)
            else:
                child_chunks = self._create_chunks(text or "", document_id, filename)
            parent_chunks = []

        meta = DocumentMetadata(
            filename=filename,
            file_type=DocumentType(extension[1:]),
            file_size=file_size,
            page_count=page_count,
            chunk_count=len(child_chunks),
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)",
                (document_id, meta.filename, meta.file_type.value, meta.file_size,
                 meta.page_count, meta.chunk_count, meta.uploaded_at.isoformat())
            )
            if parent_chunks:
                conn.executemany(
                    "INSERT OR REPLACE INTO parent_chunks VALUES (?,?,?,?,?,?)",
                    [(p.id, p.document_id, p.filename, p.content, p.page, p.chunk_index)
                     for p in parent_chunks]
                )
        self._documents[document_id] = meta

        logger.info(
            f"Processed {filename}: {len(child_chunks)} child chunks, "
            f"{len(parent_chunks)} parent chunks"
        )
        return document_id, child_chunks, parent_chunks

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_document(
        self, content: bytes, extension: str
    ) -> Tuple[Optional[str], Optional[int], Optional[List[Tuple[int, str]]]]:
        try:
            if extension == '.pdf':
                return self._parse_pdf(content)
            elif extension == '.docx':
                text, _ = self._parse_docx(content)
                return text, None, None
            elif extension in ('.txt', '.md'):
                text, _ = self._parse_text(content)
                return text, None, None
            else:
                raise UnsupportedFileTypeError(extension)
        except UnsupportedFileTypeError:
            raise
        except Exception as e:
            logger.error(f"Parse error: {e}")
            raise DocumentProcessingError(f"Could not parse document: {e}")

    def _parse_pdf(self, content: bytes) -> Tuple[str, int, List[Tuple[int, str]]]:
        page_map: List[Tuple[int, str]] = []
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""

                    # Table extraction → Markdown
                    tables = page.extract_tables() or []
                    table_md_parts = []
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        rows_md = []
                        for j, row in enumerate(table):
                            cells = [str(c or "").strip() for c in row]
                            rows_md.append("| " + " | ".join(cells) + " |")
                            if j == 0:
                                rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
                        table_md_parts.append("\n".join(rows_md))
                    combined = (page_text + ("\n\n" + "\n\n".join(table_md_parts) if table_md_parts else "")).strip()

                    # OCR fallback for empty pages
                    if not combined and self.settings.enable_ocr:
                        combined = self._ocr_page(page)

                    if combined:
                        page_map.append((i, combined))

        except Exception as e:
            logger.warning(f"pdfplumber failed, falling back to PyPDF2: {e}")
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                for i, page in enumerate(reader.pages, 1):
                    t = (page.extract_text() or "").strip()
                    if t:
                        page_map.append((i, t))
            except Exception as e2:
                raise DocumentProcessingError(f"PDF unreadable: {e2}")

        full_text = "\n\n".join(t for _, t in page_map)
        return full_text, len(page_map), page_map

    def _ocr_page(self, page) -> str:
        try:
            import pytesseract
            img = page.to_image(resolution=200).original
            return pytesseract.image_to_string(img, lang=self.settings.ocr_language)
        except ImportError:
            logger.warning("OCR requested but pytesseract/Pillow not installed")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _parse_docx(self, content: bytes) -> Tuple[str, None]:
        try:
            doc = DocxDocument(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return '\n\n'.join(paragraphs), None
        except Exception as e:
            raise DocumentProcessingError(f"DOCX unreadable: {e}")

    def _parse_text(self, content: bytes) -> Tuple[str, None]:
        for enc in ('utf-8', 'cp1254', 'latin-1'):
            try:
                return content.decode(enc), None
            except UnicodeDecodeError:
                continue
        return content.decode('latin-1', errors='replace'), None

    # ── Parent-child chunking ─────────────────────────────────────────────────

    def _create_parent_child_chunks(
        self,
        page_map: List[Tuple[int, str]],
        document_id: str,
        filename: str,
    ) -> Tuple[List[DocumentChunk], List[ParentChunk]]:
        child_chunks: List[DocumentChunk] = []
        parent_chunks: List[ParentChunk] = []
        child_idx = parent_idx = 0

        parent_size = self.settings.parent_chunk_size
        child_size  = self.settings.child_chunk_size

        for page_num, page_text in page_map:
            page_text = self._clean_text(page_text)
            if not page_text:
                continue

            for pt in self._split_text_into_chunks(page_text, chunk_size=parent_size, chunk_overlap=100):
                pt = pt.strip()
                if not pt:
                    continue
                parent_id = f"{document_id}_p{parent_idx}"
                parent_chunks.append(ParentChunk(
                    id=parent_id,
                    content=pt,
                    document_id=document_id,
                    filename=filename,
                    page=page_num,
                    chunk_index=parent_idx,
                ))

                for ct in self._split_text_into_chunks(pt, chunk_size=child_size, chunk_overlap=30):
                    ct = ct.strip()
                    if ct:
                        child_chunks.append(DocumentChunk(
                            content=ct,
                            document_id=document_id,
                            filename=filename,
                            chunk_index=child_idx,
                            page=page_num,
                            parent_id=parent_id,
                        ))
                        child_idx += 1
                parent_idx += 1

        return child_chunks, parent_chunks

    # ── Standard chunking ─────────────────────────────────────────────────────

    def _create_chunks_with_pages(
        self, page_map: List[Tuple[int, str]], document_id: str, filename: str
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        idx = 0
        for page_num, page_text in page_map:
            page_text = self._clean_text(page_text)
            if not page_text:
                continue
            for ct in self._split_text_into_chunks(page_text):
                if ct.strip():
                    chunks.append(DocumentChunk(
                        content=ct.strip(), document_id=document_id,
                        filename=filename, chunk_index=idx, page=page_num
                    ))
                    idx += 1
        return chunks

    def _create_chunks(self, text: str, document_id: str, filename: str) -> List[DocumentChunk]:
        text = self._clean_text(text)
        if not text:
            return []
        return [
            DocumentChunk(content=t.strip(), document_id=document_id, filename=filename, chunk_index=i)
            for i, t in enumerate(self._split_text_into_chunks(text)) if t.strip()
        ]

    def _split_text_into_chunks(
        self, text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        chunk_size    = chunk_size    or self.settings.chunk_size
        chunk_overlap = chunk_overlap or self.settings.chunk_overlap
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = []
        cur_len = 0

        for s in sentences:
            s_len = len(s)
            if s_len > chunk_size:
                if current:
                    chunks.append(" ".join(current))
                chunks.extend(self._hard_split(s, chunk_size, chunk_overlap))
                current, cur_len = [], 0
                continue
            if cur_len + s_len + 1 > chunk_size and current:
                chunks.append(" ".join(current))
                overlap: List[str] = []
                ov_len = 0
                for prev in reversed(current):
                    if ov_len + len(prev) > chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    ov_len += len(prev) + 1
                current, cur_len = overlap, ov_len
            current.append(s)
            cur_len += s_len + 1

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences: List[str] = []
        for para in text.split('\n\n'):
            para = para.strip()
            if para:
                sentences.extend(p.strip() for p in _SENTENCE_END.split(para) if p.strip())
        return sentences

    def _hard_split(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        i = 0
        while i < len(words):
            cw, length, j = [], 0, i
            while j < len(words) and length + len(words[j]) + 1 <= chunk_size:
                cw.append(words[j]); length += len(words[j]) + 1; j += 1
            if not cw:
                cw = [words[i]]; j = i + 1
            chunks.append(" ".join(cw))
            back, k = 0, j - 1
            while k >= i and back < overlap:
                back += len(words[k]) + 1; k -= 1
            i = k + 2 if k + 2 <= j else j
        return chunks

    def _clean_text(self, text: str) -> str:
        return _MULTI_NEWLINE.sub('\n\n', _MULTI_SPACE.sub(' ', text)).strip()

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get_documents(self) -> dict[str, DocumentMetadata]:
        return self._documents

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        return self._documents.get(document_id)

    async def delete_document(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False
        for ext in self.SUPPORTED_EXTENSIONS:
            fp = self.documents_path / f"{document_id}{ext}"
            if fp.exists():
                fp.unlink()
                break
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE id=?",      (document_id,))
            conn.execute("DELETE FROM parent_chunks WHERE document_id=?", (document_id,))
        del self._documents[document_id]
        return True

    def is_healthy(self) -> bool:
        return self.documents_path.exists()
