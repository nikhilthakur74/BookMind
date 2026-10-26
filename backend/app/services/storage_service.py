"""Local storage and file management service for BookMind."""

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional
from app.core.config import settings
from app.schemas.book import (
    BookMetadata,
    BookSummary,
    ChunkMetadata,
    PageExtractionReport,
)


def compute_file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of binary content."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters.

    Never trusts the user-supplied filename as a filesystem path.
    """
    raw_name = Path(filename).name
    # Replace anything other than alphanumeric, dots, underscores, and dashes
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned or cleaned in (".", ".."):
        cleaned = "unnamed_document.pdf"
    return cleaned[:120]


def get_book_dir(book_id: str) -> Path:
    """Return the isolated storage directory for a specific book."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", book_id)
    return settings.books_dir / safe_id


def find_book_by_hash(file_hash: str) -> Optional[BookMetadata]:
    """Check if a book with the identical file hash has already been ingested."""
    books_dir = settings.books_dir
    if not books_dir.exists():
        return None

    # Scan directories
    for child in books_dir.iterdir():
        if child.is_dir():
            meta_path = child / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("file_hash") == file_hash:
                        return BookMetadata.model_validate(data)
                except Exception:
                    continue
    return None


def save_book_assets(
    book_id: str,
    pdf_bytes: bytes,
    metadata: BookMetadata,
    chunks: List[ChunkMetadata],
    page_reports: List[PageExtractionReport],
) -> Path:
    """Save raw PDF, metadata, chunks, and page reports into an isolated book directory."""
    target_dir = get_book_dir(book_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Save original PDF as a fixed safe filename
        pdf_path = target_dir / "document.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # 2. Save metadata JSON
        meta_path = target_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        # 3. Save chunks JSON
        chunks_path = target_dir / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            chunks_data = [chunk.model_dump(mode="json") for chunk in chunks]
            json.dump(chunks_data, f, indent=2)

        # 4. Save page extraction reports JSON
        pages_path = target_dir / "page_reports.json"
        with open(pages_path, "w", encoding="utf-8") as f:
            reports_data = [report.model_dump(mode="json") for report in page_reports]
            json.dump(reports_data, f, indent=2)

        return target_dir
    except Exception:
        # In case of partial writing error, clean up
        cleanup_book_dir(book_id)
        raise


def get_book_metadata(book_id: str) -> Optional[BookMetadata]:
    """Retrieve metadata for a specific book ID."""
    target_dir = get_book_dir(book_id)
    meta_path = target_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BookMetadata.model_validate(data)
    except Exception:
        return None


def save_book_metadata(book_id: str, metadata: BookMetadata) -> None:
    """Update and persist metadata for an existing book."""
    target_dir = get_book_dir(book_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    meta_path = target_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))


def get_book_page_reports(book_id: str) -> Optional[List[PageExtractionReport]]:
    """Retrieve detailed page extraction reports for a book."""
    target_dir = get_book_dir(book_id)
    pages_path = target_dir / "page_reports.json"
    if not pages_path.exists():
        return None
    try:
        with open(pages_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PageExtractionReport.model_validate(p) for p in data]
    except Exception:
        return None


def get_book_chunks(book_id: str) -> Optional[List[ChunkMetadata]]:
    """Retrieve all text chunks for a specific book ID."""
    target_dir = get_book_dir(book_id)
    chunks_path = target_dir / "chunks.json"
    if not chunks_path.exists():
        return None
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ChunkMetadata.model_validate(c) for c in data]
    except Exception:
        return None


def list_all_books() -> List[BookSummary]:
    """Return summary records of all stored books sorted by creation time descending."""
    books_dir = settings.books_dir
    if not books_dir.exists():
        return []

    results: List[BookSummary] = []
    for child in books_dir.iterdir():
        if child.is_dir():
            meta_path = child / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = BookMetadata.model_validate(data)
                    results.append(
                        BookSummary(
                            book_id=meta.book_id,
                            original_filename=meta.original_filename,
                            file_size_bytes=meta.file_size_bytes,
                            page_count=meta.page_count,
                            chunk_count=meta.chunk_count,
                            extraction_status=meta.extraction_status,
                            needs_ocr=meta.needs_ocr,
                            ocr_warning=meta.ocr_warning,
                            created_at=meta.created_at,
                            indexing_status=meta.indexing_status,
                            indexed_chunk_count=meta.indexed_chunk_count,
                        )
                    )
                except Exception:
                    continue

    results.sort(key=lambda x: x.created_at, reverse=True)
    return results


def delete_book(book_id: str) -> bool:
    """Delete a book and its associated directory and files."""
    target_dir = get_book_dir(book_id)
    if not target_dir.exists():
        return False
    try:
        shutil.rmtree(target_dir)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to delete book directory {target_dir}: {e}") from e


def cleanup_book_dir(book_id: str) -> None:
    """Safely remove a book directory during error recovery."""
    target_dir = get_book_dir(book_id)
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
        except Exception:
            pass
