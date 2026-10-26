"""Automated tests for Phase 2: PDF ingestion, chunking, OCR detection, and books API."""

import io
from typing import List, Optional
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def build_synthetic_pdf(
    pages: List[str],
    toc: Optional[List] = None,
    image_pages: Optional[List[int]] = None,
) -> bytes:
    """Generate an in-memory synthetic PDF for deterministic test scenarios.

    Args:
        pages: List of text content strings, one per page.
        toc: Optional table of contents list [[level, title, page_number], ...].
        image_pages: 1-based page numbers where an embedded image should be inserted.
    """
    doc = pymupdf.open()
    image_pages_set = set(image_pages or [])

    for idx, page_text in enumerate(pages):
        page_num = idx + 1
        page = doc.new_page()

        # Insert text if provided
        if page_text:
            page.insert_text((50, 50), page_text, fontsize=12)

        # Insert synthetic image if requested
        if page_num in image_pages_set:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 20, 20), False)
            page.insert_image(pymupdf.Rect(50, 100, 200, 250), pixmap=pixmap)

    if toc:
        doc.set_toc(toc)

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Ensure each test runs against a clean, isolated temporary storage directory."""
    temp_upload = tmp_path / "uploads"
    temp_upload.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(temp_upload))
    yield temp_upload


def test_upload_valid_pdf_with_chapters():
    """Test standard upload: parses 1-based pages, detects chapters, and produces chunks."""
    page_1 = (
        "Chapter 1: The Foundations of Thought\n\n"
        "Philosophy begins in wonder and inquiry. When we explore long-form literature, "
        "we encounter complex thematic structures that cannot be reduced to isolated keywords. "
        "Instead, ideas unfold across interconnected chapters and cohesive paragraphs."
    )
    page_2 = (
        "Chapter 2: The Horizon of Understanding\n\n"
        "Deep reading demands sustained attention and contextual awareness. "
        "Every page connects to the author's overarching thesis, requiring synthetic retrieval."
    )

    pdf_bytes = build_synthetic_pdf(
        pages=[page_1, page_2],
        toc=[
            [1, "Chapter 1: The Foundations of Thought", 1],
            [1, "Chapter 2: The Horizon of Understanding", 2],
        ],
    )

    response = client.post(
        "/api/books/upload",
        files={"file": ("foundations_book.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    book = data["book"]

    assert book["original_filename"] == "foundations_book.pdf"
    assert book["safe_filename"] == "foundations_book.pdf"
    assert book["page_count"] == 2
    assert book["chunk_count"] >= 2
    assert book["extraction_status"] == "completed"
    assert book["needs_ocr"] is False
    assert len(book["headings"]) >= 2
    assert book["headings"][0]["title"] == "Chapter 1: The Foundations of Thought"
    assert book["headings"][0]["page_number"] == 1
    assert book["headings"][0]["confidence"] == "high"

    # Verify sample chunks metadata
    assert len(data["sample_chunks"]) > 0
    first_chunk = data["sample_chunks"][0]
    assert first_chunk["page_start"] == 1
    assert first_chunk["chapter"] == "Chapter 1: The Foundations of Thought"
    assert "chunk_id" in first_chunk


def test_scanned_image_only_pdf_flags_ocr_warning():
    """Test OCR detection: image-only pages must be flagged and not pretend to be processed cleanly."""
    page_1_text = "Preface and overview of the scanned manuscript.\n\nHere is readable text."
    page_2_scanned = ""  # Zero extractable text, purely image

    pdf_bytes = build_synthetic_pdf(
        pages=[page_1_text, page_2_scanned],
        image_pages=[2],
    )

    response = client.post(
        "/api/books/upload",
        files={"file": ("scanned_volume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    book = data["book"]

    assert book["needs_ocr"] is True
    assert book["extraction_status"] == "completed_with_warnings"
    assert book["ocr_needed_pages"] == [2]
    assert "OCR is recommended for 1 page(s) ([2])" in book["ocr_warning"]


def test_empty_page_detection():
    """Test blank page tracking without false OCR alarms."""
    page_1 = "Standard text on page one."
    page_2 = ""  # Completely blank: no text, no images

    pdf_bytes = build_synthetic_pdf(pages=[page_1, page_2])

    response = client.post(
        "/api/books/upload",
        files={"file": ("partially_blank.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 201
    book = response.json()["book"]
    assert book["empty_pages"] == [2]
    assert book["needs_ocr"] is False


def test_prevent_duplicate_uploads():
    """Test duplicate detection via file content hash returns 409 Conflict."""
    pdf_bytes = build_synthetic_pdf(["Content for duplicate test."])

    # First upload succeeds
    res1 = client.post(
        "/api/books/upload",
        files={"file": ("original.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res1.status_code == 201
    book_id = res1.json()["book"]["book_id"]

    # Second upload with identical bytes must return 409
    res2 = client.post(
        "/api/books/upload",
        files={"file": ("duplicate_upload.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res2.status_code == 409
    detail = res2.json()["detail"]
    assert "Duplicate book detected" in detail["message"]
    assert detail["existing_book_id"] == book_id


def test_reject_invalid_file_extension():
    """Test non-PDF file extension is rejected with 400 Bad Request."""
    response = client.post(
        "/api/books/upload",
        files={"file": ("notes.txt", io.BytesIO(b"Plain text content"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type '.txt'" in response.json()["detail"]


def test_reject_empty_file():
    """Test 0-byte upload is rejected with 400 Bad Request."""
    response = client.post(
        "/api/books/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_reject_corrupt_non_pdf_file():
    """Test non-PDF binary masquerading as PDF is rejected with 400 Bad Request."""
    corrupt_bytes = b"NOT_A_PDF_HEADER and random corrupted bytes"
    response = client.post(
        "/api/books/upload",
        files={"file": ("fake.pdf", io.BytesIO(corrupt_bytes), "application/pdf")},
    )
    assert response.status_code == 400
    assert "does not match PDF standard" in response.json()["detail"]


def test_list_and_get_book():
    """Test listing books and retrieving individual book details."""
    pdf_bytes = build_synthetic_pdf(["Book Alpha content."])

    upload_res = client.post(
        "/api/books/upload",
        files={"file": ("book_alpha.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    book_id = upload_res.json()["book"]["book_id"]

    # List books
    list_res = client.get("/api/books")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(b["book_id"] == book_id for b in list_data["books"])

    # Get single book details
    get_res = client.get(f"/api/books/{book_id}")
    assert get_res.status_code == 200
    assert get_res.json()["book"]["book_id"] == book_id


def test_get_book_chunks_paginated():
    """Test retrieving chunks with pagination parameters."""
    multi_paragraph = "\n\n".join([f"Paragraph number {i} discussing themes." for i in range(15)])
    pdf_bytes = build_synthetic_pdf([multi_paragraph])

    upload_res = client.post(
        "/api/books/upload",
        files={"file": ("multi_chunk_book.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    book_id = upload_res.json()["book"]["book_id"]

    # Fetch paginated chunks
    chunks_res = client.get(f"/api/books/{book_id}/chunks?skip=0&limit=2")
    assert chunks_res.status_code == 200
    chunk_data = chunks_res.json()
    assert chunk_data["book_id"] == book_id
    assert chunk_data["limit"] == 2
    assert len(chunk_data["chunks"]) <= 2


def test_delete_book():
    """Test deleting a book removes its files and makes subsequent lookups return 404."""
    pdf_bytes = build_synthetic_pdf(["Temporary book content to be deleted."])

    upload_res = client.post(
        "/api/books/upload",
        files={"file": ("temp_book.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 201
    book_id = upload_res.json()["book"]["book_id"]

    # Delete book
    del_res = client.delete(f"/api/books/{book_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Subsequent GET returns 404
    get_res = client.get(f"/api/books/{book_id}")
    assert get_res.status_code == 404


def test_heuristic_heading_detection_without_toc():
    """Test heuristic heading detection when no PDF TOC bookmark outline exists."""
    page_text = (
        "Chapter 3: The Architecture of Memory\n\n"
        "Memory serves as the cognitive foundation for contextual continuity. "
        "Without Table of Contents metadata, regex and layout heuristics identify this title."
    )
    pdf_bytes = build_synthetic_pdf(pages=[page_text], toc=None)

    response = client.post(
        "/api/books/upload",
        files={"file": ("heuristic_headings.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    headings = data["book"]["headings"]

    assert len(headings) >= 1
    found_heading = headings[0]
    assert "Chapter 3: The Architecture of Memory" in found_heading["title"]
    assert found_heading["confidence"] == "heuristic"
    assert found_heading["source"] == "layout_heuristic"


def test_file_size_limit_enforced(monkeypatch):
    """Test that file exceeding MAX_UPLOAD_SIZE_BYTES returns 413 Payload Too Large."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 500)  # Limit to 500 bytes

    oversized_pdf = build_synthetic_pdf(["This is a test PDF that exceeds the 500 byte limit."])
    assert len(oversized_pdf) > 500

    response = client.post(
        "/api/books/upload",
        files={"file": ("oversized.pdf", io.BytesIO(oversized_pdf), "application/pdf")},
    )
    assert response.status_code == 413
    assert "Uploaded file exceeds maximum limit" in response.json()["detail"]

