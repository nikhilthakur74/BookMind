"""PDF parsing, OCR detection, heading heuristics, and text chunking using PyMuPDF."""

import re
from typing import Dict, List, Optional, Tuple
import pymupdf

from app.schemas.book import ChunkMetadata, HeadingInfo, PageExtractionReport


# Regex patterns for heuristic chapter and section detection
HEADING_PATTERNS = [
    # Chapter N: Title
    (
        re.compile(
            r"^(?:CHAPTER|Chapter)\s+([IVXLCDM\d]+|[A-Za-z]+)(?:[:.\s\-–—]+(.*))?$",
            re.IGNORECASE,
        ),
        1,  # Level 1 (Chapter)
    ),
    # Part N: Title
    (
        re.compile(
            r"^(?:PART|Part|BOOK|Book)\s+([IVXLCDM\d]+|[A-Za-z]+)(?:[:.\s\-–—]+(.*))?$",
            re.IGNORECASE,
        ),
        1,  # Level 1 (Part)
    ),
    # Section N.N: Title
    (
        re.compile(
            r"^(?:SECTION|Section)\s+([IVXLCDM\d]+|\d+(?:\.\d+)*)(?:[:.\s\-–—]+(.*))?$",
            re.IGNORECASE,
        ),
        2,  # Level 2 (Section)
    ),
    # Common standard book divisions
    (
        re.compile(
            r"^(?:PROLOGUE|Prologue|EPILOGUE|Epilogue|INTRODUCTION|Introduction|PREFACE|Preface|CONCLUSION|Conclusion|APPENDIX|Appendix)(?:[:.\s\-–—]+(.*))?$",
            re.IGNORECASE,
        ),
        1,  # Level 1
    ),
]


def extract_document_pages(
    doc: pymupdf.Document, ocr_min_chars: int = 50
) -> Tuple[List[Dict], List[PageExtractionReport], List[int], List[int]]:
    """Extract text page-by-page while preserving 1-based page numbers.

    Detects empty or image-only pages to identify when OCR is required.
    Returns:
        (pages_data, page_reports, ocr_needed_pages, empty_pages)
    """
    pages_data: List[Dict] = []
    page_reports: List[PageExtractionReport] = []
    ocr_needed_pages: List[int] = []
    empty_pages: List[int] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-based page number

        # Extract text content
        raw_text = page.get_text("text") or ""
        cleaned_text = raw_text.strip()
        char_count = len(cleaned_text)
        word_count = len(cleaned_text.split())

        # Inspect embedded images
        image_list = page.get_images(full=True)
        image_count = len(image_list)
        has_images = image_count > 0

        # OCR necessity detection
        needs_ocr = False
        is_empty = False
        reason = "normal"

        if char_count < ocr_min_chars:
            if has_images:
                needs_ocr = True
                reason = "image_only"
                ocr_needed_pages.append(page_num)
            elif char_count == 0:
                is_empty = True
                reason = "empty_page"
                empty_pages.append(page_num)
            else:
                reason = "sparse_text"

        report = PageExtractionReport(
            page_number=page_num,
            char_count=char_count,
            word_count=word_count,
            has_images=has_images,
            image_count=image_count,
            is_empty=is_empty,
            needs_ocr=needs_ocr,
            reason=reason,
        )
        page_reports.append(report)

        pages_data.append(
            {
                "page_number": page_num,
                "text": raw_text,
                "cleaned_text": cleaned_text,
                "needs_ocr": needs_ocr,
            }
        )

    return pages_data, page_reports, ocr_needed_pages, empty_pages


def detect_document_headings(
    doc: pymupdf.Document, pages_data: List[Dict]
) -> List[HeadingInfo]:
    """Detect chapter and section headings.

    First checks Table of Contents (TOC/bookmarks) for high-confidence headings.
    Augments with pattern-based heuristics, preserving uncertainty.
    """
    headings: List[HeadingInfo] = []
    seen_titles: Dict[Tuple[int, str], HeadingInfo] = {}

    # Pass 1: Table of Contents (high confidence)
    try:
        toc = doc.get_toc()  # [[lvl, title, page_num], ...]
        if toc:
            for item in toc:
                if len(item) >= 3:
                    lvl, title, page_num = item[0], item[1], item[2]
                    title_clean = str(title).strip()
                    if title_clean and page_num > 0:
                        h = HeadingInfo(
                            title=title_clean,
                            level=int(lvl),
                            page_number=int(page_num),
                            confidence="high",
                            source="toc",
                        )
                        key = (int(page_num), title_clean.lower())
                        seen_titles[key] = h
                        headings.append(h)
    except Exception:
        pass

    # Pass 2: Layout / regex heuristics for lines on each page
    for p_data in pages_data:
        page_num = p_data["page_number"]
        page_text = p_data["cleaned_text"]
        if not page_text:
            continue

        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        for line in lines:
            # Headings are generally concise
            if len(line) > 120:
                continue

            for pattern, level in HEADING_PATTERNS:
                if pattern.match(line):
                    clean_heading = line.strip()
                    key = (page_num, clean_heading.lower())
                    if key not in seen_titles:
                        h = HeadingInfo(
                            title=clean_heading,
                            level=level,
                            page_number=page_num,
                            confidence="heuristic",
                            source="layout_heuristic",
                        )
                        seen_titles[key] = h
                        headings.append(h)
                    break

    # Sort headings chronologically by page number
    headings.sort(key=lambda x: (x.page_number, x.level))
    return headings


def split_into_sentences(text: str) -> List[str]:
    """Split paragraph into sentence units using boundary punctuation heuristics."""
    # Split on sentence terminals followed by whitespace, keeping delimiters
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text]


def create_overlapping_chunks(
    book_id: str,
    pages_data: List[Dict],
    headings: List[HeadingInfo],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[ChunkMetadata]:
    """Create meaningful overlapping text chunks preserving sentences, paragraphs,

    original 1-based page numbers, and active chapter/section headings.
    """
    # Build page-to-headings mapping
    page_heading_map: Dict[int, List[HeadingInfo]] = {}
    for h in headings:
        page_heading_map.setdefault(h.page_number, []).append(h)

    current_chapter: Optional[str] = None
    current_section: Optional[str] = None
    current_confidence: Optional[str] = None

    chunks: List[ChunkMetadata] = []
    chunk_index = 0

    # Buffer holds units: List of tuples (text_unit, page_number)
    buffer: List[Tuple[str, int]] = []
    buffer_len = 0

    def flush_chunk(force_emit: bool = False):
        nonlocal buffer, buffer_len, chunk_index
        if not buffer:
            return

        chunk_text = " ".join(item[0] for item in buffer).strip()
        if not chunk_text:
            buffer = []
            buffer_len = 0
            return

        min_page = min(item[1] for item in buffer)
        max_page = max(item[1] for item in buffer)

        chunk = ChunkMetadata(
            chunk_id=f"{book_id}_c{chunk_index:04d}",
            book_id=book_id,
            chunk_index=chunk_index,
            text=chunk_text,
            page_start=min_page,
            page_end=max_page,
            chapter=current_chapter,
            section=current_section,
            heading_confidence=current_confidence,
            char_count=len(chunk_text),
            word_count=len(chunk_text.split()),
        )
        chunks.append(chunk)
        chunk_index += 1

        if force_emit:
            buffer = []
            buffer_len = 0
            return

        # Prepare overlapping buffer for next chunk
        # Keep trailing units up to target overlap size
        new_buffer: List[Tuple[str, int]] = []
        accumulated_overlap = 0

        for item in reversed(buffer):
            unit_len = len(item[0]) + 1
            if (
                accumulated_overlap + unit_len <= chunk_overlap
                or not new_buffer
            ):
                new_buffer.insert(0, item)
                accumulated_overlap += unit_len
            else:
                break

        # If the overlap includes the entire buffer, drop at least one item to guarantee forward progress
        if len(new_buffer) >= len(buffer):
            new_buffer = new_buffer[1:]

        buffer = new_buffer
        buffer_len = sum(len(item[0]) + 1 for item in buffer)

    for p in pages_data:
        page_num = p["page_number"]
        page_text = p["cleaned_text"]

        # Update active headings on this page
        if page_num in page_heading_map:
            for h in page_heading_map[page_num]:
                if h.level == 1:
                    # Flush previous buffer at chapter transition so each chapter starts cleanly
                    if buffer:
                        flush_chunk(force_emit=True)
                    current_chapter = h.title
                    current_confidence = h.confidence
                elif h.level >= 2:
                    current_section = h.title
                    current_confidence = h.confidence

        if not page_text:
            continue

        # Split page into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]

        for para in paragraphs:
            # If paragraph fits comfortably, treat as unit; otherwise split into sentences
            if len(para) <= chunk_size // 2:
                units = [para]
            else:
                units = split_into_sentences(para)

            for unit in units:
                unit_len = len(unit) + 1  # count space separator

                # If adding unit exceeds chunk_size and buffer already has content, emit chunk
                if buffer and (buffer_len + unit_len > chunk_size):
                    flush_chunk()

                buffer.append((unit, page_num))
                buffer_len += unit_len

    # Flush any remaining buffer text at document end
    if buffer:
        flush_chunk(force_emit=True)

    return chunks


def process_pdf_bytes(
    book_id: str,
    original_filename: str,
    pdf_bytes: bytes,
    file_hash: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    ocr_min_chars: int = 50,
) -> Tuple[List[ChunkMetadata], List[PageExtractionReport], List[HeadingInfo], Dict]:
    """Orchestrate end-to-end PDF processing:

    Validates document structure, extracts pages, detects OCR necessity,
    identifies headings, and produces structured overlapping chunks.
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(
            f"Invalid or corrupted PDF file. PyMuPDF could not read document: {e}"
        ) from e

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise ValueError("PDF document contains 0 pages.")

    try:
        # Step 1: Extract pages and assess OCR necessity
        pages_data, page_reports, ocr_needed_pages, empty_pages = (
            extract_document_pages(doc, ocr_min_chars=ocr_min_chars)
        )

        # Step 2: Detect headings
        headings = detect_document_headings(doc, pages_data)

        # Step 3: Create overlapping chunks
        chunks = create_overlapping_chunks(
            book_id=book_id,
            pages_data=pages_data,
            headings=headings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Determine overall extraction status
        needs_ocr = len(ocr_needed_pages) > 0
        if needs_ocr:
            extraction_status = "completed_with_warnings"
            ocr_warning = (
                f"OCR is recommended for {len(ocr_needed_pages)} page(s) "
                f"({ocr_needed_pages}) which appear to contain images or scans with minimal extractable text."
            )
        else:
            extraction_status = "completed"
            ocr_warning = None

        stats = {
            "page_count": page_count,
            "chunk_count": len(chunks),
            "needs_ocr": needs_ocr,
            "ocr_warning": ocr_warning,
            "ocr_needed_pages": ocr_needed_pages,
            "empty_pages": empty_pages,
            "extraction_status": extraction_status,
        }

        return chunks, page_reports, headings, stats
    finally:
        doc.close()
