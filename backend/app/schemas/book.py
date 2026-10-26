"""Pydantic schemas for PDF ingestion, books, chunks, and extraction metadata."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class HeadingInfo(BaseModel):
    """Detected chapter or section heading with confidence tracking."""

    title: str = Field(..., description="Extracted heading title")
    level: int = Field(default=1, description="Hierarchical level: 1=Chapter/Part, 2=Section/Subheading")
    page_number: int = Field(..., description="Original 1-based page number where heading appears")
    confidence: str = Field(
        ...,
        description="Confidence classification: 'high' (from PDF TOC/bookmarks) or 'heuristic' (font/pattern detection)",
    )
    source: str = Field(
        ...,
        description="Source of detection: 'toc' or 'layout_heuristic'",
    )


class PageExtractionReport(BaseModel):
    """Extraction analysis and OCR necessity check for an individual page."""

    page_number: int = Field(..., description="Original 1-based page number")
    char_count: int = Field(..., description="Number of text characters extracted")
    word_count: int = Field(..., description="Number of words extracted")
    has_images: bool = Field(..., description="Whether the page contains raster or vector images")
    image_count: int = Field(default=0, description="Number of embedded images detected on this page")
    is_empty: bool = Field(default=False, description="True if no text and no images found")
    needs_ocr: bool = Field(
        default=False,
        description="True if page contains images with minimal/zero extractable text (likely scanned)",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Classification reason: 'normal', 'image_only', 'empty_page', or 'sparse_text'",
    )


class ChunkMetadata(BaseModel):
    """Meaningful overlapping text chunk with associated document metadata."""

    chunk_id: str = Field(..., description="Stable deterministic chunk identifier, e.g. b_xyz_c0001")
    book_id: str = Field(..., description="ID of the parent book")
    chunk_index: int = Field(..., description="0-based sequence index of the chunk")
    text: str = Field(..., description="Extracted and cleaned chunk text content")
    page_start: int = Field(..., description="1-based starting page number of the chunk text")
    page_end: int = Field(..., description="1-based ending page number of the chunk text")
    chapter: Optional[str] = Field(default=None, description="Current chapter title or null if unassigned")
    section: Optional[str] = Field(default=None, description="Current section title or null if unassigned")
    heading_confidence: Optional[str] = Field(
        default=None,
        description="Confidence of the associated heading: 'high', 'heuristic', or null",
    )
    char_count: int = Field(..., description="Total character count of the chunk")
    word_count: int = Field(..., description="Total word count of the chunk")


class BookMetadata(BaseModel):
    """Full metadata stored for an ingested book."""

    book_id: str = Field(..., description="Stable unique book identifier")
    original_filename: str = Field(..., description="Original uploaded filename")
    safe_filename: str = Field(..., description="Sanitized safe filename used in metadata")
    file_hash: str = Field(..., description="SHA-256 hash of the uploaded PDF file content")
    file_size_bytes: int = Field(..., description="Size of the uploaded PDF file in bytes")
    page_count: int = Field(..., description="Total number of pages in the PDF document")
    chunk_count: int = Field(..., description="Total number of text chunks generated")
    extraction_status: str = Field(
        ...,
        description="Status: 'completed' (full text), 'completed_with_warnings' (OCR required), 'failed'",
    )
    needs_ocr: bool = Field(
        default=False,
        description="Flag indicating if one or more pages appear scanned or image-only",
    )
    ocr_warning: Optional[str] = Field(
        default=None,
        description="Human-readable warning explaining which pages require OCR",
    )
    ocr_needed_pages: List[int] = Field(
        default_factory=list,
        description="List of 1-based page numbers requiring OCR",
    )
    empty_pages: List[int] = Field(
        default_factory=list,
        description="List of 1-based page numbers that are completely empty",
    )
    headings: List[HeadingInfo] = Field(
        default_factory=list,
        description="Extracted or detected document headings",
    )
    created_at: datetime = Field(..., description="UTC timestamp of ingestion")
    indexing_status: str = Field(
        default="not_indexed",
        description="Vector & lexical indexing status: 'not_indexed', 'indexing', 'indexed', 'failed'",
    )
    indexed_chunk_count: int = Field(
        default=0,
        description="Number of chunks successfully indexed in Qdrant and BM25",
    )
    indexing_error: Optional[str] = Field(
        default=None,
        description="Detailed failure message if indexing encountered an error",
    )
    indexed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when indexing completed",
    )


class BookSummary(BaseModel):
    """Summary item for book catalog listing."""

    book_id: str
    original_filename: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    extraction_status: str
    needs_ocr: bool
    ocr_warning: Optional[str] = None
    created_at: datetime
    indexing_status: str = "not_indexed"
    indexed_chunk_count: int = 0


class BookListResponse(BaseModel):
    """Response returned by GET /api/books."""

    total: int = Field(..., description="Total number of uploaded books")
    books: List[BookSummary] = Field(..., description="List of book summaries")


class BookDetailResponse(BaseModel):
    """Detailed response returned on upload and GET /api/books/{book_id}."""

    book: BookMetadata = Field(..., description="Full book metadata and extraction report")
    page_reports: Optional[List[PageExtractionReport]] = Field(
        default=None,
        description="Page-by-page extraction details",
    )
    sample_chunks: Optional[List[ChunkMetadata]] = Field(
        default=None,
        description="Preview of the first few chunks (up to 3)",
    )


class ChunkListResponse(BaseModel):
    """Paginated response returned by GET /api/books/{book_id}/chunks."""

    book_id: str
    total_chunks: int
    skip: int
    limit: int
    chunks: List[ChunkMetadata]


class BookDeleteResponse(BaseModel):
    """Response returned upon deleting a book."""

    book_id: str
    deleted: bool
    message: str
