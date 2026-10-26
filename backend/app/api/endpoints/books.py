"""Books API endpoints for uploading, listing, retrieving, and deleting ingested books."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.core.config import settings
from app.schemas.book import (
    BookDeleteResponse,
    BookDetailResponse,
    BookListResponse,
    BookMetadata,
    ChunkListResponse,
)
from app.schemas.retrieval import (
    IndexingStatusResponse,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResponse,
)
from app.schemas.hierarchy import (
    BookHierarchy,
    HierarchyGenerateRequest,
    HierarchyStatusResponse,
)
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
)
from app.services import (
    bm25_service,
    chat_service,
    conversation_service,
    hierarchy_service,
    indexing_service,
    pdf_service,
    qdrant_service,
    retrieval_service,
    storage_service,
)

router = APIRouter(prefix="/books", tags=["Books"])


@router.post(
    "/upload",
    response_model=BookDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest a PDF Book",
    description=(
        "Upload a PDF book. Validates file integrity and size, extracts text page by page, "
        "detects OCR requirements for scanned/image pages, identifies chapter headings heuristically, "
        "and generates sentence/paragraph-aware overlapping text chunks."
    ),
)
async def upload_book(file: UploadFile = File(...)) -> BookDetailResponse:
    """Handle PDF upload, parsing, OCR detection, chunking, and local asset storage."""
    # 1. Validate file presence and filename extension
    filename = file.filename or "unknown.pdf"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Only PDF documents (.pdf) are permitted.",
        )

    # 2. Read file content with size limit enforcement
    max_size = settings.MAX_UPLOAD_SIZE_BYTES
    content_chunks = []
    total_bytes = 0

    chunk_size = 1024 * 1024  # 1MB buffer
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum limit of {max_size // (1024 * 1024)} MB.",
            )
        content_chunks.append(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    pdf_bytes = b"".join(content_chunks)

    # 3. Validate PDF magic bytes (%PDF-)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: file header does not match PDF standard (%PDF-).",
        )

    # 4. Check for duplicate upload via SHA-256 hash
    file_hash = storage_service.compute_file_hash(pdf_bytes)
    existing_book = storage_service.find_book_by_hash(file_hash)
    if existing_book:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Duplicate book detected: an identical file has already been ingested.",
                "existing_book_id": existing_book.book_id,
                "original_filename": existing_book.original_filename,
                "file_hash": existing_book.file_hash,
            },
        )

    # 5. Generate stable, unique book ID and sanitized filename
    book_id = f"book_{file_hash[:8]}_{uuid.uuid4().hex[:6]}"
    safe_name = storage_service.sanitize_filename(filename)

    # 6. Parse and extract PDF using PyMuPDF
    try:
        chunks, page_reports, headings, stats = pdf_service.process_pdf_bytes(
            book_id=book_id,
            original_filename=filename,
            pdf_bytes=pdf_bytes,
            file_hash=file_hash,
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
            ocr_min_chars=settings.OCR_THRESHOLD_MIN_CHARS_PER_PAGE,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during PDF extraction: {e}",
        ) from e

    # 7. Assemble metadata record
    metadata = BookMetadata(
        book_id=book_id,
        original_filename=filename,
        safe_filename=safe_name,
        file_hash=file_hash,
        file_size_bytes=total_bytes,
        page_count=stats["page_count"],
        chunk_count=stats["chunk_count"],
        extraction_status=stats["extraction_status"],
        needs_ocr=stats["needs_ocr"],
        ocr_warning=stats["ocr_warning"],
        ocr_needed_pages=stats["ocr_needed_pages"],
        empty_pages=stats["empty_pages"],
        headings=headings,
        created_at=datetime.now(timezone.utc),
    )

    # 8. Persist book assets locally
    try:
        storage_service.save_book_assets(
            book_id=book_id,
            pdf_bytes=pdf_bytes,
            metadata=metadata,
            chunks=chunks,
            page_reports=page_reports,
        )
    except Exception as e:
        storage_service.cleanup_book_dir(book_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist book storage assets: {e}",
        ) from e

    return BookDetailResponse(
        book=metadata,
        page_reports=page_reports,
        sample_chunks=chunks[:3] if chunks else [],
    )


@router.get(
    "",
    response_model=BookListResponse,
    summary="List All Ingested Books",
    description="Returns a list of all ingested books with page counts, chunk counts, and OCR status.",
)
def list_books() -> BookListResponse:
    """Retrieve catalog of all ingested books."""
    books = storage_service.list_all_books()
    return BookListResponse(total=len(books), books=books)


@router.get(
    "/{book_id}",
    response_model=BookDetailResponse,
    summary="Get Book Details",
    description="Retrieve full metadata, heading structure, and page reports for a specific book.",
)
def get_book(book_id: str) -> BookDetailResponse:
    """Retrieve detailed book record by book ID."""
    metadata = storage_service.get_book_metadata(book_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found.",
        )

    page_reports = storage_service.get_book_page_reports(book_id)
    chunks = storage_service.get_book_chunks(book_id)

    return BookDetailResponse(
        book=metadata,
        page_reports=page_reports,
        sample_chunks=chunks[:3] if chunks else [],
    )


@router.get(
    "/{book_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get Book Chunks",
    description="Retrieve paginated text chunks with metadata for a specific book.",
)
def get_book_chunks(
    book_id: str,
    skip: int = Query(default=0, ge=0, description="Offset index to start returning chunks"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of chunks to return"),
) -> ChunkListResponse:
    """Retrieve chunks belonging to a book with pagination."""
    metadata = storage_service.get_book_metadata(book_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found.",
        )

    all_chunks = storage_service.get_book_chunks(book_id) or []
    total = len(all_chunks)
    sliced = all_chunks[skip : skip + limit]

    return ChunkListResponse(
        book_id=book_id,
        total_chunks=total,
        skip=skip,
        limit=limit,
        chunks=sliced,
    )


@router.delete(
    "/{book_id}",
    response_model=BookDeleteResponse,
    summary="Delete a Book",
    description="Deletes a book, its original PDF, chunks, and metadata from local storage.",
)
def delete_book(book_id: str) -> BookDeleteResponse:
    """Delete a book and all associated files."""
    metadata = storage_service.get_book_metadata(book_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found.",
        )

    # Clean up Qdrant vectors, BM25 index, hierarchy, and conversations
    try:
        qdrant_service.get_qdrant_service().delete_book_vectors(book_id)
        bm25_service.get_bm25_service().delete_book_index(book_id)
        hierarchy_service.get_hierarchy_service().delete_hierarchy(book_id)
        conversation_service.get_conversation_service().delete_all_book_conversations(book_id)
    except Exception:
        pass

    success = storage_service.delete_book(book_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete book files for '{book_id}'.",
        )

    return BookDeleteResponse(
        book_id=book_id,
        deleted=True,
        message=f"Book '{metadata.original_filename}' ({book_id}) and all extracted chunks were deleted.",
    )


@router.post(
    "/{book_id}/index",
    response_model=BookMetadata,
    summary="Index Book Chunks into Vector & Lexical Stores",
    description="Generates BGE-M3 dense vectors, stores them in Qdrant, and builds an Okapi BM25 index.",
)
def index_book(
    book_id: str,
    force: bool = Query(default=False, description="Force re-indexing even if already indexed"),
) -> BookMetadata:
    """Trigger embedding generation, Qdrant vector upsert, and BM25 index construction."""
    try:
        service = indexing_service.get_indexing_service()
        return service.index_book(book_id=book_id, force=force)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed for book '{book_id}': {e}",
        ) from e


@router.get(
    "/{book_id}/indexing-status",
    response_model=IndexingStatusResponse,
    summary="Get Book Indexing Status",
    description="Check whether a book has been indexed into Qdrant and BM25 stores.",
)
def get_indexing_status(book_id: str) -> IndexingStatusResponse:
    """Retrieve indexing status and indexed chunk count."""
    try:
        service = indexing_service.get_indexing_service()
        return service.get_status(book_id=book_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post(
    "/{book_id}/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve Relevant Book Passages",
    description=(
        "Execute hybrid (Reciprocal Rank Fusion), dense semantic (BGE-M3), "
        "or lexical (BM25) search strictly isolated to the specified book."
    ),
)
def retrieve_book_passages(book_id: str, request: RetrievalRequest) -> RetrievalResponse:
    """Execute isolated retrieval against a specific book catalog."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found.",
        )

    service = retrieval_service.get_retrieval_service()

    try:
        if request.mode == RetrievalMode.SEMANTIC:
            results = service.retrieve_semantic(
                query=request.query,
                book_id=book_id,
                top_k=request.top_k,
            )
        elif request.mode == RetrievalMode.BM25:
            results = service.retrieve_bm25(
                query=request.query,
                book_id=book_id,
                top_k=request.top_k,
            )
        else:
            results = service.retrieve_hybrid(
                query=request.query,
                book_id=book_id,
                top_k=request.top_k,
                semantic_top_k=request.semantic_top_k,
                bm25_top_k=request.bm25_top_k,
                rrf_k=request.rrf_k,
            )

        return RetrievalResponse(
            book_id=book_id,
            query=request.query,
            mode=request.mode.value,
            top_k=request.top_k,
            total_found=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {e}",
        ) from e


@router.post(
    "/{book_id}/hierarchy/generate",
    response_model=BookHierarchy,
    summary="Generate or Resume Hierarchical Book Understanding",
    description="Synthesizes Level 2 Section Summaries, Level 3 Chapter Cruxes, and Level 4 Book Thesis.",
)
async def generate_hierarchy(
    book_id: str,
    request: Optional[HierarchyGenerateRequest] = None,
) -> BookHierarchy:
    """Trigger or resume hierarchical synthesis for a book."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found.",
        )

    force_restart = request.force_restart if request else False
    service = hierarchy_service.get_hierarchy_service()
    try:
        return await service.generate_hierarchy(book_id=book_id, force_restart=force_restart)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        err_msg = str(e)
        if "Ollama" in err_msg or "not installed" in err_msg or "unreachable" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Hierarchical generation unavailable: {e}",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hierarchical generation error: {e}",
        ) from e


@router.get(
    "/{book_id}/hierarchy",
    response_model=BookHierarchy,
    summary="Get Hierarchical Book Understanding",
    description="Retrieve all four knowledge levels for an ingested book.",
)
def get_hierarchy(book_id: str) -> BookHierarchy:
    """Retrieve full four-level hierarchical knowledge."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book with ID '{book_id}' was not found.",
        )

    service = hierarchy_service.get_hierarchy_service()
    hierarchy = service.load_hierarchy(book_id)
    if not hierarchy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hierarchy for book '{book_id}' has not been generated yet.",
        )
    return hierarchy


@router.get(
    "/{book_id}/hierarchy/status",
    response_model=HierarchyStatusResponse,
    summary="Get Hierarchical Generation Status",
    description="Inspect progress and completion counts of the hierarchical summarization pipeline.",
)
def get_hierarchy_status(book_id: str) -> HierarchyStatusResponse:
    """Check progress of hierarchical summarization."""
    try:
        service = hierarchy_service.get_hierarchy_service()
        return service.get_status(book_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post(
    "/{book_id}/chat",
    response_model=ChatResponse,
    summary="Ask a Grounded Question About a Book",
    description="Conversational RAG combining hybrid retrieval, hierarchical summaries, and structured citations.",
)
async def chat_with_book(book_id: str, request: ChatRequest) -> ChatResponse:
    """Execute conversational inquiry against a book with verifiable evidence citations."""
    try:
        service = chat_service.get_chat_service()
        return await service.answer_question(book_id=book_id, request=request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except RuntimeError as e:
        err_msg = str(e)
        if (
            "GROQ_API_KEY" in err_msg
            or "API key is not configured" in err_msg
            or "Ollama" in err_msg
            or "not installed" in err_msg
            or "unreachable" in err_msg
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=err_msg,
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat synthesis error: {e}",
        ) from e


@router.get(
    "/{book_id}/conversations",
    response_model=List[ConversationHistory],
    summary="List Conversations for a Book",
    description="List all active conversation threads associated with a book.",
)
def list_conversations(book_id: str) -> List[ConversationHistory]:
    """List conversation threads for a book."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' was not found.")
    conv_svc = conversation_service.get_conversation_service()
    return conv_svc.list_conversations(book_id)


@router.get(
    "/{book_id}/conversations/{conversation_id}",
    response_model=ConversationHistory,
    summary="Get Conversation Thread",
    description="Retrieve a specific conversation history thread for a book.",
)
def get_conversation(book_id: str, conversation_id: str) -> ConversationHistory:
    """Retrieve full conversation history for a thread."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' was not found.")
    conv_svc = conversation_service.get_conversation_service()
    conv = conv_svc.get_conversation(book_id, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' was not found.",
        )
    return conv

