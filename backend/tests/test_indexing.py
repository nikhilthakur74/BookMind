"""Tests for BookMind indexing service, status tracking, idempotency, and failure handling."""

import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.book import BookMetadata, ChunkMetadata
from app.schemas.retrieval import IndexingStatusResponse
from app.services import (
    bm25_service,
    embedding_service,
    indexing_service,
    qdrant_service,
    storage_service,
)
from app.services.embedding_service import DeterministicMockEmbeddingService
from qdrant_client import QdrantClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Set up isolated temporary directories and mock embedding/Qdrant services."""
    # Point books_dir and qdrant storage to tmp_path
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "QDRANT_STORAGE_DIR", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")

    # In-memory Qdrant client for zero lock contention and isolated tests
    mem_qdrant_client = QdrantClient(location=":memory:")
    test_qdrant_svc = qdrant_service.QdrantService(
        client=mem_qdrant_client,
        collection_name="test_indexing_collection",
        dimension=1024,
    )
    test_emb_svc = DeterministicMockEmbeddingService(dimension=1024)
    test_bm25_svc = bm25_service.BM25Service(k1=1.5, b=0.75)

    test_idx_svc = indexing_service.IndexingService(
        qdrant_service=test_qdrant_svc,
        embedding_service=test_emb_svc,
        bm25_service=test_bm25_svc,
    )

    monkeypatch.setattr(qdrant_service, "get_qdrant_service", lambda *args, **kwargs: test_qdrant_svc)
    monkeypatch.setattr(embedding_service, "get_embedding_service", lambda *args, **kwargs: test_emb_svc)
    monkeypatch.setattr(bm25_service, "get_bm25_service", lambda *args, **kwargs: test_bm25_svc)
    monkeypatch.setattr(indexing_service, "get_indexing_service", lambda *args, **kwargs: test_idx_svc)

    yield {
        "qdrant": test_qdrant_svc,
        "embedding": test_emb_svc,
        "bm25": test_bm25_svc,
        "indexing": test_idx_svc,
    }


def create_sample_book(book_id: str, num_chunks: int = 5) -> tuple:
    """Helper creating a sample stored book with extracted chunks."""
    from datetime import datetime, timezone

    meta = BookMetadata(
        book_id=book_id,
        original_filename=f"{book_id}.pdf",
        safe_filename=f"{book_id}.pdf",
        file_hash=f"hash_{book_id}",
        file_size_bytes=1024 * 100,
        page_count=num_chunks,
        chunk_count=num_chunks,
        extraction_status="completed",
        needs_ocr=False,
        headings=[],
        created_at=datetime.now(timezone.utc),
        indexing_status="not_indexed",
        indexed_chunk_count=0,
    )

    chunks = []
    for i in range(num_chunks):
        chunks.append(
            ChunkMetadata(
                chunk_id=f"{book_id}_c{i:04d}",
                book_id=book_id,
                chunk_index=i,
                text=f"Sample text passage for {book_id} chapter {i + 1}. Contains semantic concept number {i}.",
                page_start=i + 1,
                page_end=i + 1,
                chapter=f"Chapter {i + 1}",
                section=f"Section {i + 1}.1",
                heading_confidence="high",
                char_count=80,
                word_count=12,
            )
        )

    storage_service.save_book_assets(
        book_id=book_id,
        pdf_bytes=b"%PDF-sample-data",
        metadata=meta,
        chunks=chunks,
        page_reports=[],
    )
    return meta, chunks


def test_indexing_workflow_success(setup_test_environment):
    """Test full indexing lifecycle: status transition to 'indexed', Qdrant storage, and BM25 index."""
    idx_svc = setup_test_environment["indexing"]
    qdrant_svc = setup_test_environment["qdrant"]

    book_id = "book_workflow_01"
    create_sample_book(book_id, num_chunks=4)

    # Initial status
    status_initial = idx_svc.get_status(book_id)
    assert status_initial.indexing_status == "not_indexed"
    assert status_initial.indexed_chunk_count == 0

    # Execute indexing
    updated_meta = idx_svc.index_book(book_id)
    assert updated_meta.indexing_status == "indexed"
    assert updated_meta.indexed_chunk_count == 4
    assert updated_meta.indexed_at is not None
    assert updated_meta.indexing_error is None

    # Verify Qdrant points count
    points_count = qdrant_svc.count_book_vectors(book_id)
    assert points_count == 4

    # Verify BM25 index file exists
    bm25_file = storage_service.get_book_dir(book_id) / "bm25_index.json"
    assert bm25_file.exists()


def test_indexing_idempotency_prevents_duplicate_vectors(setup_test_environment):
    """Test that repeatedly indexing the same book avoids duplicate vector points."""
    idx_svc = setup_test_environment["indexing"]
    qdrant_svc = setup_test_environment["qdrant"]

    book_id = "book_idempotent_02"
    create_sample_book(book_id, num_chunks=5)

    # First index pass
    idx_svc.index_book(book_id)
    assert qdrant_svc.count_book_vectors(book_id) == 5

    # Second index pass with force=True (re-indexing)
    idx_svc.index_book(book_id, force=True)
    # Point count MUST remain 5, not 10!
    assert qdrant_svc.count_book_vectors(book_id) == 5


def test_indexing_failure_handling(setup_test_environment, monkeypatch):
    """Test that embedding or database errors transition status to 'failed' and preserve error message."""
    idx_svc = setup_test_environment["indexing"]

    book_id = "book_fail_03"
    create_sample_book(book_id, num_chunks=3)

    # Simulate embedding failure
    def failing_embed(*args, **kwargs):
        raise RuntimeError("Simulated GPU out-of-memory error during embedding")

    monkeypatch.setattr(idx_svc._embedding_service, "embed_documents", failing_embed)

    with pytest.raises(RuntimeError) as exc_info:
        idx_svc.index_book(book_id)

    assert "Simulated GPU out-of-memory error" in str(exc_info.value)

    # Verify status is recorded as failed
    status = idx_svc.get_status(book_id)
    assert status.indexing_status == "failed"
    assert "Simulated GPU out-of-memory error" in status.indexing_error


def test_indexing_status_api_endpoint(setup_test_environment):
    """Test GET /api/books/{book_id}/indexing-status endpoint."""
    book_id = "book_api_status_04"
    create_sample_book(book_id, num_chunks=3)

    resp = client.get(f"/api/books/{book_id}/indexing-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["book_id"] == book_id
    assert data["indexing_status"] == "not_indexed"
    assert data["total_chunks"] == 3

    # Trigger indexing via API
    index_resp = client.post(f"/api/books/{book_id}/index")
    assert index_resp.status_code == 200
    meta_data = index_resp.json()
    assert meta_data["indexing_status"] == "indexed"
    assert meta_data["indexed_chunk_count"] == 3

    # Check status again
    resp2 = client.get(f"/api/books/{book_id}/indexing-status")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["indexing_status"] == "indexed"
    assert data2["indexed_chunk_count"] == 3
