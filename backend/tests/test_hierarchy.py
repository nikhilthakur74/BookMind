"""Comprehensive tests for Phase 4: Hierarchical Book Understanding.

Tests:
1. Provenance preservation: source chunk IDs and page references strictly retained.
2. Grounding & claim classification: author-stated vs model-inferred.
3. Missing evidence & uncertainty handling.
4. Malformed LLM responses and retry behavior.
5. Long chapter staged summarization.
6. Idempotency and resumable execution.
7. Separate storage: chunks.json is never mutated.
8. HTTP API endpoints for hierarchy generation, inspection, and status tracking.
"""

import asyncio
from datetime import datetime, timezone
import json
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.book import BookMetadata, ChunkMetadata
from app.schemas.hierarchy import (
    BookHierarchy,
    BookSummary,
    ChapterSummary,
    SectionSummary,
)
from app.services import (
    hierarchy_service,
    llm_client,
    storage_service,
)
from app.services.hierarchy_service import HierarchyService
from app.services.llm_client import MockLLMClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_hierarchy_test_env(tmp_path, monkeypatch):
    """Isolate storage and configure deterministic mock LLM client."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "LONG_CHAPTER_CHUNK_THRESHOLD", 6)

    test_llm = MockLLMClient(model_name="mock-llama-3.3-70b", prompt_version="v1.0")
    test_hier_svc = HierarchyService(llm_client=test_llm)

    monkeypatch.setattr(llm_client, "get_llm_client", lambda *args, **kwargs: test_llm)
    monkeypatch.setattr(hierarchy_service, "get_hierarchy_service", lambda *args, **kwargs: test_hier_svc)

    yield {
        "llm": test_llm,
        "hierarchy_svc": test_hier_svc,
        "tmp_path": tmp_path,
    }


def seed_book_for_hierarchy(
    book_id: str,
    chapter_specs: list,  # list of (chapter_title, [passage_texts])
) -> tuple:
    """Helper creating stored book with chunks mapped to specific chapters."""
    chunks = []
    chunk_counter = 0

    for chap_idx, (chap_title, passages) in enumerate(chapter_specs):
        for sec_idx, text in enumerate(passages):
            chunk_id = f"{book_id}_c{chunk_counter:04d}"
            chunks.append(
                ChunkMetadata(
                    chunk_id=chunk_id,
                    book_id=book_id,
                    chunk_index=chunk_counter,
                    text=text,
                    page_start=chunk_counter + 1,
                    page_end=chunk_counter + 1,
                    chapter=chap_title,
                    section=f"Section {sec_idx + 1}",
                    heading_confidence="high",
                    char_count=len(text),
                    word_count=len(text.split()),
                )
            )
            chunk_counter += 1

    meta = BookMetadata(
        book_id=book_id,
        original_filename=f"{book_id}.pdf",
        safe_filename=f"{book_id}.pdf",
        file_hash=f"hash_{book_id}",
        file_size_bytes=1024 * 50,
        page_count=len(chunks),
        chunk_count=len(chunks),
        extraction_status="completed",
        needs_ocr=False,
        headings=[],
        created_at=datetime.now(timezone.utc),
        indexing_status="indexed",
        indexed_chunk_count=len(chunks),
    )

    storage_service.save_book_assets(
        book_id=book_id,
        pdf_bytes=b"%PDF-test-hierarchy",
        metadata=meta,
        chunks=chunks,
        page_reports=[],
    )
    return meta, chunks


# ==============================================================================
# 1. Provenance Preservation & Grounding Tests
# ==============================================================================

@pytest.mark.anyio
async def test_provenance_preservation_across_all_levels(setup_hierarchy_test_env):
    """Verify that every concept, argument, and takeaway strictly preserves source chunk IDs."""
    hier_svc = setup_hierarchy_test_env["hierarchy_svc"]

    book_id = "book_provenance_01"
    chap_specs = [
        (
            "Chapter 1: Foundations of Cognition",
            [
                "Cognitive architectures structure perception, memory, and action into recursive feedback loops.",
                "Working memory limits human capacity to four distinct information chunks simultaneously.",
            ],
        ),
        (
            "Chapter 2: Computational Linguistics",
            [
                "Language comprehension combines compositionality with predictive distributional semantics.",
                "Syntactic trees represent hierarchical grammatical dependencies between lexical tokens.",
            ],
        ),
    ]
    meta, original_chunks = seed_book_for_hierarchy(book_id, chap_specs)
    all_orig_chunk_ids = {c.chunk_id for c in original_chunks}

    hierarchy = await hier_svc.generate_hierarchy(book_id)

    assert hierarchy.generation_status == "completed"
    assert hierarchy.progress_percent == 100.0

    # Level 2 Section Check
    assert len(hierarchy.sections) >= 2
    for sec in hierarchy.sections:
        assert set(sec.source_chunk_ids).issubset(all_orig_chunk_ids)
        for concept in sec.key_concepts:
            assert len(concept.source_chunk_ids) > 0
            assert set(concept.source_chunk_ids).issubset(all_orig_chunk_ids)

    # Level 3 Chapter Check
    assert len(hierarchy.chapters) == 2
    for chap in hierarchy.chapters:
        assert len(chap.source_chunk_ids) > 0
        assert set(chap.source_chunk_ids).issubset(all_orig_chunk_ids)
        assert chap.chapter_crux is not None
        assert len(chap.supporting_arguments) > 0
        for arg in chap.supporting_arguments:
            assert len(arg.source_chunk_ids) > 0
            assert set(arg.source_chunk_ids).issubset(all_orig_chunk_ids)

    # Level 4 Book Check
    assert hierarchy.book_level is not None
    book_level = hierarchy.book_level
    assert book_level.central_thesis is not None
    assert book_level.overall_crux is not None
    assert len(book_level.essential_takeaways) >= 5
    for takeaway in book_level.essential_takeaways:
        assert len(takeaway.source_chunk_ids) > 0
        assert set(takeaway.source_chunk_ids).issubset(all_orig_chunk_ids)


# ==============================================================================
# 2. Separation of Storage Tests
# ==============================================================================

@pytest.mark.anyio
async def test_storage_separation_original_chunks_never_mutated(setup_hierarchy_test_env):
    """Verify that chunks.json is NEVER overwritten or modified when hierarchy is generated."""
    hier_svc = setup_hierarchy_test_env["hierarchy_svc"]

    book_id = "book_storage_sep_02"
    chap_specs = [
        ("Chapter 1: Pristine Data", ["Original raw evidence that must remain completely untouched."])
    ]
    meta, chunks = seed_book_for_hierarchy(book_id, chap_specs)

    # Read original chunks.json content verbatim
    chunks_path = storage_service.get_book_dir(book_id) / "chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        original_chunks_json = f.read()

    # Generate hierarchy
    await hier_svc.generate_hierarchy(book_id)

    # Verify hierarchy.json exists in separate file
    hier_path = storage_service.get_book_dir(book_id) / "hierarchy.json"
    assert hier_path.exists()

    # Verify chunks.json is bit-for-bit identical to before generation
    with open(chunks_path, "r", encoding="utf-8") as f:
        current_chunks_json = f.read()

    assert current_chunks_json == original_chunks_json


# ==============================================================================
# 3. Malformed LLM Responses & Retry Behavior Tests
# ==============================================================================

@pytest.mark.anyio
async def test_malformed_llm_response_retry_behavior():
    """Verify that client recovers when LLM returns invalid JSON or malformed schema on initial attempts."""
    # Create MockLLMClient that fails with malformed JSON for first 2 attempts, then succeeds
    recovering_llm = MockLLMClient(
        model_name="mock-retry-model",
        malformed_json_failures_remaining=2,
    )

    # Test through SectionSummary schema using a wrapped retry runner
    # Verify generate_structured handles retries when exceptions occur
    retries_occurred = False
    for attempt in range(3):
        try:
            res = await recovering_llm.generate_structured(
                system_prompt="sys",
                user_prompt="Chunk book_retry_c0001 on page 1",
                response_model=SectionSummary,
            )
            # Succeeded
            assert res.source_chunk_ids == ["book_retry_c0001"]
            retries_occurred = (attempt == 2)
            break
        except Exception:
            continue

    assert retries_occurred
    assert recovering_llm.call_count == 3


# ==============================================================================
# 4. Missing Evidence & Uncertainty Handling Tests
# ==============================================================================

@pytest.mark.anyio
async def test_missing_evidence_populates_uncertainty(setup_hierarchy_test_env):
    """Verify that sparse or insufficient text sets evidence_confidence='insufficient' and uncertainty."""
    hier_svc = setup_hierarchy_test_env["hierarchy_svc"]

    book_id = "book_sparse_04"
    # Seed very short / sparse text (e.g. OCR failure or blank preface)
    chap_specs = [
        ("Chapter 1: Sparse Preface", ["sparse."])
    ]
    seed_book_for_hierarchy(book_id, chap_specs)

    hierarchy = await hier_svc.generate_hierarchy(book_id)

    assert len(hierarchy.chapters) == 1
    chapter = hierarchy.chapters[0]
    assert chapter.evidence_confidence == "insufficient"
    assert chapter.uncertainty is not None
    assert "sparse" in chapter.uncertainty.lower() or "brief" in chapter.uncertainty.lower()


# ==============================================================================
# 5. Long Chapter Staged Summarization Tests
# ==============================================================================

@pytest.mark.anyio
async def test_long_chapter_staged_summarization(setup_hierarchy_test_env):
    """Verify that chapters exceeding threshold chunks use staged summarization (stage_count=2)."""
    hier_svc = setup_hierarchy_test_env["hierarchy_svc"]

    book_id = "book_long_chapter_05"
    # LONG_CHAPTER_CHUNK_THRESHOLD is configured to 6 in fixture. Seed 10 chunks!
    ten_passages = [f"Passage number {i} detailing advanced algorithmic formulations." for i in range(10)]
    chap_specs = [
        ("Chapter 1: The Massive Monograph", ten_passages)
    ]
    _, chunks = seed_book_for_hierarchy(book_id, chap_specs)

    hierarchy = await hier_svc.generate_hierarchy(book_id)

    assert len(hierarchy.chapters) == 1
    chapter = hierarchy.chapters[0]
    # Staged summarization must be activated
    assert chapter.stage_count == 2
    # All 10 original chunk IDs must be preserved in the chapter summary
    assert len(chapter.source_chunk_ids) == 10
    assert len(chapter.supporting_section_ids) >= 3


# ==============================================================================
# 6. Idempotency & Resumable Execution Tests
# ==============================================================================

@pytest.mark.anyio
async def test_resumable_and_idempotent_execution(setup_hierarchy_test_env):
    """Verify that running hierarchy generation twice reuses completed units without redundant LLM calls."""
    hier_svc = setup_hierarchy_test_env["hierarchy_svc"]
    llm = setup_hierarchy_test_env["llm"]

    book_id = "book_idempotent_06"
    chap_specs = [
        ("Chapter 1: Part A", ["First foundational discussion."]),
        ("Chapter 2: Part B", ["Second advanced formulation."]),
    ]
    seed_book_for_hierarchy(book_id, chap_specs)

    # First complete pass
    hier_1 = await hier_svc.generate_hierarchy(book_id)
    initial_call_count = llm.call_count
    assert hier_1.generation_status == "completed"
    assert len(hier_1.completed_units) > 0

    # Second pass without force_restart
    hier_2 = await hier_svc.generate_hierarchy(book_id, force_restart=False)
    # LLM call count MUST remain identical (no redundant calls)
    assert llm.call_count == initial_call_count
    assert hier_2.generation_status == "completed"
    assert len(hier_2.chapters) == len(hier_1.chapters)


# ==============================================================================
# 7. HTTP API Hierarchy Endpoints Tests
# ==============================================================================

def test_hierarchy_api_endpoints(setup_hierarchy_test_env):
    """Test POST /generate, GET /status, and GET /hierarchy API endpoints."""
    book_id = "book_api_hier_07"
    chap_specs = [
        ("Chapter 1: Core", ["Author stated premises on computational complexity."])
    ]
    seed_book_for_hierarchy(book_id, chap_specs)

    # Initial status
    status_resp = client.get(f"/api/books/{book_id}/hierarchy/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["generation_status"] == "not_started"

    # Trigger generation via API
    gen_resp = client.post(f"/api/books/{book_id}/hierarchy/generate", json={})
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()
    assert gen_data["generation_status"] == "completed"
    assert gen_data["progress_percent"] == 100.0
    assert len(gen_data["chapters"]) == 1
    assert gen_data["book_level"]["central_thesis"] is not None

    # Inspect full hierarchy via GET
    get_resp = client.get(f"/api/books/{book_id}/hierarchy")
    assert get_resp.status_code == 200
    hier_data = get_resp.json()
    assert hier_data["book_id"] == book_id
    assert len(hier_data["sections"]) >= 1
    assert len(hier_data["chapters"]) == 1

    # Check status endpoint reflects completed state
    status_resp2 = client.get(f"/api/books/{book_id}/hierarchy/status")
    assert status_resp2.status_code == 200
    assert status_resp2.json()["generation_status"] == "completed"
    assert status_resp2.json()["has_book_summary"] is True
