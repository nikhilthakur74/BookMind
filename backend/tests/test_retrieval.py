"""Comprehensive tests for BM25 keyword retrieval, semantic search, Reciprocal Rank Fusion,
strict book isolation, and metadata preservation.
"""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.main import app
from app.schemas.book import BookMetadata, ChunkMetadata
from app.schemas.retrieval import RetrievalMode, RetrievalRequest
from app.services import (
    bm25_service,
    embedding_service,
    fusion_service,
    indexing_service,
    qdrant_service,
    retrieval_service,
    storage_service,
)
from app.services.bm25_service import BM25Index, BM25Service
from app.services.embedding_service import DeterministicMockEmbeddingService
from app.services.fusion_service import reciprocal_rank_fusion
from app.services.qdrant_service import QdrantService
from app.services.retrieval_service import RetrievalService

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_retrieval(tmp_path, monkeypatch):
    """Provide isolated environment with in-memory Qdrant and deterministic mock embeddings."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "QDRANT_STORAGE_DIR", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")

    mem_qdrant_client = QdrantClient(location=":memory:")
    test_qdrant_svc = QdrantService(
        client=mem_qdrant_client,
        collection_name="test_retrieval_collection",
        dimension=1024,
    )
    test_emb_svc = DeterministicMockEmbeddingService(dimension=1024)
    test_bm25_svc = BM25Service(k1=1.5, b=0.75)
    test_idx_svc = indexing_service.IndexingService(
        qdrant_service=test_qdrant_svc,
        embedding_service=test_emb_svc,
        bm25_service=test_bm25_svc,
    )
    test_ret_svc = RetrievalService(
        qdrant_service=test_qdrant_svc,
        embedding_service=test_emb_svc,
        bm25_service=test_bm25_svc,
    )

    monkeypatch.setattr(qdrant_service, "get_qdrant_service", lambda *args, **kwargs: test_qdrant_svc)
    monkeypatch.setattr(embedding_service, "get_embedding_service", lambda *args, **kwargs: test_emb_svc)
    monkeypatch.setattr(bm25_service, "get_bm25_service", lambda *args, **kwargs: test_bm25_svc)
    monkeypatch.setattr(indexing_service, "get_indexing_service", lambda *args, **kwargs: test_idx_svc)
    monkeypatch.setattr(retrieval_service, "get_retrieval_service", lambda *args, **kwargs: test_ret_svc)

    yield {
        "qdrant": test_qdrant_svc,
        "embedding": test_emb_svc,
        "bm25": test_bm25_svc,
        "indexing": test_idx_svc,
        "retrieval": test_ret_svc,
    }


def seed_test_book(
    book_id: str,
    passages: list,
    indexing_svc: indexing_service.IndexingService,
) -> tuple:
    """Helper creating and indexing a book with custom text passages."""
    meta = BookMetadata(
        book_id=book_id,
        original_filename=f"{book_id}.pdf",
        safe_filename=f"{book_id}.pdf",
        file_hash=f"hash_{book_id}",
        file_size_bytes=1024 * 50,
        page_count=len(passages),
        chunk_count=len(passages),
        extraction_status="completed",
        needs_ocr=False,
        headings=[],
        created_at=datetime.now(timezone.utc),
        indexing_status="not_indexed",
        indexed_chunk_count=0,
    )

    chunks = []
    for i, text in enumerate(passages):
        chunks.append(
            ChunkMetadata(
                chunk_id=f"{book_id}_c{i:04d}",
                book_id=book_id,
                chunk_index=i,
                text=text,
                page_start=i + 1,
                page_end=i + 1,
                chapter=f"Chapter {i + 1}",
                section=f"Section {i + 1}.1",
                heading_confidence="high",
                char_count=len(text),
                word_count=len(text.split()),
            )
        )

    storage_service.save_book_assets(
        book_id=book_id,
        pdf_bytes=b"%PDF-test",
        metadata=meta,
        chunks=chunks,
        page_reports=[],
    )

    indexing_svc.index_book(book_id)
    return meta, chunks


# ==============================================================================
# 1. Independent BM25 Ranking Tests
# ==============================================================================

def test_bm25_deterministic_ranking():
    """Verify that exact keyword matches score highest with Okapi BM25."""
    c1 = ChunkMetadata(
        chunk_id="c1",
        book_id="b1",
        chunk_index=0,
        text="Quantum superposition allows qubits to exist in multiple simultaneous states.",
        page_start=1,
        page_end=1,
        chapter="Chapter 1",
        section="Intro",
        heading_confidence="high",
        char_count=80,
        word_count=10,
    )
    c2 = ChunkMetadata(
        chunk_id="c2",
        book_id="b1",
        chunk_index=1,
        text="Classical boolean gates like AND, OR, and NOT operate on deterministic bits.",
        page_start=2,
        page_end=2,
        chapter="Chapter 2",
        section="Logic",
        heading_confidence="high",
        char_count=78,
        word_count=11,
    )
    c3 = ChunkMetadata(
        chunk_id="c3",
        book_id="b1",
        chunk_index=2,
        text="Superposition is combined with quantum entanglement for cryptographic protocols.",
        page_start=3,
        page_end=3,
        chapter="Chapter 3",
        section="Entanglement",
        heading_confidence="high",
        char_count=82,
        word_count=9,
    )

    idx = BM25Index("b1", k1=1.5, b=0.75)
    idx.add_chunks([c1, c2, c3])

    # Query for boolean gates -> c2 must be top
    scores_boolean = idx.score("boolean gates")
    assert len(scores_boolean) > 0
    assert scores_boolean[0][0] == "c2"

    # Query for cryptographic protocols -> c3 must be top
    scores_crypto = idx.score("cryptographic protocols")
    assert len(scores_crypto) > 0
    assert scores_crypto[0][0] == "c3"


def test_bm25_document_length_normalization():
    """Verify that BM25 rewards concise documents over longer ones with same term frequency."""
    c_short = ChunkMetadata(
        chunk_id="short_doc",
        book_id="b_norm",
        chunk_index=0,
        text="Neural networks learn gradient representations.",
        page_start=1,
        page_end=1,
        char_count=45,
        word_count=5,
    )
    c_long = ChunkMetadata(
        chunk_id="long_doc",
        book_id="b_norm",
        chunk_index=1,
        text=(
            "In this introductory section we provide an extensive historical overview of the evolution of "
            "computing machines before we describe how neural networks learn gradient representations across epochs."
        ),
        page_start=2,
        page_end=2,
        char_count=200,
        word_count=27,
    )

    idx = BM25Index("b_norm", k1=1.5, b=0.75)
    idx.add_chunks([c_short, c_long])

    scores = idx.score("neural networks gradient")
    # Concise doc has smaller |d| / avgdl, thus lower denominator in TF component -> higher score
    assert scores[0][0] == "short_doc"
    assert scores[0][1] > scores[1][1]


# ==============================================================================
# 2. Reciprocal Rank Fusion (RRF) Logic & Consensus Reward Tests
# ==============================================================================

def test_rrf_mathematical_precision():
    """Verify RRF formula precision: RRF = sum(1 / (k + rank))."""
    from app.schemas.retrieval import SearchResultItem

    def make_item(cid: str, score: float, rank: int, method: str) -> SearchResultItem:
        chunk = ChunkMetadata(
            chunk_id=cid,
            book_id="b_rrf",
            chunk_index=ord(cid[-1]),
            text=f"Text for {cid}",
            page_start=1,
            page_end=1,
            char_count=20,
            word_count=3,
        )
        return SearchResultItem(chunk=chunk, score=score, rank=rank, method=method)

    # Document A is Rank 1 in Semantic and Rank 1 in BM25
    # Document B is Rank 2 in Semantic and absent in BM25
    # Document C is absent in Semantic and Rank 2 in BM25
    sem_list = [make_item("doc_A", 0.95, 1, "semantic"), make_item("doc_B", 0.85, 2, "semantic")]
    bm25_list = [make_item("doc_A", 4.2, 1, "bm25"), make_item("doc_C", 3.1, 2, "bm25")]

    fused = reciprocal_rank_fusion(
        semantic_results=sem_list,
        bm25_results=bm25_list,
        rrf_k=60,
        top_k=5,
    )

    # Expected scores:
    # doc_A: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278688
    # doc_B: 1/(60+2) = 1/62 ≈ 0.01612903
    # doc_C: 1/(60+2) = 1/62 ≈ 0.01612903
    assert len(fused) == 3
    assert fused[0].chunk.chunk_id == "doc_A"
    assert abs(fused[0].rrf_score - (2.0 / 61.0)) < 1e-6
    assert fused[0].rrf_rank == 1
    assert fused[0].semantic_rank == 1
    assert fused[0].bm25_rank == 1

    assert fused[1].chunk.chunk_id == "doc_B"
    assert abs(fused[1].rrf_score - (1.0 / 62.0)) < 1e-6
    assert fused[1].semantic_rank == 2
    assert fused[1].bm25_rank is None


# ==============================================================================
# 3. Book Filtering & Strict Evidence Isolation Tests
# ==============================================================================

def test_strict_book_isolation_in_all_retrieval_modes(setup_test_retrieval):
    """Verify that querying Book A NEVER returns chunks from Book B, even with identical query terms."""
    idx_svc = setup_test_retrieval["indexing"]
    ret_svc = setup_test_retrieval["retrieval"]

    # Book 1: Biology focus
    passages_1 = [
        "Mitochondria generate ATP through oxidative phosphorylation inside eukaryotic cells.",
        "Ribosomes synthesize proteins by translating messenger RNA sequences.",
    ]
    seed_test_book("book_bio_01", passages_1, idx_svc)

    # Book 2: Physics focus, but also mentions cellular automata
    passages_2 = [
        "Thermodynamics principles dictate entropy increase and energy conservation in physical cells.",
        "Quantum mechanics describes atomic electron shells and photon emissions.",
    ]
    seed_test_book("book_phys_02", passages_2, idx_svc)

    # 1. Semantic search for Book 1
    sem_res_1 = ret_svc.retrieve_semantic("mitochondria ATP cells", book_id="book_bio_01", top_k=5)
    assert len(sem_res_1) > 0
    for item in sem_res_1:
        assert item.chunk.book_id == "book_bio_01"

    # 2. BM25 search for Book 1
    bm25_res_1 = ret_svc.retrieve_bm25("cells", book_id="book_bio_01", top_k=5)
    assert len(bm25_res_1) > 0
    for item in bm25_res_1:
        assert item.chunk.book_id == "book_bio_01"

    # 3. Hybrid search for Book 2 using a query containing 'cells'
    hybrid_res_2 = ret_svc.retrieve_hybrid("energy conservation in cells", book_id="book_phys_02", top_k=5)
    assert len(hybrid_res_2) > 0
    for item in hybrid_res_2:
        assert item.chunk.book_id == "book_phys_02"
        assert item.chunk.book_id != "book_bio_01"


# ==============================================================================
# 4. Metadata Preservation Tests
# ==============================================================================

def test_chunk_metadata_complete_preservation(setup_test_retrieval):
    """Verify that all metadata fields are preserved intact throughout retrieval."""
    idx_svc = setup_test_retrieval["indexing"]
    ret_svc = setup_test_retrieval["retrieval"]

    passages = ["Detailed architectural guidelines for distributed consensus algorithms."]
    _, chunks = seed_test_book("book_meta_preserve", passages, idx_svc)
    orig_chunk = chunks[0]

    # Test Hybrid Retrieval
    results = ret_svc.retrieve_hybrid(
        query="distributed consensus",
        book_id="book_meta_preserve",
        top_k=1,
    )

    assert len(results) == 1
    ret_chunk = results[0].chunk

    assert ret_chunk.chunk_id == orig_chunk.chunk_id
    assert ret_chunk.book_id == orig_chunk.book_id
    assert ret_chunk.chunk_index == orig_chunk.chunk_index
    assert ret_chunk.text == orig_chunk.text
    assert ret_chunk.page_start == orig_chunk.page_start
    assert ret_chunk.page_end == orig_chunk.page_end
    assert ret_chunk.chapter == orig_chunk.chapter
    assert ret_chunk.section == orig_chunk.section
    assert ret_chunk.heading_confidence == orig_chunk.heading_confidence
    assert ret_chunk.char_count == orig_chunk.char_count
    assert ret_chunk.word_count == orig_chunk.word_count


# ==============================================================================
# 5. Configurable Top-K Tests
# ==============================================================================

def test_configurable_top_k(setup_test_retrieval):
    """Verify that retrieval strictly respects requested top_k parameter."""
    idx_svc = setup_test_retrieval["indexing"]
    ret_svc = setup_test_retrieval["retrieval"]

    passages = [
        f"Database transaction isolation level definition passage number {i}." for i in range(10)
    ]
    seed_test_book("book_topk_test", passages, idx_svc)

    for k in (2, 4, 7):
        res = ret_svc.retrieve_hybrid("database transaction isolation", book_id="book_topk_test", top_k=k)
        assert len(res) == k
        # Ensure rankings are sequential 1..k
        assert [item.rrf_rank for item in res] == list(range(1, k + 1))


# ==============================================================================
# 6. HTTP API Retrieval Tests
# ==============================================================================

def test_retrieval_api_endpoint(setup_test_retrieval):
    """Test POST /api/books/{book_id}/retrieve endpoint with hybrid, semantic, and bm25 modes."""
    idx_svc = setup_test_retrieval["indexing"]

    passages = [
        "Recursive neural networks process hierarchical graph and tree representations.",
        "Convolutional neural networks extract local spatial features from multi-channel images.",
    ]
    seed_test_book("book_api_ret_01", passages, idx_svc)

    # 1. Hybrid search
    resp_hybrid = client.post(
        "/api/books/book_api_ret_01/retrieve",
        json={"query": "neural networks representations", "mode": "hybrid", "top_k": 2},
    )
    assert resp_hybrid.status_code == 200
    data_hybrid = resp_hybrid.json()
    assert data_hybrid["book_id"] == "book_api_ret_01"
    assert data_hybrid["mode"] == "hybrid"
    assert len(data_hybrid["results"]) == 2
    assert "rrf_score" in data_hybrid["results"][0]

    # 2. Semantic search
    resp_sem = client.post(
        "/api/books/book_api_ret_01/retrieve",
        json={"query": "convolutional features", "mode": "semantic", "top_k": 1},
    )
    assert resp_sem.status_code == 200
    data_sem = resp_sem.json()
    assert data_sem["mode"] == "semantic"
    assert len(data_sem["results"]) == 1
    assert data_sem["results"][0]["method"] == "semantic"

    # 3. BM25 search
    resp_bm25 = client.post(
        "/api/books/book_api_ret_01/retrieve",
        json={"query": "recursive graph tree", "mode": "bm25", "top_k": 1},
    )
    assert resp_bm25.status_code == 200
    data_bm25 = resp_bm25.json()
    assert data_bm25["mode"] == "bm25"
    assert len(data_bm25["results"]) == 1
    assert data_bm25["results"][0]["method"] == "bm25"
    assert data_bm25["results"][0]["chunk"]["chunk_id"] == "book_api_ret_01_c0000"

    # 4. Non-existent book ID returns 404
    resp_404 = client.post(
        "/api/books/non_existent_book/retrieve",
        json={"query": "test query"},
    )
    assert resp_404.status_code == 404
