"""Comprehensive tests for Phase 5: Conversational RAG API.

Tests:
1. Question classification: factual, conceptual, chapter summary, book crux, comparison, quiz.
2. Grounded answer formatting: crux-first rule, multi-passage synthesis, confidence score.
3. Structured citation verification: chunk IDs and page references strictly match retrieved chunks.
4. Book boundary isolation: Book A queries never retrieve or cite Book B evidence.
5. Unanswerable inquiries: explicit insufficient-evidence response and uncertainty acknowledgment.
6. Cross-chapter comparison: balanced multi-concept evidence retrieval and comparative synthesis.
7. Grounded quiz generation: grounded self-test questions with explanations.
8. Conversation lifecycle: persistent multi-turn thread storage and resumption.
9. Missing API credentials: HTTP 503 response when Groq key is absent.
10. Failed LLM request handling: graceful HTTP 500 error propagation.
11. Reranker integration: reranker rescoring applied when enabled.
"""

from datetime import datetime, timezone
import json
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.main import app
from app.schemas.book import BookMetadata, ChunkMetadata
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    QuestionType,
    StructuredAnswer,
)
from app.services import (
    bm25_service,
    chat_service,
    conversation_service,
    embedding_service,
    hierarchy_service,
    indexing_service,
    llm_client,
    qdrant_service,
    reranker_service,
    retrieval_service,
    storage_service,
)
from app.services.bm25_service import BM25Service
from app.services.chat_service import ChatService, classify_question
from app.services.conversation_service import ConversationService
from app.services.embedding_service import DeterministicMockEmbeddingService
from app.services.hierarchy_service import HierarchyService
from app.services.indexing_service import IndexingService
from app.services.llm_client import BaseLLMClient, MockLLMClient
from app.services.qdrant_service import QdrantService
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_chat_test_env(tmp_path, monkeypatch):
    """Provide isolated environment with in-memory Qdrant, mock embeddings, and mock LLM."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "QDRANT_STORAGE_DIR", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "ENABLE_RERANKER", False)

    mem_qdrant_client = QdrantClient(location=":memory:")
    test_qdrant_svc = QdrantService(
        client=mem_qdrant_client,
        collection_name="test_chat_collection",
        dimension=1024,
    )
    test_emb_svc = DeterministicMockEmbeddingService(dimension=1024)
    test_bm25_svc = BM25Service(k1=1.5, b=0.75)
    test_idx_svc = IndexingService(
        qdrant_service=test_qdrant_svc,
        embedding_service=test_emb_svc,
        bm25_service=test_bm25_svc,
    )
    test_ret_svc = RetrievalService(
        qdrant_service=test_qdrant_svc,
        embedding_service=test_emb_svc,
        bm25_service=test_bm25_svc,
    )
    test_llm = MockLLMClient(model_name="mock-llama-3.3-70b", prompt_version="v1.0")
    test_hier_svc = HierarchyService(llm_client=test_llm)
    test_conv_svc = ConversationService()
    test_rerank_svc = RerankerService()
    test_chat_svc = ChatService(
        retrieval_svc=test_ret_svc,
        hierarchy_svc=test_hier_svc,
        conv_svc=test_conv_svc,
        rerank_svc=test_rerank_svc,
        llm=test_llm,
    )

    monkeypatch.setattr(qdrant_service, "get_qdrant_service", lambda *args, **kwargs: test_qdrant_svc)
    monkeypatch.setattr(embedding_service, "get_embedding_service", lambda *args, **kwargs: test_emb_svc)
    monkeypatch.setattr(bm25_service, "get_bm25_service", lambda *args, **kwargs: test_bm25_svc)
    monkeypatch.setattr(indexing_service, "get_indexing_service", lambda *args, **kwargs: test_idx_svc)
    monkeypatch.setattr(retrieval_service, "get_retrieval_service", lambda *args, **kwargs: test_ret_svc)
    monkeypatch.setattr(llm_client, "get_llm_client", lambda *args, **kwargs: test_llm)
    monkeypatch.setattr(hierarchy_service, "get_hierarchy_service", lambda *args, **kwargs: test_hier_svc)
    monkeypatch.setattr(conversation_service, "get_conversation_service", lambda *args, **kwargs: test_conv_svc)
    monkeypatch.setattr(reranker_service, "get_reranker_service", lambda *args, **kwargs: test_rerank_svc)
    monkeypatch.setattr(chat_service, "get_chat_service", lambda *args, **kwargs: test_chat_svc)

    yield {
        "llm": test_llm,
        "chat_svc": test_chat_svc,
        "idx_svc": test_idx_svc,
        "ret_svc": test_ret_svc,
        "tmp_path": tmp_path,
    }


def seed_and_index_book(
    book_id: str,
    passages: list,  # list of (chapter, text)
) -> tuple:
    """Helper creating and indexing a book in isolated test storage and Qdrant."""
    chunks = []
    for i, (chap, text) in enumerate(passages):
        cid = f"{book_id}_c{i:04d}"
        chunks.append(
            ChunkMetadata(
                chunk_id=cid,
                book_id=book_id,
                chunk_index=i,
                text=text,
                page_start=i + 1,
                page_end=i + 1,
                chapter=chap,
                section=f"Section {i + 1}",
                heading_confidence="high",
                char_count=len(text),
                word_count=len(text.split()),
            )
        )

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
        pdf_bytes=b"%PDF-chat-test",
        metadata=meta,
        chunks=chunks,
        page_reports=[],
    )

    # Index into Qdrant & BM25
    idx_svc = indexing_service.get_indexing_service()
    idx_svc.index_book(book_id, force=True)

    return meta, chunks


# ==============================================================================
# 1. Question Classification Tests
# ==============================================================================

def test_question_classification():
    """Verify classification into all 6 specified question categories."""
    assert classify_question("What year was the Roman empire established?") == QuestionType.FACTUAL
    assert classify_question("Why does quantum superposition degrade in open systems?") == QuestionType.CONCEPTUAL
    assert classify_question("Can you summarize chapter 3 for me?") == QuestionType.CHAPTER_SUMMARY
    assert classify_question("What is the central crux of the book?") == QuestionType.BOOK_CRUX
    assert classify_question("Compare supervised learning versus unsupervised learning") == QuestionType.CROSS_CHAPTER_COMPARISON
    assert classify_question("Quiz me on the concepts in this section") == QuestionType.QUIZ


# ==============================================================================
# 2. Answer Formatting & Crux-First Tests
# ==============================================================================

def test_chat_answer_formatting_crux_first():
    """Verify crux-first answering rule, synthesized detail, and confidence score."""
    book_id = "book_arch_01"
    passages = [
        ("Chapter 1: Principles", "Distributed consensus requires quorum agreement across nodes to prevent split-brain states."),
        ("Chapter 1: Principles", "By contrast, centralized systems rely on a single primary coordinator which introduces a single point of failure."),
    ]
    seed_and_index_book(book_id, passages)

    payload = {
        "message": "What is the primary requirement for distributed consensus?",
        "include_hierarchy": False,
    }
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["book_id"] == book_id
    assert data["conversation_id"] is not None
    assert data["crux_summary"] != ""
    # Enforce crux-first answering rule: the answer must lead with the crux summary
    assert data["answer"].startswith(data["crux_summary"])
    assert data["confidence"] == "high"
    assert data["insufficient_evidence"] is False
    assert len(data["citations"]) > 0


# ==============================================================================
# 3. Citation Verification & Grounding Tests
# ==============================================================================

def test_chat_structured_citations_grounding():
    """Verify citations contain valid chunk IDs, accurate page numbers, and real snippets."""
    book_id = "book_cit_01"
    passages = [
        ("Chapter 2: Concurrency", "Optimistic concurrency control checks transaction validity at commit time rather than locking upfront."),
        ("Chapter 2: Concurrency", "Pessimistic locking prevents conflicting writes by acquiring exclusive locks on read operations."),
    ]
    _, chunks = seed_and_index_book(book_id, passages)
    valid_cids = {c.chunk_id for c in chunks}

    payload = {
        "message": "Explain how optimistic concurrency control works.",
    }
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    citations = data["citations"]
    assert len(citations) >= 1
    for cit in citations:
        assert cit["chunk_id"] in valid_cids
        assert cit["page_start"] >= 1
        assert cit["page_end"] >= cit["page_start"]
        assert cit["chapter"] != ""
        assert len(cit["quoted_snippet"]) > 0


# ==============================================================================
# 4. Strict Book Isolation Tests
# ==============================================================================

def test_chat_strict_book_isolation():
    """Verify that queries to Book A NEVER retrieve or cite chunks from Book B."""
    book_a = "book_quantum_comp"
    passages_a = [
        ("Chapter 1: Qubits", "Quantum processors utilize superconducting circuits to maintain quantum state superposition."),
    ]
    seed_and_index_book(book_a, passages_a)

    book_b = "book_roman_hist"
    passages_b = [
        ("Chapter 1: Julius Caesar", "The Roman Senate held legislative authority during the late Republican period."),
    ]
    seed_and_index_book(book_b, passages_b)

    # Inquire in Book A about Book B's exclusive topic
    payload = {
        "message": "What authority did the Roman Senate hold?",
    }
    response = client.post(f"/api/books/{book_a}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Citations in Book A must NEVER contain any chunk from Book B
    for cit in data["citations"]:
        assert not cit["chunk_id"].startswith("book_roman_hist")
        assert cit["chunk_id"].startswith("book_quantum_comp")


# ==============================================================================
# 5. Insufficient Evidence & Uncertainty Tests
# ==============================================================================

def test_chat_unanswerable_question_insufficient_evidence():
    """Verify explicit insufficient-evidence response when book lacks relevant material."""
    book_id = "book_data_eng"
    passages = [
        ("Chapter 1: Pipelines", "Data pipelines extract data from transactional databases and load it into analytical warehouses."),
    ]
    seed_and_index_book(book_id, passages)

    payload = {
        "message": "What is the secret recipe for baking a chocolate cake?",
    }
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["insufficient_evidence"] is True
    assert data["confidence"] == "insufficient"
    assert "does not contain sufficient evidence" in data["answer"].lower()
    assert data["citations"] == []


# ==============================================================================
# 6. Cross-Chapter Comparison Tests
# ==============================================================================

def test_chat_cross_chapter_comparison():
    """Verify dual-retrieval and comparative crux for cross-chapter questions."""
    book_id = "book_ml_systems"
    passages = [
        ("Chapter 1: Supervised Learning", "Supervised learning maps inputs to ground-truth labeled targets using gradient descent."),
        ("Chapter 2: Unsupervised Learning", "Unsupervised learning discovers underlying latent structure from unannotated observations."),
    ]
    seed_and_index_book(book_id, passages)

    payload = {
        "message": "Compare supervised learning versus unsupervised learning.",
    }
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["question_type"] == QuestionType.CROSS_CHAPTER_COMPARISON
    assert "Comparative Crux" in data["crux_summary"]
    assert len(data["citations"]) >= 1


# ==============================================================================
# 7. Grounded Quiz Generation Tests
# ==============================================================================

def test_chat_quiz_generation():
    """Verify quiz classification and grounded self-test generation."""
    book_id = "book_biology"
    passages = [
        ("Chapter 1: Cellular Respiration", "Mitochondria generate adenosine triphosphate through oxidative phosphorylation."),
    ]
    seed_and_index_book(book_id, passages)

    payload = {
        "message": "Generate a quiz on cellular respiration to test me.",
    }
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["question_type"] == QuestionType.QUIZ
    assert "Quiz Question" in data["crux_summary"]
    assert "Correct Answer:" in data["answer"]


# ==============================================================================
# 8. Conversation History Lifecycle & Multi-turn Resumption
# ==============================================================================

def test_conversation_history_lifecycle():
    """Verify persistent local conversation storage, multi-turn resumption, and retrieval."""
    book_id = "book_dialogue_01"
    passages = [
        ("Chapter 1: Networking", "The TCP three-way handshake establishes a reliable stream transmission connection."),
        ("Chapter 2: Flow Control", "Sliding window protocols regulate the transmission rate to prevent receiver buffer overflow."),
    ]
    seed_and_index_book(book_id, passages)

    # Turn 1: Initial question (no conversation_id)
    t1_resp = client.post(
        f"/api/books/{book_id}/chat",
        json={"message": "How does the TCP handshake work?"},
    )
    assert t1_resp.status_code == 200
    t1_data = t1_resp.json()
    conv_id = t1_data["conversation_id"]
    assert conv_id is not None

    # Turn 2: Follow-up question using the same conversation_id
    t2_resp = client.post(
        f"/api/books/{book_id}/chat",
        json={
            "conversation_id": conv_id,
            "message": "What mechanism prevents buffer overflow?",
        },
    )
    assert t2_resp.status_code == 200
    t2_data = t2_resp.json()
    assert t2_data["conversation_id"] == conv_id

    # Retrieve conversation thread directly
    conv_detail_resp = client.get(f"/api/books/{book_id}/conversations/{conv_id}")
    assert conv_detail_resp.status_code == 200
    conv_data = conv_detail_resp.json()
    assert conv_data["conversation_id"] == conv_id
    assert conv_data["book_id"] == book_id
    # 2 user questions + 2 assistant answers = 4 messages
    assert len(conv_data["messages"]) == 4
    assert conv_data["messages"][0]["role"] == "user"
    assert conv_data["messages"][1]["role"] == "assistant"
    assert conv_data["messages"][2]["role"] == "user"
    assert conv_data["messages"][3]["role"] == "assistant"

    # List conversations for the book
    list_resp = client.get(f"/api/books/{book_id}/conversations")
    assert list_resp.status_code == 200
    conversations = list_resp.json()
    assert len(conversations) >= 1
    found = [c for c in conversations if c["conversation_id"] == conv_id]
    assert len(found) == 1
    assert len(found[0]["messages"]) == 4


# ==============================================================================
# 9. Missing API Credentials & Error Handling
# ==============================================================================

def test_missing_groq_api_credentials(monkeypatch):
    """Verify HTTP 503 is returned when Groq provider is configured without an API key."""
    book_id = "book_creds_01"
    passages = [
        ("Chapter 1: Security", "Cryptographic keys must be protected against unauthorized disclosure."),
    ]
    seed_and_index_book(book_id, passages)

    # Configure Groq provider with missing API key
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")

    # Reconstruct ChatService using default factory to trigger GroqLLMClient check
    chat_service.reset_chat_service()

    payload = {"message": "How should cryptographic keys be protected?"}
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_failed_llm_request_handling(monkeypatch):
    """Verify graceful HTTP 500 error propagation on unrecoverable LLM failures."""
    book_id = "book_err_01"
    passages = [
        ("Chapter 1: Faults", "Byzantine faults represent arbitrary failure modes in distributed environments."),
    ]
    seed_and_index_book(book_id, passages)

    class FailingLLMClient(MockLLMClient):
        async def generate_structured(self, system_prompt, user_prompt, response_model, max_retries=None):
            raise RuntimeError("Upstream LLM network connection unexpectedly severed")

    failing_llm = FailingLLMClient(model_name="failing-model", prompt_version="v1.0")
    failing_chat_svc = ChatService(
        retrieval_svc=retrieval_service.get_retrieval_service(),
        hierarchy_svc=hierarchy_service.get_hierarchy_service(),
        conv_svc=conversation_service.get_conversation_service(),
        rerank_svc=reranker_service.get_reranker_service(),
        llm=failing_llm,
    )
    monkeypatch.setattr(chat_service, "get_chat_service", lambda *args, **kwargs: failing_chat_svc)

    payload = {"message": "What is a Byzantine fault?"}
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 500
    assert "Upstream LLM network connection" in response.json()["detail"]


# ==============================================================================
# 10. Nonexistent Book and Reranker Handling
# ==============================================================================

def test_chat_nonexistent_book():
    """Verify HTTP 404 is returned when attempting to chat with a non-existent book."""
    response = client.post(
        "/api/books/nonexistent_book_id_9999/chat",
        json={"message": "Hello?"},
    )
    assert response.status_code == 404


def test_chat_with_reranker_enabled():
    """Verify chat functions properly with reranking enabled."""
    book_id = "book_rerank_01"
    passages = [
        ("Chapter 1: Storage", "Append-only log storage formats provide sequential write performance."),
        ("Chapter 1: Storage", "B-trees provide predictable logarithmic read latency for random access."),
    ]
    seed_and_index_book(book_id, passages)

    payload = {
        "message": "Which storage format gives sequential write performance?",
        "enable_reranker": True,
        "top_k": 3,
    }
    response = client.post(f"/api/books/{book_id}/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] != ""
    assert len(data["citations"]) > 0
