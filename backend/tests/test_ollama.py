"""Tests for Local Ollama LLM provider integration.

Covers:
- Provider selection and configuration validation
- Local-mode isolation from Groq (no cloud calls, no token leakage)
- Structured JSON output request and response parsing
- Automatic retry on malformed JSON or validation errors
- Graceful handling of unavailable Ollama service (connection errors)
- Missing model detection and remediation guidance
- Timeout handling and request resilience
- Citation preservation across chat queries
- Hierarchy generation failure recovery and resumable execution
- Health diagnostic endpoint integration
- Real local Ollama inference integration verification (when Ollama is live)
"""

import asyncio
from datetime import datetime, timezone
import json
from unittest.mock import MagicMock
import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.book import BookMetadata, ChunkMetadata
from app.schemas.chat import ChatRequest, StructuredAnswer
from app.schemas.health import LLMHealthResponse
from app.schemas.hierarchy import (
    BookHierarchy,
    BookSummary,
    ChapterSummary,
    ConceptRecord,
    DefinitionRecord,
    SectionSummary,
)
from app.services import chat_service, hierarchy_service, llm_client, storage_service
from app.services.chat_service import ChatService
from app.services.hierarchy_service import HierarchyService
from app.services.llm_client import (
    GroqLLMClient,
    MockLLMClient,
    OllamaLLMClient,
    get_llm_client,
    reset_llm_client,
)
from app.api.endpoints.health import check_llm_health

client = TestClient(app)


# ==============================================================================
# Helper Mock Transports
# ==============================================================================

def make_mock_ollama_transport(
    chat_responses=None,
    tags_models=None,
    status_code=200,
    raise_exc=None,
):
    """Build a deterministic httpx.MockTransport for Ollama endpoints."""
    if tags_models is None:
        tags_models = ["qwen2.5:3b"]

    responses_iter = iter(chat_responses or [])

    def handler(request: httpx.Request) -> httpx.Response:
        if raise_exc:
            raise raise_exc

        url_str = str(request.url)

        if "/api/tags" in url_str:
            body = {"models": [{"name": m, "model": m} for m in tags_models]}
            return httpx.Response(200, json=body)

        if "/api/chat" in url_str:
            if status_code != 200:
                return httpx.Response(status_code, text="Ollama internal error")

            resp_item = next(responses_iter, None)
            if resp_item is None:
                resp_item = {"crux": "Default crux response", "citations": []}

            if isinstance(resp_item, Exception):
                raise resp_item

            if isinstance(resp_item, str):
                content = resp_item
            else:
                content = json.dumps(resp_item)

            return httpx.Response(
                200,
                json={
                    "model": "qwen2.5:3b",
                    "message": {"role": "assistant", "content": content},
                    "done": True,
                },
            )

        return httpx.Response(404, text="Not Found")

    return httpx.MockTransport(handler)


# ==============================================================================
# 1. Provider Selection & Configuration Tests
# ==============================================================================

def test_provider_selection_ollama(monkeypatch):
    """Verify get_llm_client instantiates OllamaLLMClient when provider is 'ollama'."""
    reset_llm_client()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")

    client_instance = get_llm_client(force_new=True)
    assert isinstance(client_instance, OllamaLLMClient)
    assert client_instance.model_name == "qwen2.5:3b"
    assert client_instance.base_url == "http://localhost:11434"
    reset_llm_client()


def test_provider_selection_groq(monkeypatch):
    """Verify get_llm_client instantiates GroqLLMClient when provider is 'groq' with API key."""
    reset_llm_client()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_mock_key_12345")
    monkeypatch.setattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")

    client_instance = get_llm_client(force_new=True)
    assert isinstance(client_instance, GroqLLMClient)
    assert client_instance.model_name == "llama-3.3-70b-versatile"
    reset_llm_client()


def test_provider_selection_mock(monkeypatch):
    """Verify get_llm_client instantiates MockLLMClient when provider is 'mock'."""
    reset_llm_client()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    client_instance = get_llm_client(force_new=True)
    assert isinstance(client_instance, MockLLMClient)
    reset_llm_client()


def test_provider_selection_invalid(monkeypatch):
    """Verify get_llm_client raises ValueError for unsupported provider."""
    reset_llm_client()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "invalid_provider")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm_client(force_new=True)
    reset_llm_client()


# ==============================================================================
# 2. Local Mode Isolation from Groq Tests
# ==============================================================================

def test_local_mode_isolation_from_groq(monkeypatch):
    """Ensure that local Ollama mode NEVER requires, accesses, or leaks to Groq."""
    reset_llm_client()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    # Empty Groq API key: Ollama must initialize without complaining about Groq credentials
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")

    client_instance = get_llm_client(force_new=True)
    assert isinstance(client_instance, OllamaLLMClient)
    assert not isinstance(client_instance, GroqLLMClient)
    reset_llm_client()


# ==============================================================================
# 3. Request & Response Parsing (Schema Constrained Outputs)
# ==============================================================================

@pytest.mark.anyio
async def test_ollama_request_and_response_parsing():
    """Verify that OllamaLLMClient sends JSON schema format and parses valid Pydantic response."""
    captured_requests = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        url_str = str(request.url)
        if "/api/tags" in url_str:
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        if "/api/chat" in url_str:
            payload = json.loads(request.read())
            assert "format" in payload
            assert payload["model"] == "qwen2.5:3b"
            assert payload["options"]["num_ctx"] == 4096
            assert payload["options"]["num_predict"] == 2048

            valid_section = {
                "section_id": "sec_01",
                "chapter_title": "Chapter 1",
                "section_title": "Introduction",
                "page_start": 1,
                "page_end": 4,
                "concise_summary": "Clear summary of the section.",
                "key_concepts": [],
                "definitions": [],
                "reasoning_chain": ["Premise 1", "Conclusion 1"],
                "source_chunk_ids": ["chunk_01"],
            }
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": json.dumps(valid_section)}},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            http_client=mock_http,
        )

        res = await ollama.generate_structured(
            system_prompt="Summarize section.",
            user_prompt="Evidence text...",
            response_model=SectionSummary,
        )

        assert isinstance(res, SectionSummary)
        assert res.section_id == "sec_01"
        assert res.concise_summary == "Clear summary of the section."


@pytest.mark.anyio
async def test_ollama_strips_markdown_code_fences():
    """Verify that OllamaLLMClient cleanly strips ```json code fences if model emits them."""
    fenced_content = "```json\n{\"crux\": \"Core idea\", \"detailed_synthesis\": \"Details\", \"citations\": [], \"insufficient_evidence\": false, \"confidence\": \"high\"}\n```"

    transport = make_mock_ollama_transport(chat_responses=[fenced_content])
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            http_client=mock_http,
        )

        res = await ollama.generate_structured(
            system_prompt="Answer query.",
            user_prompt="Question text.",
            response_model=StructuredAnswer,
        )
        assert res.crux == "Core idea"
        assert res.confidence == "high"


# ==============================================================================
# 4. Malformed Structured Output & Retry Behavior Tests
# ==============================================================================

@pytest.mark.anyio
async def test_ollama_malformed_json_retry_success():
    """Verify that client recovers when Ollama produces malformed JSON on attempt 1 and valid on attempt 2."""
    malformed = "NOT_VALID_JSON{foo"
    valid = json.dumps({
        "crux": "Recovered crux after retry",
        "detailed_synthesis": "Correct synthesis",
        "citations": [],
        "insufficient_evidence": False,
        "confidence": "high",
    })

    transport = make_mock_ollama_transport(chat_responses=[malformed, valid])
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            http_client=mock_http,
            max_retries=3,
        )

        res = await ollama.generate_structured(
            system_prompt="Synthesize answer.",
            user_prompt="Query",
            response_model=StructuredAnswer,
        )
        assert res.crux == "Recovered crux after retry"


@pytest.mark.anyio
async def test_ollama_malformed_json_retry_exhaustion():
    """Verify that persistent malformed JSON raises RuntimeError and is never treated as success."""
    malformed = "INVALID_JSON_EVERY_ATTEMPT"

    transport = make_mock_ollama_transport(chat_responses=[malformed, malformed, malformed])
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            http_client=mock_http,
            max_retries=2,
        )

        with pytest.raises(RuntimeError, match="Failed to generate valid structured response"):
            await ollama.generate_structured(
                system_prompt="Synthesize.",
                user_prompt="Query",
                response_model=StructuredAnswer,
            )


# ==============================================================================
# 5. Missing Model & Unavailable Service Diagnostics
# ==============================================================================

@pytest.mark.anyio
async def test_ollama_unavailable_server():
    """Verify clear, actionable error when Ollama server is unreachable."""
    transport = make_mock_ollama_transport(raise_exc=httpx.ConnectError("Connection refused"))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            base_url="http://localhost:11434",
            model_name="qwen2.5:3b",
            http_client=mock_http,
        )

        with pytest.raises(RuntimeError, match="Ollama service is unreachable"):
            await ollama.generate_structured(
                system_prompt="Test",
                user_prompt="Test",
                response_model=StructuredAnswer,
            )


@pytest.mark.anyio
async def test_ollama_missing_model():
    """Verify descriptive error with 'ollama pull' remediation when model is not installed."""
    # Tags returns list of models that does NOT include 'qwen2.5:3b'
    transport = make_mock_ollama_transport(tags_models=["llama3.1:8b", "mistral:7b"])
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            http_client=mock_http,
        )

        with pytest.raises(RuntimeError, match="ollama pull qwen2.5:3b"):
            await ollama.generate_structured(
                system_prompt="Test",
                user_prompt="Test",
                response_model=StructuredAnswer,
            )


# ==============================================================================
# 6. Timeout Handling
# ==============================================================================

@pytest.mark.anyio
async def test_ollama_timeout_handling():
    """Verify TimeoutError is raised when Ollama requests persistently time out."""
    transport = make_mock_ollama_transport(
        chat_responses=[
            httpx.TimeoutException("Read timeout"),
            httpx.TimeoutException("Read timeout"),
        ]
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            timeout=1.0,
            max_retries=2,
            http_client=mock_http,
        )

        with pytest.raises(TimeoutError, match="timed out"):
            await ollama.generate_structured(
                system_prompt="Test",
                user_prompt="Test",
                response_model=StructuredAnswer,
            )


# ==============================================================================
# 7. Health Endpoint & Provider Diagnostics Tests
# ==============================================================================

def test_health_endpoints_with_mocked_ollama(monkeypatch):
    """Test GET /api/health and GET /api/health/llm reporting provider diagnostics."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    # 1. Base /api/health
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "llm" in data
    assert data["llm"]["provider"] == "mock"
    assert data["llm"]["is_reachable"] is True
    assert data["llm"]["is_model_installed"] is True

    # 2. Deep diagnostic /api/health/llm
    llm_resp = client.get("/api/health/llm")
    assert llm_resp.status_code == 200
    llm_data = llm_resp.json()
    assert llm_data["provider"] == "mock"
    assert llm_data["generation_successful"] is True
    assert llm_data["test_latency_ms"] is not None


# ==============================================================================
# 8. Citation Preservation in Chat
# ==============================================================================

@pytest.mark.anyio
async def test_chat_citation_preservation_with_ollama(tmp_path, monkeypatch):
    """Verify ChatService correctly aligns citations returned by Ollama client with original chunks."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")

    book_id = "book_cit_ollama_01"
    book_dir = tmp_path / "uploads" / "books" / book_id
    book_dir.mkdir(parents=True, exist_ok=True)

    # Save mock book chunks
    meta = BookMetadata(
        book_id=book_id,
        original_filename="test.pdf",
        safe_filename="test.pdf",
        file_hash="mock_hash_123",
        file_size_bytes=1000,
        page_count=5,
        chunk_count=2,
        extraction_status="completed",
        created_at=datetime.now(timezone.utc),
    )
    from app.schemas.retrieval import HybridSearchResultItem

    c1 = ChunkMetadata(
        chunk_id="chunk_ollama_001",
        book_id=book_id,
        chunk_index=0,
        text="The core algorithm processes data deterministically.",
        page_start=1,
        page_end=1,
        chapter="Chapter 1",
        char_count=52,
        word_count=6,
    )
    c2 = ChunkMetadata(
        chunk_id="chunk_ollama_002",
        book_id=book_id,
        chunk_index=1,
        text="Secondary mechanism provides adaptive feedback.",
        page_start=2,
        page_end=2,
        chapter="Chapter 1",
        char_count=48,
        word_count=5,
    )

    storage_service.save_book_assets(
        book_id=book_id,
        pdf_bytes=b"%PDF-test",
        metadata=meta,
        chunks=[c1, c2],
        page_reports=[],
    )

    # Mock Ollama returning a structured answer with chunk citations
    answer_payload = {
        "crux": "The algorithm executes deterministically.",
        "detailed_synthesis": "As demonstrated in chunk_ollama_001, the system operates deterministically.",
        "citations": [
            {
                "chunk_id": "chunk_ollama_001",
                "page_start": 1,
                "page_end": 1,
                "chapter": "Chapter 1",
                "section": None,
                "quoted_snippet": "The core algorithm processes data deterministically.",
            }
        ],
        "insufficient_evidence": False,
        "confidence": "high",
    }

    transport = make_mock_ollama_transport(chat_responses=[json.dumps(answer_payload)])
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as mock_http:
        ollama = OllamaLLMClient(
            model_name="qwen2.5:3b",
            base_url="http://localhost:11434",
            http_client=mock_http,
        )

        mock_retrieval = MagicMock()
        mock_retrieval.retrieve_hybrid.return_value = [
            HybridSearchResultItem(chunk=c1, rrf_score=0.03, rrf_rank=1),
            HybridSearchResultItem(chunk=c2, rrf_score=0.02, rrf_rank=2),
        ]

        chat_svc = ChatService(retrieval_svc=mock_retrieval, llm=ollama)
        req = ChatRequest(message="How does the core algorithm process data?")
        resp = await chat_svc.answer_question(book_id=book_id, request=req)

        assert resp.crux_summary == "The algorithm executes deterministically."
        assert len(resp.citations) >= 1
        assert resp.citations[0].chunk_id == "chunk_ollama_001"
        assert resp.citations[0].page_start == 1


# ==============================================================================
# 9. Hierarchy Failure & Resumable Idempotent Execution
# ==============================================================================

@pytest.mark.anyio
async def test_hierarchy_failure_and_resumption(tmp_path, monkeypatch):
    """Verify that a failure during hierarchy generation saves completed units and resumes without recomputing."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    book_id = "book_hier_resumable_01"
    meta = BookMetadata(
        book_id=book_id,
        original_filename="test.pdf",
        safe_filename="test.pdf",
        file_hash="mock_hash_456",
        file_size_bytes=1000,
        page_count=4,
        chunk_count=2,
        extraction_status="completed",
        created_at=datetime.now(timezone.utc),
    )

    c1 = ChunkMetadata(
        chunk_id="chunk_h_001",
        book_id=book_id,
        chunk_index=0,
        text="Section 1 foundational text.",
        page_start=1,
        page_end=1,
        chapter="Chapter 1",
        char_count=28,
        word_count=4,
    )
    c2 = ChunkMetadata(
        chunk_id="chunk_h_002",
        book_id=book_id,
        chunk_index=1,
        text="Section 2 advanced text.",
        page_start=2,
        page_end=2,
        chapter="Chapter 2",
        char_count=24,
        word_count=4,
    )
    storage_service.save_book_assets(
        book_id=book_id,
        pdf_bytes=b"%PDF-test",
        metadata=meta,
        chunks=[c1, c2],
        page_reports=[],
    )

    # Create a mock LLM that succeeds on sections but fails on chapter synthesis
    call_counts = {"generate": 0}

    class FailingOnChapterLLM(MockLLMClient):
        async def generate_structured(self, system_prompt, user_prompt, response_model, max_retries=None):
            call_counts["generate"] += 1
            if response_model.__name__ == "ChapterSummary" and call_counts["generate"] == 3:
                raise RuntimeError("Simulated failure during chapter generation")
            return await super().generate_structured(system_prompt, user_prompt, response_model, max_retries)

    failing_llm = FailingOnChapterLLM()
    hier_svc = HierarchyService(llm_client=failing_llm)

    # First attempt: expect failure on chapter
    with pytest.raises(RuntimeError, match="Simulated failure during chapter generation"):
        await hier_svc.generate_hierarchy(book_id)

    # Check status: should be "failed" and have completed sections in completed_units
    saved_hier = hier_svc.load_hierarchy(book_id)
    assert saved_hier is not None
    assert saved_hier.generation_status == "failed"
    assert len(saved_hier.completed_units) >= 2  # Section units succeeded!

    # Second attempt with normal LLM: resumes from where it left off!
    working_llm = MockLLMClient()
    resumed_svc = HierarchyService(llm_client=working_llm)
    completed_hier = await resumed_svc.generate_hierarchy(book_id, force_restart=False)

    assert completed_hier.generation_status == "completed"
    assert completed_hier.progress_percent == 100.0
    assert len(completed_hier.sections) == 2
    assert len(completed_hier.chapters) == 2
    assert completed_hier.book_level is not None


# ==============================================================================
# 10. Live Local Ollama Verification (Optional / Gated by Reachability)
# ==============================================================================

@pytest.mark.anyio
async def test_live_local_ollama_health_if_running():
    """Live integration probe against local Ollama at http://localhost:11434 with qwen2.5:3b."""
    health = await check_llm_health(run_generation_test=True, settings=settings)

    if not health.is_reachable:
        pytest.skip("Local Ollama is not running; skipping live integration probe.")

    assert health.provider == "ollama"
    assert health.model == "qwen2.5:3b"
    assert health.is_reachable is True
    assert health.is_model_installed is True
    assert health.generation_successful is True
    assert health.test_latency_ms is not None
    print(f"\n[LIVE OLLAMA] Model: {health.model} | Latency: {health.test_latency_ms} ms")
