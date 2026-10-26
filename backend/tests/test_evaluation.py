"""Comprehensive tests for BookMind Phase 6: Reproducible RAG Evaluation Framework.

Tests:
1. Retrieval metrics with deterministic known values: Recall@k, Precision@k, MRR, HitRate@k.
2. Generation metrics: Faithfulness, Unsupported-Claim Rate, Answer Relevance, Answer Correctness, Context Relevance, Abstention Quality.
3. Citation metrics: Citation Validity, Precision, and Recall against ground-truth expected chunks.
4. Crux conceptual completeness rubric: Individual scores for all 5 dimensions.
5. Dataset lifecycle: Draft generation, manual review status transition (draft -> validated/rejected).
6. Multi-pipeline evaluation runner: Benchmarking Baseline, Hybrid, and Hierarchical RAG.
7. Failure case diagnosis and inspection.
8. JSON & CSV export generation.
9. FastAPI endpoints for datasets, runs, item validation, and exports.
"""

from datetime import datetime, timezone
import json
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.core.config import settings
from app.evaluation.dataset_service import EvaluationDatasetService, get_dataset_service
from app.evaluation.export import ExportService
from app.evaluation.metrics.crux_rubric import (
    evaluate_central_thesis,
    evaluate_cross_chapter_connections,
    evaluate_crux_rubric,
    evaluate_essential_concepts,
    evaluate_important_qualifications,
    evaluate_supporting_reasoning,
)
from app.evaluation.metrics.generation import (
    calculate_generation_metrics,
    compute_abstention_quality,
    compute_answer_correctness,
    compute_answer_relevance,
    compute_citation_metrics,
    compute_context_relevance,
    compute_faithfulness,
)
from app.evaluation.metrics.retrieval import (
    calculate_retrieval_metrics,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)
from app.evaluation.runner import EvaluationRunner, get_evaluation_runner
from app.evaluation.schemas import (
    DatasetGenerateRequest,
    EvaluationDataset,
    EvaluationItem,
    EvaluationQuestionType,
    ExperimentRunRequest,
    ItemUpdateRequest,
    PipelineType,
    ValidationStatus,
)
from app.main import app
from app.schemas.book import BookMetadata, ChunkMetadata
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
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.embedding_service import DeterministicMockEmbeddingService
from app.services.hierarchy_service import HierarchyService
from app.services.indexing_service import IndexingService
from app.services.llm_client import MockLLMClient
from app.services.qdrant_service import QdrantService
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService

client = TestClient(app)


# ==============================================================================
# Fixtures & Environment Isolation
# ==============================================================================

@pytest.fixture(autouse=True)
def setup_eval_test_env(tmp_path, monkeypatch):
    """Provide isolated environment with in-memory Qdrant and mock services."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "QDRANT_STORAGE_DIR", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    mem_qdrant_client = QdrantClient(location=":memory:")
    test_qdrant_svc = QdrantService(
        client=mem_qdrant_client,
        collection_name="test_eval_collection",
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
    test_ds_svc = EvaluationDatasetService(llm_client=test_llm)
    test_runner = EvaluationRunner(
        retrieval_svc=test_ret_svc,
        hierarchy_svc=test_hier_svc,
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
    monkeypatch.setattr(
        "app.evaluation.dataset_service.get_dataset_service",
        lambda *args, **kwargs: test_ds_svc,
    )
    monkeypatch.setattr(
        "app.evaluation.runner.get_evaluation_runner",
        lambda *args, **kwargs: test_runner,
    )

    yield {
        "llm": test_llm,
        "runner": test_runner,
        "ds_svc": test_ds_svc,
        "tmp_path": tmp_path,
    }


def seed_book_with_chunks(book_id: str, passages: list) -> tuple:
    """Helper creating and indexing a book in isolated test storage."""
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
        pdf_bytes=b"%PDF-eval-test",
        metadata=meta,
        chunks=chunks,
        page_reports=[],
    )

    idx_svc = indexing_service.get_indexing_service()
    idx_svc.index_book(book_id, force=True)
    return meta, chunks


# ==============================================================================
# 1. Retrieval Metrics Deterministic Tests
# ==============================================================================

def test_retrieval_metrics_exact_values():
    """Verify Recall@k, Precision@k, MRR, and HitRate@k against known IR ground truth."""
    # Scenario: Expected relevant chunks are C1, C2, C3.
    # Retrieved chunks in order: [C9, C1, C4, C2, C8] with k=5.
    retrieved = ["c9", "c1", "c4", "c2", "c8"]
    expected = ["c1", "c2", "c3"]

    # 1. Recall@5: 2 of 3 expected items found -> 2/3 = 0.6667
    assert compute_recall_at_k(retrieved, expected, k=5) == 0.6667

    # 2. Precision@5: 2 of 5 retrieved items relevant -> 2/5 = 0.4000
    assert compute_precision_at_k(retrieved, expected, k=5) == 0.4000

    # 3. MRR: First relevant item is C1 at rank 2 (index 1) -> 1/2 = 0.5000
    assert compute_mrr(retrieved, expected, k=5) == 0.5000

    # 4. Hit Rate@5: At least one relevant item in top-5 -> 1.0
    assert compute_hit_rate_at_k(retrieved, expected, k=5) == 1.0

    # Edge Case: Zero matches
    assert compute_recall_at_k(["x1", "x2"], expected, k=2) == 0.0
    assert compute_precision_at_k(["x1", "x2"], expected, k=2) == 0.0
    assert compute_mrr(["x1", "x2"], expected, k=2) == 0.0
    assert compute_hit_rate_at_k(["x1", "x2"], expected, k=2) == 0.0

    # Edge Case: Empty expected (e.g. unanswerable query with no target chunks)
    assert compute_recall_at_k(retrieved, [], k=5) == 1.0
    assert compute_hit_rate_at_k(retrieved, [], k=5) == 1.0


# ==============================================================================
# 2. Generation & Citation Metrics Tests
# ==============================================================================

def test_faithfulness_and_unsupported_claim_rate():
    """Verify faithfulness and unsupported claim rate scoring."""
    context = [
        "Distributed consensus requires quorum agreement across nodes to prevent split-brain anomalies.",
        "Leader election in Raft transitions nodes between follower, candidate, and leader states.",
    ]

    # Faithful answer closely matching context tokens
    faithful_answer = (
        "Distributed consensus requires quorum agreement across cluster nodes to prevent split-brain anomalies. "
        "In Raft, leader election transitions nodes between follower, candidate, and leader states."
    )
    faithfulness = compute_faithfulness(faithful_answer, context)
    assert faithfulness == 1.0

    # Hallucinated / unsupported answer
    hallucinated_answer = (
        "The system uses Byzantine quantum teleportation across subterranean optical cables. "
        "Every transaction executes via asynchronous proof of stake blockchains."
    )
    unfaithful_score = compute_faithfulness(hallucinated_answer, context)
    assert unfaithful_score < 0.25

    # Abstention answer is inherently 100% faithful
    abstention_answer = "The text does not contain sufficient evidence to answer."
    assert compute_faithfulness(abstention_answer, context) == 1.0


def test_answer_relevance_and_correctness():
    """Verify relevance to question and correctness against reference answer."""
    question = "How does the Raft protocol conduct leader election?"
    reference = "Raft conducts leader election by transitioning candidate nodes and requesting quorum votes."
    candidate = "Raft conducts leader election by transitioning nodes from follower to candidate and requesting votes."

    relevance = compute_answer_relevance(question, candidate)
    assert relevance >= 0.80

    correctness = compute_answer_correctness(candidate, reference)
    assert correctness >= 0.60


def test_abstention_quality_metric():
    """Verify abstention quality for both unanswerable and answerable items."""
    # 1. Unanswerable item: model correctly abstains -> 1.0
    assert compute_abstention_quality(
        is_answerable=False,
        insufficient_evidence=True,
        answer_text="The text does not contain sufficient evidence to answer.",
    ) == 1.0

    # 2. Unanswerable item: model hallucinated an answer -> 0.0
    assert compute_abstention_quality(
        is_answerable=False,
        insufficient_evidence=False,
        answer_text="Here is the recipe for baking chocolate cake.",
    ) == 0.0

    # 3. Answerable item: model answered -> 1.0
    assert compute_abstention_quality(
        is_answerable=True,
        insufficient_evidence=False,
        answer_text="Quorum requires a strict majority of active nodes.",
    ) == 1.0

    # 4. Answerable item: model falsely abstained -> 0.0
    assert compute_abstention_quality(
        is_answerable=True,
        insufficient_evidence=True,
        answer_text="The text does not contain sufficient evidence.",
    ) == 0.0


def test_citation_metrics():
    """Verify citation validity, precision, and recall."""
    retrieved = ["c001", "c002", "c003"]
    expected = ["c001", "c002"]

    # 2 citations: 1 valid and correct (c001), 1 hallucinated chunk (c999)
    citations = [
        {"chunk_id": "c001", "page_start": 1, "page_end": 1},
        {"chunk_id": "c999", "page_start": 5, "page_end": 5},
    ]

    metrics = compute_citation_metrics(
        citations=citations,
        retrieved_chunk_ids=retrieved,
        expected_chunk_ids=expected,
        is_answerable=True,
    )

    # 1 out of 2 citations is in retrieved set -> validity = 0.50
    assert metrics.citation_validity == 0.50
    # 1 out of 2 citations is in expected set -> precision = 0.50
    assert metrics.citation_precision == 0.50
    # 1 out of 2 expected chunks cited -> recall = 0.50
    assert metrics.citation_recall == 0.50


# ==============================================================================
# 3. Crux Conceptual Completeness Rubric Tests
# ==============================================================================

def test_crux_rubric_individual_dimensions():
    """Verify that all 5 crux conceptual completeness rubric dimensions are individually scored."""
    answer = (
        "Direct Crux: The author's central finding directly addresses quorum consensus.\n\n"
        "According to the text, distributed consensus requires strict quorum agreement "
        "because this ensures split-brain anomalies are prevented. However, this is only valid "
        "unless network partitions isolate the leader, whereas decentralized protocols differ."
    )
    essential_concepts = ["quorum consensus", "split-brain"]

    rubric = evaluate_crux_rubric(
        answer=answer,
        essential_concepts=essential_concepts,
        question_type="comparison",
        crux_summary="Direct Crux: The author's central finding directly addresses quorum consensus.",
    )

    # 1. Essential concepts coverage (both concepts present)
    assert rubric.essential_concepts_coverage == 1.0
    # 2. Central thesis representation (starts with Direct Crux)
    assert rubric.central_thesis_representation == 1.0
    # 3. Important qualifications retained (contains 'however', 'unless', 'whereas')
    assert rubric.important_qualifications_retained >= 0.70
    # 4. Supporting reasoning included (contains 'because', 'requires', 'ensures')
    assert rubric.supporting_reasoning_included >= 0.70
    # 5. Cross-chapter connections (contains 'whereas', 'differ')
    assert rubric.cross_chapter_connections == 1.0

    # Ensure individual scores are visible in rubric_breakdown
    assert "essential_concepts" in rubric.rubric_breakdown
    assert "central_thesis" in rubric.rubric_breakdown
    assert "important_qualifications" in rubric.rubric_breakdown
    assert "supporting_reasoning" in rubric.rubric_breakdown
    assert "cross_chapter_connections" in rubric.rubric_breakdown


# ==============================================================================
# 4. Dataset Lifecycle & Manual Validation Workflow
# ==============================================================================

def test_dataset_lifecycle_manual_validation():
    """Verify draft generation, manual review status transitions (draft -> validated/rejected)."""
    book_id = "book_eval_ds_01"
    passages = [
        ("Chapter 1: Foundations", "Linearizability guarantees that operations appear to execute instantaneously."),
        ("Chapter 2: Partitions", "Under network partitions, distributed systems must trade off consistency or availability."),
    ]
    seed_book_with_chunks(book_id, passages)

    ds_svc = get_dataset_service()

    # 1. Generate draft dataset with LLM assistance
    dataset = ds_svc.generate_draft_dataset(book_id)
    assert dataset.dataset_id is not None
    assert len(dataset.items) >= 4

    # Crucial rule: Generated items MUST be marked 'draft'
    for item in dataset.items:
        assert item.validation_status == ValidationStatus.DRAFT

    # 2. Manually review and validate Item 1
    target_item_id = dataset.items[0].item_id
    update_req = ItemUpdateRequest(
        validation_status=ValidationStatus.VALIDATED,
        notes="Reviewed and approved by domain expert.",
    )
    updated = ds_svc.update_item(book_id, dataset.dataset_id, target_item_id, update_req)
    assert updated is not None
    assert updated.validation_status == ValidationStatus.VALIDATED
    assert updated.notes == "Reviewed and approved by domain expert."

    # 3. Reject Item 2
    item2_id = dataset.items[1].item_id
    reject_req = ItemUpdateRequest(
        validation_status=ValidationStatus.REJECTED,
        notes="Question ambiguous; rejected.",
    )
    rejected = ds_svc.update_item(book_id, dataset.dataset_id, item2_id, reject_req)
    assert rejected.validation_status == ValidationStatus.REJECTED

    # 4. Reload dataset from storage and verify persistence
    reloaded = ds_svc.get_dataset(book_id, dataset.dataset_id)
    assert reloaded is not None
    item1 = [i for i in reloaded.items if i.item_id == target_item_id][0]
    assert item1.validation_status == ValidationStatus.VALIDATED


# ==============================================================================
# 5. Multi-Pipeline Runner & Failure Case Diagnosis
# ==============================================================================

@pytest.mark.anyio
async def test_multi_pipeline_runner_and_failures():
    """Verify comparative execution across Baseline, Hybrid, and Hierarchical RAG."""
    book_id = "book_pipe_test_01"
    passages = [
        ("Chapter 1: Storage", "LSM trees optimize write amplification by appending to an immutable memtable."),
        ("Chapter 2: Indexing", "B-trees maintain balanced search trees with predictable random access page latencies."),
    ]
    _, chunks = seed_book_with_chunks(book_id, passages)

    # Create explicit test dataset with answerable and unanswerable questions
    item1 = EvaluationItem(
        item_id="q1",
        question="How do LSM trees optimize write amplification?",
        question_type=EvaluationQuestionType.FACTUAL,
        reference_answer="LSM trees optimize write amplification by appending writes sequentially to an immutable memtable.",
        expected_chunk_ids=[chunks[0].chunk_id],
        expected_page_numbers=[chunks[0].page_start],
        is_answerable=True,
        essential_concepts=["LSM trees", "write amplification"],
        validation_status=ValidationStatus.VALIDATED,
    )
    item2 = EvaluationItem(
        item_id="q2",
        question="What is the recipe for baking chocolate cake?",
        question_type=EvaluationQuestionType.UNANSWERABLE,
        reference_answer="The book does not discuss baking chocolate cake.",
        expected_chunk_ids=[],
        expected_page_numbers=[],
        is_answerable=False,
        essential_concepts=["insufficient evidence"],
        validation_status=ValidationStatus.VALIDATED,
    )
    dataset = EvaluationDataset(
        dataset_id="ds_pipe_test",
        book_id=book_id,
        version="1.0.0",
        description="Test dataset for pipeline comparison",
        items=[item1, item2],
    )
    ds_svc = get_dataset_service()
    ds_svc.save_dataset(dataset)

    runner = get_evaluation_runner()

    # Benchmark Baseline RAG
    base_run = await runner.evaluate_dataset(
        dataset=dataset,
        pipeline_type=PipelineType.BASELINE,
        top_k=3,
        validated_only=True,
    )
    assert base_run.pipeline_type == PipelineType.BASELINE
    assert len(base_run.sample_results) == 2
    assert base_run.aggregate_metrics.mean_recall_at_k > 0.0

    # Benchmark Hierarchical RAG
    hier_run = await runner.evaluate_dataset(
        dataset=dataset,
        pipeline_type=PipelineType.HIERARCHICAL,
        top_k=3,
        validated_only=True,
    )
    assert hier_run.pipeline_type == PipelineType.HIERARCHICAL
    assert len(hier_run.sample_results) == 2
    assert hier_run.aggregate_metrics.mean_central_thesis_representation >= 0.80

    # Check Run Persistence
    saved_run = runner.get_run(book_id, hier_run.run_id)
    assert saved_run is not None
    assert saved_run.run_id == hier_run.run_id


# ==============================================================================
# 6. JSON and CSV Export Tests
# ==============================================================================

def test_export_json_and_csv():
    """Verify serialization to formatted JSON and CSV with visible individual metric columns."""
    ret_m = calculate_retrieval_metrics(["c1"], ["c1"], k=3)
    gen_m = calculate_generation_metrics(
        question="What is quorum?",
        generated_answer="Quorum is a majority agreement.",
        reference_answer="Quorum is majority consensus.",
        retrieved_contexts=["Quorum is majority consensus."],
        retrieved_chunk_ids=["c1"],
        expected_chunk_ids=["c1"],
        is_answerable=True,
        insufficient_evidence=False,
        citations=[{"chunk_id": "c1", "page_start": 1, "page_end": 1}],
        essential_concepts=["quorum"],
        question_type="factual",
    )

    from app.evaluation.schemas import AggregateMetrics, SampleEvaluationResult, ExperimentRun
    sample = SampleEvaluationResult(
        item_id="q01",
        question="What is quorum?",
        question_type=EvaluationQuestionType.FACTUAL,
        pipeline_type=PipelineType.HIERARCHICAL,
        retrieved_chunk_ids=["c1"],
        generated_answer="Quorum is a majority agreement.",
        retrieval_metrics=ret_m,
        generation_metrics=gen_m,
    )
    agg = AggregateMetrics(
        sample_count=1,
        mean_recall_at_k=1.0,
        mean_precision_at_k=0.33,
        mean_mrr=1.0,
        mean_hit_rate_at_k=1.0,
        mean_faithfulness=1.0,
        mean_unsupported_claim_rate=0.0,
        mean_answer_relevance=1.0,
        mean_answer_correctness=0.8,
        mean_context_relevance=1.0,
        mean_abstention_quality=1.0,
        mean_citation_validity=1.0,
        mean_citation_precision=1.0,
        mean_citation_recall=1.0,
        mean_essential_concepts_coverage=1.0,
        mean_central_thesis_representation=1.0,
        mean_important_qualifications_retained=1.0,
        mean_supporting_reasoning_included=1.0,
        mean_cross_chapter_connections=1.0,
        failure_case_count=0,
        mean_latency_ms=45.0,
    )
    run = ExperimentRun(
        run_id="run_test_export",
        book_id="book_test",
        dataset_id="ds_test",
        dataset_version="1.0.0",
        pipeline_type=PipelineType.HIERARCHICAL,
        aggregate_metrics=agg,
        sample_results=[sample],
    )

    # 1. JSON Export
    json_out = ExportService.export_run_json(run)
    parsed = json.loads(json_out)
    assert parsed["run_id"] == "run_test_export"
    assert parsed["aggregate_metrics"]["mean_recall_at_k"] == 1.0

    # 2. CSV Export
    csv_out = ExportService.export_run_samples_csv(run)
    assert "item_id" in csv_out
    assert "recall_at_k" in csv_out
    assert "essential_concepts_coverage" in csv_out
    assert "q01" in csv_out


# ==============================================================================
# 7. FastAPI Endpoint Integration Tests
# ==============================================================================

def test_evaluation_api_endpoints():
    """Verify HTTP endpoints for dataset generation, item validation, run execution, and export."""
    book_id = "book_api_eval_01"
    passages = [
        ("Chapter 1: Overview", "The primary objective of BookMind is grounded synthesis of long-form literature."),
    ]
    seed_book_with_chunks(book_id, passages)

    # 1. POST /datasets/generate
    gen_resp = client.post(f"/api/books/{book_id}/evaluations/datasets/generate")
    assert gen_resp.status_code == 201
    ds_data = gen_resp.json()
    dataset_id = ds_data["dataset_id"]
    item_id = ds_data["items"][0]["item_id"]

    # 2. GET /datasets
    list_resp = client.get(f"/api/books/{book_id}/evaluations/datasets")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

    # 3. PUT /datasets/{dataset_id}/items/{item_id} (manual validation)
    put_resp = client.put(
        f"/api/books/{book_id}/evaluations/datasets/{dataset_id}/items/{item_id}",
        json={"validation_status": "validated", "notes": "Approved for testing"},
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["validation_status"] == "validated"

    # 4. POST /run (Execute benchmark on validated items)
    run_payload = {
        "dataset_id": dataset_id,
        "pipelines": ["baseline_rag", "hierarchical_rag"],
        "top_k": 3,
        "validated_only": True,
        "enable_llm_judge": False,
    }
    run_resp = client.post(f"/api/books/{book_id}/evaluations/run", json=run_payload)
    assert run_resp.status_code == 200
    report_data = run_resp.json()
    assert "baseline_rag" in report_data["runs"]
    assert "hierarchical_rag" in report_data["runs"]
    baseline_run_id = report_data["runs"]["baseline_rag"]["run_id"]

    # 5. GET /runs/{run_id}
    detail_resp = client.get(f"/api/books/{book_id}/evaluations/runs/{baseline_run_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["run_id"] == baseline_run_id

    # 6. GET /runs/{run_id}/export?format=csv
    csv_resp = client.get(f"/api/books/{book_id}/evaluations/runs/{baseline_run_id}/export?format=csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "item_id" in csv_resp.text
