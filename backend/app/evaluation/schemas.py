"""Pydantic schemas for BookMind Phase 6 Reproducible RAG Evaluation Framework."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


def now_utc() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class EvaluationQuestionType(str, Enum):
    """Supported question categories for systematic evaluation."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    SUMMARIZATION = "summarization"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    UNANSWERABLE = "unanswerable"


class ValidationStatus(str, Enum):
    """Validation lifecycle of an evaluation item or dataset."""

    DRAFT = "draft"
    VALIDATED = "validated"
    REJECTED = "rejected"


class PipelineType(str, Enum):
    """RAG pipeline variants under comparison."""

    BASELINE = "baseline_rag"
    HYBRID = "hybrid_rag"
    HIERARCHICAL = "hierarchical_rag"
    HIERARCHICAL_RERANK = "hierarchical_rag_rerank"


# ==============================================================================
# Evaluation Dataset Schemas
# ==============================================================================

class EvaluationItem(BaseModel):
    """Single test query in an evaluation dataset with ground-truth labels."""

    item_id: str = Field(..., description="Unique ID for this question (e.g. q_001)")
    question: str = Field(..., min_length=3, description="Evaluation question")
    question_type: EvaluationQuestionType = Field(..., description="Category of the question")
    reference_answer: str = Field(..., description="Ground-truth reference answer")
    expected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs that contain the necessary grounding evidence",
    )
    expected_page_numbers: List[int] = Field(
        default_factory=list,
        description="1-based page numbers containing the evidence",
    )
    is_answerable: bool = Field(
        default=True,
        description="Whether the book is expected to contain sufficient evidence to answer",
    )
    essential_concepts: List[str] = Field(
        default_factory=list,
        description="Optional list of essential concepts or claims that must be covered",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.DRAFT,
        description="Whether this item has been manually reviewed and validated as ground truth",
    )
    notes: Optional[str] = Field(default=None, description="Reviewer notes or rationale")
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class EvaluationDataset(BaseModel):
    """Versioned collection of evaluation items for a book."""

    dataset_id: str = Field(..., description="Unique dataset identifier (e.g. ds_book123_v1)")
    book_id: str = Field(..., description="Associated book ID")
    version: str = Field(default="1.0.0", description="Semantic dataset version")
    description: str = Field(..., description="Dataset purpose and description")
    items: List[EvaluationItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class ItemUpdateRequest(BaseModel):
    """Payload to review or update validation status of an item."""

    validation_status: ValidationStatus
    reference_answer: Optional[str] = None
    expected_chunk_ids: Optional[List[str]] = None
    expected_page_numbers: Optional[List[int]] = None
    essential_concepts: Optional[List[str]] = None
    is_answerable: Optional[bool] = None
    notes: Optional[str] = None


class DatasetGenerateRequest(BaseModel):
    """Request to automatically generate a draft evaluation dataset using LLM assistance."""

    questions_per_type: int = Field(default=2, ge=1, le=10, description="Draft questions per category")
    include_types: Optional[List[EvaluationQuestionType]] = Field(
        default=None,
        description="Subset of question types to generate (all if omitted)",
    )


# ==============================================================================
# Metric Schemas
# ==============================================================================

class RetrievalMetrics(BaseModel):
    """Information retrieval performance metrics against expected chunk IDs."""

    recall_at_k: float = Field(..., ge=0.0, le=1.0, description="Fraction of expected chunks retrieved in top-k")
    precision_at_k: float = Field(..., ge=0.0, le=1.0, description="Fraction of retrieved top-k chunks that are relevant")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank of first relevant chunk")
    hit_rate_at_k: float = Field(..., ge=0.0, le=1.0, description="1.0 if at least one relevant chunk in top-k, else 0.0")
    k: int = Field(default=5)


class CitationMetrics(BaseModel):
    """Verification and correctness metrics for generated citations."""

    citation_validity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of citations linking to valid retrieved chunks with accurate page numbers",
    )
    citation_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of cited chunks that belong to expected ground-truth chunks",
    )
    citation_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of expected ground-truth chunks that were cited",
    )
    total_citations: int = Field(default=0)
    valid_citations: int = Field(default=0)


class CruxRubricResult(BaseModel):
    """Conceptual completeness rubric scores kept visible as individual dimensions."""

    essential_concepts_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of required essential concepts covered in answer",
    )
    central_thesis_representation: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating accurate articulation of central thesis",
    )
    important_qualifications_retained: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating preservation of boundary conditions and caveats",
    )
    supporting_reasoning_included: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating presence of causal mechanisms and logical explanation",
    )
    cross_chapter_connections: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating evidenced synthesis across multiple chapters/concepts",
    )
    rubric_breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed evidence and notes for each conceptual dimension",
    )


class GenerationMetrics(BaseModel):
    """Answer quality, faithfulness, relevance, and abstention metrics."""

    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Fraction of claims grounded in retrieved context")
    unsupported_claim_rate: float = Field(..., ge=0.0, le=1.0, description="1.0 - faithfulness (hallucination rate)")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="Direct relevance of answer to user question")
    answer_correctness: float = Field(..., ge=0.0, le=1.0, description="Agreement with validated reference answer")
    context_relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance of retrieved context to the query")
    abstention_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="1.0 if abstained on unanswerable or answered on answerable; 0.0 otherwise",
    )
    citations: CitationMetrics
    crux_rubric: CruxRubricResult


# ==============================================================================
# Sample & Run Schemas
# ==============================================================================

class SampleEvaluationResult(BaseModel):
    """Evaluation result for a single question under a specific pipeline."""

    item_id: str
    question: str
    question_type: EvaluationQuestionType
    pipeline_type: PipelineType
    retrieved_chunk_ids: List[str]
    generated_answer: str
    crux_summary: Optional[str] = None
    generated_citations: List[Dict[str, Any]] = Field(default_factory=list)
    insufficient_evidence: bool = False
    confidence: str = "high"
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    latency_ms: int = 0
    token_usage: Optional[Dict[str, int]] = None
    is_failure: bool = False
    failure_reasons: List[str] = Field(default_factory=list)


class FailureCase(BaseModel):
    """Detailed diagnosis of an individual failure case."""

    item_id: str
    question: str
    question_type: EvaluationQuestionType
    pipeline_type: PipelineType
    failure_reasons: List[str]
    retrieved_chunk_ids: List[str]
    expected_chunk_ids: List[str]
    recall_at_k: float
    faithfulness: float
    unsupported_claim_rate: float
    citation_validity: float
    abstention_quality: float
    notes: Optional[str] = None


class AggregateMetrics(BaseModel):
    """Mean and aggregate statistics across all evaluated items in an experiment run."""

    sample_count: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_hit_rate_at_k: float
    mean_faithfulness: float
    mean_unsupported_claim_rate: float
    mean_answer_relevance: float
    mean_answer_correctness: float
    mean_context_relevance: float
    mean_abstention_quality: float
    mean_citation_validity: float
    mean_citation_precision: float
    mean_citation_recall: float
    mean_essential_concepts_coverage: float
    mean_central_thesis_representation: float
    mean_important_qualifications_retained: float
    mean_supporting_reasoning_included: float
    mean_cross_chapter_connections: float
    failure_case_count: int
    mean_latency_ms: float


class ExperimentRun(BaseModel):
    """Complete record of an evaluation experiment run with full history tracking."""

    run_id: str = Field(..., description="Unique run identifier (e.g. run_20261020_001)")
    book_id: str
    dataset_id: str
    dataset_version: str
    pipeline_type: PipelineType
    pipeline_config: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=now_utc)
    duration_seconds: float = 0.0
    aggregate_metrics: AggregateMetrics
    sample_results: List[SampleEvaluationResult] = Field(default_factory=list)
    failure_cases: List[FailureCase] = Field(default_factory=list)


class ExperimentRunRequest(BaseModel):
    """Request payload to execute an evaluation run."""

    dataset_id: str
    pipelines: List[PipelineType] = Field(
        default=[
            PipelineType.BASELINE,
            PipelineType.HYBRID,
            PipelineType.HIERARCHICAL,
        ],
        description="List of pipeline variants to benchmark",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    validated_only: bool = Field(
        default=True,
        description="Evaluate only items with validation_status == 'validated'",
    )
    enable_llm_judge: bool = Field(
        default=False,
        description="Optionally enable LLM-judge for semantic faithfulness and rubric scoring",
    )


class PipelineComparisonReport(BaseModel):
    """Comparative benchmark report contrasting all evaluated pipelines side-by-side."""

    book_id: str
    dataset_id: str
    dataset_version: str
    timestamp: datetime = Field(default_factory=now_utc)
    runs: Dict[str, ExperimentRun] = Field(default_factory=dict)
    comparative_summary: List[Dict[str, Any]] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
