"""Pydantic schemas for Phase 5 Conversational RAG API."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


def now_utc() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class QuestionType(str, Enum):
    """Categorization of user inquiry driving retrieval strategy and answer structure."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    CHAPTER_SUMMARY = "chapter_summary"
    BOOK_CRUX = "book_crux"
    CROSS_CHAPTER_COMPARISON = "cross_chapter_comparison"
    QUIZ = "quiz"


class Citation(BaseModel):
    """Structured verifiable citation linking directly to a source chunk and page range."""

    chunk_id: str = Field(..., description="ID of the grounding evidence chunk")
    page_start: int = Field(..., description="1-based starting page number")
    page_end: int = Field(..., description="1-based ending page number")
    chapter: Optional[str] = Field(default=None, description="Chapter heading if available")
    section: Optional[str] = Field(default=None, description="Section heading if available")
    quoted_snippet: str = Field(..., description="Exact or near-exact short snippet from chunk grounding the claim")


class ChatMessage(BaseModel):
    """Single turn in a persistent book conversation."""

    role: Literal["user", "assistant", "system"] = Field(...)
    content: str = Field(..., description="Message text")
    citations: Optional[List[Citation]] = Field(default=None, description="Citations supporting this response")
    question_type: Optional[QuestionType] = Field(default=None, description="Question classification")
    insufficient_evidence: bool = Field(default=False, description="Flag indicating evidence was insufficient")
    timestamp: datetime = Field(default_factory=now_utc)


class ChatRequest(BaseModel):
    """Request payload for POST /api/books/{book_id}/chat."""

    message: str = Field(..., min_length=1, max_length=4000, description="User question or inquiry")
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to continue an existing thread. Generated if omitted.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of evidence chunks to retrieve")
    enable_reranker: Optional[bool] = Field(
        default=None,
        description="Override system reranker setting (defaults to settings.ENABLE_RERANKER)",
    )
    include_hierarchy: bool = Field(
        default=True,
        description="Include relevant Level 2-4 hierarchical summaries in generation context",
    )


class ChatResponse(BaseModel):
    """Response returned by the conversational RAG endpoint."""

    conversation_id: str = Field(..., description="Active conversation thread ID")
    book_id: str = Field(..., description="Target book ID")
    question_type: QuestionType = Field(..., description="Detected question category")
    answer: str = Field(..., description="Grounded answer explaining crux first, then synthesizing details")
    crux_summary: Optional[str] = Field(default=None, description="High-level crux of the answer")
    citations: List[Citation] = Field(default_factory=list, description="Grounding source citations")
    insufficient_evidence: bool = Field(
        default=False,
        description="True if the book lacks sufficient evidence to answer the question",
    )
    confidence: Literal["high", "moderate", "low", "insufficient"] = Field(
        default="high",
        description="Grounded confidence level in answer completeness",
    )
    model_name: str = Field(..., description="Model used for synthesis")
    latency_ms: int = Field(..., description="Total processing latency in milliseconds")


class ConversationHistory(BaseModel):
    """Persisted multi-turn conversation thread for a book."""

    conversation_id: str
    book_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class StructuredAnswer(BaseModel):
    """Internal Pydantic schema enforcing structured JSON output from LLM."""

    crux: str = Field(..., description="The direct crux/core answer in 1-2 powerful sentences")
    detailed_synthesis: str = Field(..., description="Comprehensive detailed synthesis grounded in evidence")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Verifiable source citations using chunk IDs and page numbers provided in context",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="Set to true if the text does not contain enough evidence to answer",
    )
    confidence: Literal["high", "moderate", "low", "insufficient"] = Field(default="high")
