"""Pydantic schemas for Phase 4: Hierarchical Book Understanding.

Defines the Four Knowledge Levels:
1. Evidence: original chunks with stable IDs and exact page ranges (referenced).
2. Section: concise summaries, key concepts, definitions, reasoning chains, and source chunk references.
3. Chapter: chapter crux, central argument, supporting points, assumptions, examples, conclusions, and connections.
4. Book: central thesis, essential takeaways (5-10), chapter relationships, caveats, and overall crux.

Distinguishes author-stated conclusions from model-inferred connections and enforces provenance preservation.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


def now_utc() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


ClaimType = Literal["author_stated", "model_inferred"]
ConfidenceLevel = Literal["high", "moderate", "low", "insufficient"]


# ==============================================================================
# Level 2: Section Knowledge Components
# ==============================================================================

class ConceptRecord(BaseModel):
    """An essential concept extracted from text with strict source attribution."""

    name: str = Field(..., description="Name of the concept or term")
    definition: str = Field(..., description="Precise definition based on book text")
    claim_type: ClaimType = Field(
        default="author_stated",
        description="Whether the definition is directly author-stated or model-inferred",
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="IDs of source chunks grounding this concept",
    )
    page_references: List[int] = Field(
        default_factory=list,
        description="1-based page numbers where concept is introduced or elaborated",
    )


class DefinitionRecord(BaseModel):
    """A formal definition given in the book."""

    term: str = Field(..., description="Defined term")
    explanation: str = Field(..., description="Author's definition or formulation")
    verbatim_or_paraphrased: Literal["verbatim", "paraphrased"] = Field(default="paraphrased")
    source_chunk_id: str = Field(..., description="Chunk ID containing the definition")
    page_number: int = Field(..., description="1-based page number")


class SectionSummary(BaseModel):
    """Level 2: Grounded summary of a specific chapter section or chunk cluster."""

    section_id: str = Field(..., description="Unique section identifier, e.g. 'sec_ch01_01'")
    chapter_title: str = Field(..., description="Title of the parent chapter")
    section_title: Optional[str] = Field(default=None, description="Title of the section if identified")
    page_start: int = Field(..., description="1-based starting page")
    page_end: int = Field(..., description="1-based ending page")
    concise_summary: str = Field(..., description="Concise, factual summary of the section content")
    key_concepts: List[ConceptRecord] = Field(default_factory=list, description="Concepts introduced")
    definitions: List[DefinitionRecord] = Field(default_factory=list, description="Definitions provided")
    reasoning_chain: List[str] = Field(
        default_factory=list,
        description="Sequential logical steps or argumentative progression used by author",
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact source chunk IDs grounding this section summary",
    )
    uncertainty: Optional[str] = Field(
        default=None,
        description="Uncertainty notes if evidence is sparse, ambiguous, or OCR-degraded",
    )
    prompt_version: str = Field(default="v1.0")
    model_version: str = Field(default="llama-3.3-70b-versatile")
    created_at: datetime = Field(default_factory=now_utc)


# ==============================================================================
# Level 3: Chapter Knowledge Components
# ==============================================================================

class ArgumentRecord(BaseModel):
    """A core supporting argument in a chapter."""

    claim: str = Field(..., description="The claim or thesis of this argument")
    reasoning: str = Field(..., description="How the author substantiates this claim")
    claim_type: ClaimType = Field(default="author_stated")
    source_chunk_ids: List[str] = Field(default_factory=list)
    page_range: str = Field(default="", description="Human-readable page range, e.g. 'pp. 14-18'")


class AssumptionRecord(BaseModel):
    """An important underlying assumption made in the text."""

    assumption: str = Field(..., description="The assumption identified")
    explicit_or_implicit: Literal["explicit", "implicit"] = Field(default="implicit")
    source_chunk_ids: List[str] = Field(default_factory=list)


class ExampleRecord(BaseModel):
    """An illustrative example or counterexample presented by the author."""

    description: str = Field(..., description="Description of the case or scenario")
    is_counterexample: bool = Field(default=False, description="True if presented as a counterexample")
    point_illustrated: str = Field(..., description="What claim or concept this illustrates")
    source_chunk_ids: List[str] = Field(default_factory=list)


class ConclusionRecord(BaseModel):
    """A conclusion reached within the chapter."""

    statement: str = Field(..., description="The concluded outcome or thesis")
    claim_type: ClaimType = Field(default="author_stated")
    source_chunk_ids: List[str] = Field(default_factory=list)


class ConnectionRecord(BaseModel):
    """A prerequisite or forward/backward link to another chapter."""

    related_chapter_title: str = Field(..., description="Title of the referenced chapter")
    nature_of_connection: str = Field(..., description="Prerequisite, continuation, contrast, or application")
    claim_type: ClaimType = Field(default="author_stated")
    source_chunk_ids: List[str] = Field(default_factory=list)


class ChapterSummary(BaseModel):
    """Level 3: Comprehensive synthesis of a complete chapter."""

    chapter_index: int = Field(..., description="0-based sequence index of the chapter")
    chapter_title: str = Field(..., description="Authentic chapter title from book structure")
    page_start: int = Field(..., description="1-based starting page")
    page_end: int = Field(..., description="1-based ending page")
    chapter_crux: str = Field(
        ...,
        description="The central, irreducible essence or primary thesis of this chapter",
    )
    central_argument: str = Field(
        ...,
        description="Detailed explanation of the chapter's overarching argument",
    )
    supporting_arguments: List[ArgumentRecord] = Field(
        default_factory=list,
        description="Key supporting points and rationale",
    )
    essential_concepts: List[ConceptRecord] = Field(
        default_factory=list,
        description="Crucial concepts and terms defined in this chapter",
    )
    important_assumptions: List[AssumptionRecord] = Field(
        default_factory=list,
        description="Assumptions underpinning the chapter's reasoning",
    )
    examples: List[ExampleRecord] = Field(
        default_factory=list,
        description="Empirical or hypothetical examples used",
    )
    counterexamples: List[ExampleRecord] = Field(
        default_factory=list,
        description="Counterexamples or boundary cases discussed",
    )
    conclusions: List[ConclusionRecord] = Field(
        default_factory=list,
        description="Key author conclusions reached",
    )
    prerequisites_and_connections: List[ConnectionRecord] = Field(
        default_factory=list,
        description="Prerequisites and conceptual bridges to other chapters",
    )
    supporting_section_ids: List[str] = Field(
        default_factory=list,
        description="IDs of the Level 2 section summaries rolled up into this chapter",
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="All Level 1 chunk IDs grounding this chapter",
    )
    evidence_confidence: ConfidenceLevel = Field(
        default="high",
        description="Confidence in evidence sufficiency: 'high', 'moderate', 'low', 'insufficient'",
    )
    uncertainty: Optional[str] = Field(
        default=None,
        description="Clear uncertainty declaration when text is sparse or ambiguous",
    )
    stage_count: int = Field(
        default=1,
        description="Number of staged summarization passes used for long chapters",
    )
    prompt_version: str = Field(default="v1.0")
    model_version: str = Field(default="llama-3.3-70b-versatile")
    created_at: datetime = Field(default_factory=now_utc)


# ==============================================================================
# Level 4: Book Knowledge Components
# ==============================================================================

class TakeawayRecord(BaseModel):
    """An essential takeaway from the entire book."""

    point: str = Field(..., description="High-impact takeaway or principle")
    explanation: str = Field(..., description="Detailed explanation grounded in the text")
    claim_type: ClaimType = Field(default="author_stated")
    source_chunk_ids: List[str] = Field(default_factory=list)
    page_numbers: List[int] = Field(default_factory=list)


class ThemeRecord(BaseModel):
    """A recurring motif or theme spanning multiple chapters."""

    theme: str = Field(..., description="Name or description of recurring theme")
    manifestations: str = Field(..., description="How this theme evolves across the book")
    chapters_present: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)


class ChapterRelationshipRecord(BaseModel):
    """Structural and conceptual relationship between two chapters."""

    source_chapter: str = Field(..., description="Source chapter title")
    target_chapter: str = Field(..., description="Related chapter title")
    relationship_type: Literal[
        "builds_upon",
        "contrasts",
        "exemplifies",
        "generalizes",
        "chronological",
    ] = Field(default="builds_upon")
    description: str = Field(..., description="Nature of the conceptual dependency or progression")
    claim_type: ClaimType = Field(default="author_stated")
    source_chunk_ids: List[str] = Field(default_factory=list)


class BookSummary(BaseModel):
    """Level 4: Comprehensive synthesis of the entire book."""

    book_id: str = Field(..., description="Unique book identifier")
    central_thesis: str = Field(
        ...,
        description="The overarching thesis that unites and drives the entire book",
    )
    overall_crux: str = Field(
        ...,
        description="The condensed crux/essence: the fundamental insight without losing evidence",
    )
    recurring_themes: List[ThemeRecord] = Field(
        default_factory=list,
        description="Key themes that recur throughout different chapters",
    )
    essential_takeaways: List[TakeawayRecord] = Field(
        default_factory=list,
        description="5 to 10 essential takeaways directly supported by evidence",
    )
    chapter_relationships: List[ChapterRelationshipRecord] = Field(
        default_factory=list,
        description="Explicit relationships and transitions between chapters",
    )
    important_caveats: List[str] = Field(
        default_factory=list,
        description="Scope limits, qualifications, or author-stated caveats",
    )
    uncertainty: Optional[str] = Field(
        default=None,
        description="Uncertainty declaration if any chapters were missing or sparse",
    )
    supporting_chapter_titles: List[str] = Field(
        default_factory=list,
        description="List of chapters synthesizing this book summary",
    )
    source_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Aggregate of all grounding chunk IDs referenced across the book",
    )
    prompt_version: str = Field(default="v1.0")
    model_version: str = Field(default="llama-3.3-70b-versatile")
    created_at: datetime = Field(default_factory=now_utc)


# ==============================================================================
# Full Hierarchy Container & Request/Response
# ==============================================================================

class BookHierarchy(BaseModel):
    """Container holding all Four Knowledge Levels for an ingested book."""

    book_id: str
    generation_status: Literal["not_started", "in_progress", "completed", "failed"] = Field(
        default="not_started",
        description="Current state of hierarchical summarization",
    )
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    sections: List[SectionSummary] = Field(default_factory=list)
    chapters: List[ChapterSummary] = Field(default_factory=list)
    book_level: Optional[BookSummary] = Field(default=None)
    completed_units: List[str] = Field(
        default_factory=list,
        description="Completed unit keys (e.g. 'sec_xxx', 'chap_xxx') enabling idempotent resume",
    )
    error: Optional[str] = Field(default=None)
    model_name: str = Field(default="llama-3.3-70b-versatile")
    prompt_version: str = Field(default="v1.0")
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class HierarchyGenerateRequest(BaseModel):
    """Request to generate or resume hierarchical summarization."""

    force_restart: bool = Field(
        default=False,
        description="If True, clears previous progress and restarts from Level 2",
    )


class HierarchyStatusResponse(BaseModel):
    """Status summary of hierarchical knowledge generation."""

    book_id: str
    generation_status: str
    progress_percent: float
    sections_count: int
    chapters_count: int
    has_book_summary: bool
    error: Optional[str] = None
    updated_at: datetime
