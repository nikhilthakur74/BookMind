"""Pydantic schemas for Phase 3 indexing and hybrid retrieval."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from app.schemas.book import ChunkMetadata


class RetrievalMode(str, Enum):
    """Retrieval execution strategy."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    BM25 = "bm25"


class SearchResultItem(BaseModel):
    """Result item returned by a single retrieval method (semantic or BM25)."""

    chunk: ChunkMetadata = Field(..., description="Full preserved chunk metadata")
    score: float = Field(..., description="Raw retrieval score (cosine similarity or BM25 score)")
    rank: int = Field(..., description="1-based ranking in the individual retrieval method")
    method: str = Field(..., description="Retrieval method used: 'semantic' or 'bm25'")


class HybridSearchResultItem(BaseModel):
    """Fused retrieval item returned by Reciprocal Rank Fusion."""

    chunk: ChunkMetadata = Field(..., description="Full preserved chunk metadata")
    rrf_score: float = Field(..., description="Fused score calculated via Reciprocal Rank Fusion")
    rrf_rank: int = Field(..., description="1-based ranking in the final fused hybrid results")
    semantic_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score from dense semantic retrieval, or None if not in top candidates",
    )
    semantic_rank: Optional[int] = Field(
        default=None,
        description="Rank in dense semantic retrieval, or None if not in top candidates",
    )
    bm25_score: Optional[float] = Field(
        default=None,
        description="Okapi BM25 score from keyword lexical retrieval, or None if not in top candidates",
    )
    bm25_rank: Optional[int] = Field(
        default=None,
        description="Rank in lexical retrieval, or None if not in top candidates",
    )


class RetrievalRequest(BaseModel):
    """Request payload for retrieving relevant book passages."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search or inquiry query string")
    mode: RetrievalMode = Field(
        default=RetrievalMode.HYBRID,
        description="Retrieval mode: 'hybrid' (RRF combined), 'semantic' (dense BGE-M3), or 'bm25' (lexical)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of top-ranked results to return",
    )
    semantic_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of semantic candidates to fetch before RRF fusion (defaults to max(20, top_k * 2))",
    )
    bm25_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of BM25 candidates to fetch before RRF fusion (defaults to max(20, top_k * 2))",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="RRF constant k used in reciprocal rank formula (standard is 60)",
    )


class RetrievalResponse(BaseModel):
    """Response returned by the retrieval API."""

    book_id: str = Field(..., description="ID of the searched book")
    query: str = Field(..., description="Original search query")
    mode: str = Field(..., description="Mode executed ('hybrid', 'semantic', or 'bm25')")
    top_k: int = Field(..., description="Requested top_k threshold")
    total_found: int = Field(..., description="Number of results returned")
    results: List[Union[HybridSearchResultItem, SearchResultItem]] = Field(
        ...,
        description="Ranked candidate chunks with preserved metadata and scores",
    )


class IndexingStatusResponse(BaseModel):
    """Response indicating the vector and lexical indexing status of a book."""

    book_id: str
    indexing_status: str = Field(..., description="'not_indexed', 'indexing', 'indexed', 'failed'")
    indexed_chunk_count: int
    total_chunks: int
    indexing_error: Optional[str] = None
    indexed_at: Optional[datetime] = None
