"""Unified internal retrieval service for BookMind.

Provides high-level interfaces for:
1. Semantic dense vector retrieval (BGE-M3 + Qdrant)
2. Lexical keyword retrieval (Okapi BM25)
3. Hybrid retrieval via Reciprocal Rank Fusion (RRF)

Strictly enforces book boundary isolation and preserves all document chunk metadata.
"""

import logging
from typing import List, Optional

from app.core.config import settings
from app.schemas.retrieval import HybridSearchResultItem, SearchResultItem
from app.services.bm25_service import BM25Service, get_bm25_service
from app.services.embedding_service import BaseEmbeddingService, get_embedding_service
from app.services.fusion_service import reciprocal_rank_fusion
from app.services.qdrant_service import QdrantService, get_qdrant_service

logger = logging.getLogger(__name__)


class RetrievalService:
    """Internal service orchestrating semantic, lexical, and hybrid search."""

    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        bm25_service: Optional[BM25Service] = None,
    ):
        self._qdrant_service = qdrant_service or get_qdrant_service()
        self._embedding_service = embedding_service or get_embedding_service()
        self._bm25_service = bm25_service or get_bm25_service()

    @property
    def qdrant_service(self) -> QdrantService:
        return self._qdrant_service

    @property
    def embedding_service(self) -> BaseEmbeddingService:
        return self._embedding_service

    @property
    def bm25_service(self) -> BM25Service:
        return self._bm25_service

    def retrieve_semantic(
        self,
        query: str,
        book_id: str,
        top_k: int = 5,
    ) -> List[SearchResultItem]:
        """Execute independent dense vector semantic retrieval for a book."""
        if not query.strip():
            return []

        query_vector = self._embedding_service.embed_query(query)
        results = self._qdrant_service.search_semantic(
            query_vector=query_vector,
            book_id=book_id,
            top_k=top_k,
        )
        return results

    def retrieve_bm25(
        self,
        query: str,
        book_id: str,
        top_k: int = 5,
    ) -> List[SearchResultItem]:
        """Execute independent lexical keyword retrieval for a book."""
        if not query.strip():
            return []

        results = self._bm25_service.search_bm25(
            query=query,
            book_id=book_id,
            top_k=top_k,
        )
        return results

    def retrieve_hybrid(
        self,
        query: str,
        book_id: str,
        top_k: int = 5,
        semantic_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[HybridSearchResultItem]:
        """Execute hybrid search combining dense semantic and BM25 via Reciprocal Rank Fusion (RRF).

        Fetches wider candidate candidate pools from both channels (defaults to max(20, top_k * 2)),
        fuses the candidate rankings with RRF, and returns the top_k consensus passages.
        """
        if not query.strip():
            return []

        cand_sem = semantic_top_k if semantic_top_k is not None else max(20, top_k * 2)
        cand_bm25 = bm25_top_k if bm25_top_k is not None else max(20, top_k * 2)
        k_const = rrf_k if rrf_k is not None else settings.RRF_K

        # 1. Independent retrieval stages
        semantic_candidates = self.retrieve_semantic(query=query, book_id=book_id, top_k=cand_sem)
        bm25_candidates = self.retrieve_bm25(query=query, book_id=book_id, top_k=cand_bm25)

        # 2. Combine ranked candidate lists using RRF
        fused = reciprocal_rank_fusion(
            semantic_results=semantic_candidates,
            bm25_results=bm25_candidates,
            rrf_k=k_const,
            semantic_weight=1.0,
            bm25_weight=1.0,
            top_k=top_k,
        )

        return fused


_retrieval_service_instance: Optional[RetrievalService] = None


def get_retrieval_service(
    force_new: bool = False,
    qdrant_service: Optional[QdrantService] = None,
    embedding_service: Optional[BaseEmbeddingService] = None,
    bm25_service: Optional[BM25Service] = None,
) -> RetrievalService:
    """Return singleton or configured RetrievalService instance."""
    global _retrieval_service_instance
    if (
        force_new
        or qdrant_service is not None
        or embedding_service is not None
        or bm25_service is not None
    ):
        return RetrievalService(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            bm25_service=bm25_service,
        )
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance


def reset_retrieval_service() -> None:
    """Reset cached retrieval service singleton."""
    global _retrieval_service_instance
    _retrieval_service_instance = None
