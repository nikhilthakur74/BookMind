"""Reranker service for BookMind Phase 5.

Re-scores retrieved candidate chunks against the user question to maximize
relevance before context assembly and LLM answer generation.
"""

import logging
import re
from typing import List, Optional

from app.core.config import settings
from app.schemas.book import ChunkMetadata
from app.schemas.retrieval import HybridSearchResultItem

logger = logging.getLogger(__name__)


class RerankerService:
    """Reranks hybrid search candidates based on term density, heading alignment, and exact phrase matching."""

    def __init__(self, enabled: Optional[bool] = None):
        self._enabled = enabled if enabled is not None else settings.ENABLE_RERANKER

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResultItem],
        top_k: int = 5,
    ) -> List[HybridSearchResultItem]:
        """Rerank candidates using cross-scoring alignment."""
        if not self._enabled or not candidates or len(candidates) <= 1:
            return candidates[:top_k]

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        if not query_terms:
            return candidates[:top_k]

        scored_candidates = []
        for item in candidates:
            chunk = item.chunk
            text_lower = chunk.text.lower()
            text_terms = re.findall(r"\b\w+\b", text_lower)
            doc_len = max(len(text_terms), 1)

            # 1. Base RRF contribution (0.0 to 1.0 normalized)
            base_score = item.rrf_score * 30.0

            # 2. Exact query phrase match bonus
            phrase_bonus = 2.0 if query.lower().strip() in text_lower else 0.0

            # 3. Term coverage and density
            matched_terms = query_terms.intersection(text_terms)
            term_coverage = len(matched_terms) / len(query_terms)
            term_freq_sum = sum(text_lower.count(t) for t in matched_terms)
            term_density = term_freq_sum / doc_len

            # 4. Heading alignment bonus
            heading_bonus = 0.0
            if chunk.chapter and any(t in chunk.chapter.lower() for t in query_terms):
                heading_bonus += 0.5
            if chunk.section and any(t in chunk.section.lower() for t in query_terms):
                heading_bonus += 0.5

            total_rerank_score = (
                base_score
                + (term_coverage * 3.0)
                + (term_density * 2.0)
                + phrase_bonus
                + heading_bonus
            )
            scored_candidates.append((total_rerank_score, item))

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_candidates[:top_k]]


_reranker_service_instance: Optional[RerankerService] = None


def get_reranker_service(enabled: Optional[bool] = None) -> RerankerService:
    """Return singleton or configured RerankerService instance."""
    global _reranker_service_instance
    if enabled is not None or _reranker_service_instance is None:
        return RerankerService(enabled=enabled)
    return _reranker_service_instance
