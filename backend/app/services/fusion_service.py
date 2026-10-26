"""Reciprocal Rank Fusion (RRF) algorithm for combining semantic and lexical retrieval.

RRF merges rankings from dense vector search (BGE-M3) and keyword search (BM25)
without requiring score calibration between disparate distributions.
Rewards consensus documents that appear near the top of both retrieval stages.
"""

from typing import Dict, List, Optional

from app.schemas.book import ChunkMetadata
from app.schemas.retrieval import HybridSearchResultItem, SearchResultItem


def reciprocal_rank_fusion(
    semantic_results: List[SearchResultItem],
    bm25_results: List[SearchResultItem],
    rrf_k: int = 60,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int = 5,
) -> List[HybridSearchResultItem]:
    """Combine ranked semantic and BM25 results into a unified hybrid result set.

    Formula:
        RRF_score(d) = sum_{m in {semantic, bm25}} ( w_m / (k + rank_m(d)) )

    Args:
        semantic_results: Dense vector retrieval results ordered by similarity descending.
        bm25_results: Lexical BM25 retrieval results ordered by score descending.
        rrf_k: Smoothing constant k (standard default is 60).
        semantic_weight: Multiplier weight for dense semantic rank.
        bm25_weight: Multiplier weight for lexical BM25 rank.
        top_k: Maximum number of combined results to return.

    Returns:
        List of HybridSearchResultItem with full preserved chunk metadata, fused score,
        and individual retrieval ranks and scores.
    """
    scores: Dict[str, float] = {}
    chunks: Dict[str, ChunkMetadata] = {}
    sem_ranks: Dict[str, int] = {}
    sem_scores: Dict[str, float] = {}
    bm25_ranks: Dict[str, int] = {}
    bm25_scores: Dict[str, float] = {}

    # 1. Process Semantic Results
    for rank, item in enumerate(semantic_results, start=1):
        cid = item.chunk.chunk_id
        chunks[cid] = item.chunk
        sem_ranks[cid] = rank
        sem_scores[cid] = item.score
        scores[cid] = scores.get(cid, 0.0) + (semantic_weight / (rrf_k + rank))

    # 2. Process BM25 Results
    for rank, item in enumerate(bm25_results, start=1):
        cid = item.chunk.chunk_id
        chunks[cid] = item.chunk
        bm25_ranks[cid] = rank
        bm25_scores[cid] = item.score
        scores[cid] = scores.get(cid, 0.0) + (bm25_weight / (rrf_k + rank))

    if not scores:
        return []

    # 3. Deterministic Sort:
    # Primary: RRF score descending
    # Secondary: Lowest semantic rank (or 9999 if absent)
    # Tertiary: Lowest BM25 rank (or 9999 if absent)
    # Quaternary: chunk_index ascending
    def sort_key(cid: str):
        return (
            -scores[cid],
            sem_ranks.get(cid, 99999),
            bm25_ranks.get(cid, 99999),
            chunks[cid].chunk_index,
            cid,
        )

    sorted_chunk_ids = sorted(scores.keys(), key=sort_key)
    selected_ids = sorted_chunk_ids[:top_k]

    fused_results: List[HybridSearchResultItem] = []
    for rrf_rank, cid in enumerate(selected_ids, start=1):
        fused_results.append(
            HybridSearchResultItem(
                chunk=chunks[cid],
                rrf_score=scores[cid],
                rrf_rank=rrf_rank,
                semantic_score=sem_scores.get(cid),
                semantic_rank=sem_ranks.get(cid),
                bm25_score=bm25_scores.get(cid),
                bm25_rank=bm25_ranks.get(cid),
            )
        )

    return fused_results
