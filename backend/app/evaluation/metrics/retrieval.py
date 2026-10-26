"""Information retrieval metrics for RAG evaluation.

Implements standard, unit-tested Information Retrieval (IR) metrics:
- Recall@k: Fraction of ground-truth relevant chunks present in top-k retrieved chunks.
- Precision@k: Fraction of top-k retrieved chunks that belong to the ground-truth relevant set.
- MRR (Mean Reciprocal Rank): Reciprocal of the 1-based rank of the first relevant chunk in retrieved list.
- Hit Rate@k: 1.0 if at least one ground-truth relevant chunk is in top-k, else 0.0.
"""

from typing import List, Sequence, Set
from app.evaluation.schemas import RetrievalMetrics


def compute_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Compute Recall@k.

    Formula: |Retrieved_k ∩ Expected| / |Expected|
    Returns 1.0 if Expected is empty (e.g. unanswerable query with no target chunks).
    """
    expected_set: Set[str] = set(expected_chunk_ids)
    if not expected_set:
        return 1.0

    top_k_retrieved: Set[str] = set(retrieved_chunk_ids[:k])
    relevant_retrieved = top_k_retrieved.intersection(expected_set)
    return round(len(relevant_retrieved) / len(expected_set), 4)


def compute_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Compute Precision@k.

    Formula: |Retrieved_k ∩ Expected| / k
    Returns 0.0 if k <= 0 or retrieved is empty.
    """
    if k <= 0 or not retrieved_chunk_ids:
        return 0.0

    expected_set: Set[str] = set(expected_chunk_ids)
    if not expected_set:
        # If no expected chunks are relevant, any retrieved chunk is not relevant
        return 0.0

    top_k_retrieved: Set[str] = set(retrieved_chunk_ids[:k])
    relevant_retrieved = top_k_retrieved.intersection(expected_set)
    # Use effective k (denominator is k to penalize short retrieval, or len(top_k_retrieved))
    denominator = k
    return round(len(relevant_retrieved) / denominator, 4)


def compute_mrr(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    Formula: 1 / rank_of_first_relevant_item (1-based rank).
    Returns 0.0 if no relevant chunk is retrieved in top-k.
    Returns 1.0 if Expected is empty and no chunks retrieved, else 0.0.
    """
    expected_set: Set[str] = set(expected_chunk_ids)
    if not expected_set:
        return 1.0 if not retrieved_chunk_ids else 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in expected_set:
            return round(1.0 / rank, 4)

    return 0.0


def compute_hit_rate_at_k(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int = 5,
) -> float:
    """Compute Hit Rate@k.

    Formula: 1.0 if |Retrieved_k ∩ Expected| > 0 else 0.0.
    Returns 1.0 if Expected is empty.
    """
    expected_set: Set[str] = set(expected_chunk_ids)
    if not expected_set:
        return 1.0

    top_k_retrieved: Set[str] = set(retrieved_chunk_ids[:k])
    return 1.0 if bool(top_k_retrieved.intersection(expected_set)) else 0.0


def calculate_retrieval_metrics(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Aggregate all retrieval metrics into structured Pydantic model."""
    return RetrievalMetrics(
        recall_at_k=compute_recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k),
        precision_at_k=compute_precision_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k),
        mrr=compute_mrr(retrieved_chunk_ids, expected_chunk_ids, k=k),
        hit_rate_at_k=compute_hit_rate_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k),
        k=k,
    )
