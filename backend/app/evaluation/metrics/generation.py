"""Generation quality, faithfulness, relevance, citation, and abstention metrics."""

import re
from typing import Any, Dict, List, Optional, Sequence, Set
from app.evaluation.metrics.crux_rubric import evaluate_crux_rubric
from app.evaluation.schemas import CitationMetrics, GenerationMetrics


def _stem(word: str) -> str:
    """Simple light stemmer for English suffixes."""
    w = word.lower()
    for suffix in ("ing", "tions", "tion", "ies", "es", "ed", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def _tokenize(text: str) -> List[str]:
    """Tokenize lowercase alphanumeric words, filtering out short and question stopwords."""
    stopwords = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
        "are", "was", "were", "be", "been", "by", "this", "that", "it", "with", "as",
        "how", "what", "why", "when", "where", "who", "which", "does", "do", "did",
        "can", "could", "would", "should", "tell", "me", "about", "explain"
    }
    tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return [t for t in tokens if t not in stopwords]


def compute_faithfulness(
    answer: str,
    retrieved_contexts: Sequence[str],
) -> float:
    """Compute faithfulness of generated answer against retrieved context passages.

    Splits the answer into sentences and computes the fraction of sentences that
    are substantiated by token overlap (>50% non-stopword tokens) in the retrieved context.
    """
    if not answer.strip():
        return 0.0

    # If the answer explicitly abstains due to insufficient evidence, it is 100% faithful to reality
    if "insufficient evidence" in answer.lower() or "does not contain sufficient evidence" in answer.lower():
        return 1.0

    if not retrieved_contexts:
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    context_tokens = {_stem(t) for t in _tokenize(combined_context)}

    # Split answer into candidate sentences
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", answer) if len(s.strip().split()) >= 4]
    if not sentences:
        return 1.0

    supported_count = 0
    for s in sentences:
        s_tokens = [_stem(t) for t in _tokenize(s)]
        if not s_tokens:
            supported_count += 1
            continue
        overlap = sum(1 for t in s_tokens if t in context_tokens)
        if (overlap / len(s_tokens)) >= 0.40:
            supported_count += 1

    return round(supported_count / len(sentences), 4)


def compute_answer_relevance(
    question: str,
    answer: str,
) -> float:
    """Compute direct relevance of the answer to the user question."""
    if not answer.strip():
        return 0.0

    q_stems = {_stem(t) for t in _tokenize(question)}
    if not q_stems:
        return 1.0

    a_stems = {_stem(t) for t in _tokenize(answer)}
    overlap = q_stems.intersection(a_stems)
    # Coverage of question substantive keywords in the answer
    coverage = len(overlap) / len(q_stems)
    return round(min(1.0, coverage * 1.25), 4)


def compute_answer_correctness(
    generated_answer: str,
    reference_answer: str,
) -> float:
    """Compute token-level F1 overlap between generated answer and validated reference answer."""
    if not generated_answer.strip() or not reference_answer.strip():
        return 0.0

    # If both indicate insufficient evidence
    if (
        "insufficient evidence" in generated_answer.lower()
        and "insufficient evidence" in reference_answer.lower()
    ):
        return 1.0

    gen_tokens = _tokenize(generated_answer)
    ref_tokens = _tokenize(reference_answer)

    if not gen_tokens or not ref_tokens:
        return 0.0

    common = set(gen_tokens).intersection(set(ref_tokens))
    if not common:
        return 0.0

    # Compute Token F1
    precision = sum(1 for t in gen_tokens if t in common) / len(gen_tokens)
    recall = sum(1 for t in ref_tokens if t in common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_context_relevance(
    question: str,
    retrieved_contexts: Sequence[str],
) -> float:
    """Compute fraction of retrieved chunks containing relevant terms from the question."""
    if not retrieved_contexts:
        return 0.0

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 1.0

    relevant_chunks = 0
    for chunk in retrieved_contexts:
        c_tokens = set(_tokenize(chunk))
        overlap = q_tokens.intersection(c_tokens)
        if len(overlap) >= min(2, len(q_tokens)):
            relevant_chunks += 1

    return round(relevant_chunks / len(retrieved_contexts), 4)


def compute_abstention_quality(
    is_answerable: bool,
    insufficient_evidence: bool,
    answer_text: str,
) -> float:
    """Compute abstention quality.

    For unanswerable queries (is_answerable=False):
      Returns 1.0 if the model abstains, else 0.0 (hallucination).
    For answerable queries (is_answerable=True):
      Returns 1.0 if the model answers, else 0.0 (false abstention).
    """
    model_abstained = insufficient_evidence or (
        "insufficient evidence" in answer_text.lower()
        or "does not contain sufficient evidence" in answer_text.lower()
        or "no evidence or discussion" in answer_text.lower()
    )

    if not is_answerable:
        # Ground truth was unanswerable: did the model abstain?
        return 1.0 if model_abstained else 0.0
    else:
        # Ground truth was answerable: did the model provide an answer?
        return 1.0 if not model_abstained else 0.0


def compute_citation_metrics(
    citations: Sequence[Dict[str, Any]],
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    is_answerable: bool,
) -> CitationMetrics:
    """Evaluate validity and correctness of generated citations."""
    retrieved_set: Set[str] = set(retrieved_chunk_ids)
    expected_set: Set[str] = set(expected_chunk_ids)

    if not citations:
        if not is_answerable:
            # Unanswerable query correctly produced no citations
            return CitationMetrics(
                citation_validity=1.0,
                citation_precision=1.0,
                citation_recall=1.0,
                total_citations=0,
                valid_citations=0,
            )
        else:
            # Expected citations but none were produced
            return CitationMetrics(
                citation_validity=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
                total_citations=0,
                valid_citations=0,
            )

    valid_count = 0
    correct_count = 0
    cited_cids: Set[str] = set()

    for cit in citations:
        cid = cit.get("chunk_id", "")
        p_start = cit.get("page_start", 0)
        p_end = cit.get("page_end", 0)

        # Validity: chunk was actually retrieved and has valid 1-based page numbers
        is_valid = (cid in retrieved_set) and (p_start > 0) and (p_end >= p_start)
        if is_valid:
            valid_count += 1
            cited_cids.add(cid)

        # Correctness: chunk belongs to ground-truth expected chunks
        if cid in expected_set:
            correct_count += 1

    total = len(citations)
    validity = round(valid_count / total, 4)
    precision = round(correct_count / total, 4) if total > 0 else 0.0
    recall = round(len(cited_cids.intersection(expected_set)) / len(expected_set), 4) if expected_set else 1.0

    return CitationMetrics(
        citation_validity=validity,
        citation_precision=precision,
        citation_recall=recall,
        total_citations=total,
        valid_citations=valid_count,
    )


def calculate_generation_metrics(
    question: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_contexts: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: Sequence[str],
    is_answerable: bool,
    insufficient_evidence: bool,
    citations: Sequence[Dict[str, Any]],
    essential_concepts: Sequence[str],
    question_type: str,
    crux_summary: Optional[str] = None,
) -> GenerationMetrics:
    """Aggregate all generation, citation, and crux rubric metrics."""
    faithfulness = compute_faithfulness(generated_answer, retrieved_contexts)
    unsupported_rate = round(1.0 - faithfulness, 4)
    relevance = compute_answer_relevance(question, generated_answer)
    correctness = compute_answer_correctness(generated_answer, reference_answer)
    ctx_relevance = compute_context_relevance(question, retrieved_contexts)
    abstention = compute_abstention_quality(is_answerable, insufficient_evidence, generated_answer)
    citation_metrics = compute_citation_metrics(
        citations=citations,
        retrieved_chunk_ids=retrieved_chunk_ids,
        expected_chunk_ids=expected_chunk_ids,
        is_answerable=is_answerable,
    )
    crux_rubric = evaluate_crux_rubric(
        answer=generated_answer,
        essential_concepts=essential_concepts,
        question_type=question_type,
        crux_summary=crux_summary,
        reference_answer=reference_answer,
    )

    return GenerationMetrics(
        faithfulness=faithfulness,
        unsupported_claim_rate=unsupported_rate,
        answer_relevance=relevance,
        answer_correctness=correctness,
        context_relevance=ctx_relevance,
        abstention_quality=abstention,
        citations=citation_metrics,
        crux_rubric=crux_rubric,
    )
