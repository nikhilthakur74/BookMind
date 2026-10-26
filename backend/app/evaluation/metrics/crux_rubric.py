"""Crux-specific conceptual completeness rubric for deep book understanding.

Provides manually reviewable, non-collapsed evaluation of:
1. Essential concepts covered (exact/near-exact concept presence)
2. Central thesis representation (explicit statement of the author's primary argument)
3. Important qualifications retained (caveats, boundary conditions, edge cases)
4. Supporting reasoning included (mechanisms, causality, reasoning chains)
5. Cross-chapter connections (relationships between distinct chapters/themes)

CRITICAL RULE:
Do not reduce these dimensions to an unexplained single score. Keep individual scores visible.
"""

import re
from typing import Dict, List, Optional, Sequence
from app.evaluation.schemas import CruxRubricResult


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for concept matching."""
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def evaluate_essential_concepts(
    answer: str,
    essential_concepts: Sequence[str],
) -> tuple[float, Dict[str, bool]]:
    """Evaluate what fraction of essential concepts appear in the generated answer."""
    if not essential_concepts:
        # If no rubric concepts were specified, full score with empty breakdown
        return 1.0, {}

    norm_answer = _normalize(answer)
    answer_words = set(norm_answer.split())
    breakdown: Dict[str, bool] = {}
    found_count = 0

    for concept in essential_concepts:
        norm_concept = _normalize(concept)
        concept_words = norm_concept.split()
        if not concept_words:
            continue

        # Check full phrase match or high token overlap (>75%)
        if norm_concept in norm_answer:
            breakdown[concept] = True
            found_count += 1
        elif len(concept_words) > 1:
            matched_words = sum(1 for w in concept_words if w in answer_words)
            if (matched_words / len(concept_words)) >= 0.75:
                breakdown[concept] = True
                found_count += 1
            else:
                breakdown[concept] = False
        else:
            breakdown[concept] = False

    score = round(found_count / len(essential_concepts), 4) if essential_concepts else 1.0
    return score, breakdown


def evaluate_central_thesis(
    answer: str,
    crux_summary: Optional[str] = None,
    reference_answer: Optional[str] = None,
) -> tuple[float, str]:
    """Score whether the central thesis or primary finding is prominently stated."""
    if not answer.strip():
        return 0.0, "Empty answer produced"

    # Crux-first indicator or prominent thesis statement
    leads_with_crux = False
    if crux_summary and crux_summary.strip():
        leads_with_crux = answer.strip().startswith(crux_summary.strip()[:40])

    thesis_markers = [
        "crux",
        "central thesis",
        "primary finding",
        "fundamental principle",
        "core argument",
        "the author argues",
        "the author's central finding",
        "direct crux",
        "comparative crux",
    ]
    has_marker = any(m in answer.lower() for m in thesis_markers)

    if leads_with_crux or has_marker:
        return 1.0, "Central thesis prominently articulated in crux-first structure"

    # Fallback: check if first sentence carries high information density
    first_sentence = answer.split(".")[0] if "." in answer else answer
    if len(first_sentence.split()) >= 8:
        return 0.75, "Primary thesis stated in opening sentence without formal crux label"

    return 0.4, "Thesis is diffuse or buried deep within the text"


def evaluate_important_qualifications(
    answer: str,
) -> tuple[float, str]:
    """Score retention of qualifications, caveats, boundary conditions, and limits."""
    qualification_markers = [
        "unless",
        "however",
        "provided that",
        "only when",
        "subject to",
        "under",
        "caveat",
        "qualification",
        "limitation",
        "assuming",
        "in contrast to",
        "sacrificing",
        "restricted to",
        "whereas",
        "boundary",
    ]
    found = [m for m in qualification_markers if m in answer.lower()]
    if len(found) >= 2:
        return 1.0, f"Multiple qualifications preserved ({', '.join(found[:3])})"
    elif len(found) == 1:
        return 0.7, f"Single qualification preserved ({found[0]})"
    return 0.3, "Few or no explicit qualifications or boundary limits retained"


def evaluate_supporting_reasoning(
    answer: str,
) -> tuple[float, str]:
    """Score presence of mechanistic explanations and causal reasoning chains."""
    reasoning_markers = [
        "because",
        "therefore",
        "consequently",
        "in order to",
        "ensures that",
        "prevents",
        "results in",
        "by verifying",
        "mechanism",
        "operates via",
        "transitions between",
        "leads to",
        "requires",
    ]
    found = [m for m in reasoning_markers if m in answer.lower()]
    if len(found) >= 3:
        return 1.0, f"Rich mechanistic explanation present ({', '.join(found[:3])})"
    elif len(found) >= 1:
        return 0.7, f"Basic causal reasoning present ({', '.join(found[:2])})"
    return 0.3, "Assertive claims without step-by-step explanatory reasoning"


def evaluate_cross_chapter_connections(
    answer: str,
    question_type: str,
) -> tuple[float, str]:
    """Score whether multi-chapter or comparative questions synthesize relationships with evidence."""
    if question_type not in ("comparison", "multi_hop", "cross_chapter_comparison"):
        return 1.0, "N/A: Single-concept query does not require cross-chapter links"

    comparative_markers = [
        "both",
        "whereas",
        "in contrast",
        "by comparison",
        "differ from",
        "while",
        "conversely",
        "on the other hand",
        "builds upon",
    ]
    found = [m for m in comparative_markers if m in answer.lower()]
    has_dual_structure = len(found) >= 1 and len(answer.split()) > 25

    if has_dual_structure:
        return 1.0, f"Evidenced comparative synthesis across concepts ({found[0]})"
    return 0.4, "Weak or one-sided comparison lacking balanced conceptual synthesis"


def evaluate_crux_rubric(
    answer: str,
    essential_concepts: Sequence[str],
    question_type: str,
    crux_summary: Optional[str] = None,
    reference_answer: Optional[str] = None,
) -> CruxRubricResult:
    """Calculate transparent 5-dimensional conceptual completeness rubric."""
    concept_score, concept_breakdown = evaluate_essential_concepts(answer, essential_concepts)
    thesis_score, thesis_notes = evaluate_central_thesis(answer, crux_summary, reference_answer)
    qual_score, qual_notes = evaluate_important_qualifications(answer)
    reason_score, reason_notes = evaluate_supporting_reasoning(answer)
    conn_score, conn_notes = evaluate_cross_chapter_connections(answer, question_type)

    rubric_breakdown = {
        "essential_concepts": {
            "score": concept_score,
            "details": concept_breakdown,
        },
        "central_thesis": {
            "score": thesis_score,
            "notes": thesis_notes,
        },
        "important_qualifications": {
            "score": qual_score,
            "notes": qual_notes,
        },
        "supporting_reasoning": {
            "score": reason_score,
            "notes": reason_notes,
        },
        "cross_chapter_connections": {
            "score": conn_score,
            "notes": conn_notes,
        },
    }

    return CruxRubricResult(
        essential_concepts_coverage=concept_score,
        central_thesis_representation=thesis_score,
        important_qualifications_retained=qual_score,
        supporting_reasoning_included=reason_score,
        cross_chapter_connections=conn_score,
        rubric_breakdown=rubric_breakdown,
    )
