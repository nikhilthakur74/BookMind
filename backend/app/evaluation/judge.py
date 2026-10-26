"""Optional LLM-as-a-judge adapter for semantic evaluation.

IMPORTANT PRINCIPLES:
- Never present LLM-judge scores as objective ground truth.
- Track model name, prompt template, and evaluation version explicitly.
- Provide structured qualitative reasoning alongside scores.
- Runs only when explicitly enabled (enable_llm_judge=True). Deterministic custom metrics always run first.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Sequence
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.llm_client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)


class LLMJudgeScore(BaseModel):
    """Structured rating from the LLM Judge with transparent reasoning."""

    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    faithfulness_rationale: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    relevance_rationale: str
    correctness_score: float = Field(..., ge=0.0, le=1.0)
    correctness_rationale: str
    judge_model: str = Field(default="")
    judge_prompt_version: str = Field(default="")
    latency_ms: int = 0


class LLMJudge:
    """Evaluates answer faithfulness, relevance, and correctness using configured LLM."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client or get_llm_client()
        self.prompt_version = "judge-v1.0"

    async def judge_sample(
        self,
        question: str,
        generated_answer: str,
        retrieved_contexts: Sequence[str],
        reference_answer: Optional[str] = None,
    ) -> Optional[LLMJudgeScore]:
        """Perform LLM-assisted semantic assessment."""
        start = time.perf_counter()
        context_str = "\n\n".join(
            f"[Passage {i+1}]: {ctx}" for i, ctx in enumerate(retrieved_contexts[:5])
        )

        system_prompt = (
            "You are an impartial, highly rigorous evaluation judge assessing a RAG response.\n"
            "Score the following dimensions from 0.0 to 1.0 (with 1.0 being flawless):\n"
            "1. faithfulness_score: Are all factual claims in the answer supported by the retrieved passages? "
            "(If answer accurately declares insufficient evidence, score 1.0)\n"
            "2. relevance_score: Does the answer address the question asked directly and concisely?\n"
            "3. correctness_score: If a reference answer is provided, does the response convey the same core truth?\n"
            "Respond strictly in JSON matching the LLMJudgeScore schema with transparent rationales."
        )

        user_prompt = (
            f"QUESTION: {question}\n\n"
            f"RETRIEVED CONTEXT PASSAGES:\n{context_str}\n\n"
            f"GENERATED ANSWER:\n{generated_answer}\n\n"
            f"REFERENCE ANSWER:\n{reference_answer or 'N/A'}\n"
        )

        try:
            # If using mock LLM client, return deterministic judge score
            if settings.LLM_PROVIDER == "mock" or "mock" in self._llm.model_name.lower():
                latency_ms = int((time.perf_counter() - start) * 1000)
                is_insufficient = "insufficient evidence" in generated_answer.lower()
                return LLMJudgeScore(
                    faithfulness_score=1.0 if is_insufficient else 0.90,
                    faithfulness_rationale="Grounded in context; claims substantiate the primary assertion." if not is_insufficient else "Accurately declared insufficient evidence.",
                    relevance_score=0.92,
                    relevance_rationale="Directly answers the user inquiry with crux-first formulation.",
                    correctness_score=0.88,
                    correctness_rationale="High semantic alignment with ground-truth reference concepts.",
                    judge_model=self._llm.model_name,
                    judge_prompt_version=self.prompt_version,
                    latency_ms=latency_ms,
                )

            res = await self._llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=LLMJudgeScore,
            )
            res.latency_ms = int((time.perf_counter() - start) * 1000)
            res.judge_model = self._llm.model_name
            res.judge_prompt_version = self.prompt_version
            return res
        except Exception as e:
            logger.warning(f"LLM Judge evaluation failed gracefully: {e}")
            return None
