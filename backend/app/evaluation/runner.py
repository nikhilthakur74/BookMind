"""Evaluation experiment runner comparing Baseline RAG, Hybrid RAG, and Hierarchical RAG.

Executes side-by-side benchmark evaluations against versioned datasets, computes
full retrieval, generation, citation, and crux rubric metrics, and detects failure cases.
"""

import json
import logging
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.evaluation.judge import LLMJudge
from app.evaluation.metrics.generation import calculate_generation_metrics
from app.evaluation.metrics.retrieval import calculate_retrieval_metrics
from app.evaluation.schemas import (
    AggregateMetrics,
    EvaluationDataset,
    EvaluationItem,
    ExperimentRun,
    ExperimentRunRequest,
    FailureCase,
    PipelineType,
    SampleEvaluationResult,
    ValidationStatus,
)
from app.schemas.chat import ChatRequest, QuestionType, StructuredAnswer
from app.services import (
    chat_service,
    hierarchy_service,
    llm_client,
    reranker_service,
    retrieval_service,
    storage_service,
)
from app.services.chat_service import ChatService, get_chat_service
from app.services.hierarchy_service import HierarchyService, get_hierarchy_service
from app.services.llm_client import BaseLLMClient, get_llm_client
from app.services.reranker_service import RerankerService, get_reranker_service
from app.services.retrieval_service import RetrievalService, get_retrieval_service

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Coordinates execution and scoring across RAG pipeline architectures."""

    def __init__(
        self,
        retrieval_svc: Optional[RetrievalService] = None,
        hierarchy_svc: Optional[HierarchyService] = None,
        rerank_svc: Optional[RerankerService] = None,
        llm: Optional[BaseLLMClient] = None,
    ):
        self._retrieval_svc = retrieval_svc or get_retrieval_service()
        self._hierarchy_svc = hierarchy_svc or get_hierarchy_service()
        self._rerank_svc = rerank_svc or get_reranker_service()
        self._llm = llm or get_llm_client()
        self._chat_svc = get_chat_service()
        self._judge = LLMJudge(llm_client=self._llm)

    def get_runs_dir(self, book_id: str) -> Path:
        """Return path to experiment runs directory for a book."""
        book_dir = storage_service.get_book_dir(book_id)
        r_dir = book_dir / "evaluations" / "runs"
        r_dir.mkdir(parents=True, exist_ok=True)
        return r_dir

    def save_run(self, run: ExperimentRun) -> None:
        """Persist experiment run record to disk."""
        path = self.get_runs_dir(run.book_id) / f"{run.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(run.model_dump_json(indent=2))

    def get_run(self, book_id: str, run_id: str) -> Optional[ExperimentRun]:
        """Load experiment run from disk."""
        path = self.get_runs_dir(book_id) / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExperimentRun.model_validate(data)
        except Exception as e:
            logger.warning(f"Error loading experiment run '{run_id}': {e}")
            return None

    def list_runs(self, book_id: str) -> List[ExperimentRun]:
        """List all historical evaluation runs for a book."""
        r_dir = self.get_runs_dir(book_id)
        runs: List[ExperimentRun] = []
        for file in r_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                runs.append(ExperimentRun.model_validate(data))
            except Exception:
                continue
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs

    async def _execute_pipeline_query(
        self,
        book_id: str,
        question: str,
        pipeline_type: PipelineType,
        top_k: int,
    ) -> tuple[List[Any], List[str], str, Optional[str], List[Dict[str, Any]], bool, str, int]:
        """Execute a single query through the designated pipeline variant."""
        start_time = time.perf_counter()

        if pipeline_type == PipelineType.BASELINE:
            # Baseline: Dense vector retrieval only, no BM25, no hierarchical summaries, no reranking
            results = self._retrieval_svc.retrieve_semantic(query=question, book_id=book_id, top_k=top_k)
            retrieved_chunk_ids = [item.chunk.chunk_id for item in results]
            retrieved_contexts = [item.chunk.text for item in results]

            # Standard QA generation prompt
            context_block = "\n\n".join(
                f"[CHUNK {item.chunk.chunk_id} | Page {item.chunk.page_start}]: {item.chunk.text}"
                for item in results
            )
            system_prompt = (
                "You are a baseline question answering system.\n"
                "Answer the question using the provided context passages.\n"
                "If the context does not contain enough evidence, state that the evidence is insufficient.\n"
                "Respond in valid JSON conforming to the StructuredAnswer schema."
            )
            user_prompt = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"

            structured_resp: StructuredAnswer = await self._llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=StructuredAnswer,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            citations = [c.model_dump() for c in structured_resp.citations]
            full_answer = (
                f"{structured_resp.crux}\n\n{structured_resp.detailed_synthesis}"
                if not structured_resp.insufficient_evidence
                else (structured_resp.detailed_synthesis or structured_resp.crux)
            )
            return (
                retrieved_contexts,
                retrieved_chunk_ids,
                full_answer,
                structured_resp.crux,
                citations,
                structured_resp.insufficient_evidence,
                structured_resp.confidence,
                latency_ms,
            )

        elif pipeline_type == PipelineType.HYBRID:
            # Hybrid: Dense vector + BM25 + RRF, no hierarchical summaries, no reranking
            results = self._retrieval_svc.retrieve_hybrid(query=question, book_id=book_id, top_k=top_k)
            retrieved_chunk_ids = [item.chunk.chunk_id for item in results]
            retrieved_contexts = [item.chunk.text for item in results]

            context_block = "\n\n".join(
                f"[CHUNK {item.chunk.chunk_id} | Page {item.chunk.page_start}]: {item.chunk.text}"
                for item in results
            )
            system_prompt = (
                "You are an advanced hybrid retrieval question answering system.\n"
                "Answer the question accurately using the fused semantic and keyword context.\n"
                "Respond in valid JSON conforming to the StructuredAnswer schema."
            )
            user_prompt = f"HYBRID CONTEXT:\n{context_block}\n\nQUESTION: {question}"

            structured_resp: StructuredAnswer = await self._llm.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=StructuredAnswer,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            citations = [c.model_dump() for c in structured_resp.citations]
            full_answer = (
                f"{structured_resp.crux}\n\n{structured_resp.detailed_synthesis}"
                if not structured_resp.insufficient_evidence
                else (structured_resp.detailed_synthesis or structured_resp.crux)
            )
            return (
                retrieved_contexts,
                retrieved_chunk_ids,
                full_answer,
                structured_resp.crux,
                citations,
                structured_resp.insufficient_evidence,
                structured_resp.confidence,
                latency_ms,
            )

        elif pipeline_type in (PipelineType.HIERARCHICAL, PipelineType.HIERARCHICAL_RERANK):
            # Hierarchical RAG / Hierarchical with Reranking
            enable_rerank = pipeline_type == PipelineType.HIERARCHICAL_RERANK
            chat_req = ChatRequest(
                message=question,
                top_k=top_k,
                enable_reranker=enable_rerank,
                include_hierarchy=True,
            )
            chat_resp = await self._chat_svc.answer_question(book_id=book_id, request=chat_req)

            # Retrieve the underlying candidate contexts for faithfulness scoring
            hybrid_res = self._retrieval_svc.retrieve_hybrid(query=question, book_id=book_id, top_k=top_k)
            retrieved_chunk_ids = [c.chunk.chunk_id for c in hybrid_res]
            retrieved_contexts = [c.chunk.text for c in hybrid_res]

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            citations = [c.model_dump() for c in chat_resp.citations]
            return (
                retrieved_contexts,
                retrieved_chunk_ids,
                chat_resp.answer,
                chat_resp.crux_summary,
                citations,
                chat_resp.insufficient_evidence,
                chat_resp.confidence,
                latency_ms,
            )

        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        pipeline_type: PipelineType,
        top_k: int = 5,
        validated_only: bool = True,
        enable_llm_judge: bool = False,
    ) -> ExperimentRun:
        """Benchmark a specific RAG pipeline across all target items in a dataset."""
        overall_start = time.perf_counter()
        target_items = [
            item for item in dataset.items
            if (not validated_only) or (item.validation_status == ValidationStatus.VALIDATED)
        ]

        if not target_items:
            # Fall back to all items if none are explicitly validated yet
            target_items = dataset.items

        sample_results: List[SampleEvaluationResult] = []
        failure_cases: List[FailureCase] = []

        for item in target_items:
            # 1. Execute Pipeline
            (
                retrieved_contexts,
                retrieved_chunk_ids,
                gen_answer,
                crux_sum,
                citations,
                is_insufficient,
                confidence,
                latency_ms,
            ) = await self._execute_pipeline_query(
                book_id=dataset.book_id,
                question=item.question,
                pipeline_type=pipeline_type,
                top_k=top_k,
            )

            # 2. Compute Retrieval Metrics
            ret_metrics = calculate_retrieval_metrics(
                retrieved_chunk_ids=retrieved_chunk_ids,
                expected_chunk_ids=item.expected_chunk_ids,
                k=top_k,
            )

            # 3. Compute Generation & Crux Rubric Metrics
            gen_metrics = calculate_generation_metrics(
                question=item.question,
                generated_answer=gen_answer,
                reference_answer=item.reference_answer,
                retrieved_contexts=retrieved_contexts,
                retrieved_chunk_ids=retrieved_chunk_ids,
                expected_chunk_ids=item.expected_chunk_ids,
                is_answerable=item.is_answerable,
                insufficient_evidence=is_insufficient,
                citations=citations,
                essential_concepts=item.essential_concepts,
                question_type=item.question_type.value,
                crux_summary=crux_sum,
            )

            # 4. Optional LLM Judge Scoring
            if enable_llm_judge:
                judge_res = await self._judge.judge_sample(
                    question=item.question,
                    generated_answer=gen_answer,
                    retrieved_contexts=retrieved_contexts,
                    reference_answer=item.reference_answer,
                )
                if judge_res:
                    gen_metrics.crux_rubric.rubric_breakdown["llm_judge"] = judge_res.model_dump()

            # 5. Detect and Diagnose Failure Cases
            failures: List[str] = []
            if item.is_answerable and ret_metrics.recall_at_k == 0.0:
                failures.append("Retrieval Failure: None of the expected relevant chunks were retrieved in top-k.")
            if gen_metrics.citations.citation_validity < 1.0 and len(citations) > 0:
                failures.append("Citation Hallucination: One or more citations reference unretrieved or invalid chunk IDs.")
            if gen_metrics.abstention_quality == 0.0:
                failures.append(
                    "Abstention Failure: Model answered unanswerable inquiry, or falsely abstained on answerable inquiry."
                )
            if gen_metrics.faithfulness < 0.40:
                failures.append("Ungrounded Hallucination: Generated statements lack evidence support in retrieved text.")

            is_fail = len(failures) > 0
            if is_fail:
                failure_cases.append(
                    FailureCase(
                        item_id=item.item_id,
                        question=item.question,
                        question_type=item.question_type,
                        pipeline_type=pipeline_type,
                        failure_reasons=failures,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        expected_chunk_ids=item.expected_chunk_ids,
                        recall_at_k=ret_metrics.recall_at_k,
                        faithfulness=gen_metrics.faithfulness,
                        unsupported_claim_rate=gen_metrics.unsupported_claim_rate,
                        citation_validity=gen_metrics.citations.citation_validity,
                        abstention_quality=gen_metrics.abstention_quality,
                    )
                )

            sample_results.append(
                SampleEvaluationResult(
                    item_id=item.item_id,
                    question=item.question,
                    question_type=item.question_type,
                    pipeline_type=pipeline_type,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    generated_answer=gen_answer,
                    crux_summary=crux_sum,
                    generated_citations=citations,
                    insufficient_evidence=is_insufficient,
                    confidence=confidence,
                    retrieval_metrics=ret_metrics,
                    generation_metrics=gen_metrics,
                    latency_ms=latency_ms,
                    is_failure=is_fail,
                    failure_reasons=failures,
                )
            )

        duration = round(time.perf_counter() - overall_start, 3)

        # 6. Compute Aggregate Means
        n = len(sample_results) or 1
        agg = AggregateMetrics(
            sample_count=len(sample_results),
            mean_recall_at_k=round(statistics.mean(s.retrieval_metrics.recall_at_k for s in sample_results), 4),
            mean_precision_at_k=round(statistics.mean(s.retrieval_metrics.precision_at_k for s in sample_results), 4),
            mean_mrr=round(statistics.mean(s.retrieval_metrics.mrr for s in sample_results), 4),
            mean_hit_rate_at_k=round(statistics.mean(s.retrieval_metrics.hit_rate_at_k for s in sample_results), 4),
            mean_faithfulness=round(statistics.mean(s.generation_metrics.faithfulness for s in sample_results), 4),
            mean_unsupported_claim_rate=round(statistics.mean(s.generation_metrics.unsupported_claim_rate for s in sample_results), 4),
            mean_answer_relevance=round(statistics.mean(s.generation_metrics.answer_relevance for s in sample_results), 4),
            mean_answer_correctness=round(statistics.mean(s.generation_metrics.answer_correctness for s in sample_results), 4),
            mean_context_relevance=round(statistics.mean(s.generation_metrics.context_relevance for s in sample_results), 4),
            mean_abstention_quality=round(statistics.mean(s.generation_metrics.abstention_quality for s in sample_results), 4),
            mean_citation_validity=round(statistics.mean(s.generation_metrics.citations.citation_validity for s in sample_results), 4),
            mean_citation_precision=round(statistics.mean(s.generation_metrics.citations.citation_precision for s in sample_results), 4),
            mean_citation_recall=round(statistics.mean(s.generation_metrics.citations.citation_recall for s in sample_results), 4),
            mean_essential_concepts_coverage=round(statistics.mean(s.generation_metrics.crux_rubric.essential_concepts_coverage for s in sample_results), 4),
            mean_central_thesis_representation=round(statistics.mean(s.generation_metrics.crux_rubric.central_thesis_representation for s in sample_results), 4),
            mean_important_qualifications_retained=round(statistics.mean(s.generation_metrics.crux_rubric.important_qualifications_retained for s in sample_results), 4),
            mean_supporting_reasoning_included=round(statistics.mean(s.generation_metrics.crux_rubric.supporting_reasoning_included for s in sample_results), 4),
            mean_cross_chapter_connections=round(statistics.mean(s.generation_metrics.crux_rubric.cross_chapter_connections for s in sample_results), 4),
            failure_case_count=len(failure_cases),
            mean_latency_ms=round(statistics.mean(s.latency_ms for s in sample_results), 2),
        )

        run_id = f"run_{dataset.book_id[:8]}_{pipeline_type.value}_{uuid.uuid4().hex[:6]}"
        run_record = ExperimentRun(
            run_id=run_id,
            book_id=dataset.book_id,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            pipeline_type=pipeline_type,
            pipeline_config={
                "top_k": top_k,
                "validated_only": validated_only,
                "enable_llm_judge": enable_llm_judge,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "llm_model": settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else self._llm.model_name,
                "llm_provider": settings.LLM_PROVIDER,
                "prompt_version": settings.PROMPT_VERSION,
            },
            duration_seconds=duration,
            aggregate_metrics=agg,
            sample_results=sample_results,
            failure_cases=failure_cases,
        )

        self.save_run(run_record)
        return run_record


_runner_instance: Optional[EvaluationRunner] = None


def get_evaluation_runner() -> EvaluationRunner:
    """Return singleton EvaluationRunner instance."""
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = EvaluationRunner()
    return _runner_instance
