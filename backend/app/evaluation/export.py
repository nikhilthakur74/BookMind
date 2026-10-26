"""JSON and CSV export service for evaluation runs and comparison reports."""

import csv
import io
import json
from typing import Any, Dict, List
from app.evaluation.schemas import ExperimentRun, PipelineComparisonReport


class ExportService:
    """Serializes experiment runs and benchmark reports to JSON and CSV formats."""

    @staticmethod
    def export_run_json(run: ExperimentRun) -> str:
        """Export run record as indented JSON."""
        return run.model_dump_json(indent=2)

    @staticmethod
    def export_run_samples_csv(run: ExperimentRun) -> str:
        """Export individual sample results as CSV."""
        output = io.StringIO()
        fieldnames = [
            "item_id",
            "question_type",
            "question",
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "hit_rate_at_k",
            "faithfulness",
            "unsupported_claim_rate",
            "answer_relevance",
            "answer_correctness",
            "context_relevance",
            "abstention_quality",
            "citation_validity",
            "citation_precision",
            "citation_recall",
            "essential_concepts_coverage",
            "central_thesis_representation",
            "important_qualifications_retained",
            "supporting_reasoning_included",
            "cross_chapter_connections",
            "latency_ms",
            "is_failure",
            "failure_reasons",
            "generated_answer",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for s in run.sample_results:
            writer.writerow({
                "item_id": s.item_id,
                "question_type": s.question_type.value,
                "question": s.question,
                "recall_at_k": s.retrieval_metrics.recall_at_k,
                "precision_at_k": s.retrieval_metrics.precision_at_k,
                "mrr": s.retrieval_metrics.mrr,
                "hit_rate_at_k": s.retrieval_metrics.hit_rate_at_k,
                "faithfulness": s.generation_metrics.faithfulness,
                "unsupported_claim_rate": s.generation_metrics.unsupported_claim_rate,
                "answer_relevance": s.generation_metrics.answer_relevance,
                "answer_correctness": s.generation_metrics.answer_correctness,
                "context_relevance": s.generation_metrics.context_relevance,
                "abstention_quality": s.generation_metrics.abstention_quality,
                "citation_validity": s.generation_metrics.citations.citation_validity,
                "citation_precision": s.generation_metrics.citations.citation_precision,
                "citation_recall": s.generation_metrics.citations.citation_recall,
                "essential_concepts_coverage": s.generation_metrics.crux_rubric.essential_concepts_coverage,
                "central_thesis_representation": s.generation_metrics.crux_rubric.central_thesis_representation,
                "important_qualifications_retained": s.generation_metrics.crux_rubric.important_qualifications_retained,
                "supporting_reasoning_included": s.generation_metrics.crux_rubric.supporting_reasoning_included,
                "cross_chapter_connections": s.generation_metrics.crux_rubric.cross_chapter_connections,
                "latency_ms": s.latency_ms,
                "is_failure": s.is_failure,
                "failure_reasons": "; ".join(s.failure_reasons),
                "generated_answer": s.generated_answer.replace("\n", " "),
            })

        return output.getvalue()

    @staticmethod
    def export_comparison_csv(comparison: PipelineComparisonReport) -> str:
        """Export multi-pipeline comparison summary table as CSV."""
        output = io.StringIO()
        fieldnames = [
            "pipeline_type",
            "sample_count",
            "mean_recall_at_k",
            "mean_precision_at_k",
            "mean_mrr",
            "mean_hit_rate_at_k",
            "mean_faithfulness",
            "mean_unsupported_claim_rate",
            "mean_answer_relevance",
            "mean_answer_correctness",
            "mean_context_relevance",
            "mean_abstention_quality",
            "mean_citation_validity",
            "mean_citation_precision",
            "mean_citation_recall",
            "mean_essential_concepts_coverage",
            "mean_central_thesis_representation",
            "mean_important_qualifications_retained",
            "mean_supporting_reasoning_included",
            "mean_cross_chapter_connections",
            "failure_case_count",
            "mean_latency_ms",
            "duration_seconds",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for pipe_name, run in comparison.runs.items():
            agg = run.aggregate_metrics
            writer.writerow({
                "pipeline_type": pipe_name,
                "sample_count": agg.sample_count,
                "mean_recall_at_k": agg.mean_recall_at_k,
                "mean_precision_at_k": agg.mean_precision_at_k,
                "mean_mrr": agg.mean_mrr,
                "mean_hit_rate_at_k": agg.mean_hit_rate_at_k,
                "mean_faithfulness": agg.mean_faithfulness,
                "mean_unsupported_claim_rate": agg.mean_unsupported_claim_rate,
                "mean_answer_relevance": agg.mean_answer_relevance,
                "mean_answer_correctness": agg.mean_answer_correctness,
                "mean_context_relevance": agg.mean_context_relevance,
                "mean_abstention_quality": agg.mean_abstention_quality,
                "mean_citation_validity": agg.mean_citation_validity,
                "mean_citation_precision": agg.mean_citation_precision,
                "mean_citation_recall": agg.mean_citation_recall,
                "mean_essential_concepts_coverage": agg.mean_essential_concepts_coverage,
                "mean_central_thesis_representation": agg.mean_central_thesis_representation,
                "mean_important_qualifications_retained": agg.mean_important_qualifications_retained,
                "mean_supporting_reasoning_included": agg.mean_supporting_reasoning_included,
                "mean_cross_chapter_connections": agg.mean_cross_chapter_connections,
                "failure_case_count": agg.failure_case_count,
                "mean_latency_ms": agg.mean_latency_ms,
                "duration_seconds": run.duration_seconds,
            })

        return output.getvalue()
