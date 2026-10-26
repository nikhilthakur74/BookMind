"""Command-line interface for BookMind Phase 6 RAG Evaluation Framework.

Usage:
    python scripts/evaluate_rag.py --book-id <book_id> [options]

Options:
    --dataset-id <id>       Existing evaluation dataset ID (auto-generates if omitted)
    --pipelines <list>      Comma-separated pipelines: baseline_rag,hybrid_rag,hierarchical_rag,hierarchical_rag_rerank (default: baseline_rag,hybrid_rag,hierarchical_rag)
    --top-k <int>           Number of evidence chunks to retrieve (default: 5)
    --all-items             Include draft items (default evaluates validated items or all if none validated)
    --enable-llm-judge      Enable optional LLM-judge qualitative scoring
    --export-csv <path>     Save comparison summary table to CSV file
    --export-json <path>    Save full comparison report to JSON file
"""

import argparse
import asyncio
import os
import sys
from typing import List

# Add parent directory to path so imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.evaluation.dataset_service import get_dataset_service
from app.evaluation.export import ExportService
from app.evaluation.runner import get_evaluation_runner
from app.evaluation.schemas import (
    EvaluationDataset,
    ExperimentRun,
    PipelineComparisonReport,
    PipelineType,
    ValidationStatus,
)
from app.services import storage_service


def format_table(rows: List[List[str]], headers: List[str]) -> str:
    """Render a clean text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


async def main_async():
    parser = argparse.ArgumentParser(description="BookMind Phase 6 RAG Evaluation CLI")
    parser.add_argument("--book-id", required=True, help="Target book ID")
    parser.add_argument("--dataset-id", default=None, help="Evaluation dataset ID")
    parser.add_argument(
        "--pipelines",
        default="baseline_rag,hybrid_rag,hierarchical_rag",
        help="Comma-separated pipelines to benchmark",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    parser.add_argument("--all-items", action="store_true", help="Include unvalidated draft items")
    parser.add_argument("--enable-llm-judge", action="store_true", help="Enable optional LLM judge")
    parser.add_argument("--export-csv", default=None, help="Export comparison CSV to file")
    parser.add_argument("--export-json", default=None, help="Export comparison JSON to file")

    args = parser.parse_args()

    print("=" * 80)
    print("BOOKMIND PHASE 6: REPRODUCIBLE RAG EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"Target Book ID: {args.book_id}")

    meta = storage_service.get_book_metadata(args.book_id)
    if not meta:
        print(f"ERROR: Book with ID '{args.book_id}' was not found in storage.")
        sys.exit(1)

    print(f"Book: {meta.original_filename} ({meta.chunk_count} chunks, {meta.page_count} pages)")

    # 1. Dataset Resolution
    ds_svc = get_dataset_service()
    dataset = None
    if args.dataset_id:
        dataset = ds_svc.get_dataset(args.book_id, args.dataset_id)
        if not dataset:
            print(f"ERROR: Dataset '{args.dataset_id}' not found.")
            sys.exit(1)
        print(f"Loaded existing dataset: {dataset.dataset_id} ({len(dataset.items)} items)")
    else:
        # Check existing datasets
        datasets = ds_svc.list_datasets(args.book_id)
        if datasets:
            dataset = datasets[0]
            print(f"Using latest dataset: {dataset.dataset_id} ({len(dataset.items)} items)")
        else:
            print("[*] Generating draft evaluation dataset across all 6 question categories...")
            dataset = ds_svc.generate_draft_dataset(args.book_id)
            print(f"[+] Created draft dataset: {dataset.dataset_id} with {len(dataset.items)} items.")
            print("    NOTE: Draft items should be manually validated for formal publication.")

    # 2. Pipeline Resolution
    requested_pipes = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    pipeline_enums: List[PipelineType] = []
    for p in requested_pipes:
        try:
            pipeline_enums.append(PipelineType(p))
        except ValueError:
            print(f"WARNING: Unknown pipeline type '{p}'. Skipping.")

    if not pipeline_enums:
        print("ERROR: No valid pipelines specified.")
        sys.exit(1)

    print(f"Pipelines to benchmark: {[p.value for p in pipeline_enums]}")
    print(f"Retrieval Top-K: {args.top_k}")
    print(f"Validated Items Only: {not args.all_items}")
    print(f"LLM-Judge Enabled: {args.enable_llm_judge}")
    print("-" * 80)

    # 3. Execution
    runner = get_evaluation_runner()
    runs_dict = {}
    summary_rows = []

    for pipe_type in pipeline_enums:
        print(f"\n[*] Benchmarking pipeline: {pipe_type.value}...")
        run_record: ExperimentRun = await runner.evaluate_dataset(
            dataset=dataset,
            pipeline_type=pipe_type,
            top_k=args.top_k,
            validated_only=not args.all_items,
            enable_llm_judge=args.enable_llm_judge,
        )
        runs_dict[pipe_type.value] = run_record
        agg = run_record.aggregate_metrics
        print(f"    Completed in {run_record.duration_seconds}s | Mean Latency: {agg.mean_latency_ms}ms")
        print(f"    Recall@{args.top_k}: {agg.mean_recall_at_k:.1%} | Precision@{args.top_k}: {agg.mean_precision_at_k:.1%} | MRR: {agg.mean_mrr:.3f}")
        print(f"    Faithfulness: {agg.mean_faithfulness:.1%} | Abstention Quality: {agg.mean_abstention_quality:.1%}")
        print(f"    Crux Thesis Representation: {agg.mean_central_thesis_representation:.1%} | Essential Concepts: {agg.mean_essential_concepts_coverage:.1%}")
        print(f"    Failure Cases: {agg.failure_case_count}/{agg.sample_count}")

    # 4. Summary Table Output
    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK SUMMARY")
    print("=" * 80)

    table_headers = [
        "Pipeline",
        "Recall@k",
        "Prec@k",
        "MRR",
        "HitRate",
        "Faithful",
        "Relevance",
        "Abstain",
        "Thesis",
        "Concepts",
        "Fails",
        "Latency",
    ]
    table_rows = []
    for pipe_name, r in runs_dict.items():
        agg = r.aggregate_metrics
        table_rows.append([
            pipe_name,
            f"{agg.mean_recall_at_k:.1%}",
            f"{agg.mean_precision_at_k:.1%}",
            f"{agg.mean_mrr:.3f}",
            f"{agg.mean_hit_rate_at_k:.1%}",
            f"{agg.mean_faithfulness:.1%}",
            f"{agg.mean_answer_relevance:.1%}",
            f"{agg.mean_abstention_quality:.1%}",
            f"{agg.mean_central_thesis_representation:.1%}",
            f"{agg.mean_essential_concepts_coverage:.1%}",
            f"{agg.failure_case_count}/{agg.sample_count}",
            f"{agg.mean_latency_ms:.0f}ms",
        ])

    print(format_table(table_rows, table_headers))

    # 5. Crux-Specific Rubric Breakdown
    print("\n" + "-" * 80)
    print("CRUX-SPECIFIC CONCEPTUAL COMPLETENESS RUBRIC (NON-COLLAPSED)")
    print("-" * 80)
    rubric_headers = [
        "Pipeline",
        "Essential Concepts",
        "Central Thesis",
        "Qualifications",
        "Reasoning Chain",
        "Cross-Chapter Links",
    ]
    rubric_rows = []
    for pipe_name, r in runs_dict.items():
        agg = r.aggregate_metrics
        rubric_rows.append([
            pipe_name,
            f"{agg.mean_essential_concepts_coverage:.1%}",
            f"{agg.mean_central_thesis_representation:.1%}",
            f"{agg.mean_important_qualifications_retained:.1%}",
            f"{agg.mean_supporting_reasoning_included:.1%}",
            f"{agg.mean_cross_chapter_connections:.1%}",
        ])
    print(format_table(rubric_rows, rubric_headers))

    # 6. Failure Cases Inspection
    all_failures = []
    for pipe_name, r in runs_dict.items():
        for fc in r.failure_cases:
            all_failures.append((pipe_name, fc))

    if all_failures:
        print("\n" + "-" * 80)
        print(f"FAILURE CASES DETECTED ({len(all_failures)} total):")
        print("-" * 80)
        for pipe_name, fc in all_failures[:5]:
            print(f"[{pipe_name}] Item {fc.item_id} ({fc.question_type.value}): \"{fc.question}\"")
            for reason in fc.failure_reasons:
                print(f"  -> {reason}")
            print(f"  Recall: {fc.recall_at_k:.1%} | Faithfulness: {fc.faithfulness:.1%} | Abstention: {fc.abstention_quality:.1%}")
    else:
        print("\n[+] Zero critical failure cases detected across evaluated runs.")

    # 7. Exports
    comparison_report = PipelineComparisonReport(
        book_id=args.book_id,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        runs=runs_dict,
        comparative_summary=[],
        key_findings=[],
    )

    if args.export_csv:
        csv_content = ExportService.export_comparison_csv(comparison_report)
        with open(args.export_csv, "w", encoding="utf-8") as f:
            f.write(csv_content)
        print(f"\n[+] Exported comparison CSV to: {args.export_csv}")

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            f.write(comparison_report.model_dump_json(indent=2))
        print(f"[+] Exported comparison JSON to: {args.export_json}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
