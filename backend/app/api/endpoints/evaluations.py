"""Evaluation API endpoints for reproducible RAG benchmarking."""

from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, HTTPException, Query, Response, status

from app.evaluation.dataset_service import get_dataset_service
from app.evaluation.export import ExportService
from app.evaluation.runner import get_evaluation_runner
from app.evaluation.schemas import (
    DatasetGenerateRequest,
    EvaluationDataset,
    EvaluationItem,
    ExperimentRun,
    ExperimentRunRequest,
    ItemUpdateRequest,
    PipelineComparisonReport,
    PipelineType,
)
from app.services import storage_service

router = APIRouter(prefix="/books/{book_id}/evaluations", tags=["Evaluations"])


@router.post(
    "/datasets/generate",
    response_model=EvaluationDataset,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Draft Evaluation Dataset",
    description="Uses LLM assistance to generate draft questions across all 6 categories. Requires manual review.",
)
def generate_draft_dataset(
    book_id: str,
    request: Optional[DatasetGenerateRequest] = None,
) -> EvaluationDataset:
    """Generate a draft evaluation dataset from book chunks and hierarchical summaries."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    service = get_dataset_service()
    try:
        return service.generate_draft_dataset(book_id=book_id, request=request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset generation failed: {e}",
        ) from e


@router.get(
    "/datasets",
    response_model=List[EvaluationDataset],
    summary="List Evaluation Datasets",
    description="List all versioned evaluation datasets created for a book.",
)
def list_datasets(book_id: str) -> List[EvaluationDataset]:
    """List evaluation datasets for a book."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    service = get_dataset_service()
    return service.list_datasets(book_id)


@router.get(
    "/datasets/{dataset_id}",
    response_model=EvaluationDataset,
    summary="Get Evaluation Dataset",
    description="Retrieve a complete evaluation dataset with its items and validation statuses.",
)
def get_dataset(book_id: str, dataset_id: str) -> EvaluationDataset:
    """Get single evaluation dataset."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    service = get_dataset_service()
    ds = service.get_dataset(book_id, dataset_id)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation dataset '{dataset_id}' not found.",
        )
    return ds


@router.post(
    "/datasets",
    response_model=EvaluationDataset,
    status_code=status.HTTP_201_CREATED,
    summary="Create or Upload Evaluation Dataset",
    description="Create a custom versioned evaluation dataset directly.",
)
def create_dataset(book_id: str, dataset: EvaluationDataset) -> EvaluationDataset:
    """Persist custom evaluation dataset."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    service = get_dataset_service()
    dataset.book_id = book_id
    service.save_dataset(dataset)
    return dataset


@router.put(
    "/datasets/{dataset_id}/items/{item_id}",
    response_model=EvaluationItem,
    summary="Review and Validate Evaluation Item",
    description="Update an item's validation status ('draft' -> 'validated' / 'rejected') and ground-truth labels.",
)
def update_item_status(
    book_id: str,
    dataset_id: str,
    item_id: str,
    update: ItemUpdateRequest,
) -> EvaluationItem:
    """Review and transition an item from draft to validated ground truth."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    service = get_dataset_service()
    updated = service.update_item(
        book_id=book_id,
        dataset_id=dataset_id,
        item_id=item_id,
        update=update,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{item_id}' not found in dataset '{dataset_id}'.",
        )
    return updated


@router.post(
    "/run",
    response_model=PipelineComparisonReport,
    summary="Run RAG Evaluation Experiment",
    description="Executes benchmark comparison across Baseline RAG, Hybrid RAG, and Hierarchical RAG.",
)
async def run_evaluation(
    book_id: str,
    request: ExperimentRunRequest,
) -> PipelineComparisonReport:
    """Benchmark specified pipeline variants against a dataset."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    ds_service = get_dataset_service()
    dataset = ds_service.get_dataset(book_id, request.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{request.dataset_id}' not found.",
        )

    runner = get_evaluation_runner()
    runs_dict: Dict[str, ExperimentRun] = {}
    summary_rows: List[Dict[str, Any]] = []

    for pipe_type in request.pipelines:
        run_record = await runner.evaluate_dataset(
            dataset=dataset,
            pipeline_type=pipe_type,
            top_k=request.top_k,
            validated_only=request.validated_only,
            enable_llm_judge=request.enable_llm_judge,
        )
        runs_dict[pipe_type.value] = run_record
        summary_rows.append({
            "pipeline": pipe_type.value,
            "mean_recall_at_k": run_record.aggregate_metrics.mean_recall_at_k,
            "mean_precision_at_k": run_record.aggregate_metrics.mean_precision_at_k,
            "mean_mrr": run_record.aggregate_metrics.mean_mrr,
            "mean_hit_rate_at_k": run_record.aggregate_metrics.mean_hit_rate_at_k,
            "mean_faithfulness": run_record.aggregate_metrics.mean_faithfulness,
            "mean_unsupported_claim_rate": run_record.aggregate_metrics.mean_unsupported_claim_rate,
            "mean_answer_relevance": run_record.aggregate_metrics.mean_answer_relevance,
            "mean_answer_correctness": run_record.aggregate_metrics.mean_answer_correctness,
            "mean_abstention_quality": run_record.aggregate_metrics.mean_abstention_quality,
            "mean_citation_validity": run_record.aggregate_metrics.mean_citation_validity,
            "mean_essential_concepts_coverage": run_record.aggregate_metrics.mean_essential_concepts_coverage,
            "mean_central_thesis_representation": run_record.aggregate_metrics.mean_central_thesis_representation,
            "failure_case_count": run_record.aggregate_metrics.failure_case_count,
            "mean_latency_ms": run_record.aggregate_metrics.mean_latency_ms,
        })

    # Derive key findings
    key_findings = []
    if "hierarchical_rag" in runs_dict and "baseline_rag" in runs_dict:
        h_agg = runs_dict["hierarchical_rag"].aggregate_metrics
        b_agg = runs_dict["baseline_rag"].aggregate_metrics
        rec_diff = round(h_agg.mean_recall_at_k - b_agg.mean_recall_at_k, 3)
        thesis_diff = round(h_agg.mean_central_thesis_representation - b_agg.mean_central_thesis_representation, 3)
        key_findings.append(
            f"Hierarchical RAG improved Recall@{request.top_k} by {rec_diff:+.1%} compared to Baseline."
        )
        key_findings.append(
            f"Central thesis representation increased by {thesis_diff:+.1%} due to Level 3/4 summary injection."
        )

    return PipelineComparisonReport(
        book_id=book_id,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.version,
        runs=runs_dict,
        comparative_summary=summary_rows,
        key_findings=key_findings,
    )


@router.get(
    "/runs",
    response_model=List[ExperimentRun],
    summary="List Evaluation Runs",
    description="List all historical benchmark runs for a book.",
)
def list_runs(book_id: str) -> List[ExperimentRun]:
    """List historical experiment runs."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    runner = get_evaluation_runner()
    return runner.list_runs(book_id)


@router.get(
    "/runs/{run_id}",
    response_model=ExperimentRun,
    summary="Get Evaluation Run Details",
    description="Retrieve full metrics, individual sample results, and diagnosed failure cases for a run.",
)
def get_run(book_id: str, run_id: str) -> ExperimentRun:
    """Get single experiment run details."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    runner = get_evaluation_runner()
    run = runner.get_run(book_id, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found.")
    return run


@router.get(
    "/runs/{run_id}/export",
    summary="Export Evaluation Run Results",
    description="Export run results in JSON or CSV format for external analysis.",
)
def export_run(
    book_id: str,
    run_id: str,
    export_format: Literal["json", "csv"] = Query(default="csv", alias="format"),
) -> Response:
    """Export run results as CSV or JSON file download."""
    meta = storage_service.get_book_metadata(book_id)
    if not meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book '{book_id}' not found.")

    runner = get_evaluation_runner()
    run = runner.get_run(book_id, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found.")

    if export_format == "json":
        content = ExportService.export_run_json(run)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={run_id}.json"},
        )
    else:
        content = ExportService.export_run_samples_csv(run)
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={run_id}_samples.csv"},
        )
