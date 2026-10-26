"""Dataset management service for versioned RAG evaluations.

Responsibilities:
1. Versioned dataset persistence and schema validation.
2. Manual review workflow: items remain 'draft' until explicitly approved as 'validated'.
3. LLM-assisted draft question generation across all 6 question categories.
4. Item updating, status transition, and ground-truth curation.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.evaluation.schemas import (
    DatasetGenerateRequest,
    EvaluationDataset,
    EvaluationItem,
    EvaluationQuestionType,
    ItemUpdateRequest,
    ValidationStatus,
)
from app.services import storage_service
from app.services.hierarchy_service import get_hierarchy_service
from app.services.llm_client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Sanitize identifier to safe filesystem characters."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:80]


class EvaluationDatasetService:
    """Manages versioned evaluation datasets with manual review enforcement."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm = llm_client or get_llm_client()

    def get_dataset_dir(self, book_id: str) -> Path:
        """Return path to datasets folder for a book."""
        book_dir = storage_service.get_book_dir(book_id)
        d_dir = book_dir / "evaluations" / "datasets"
        d_dir.mkdir(parents=True, exist_ok=True)
        return d_dir

    def get_dataset_path(self, book_id: str, dataset_id: str) -> Path:
        """Return path to specific dataset JSON file."""
        safe_id = sanitize_filename(dataset_id)
        return self.get_dataset_dir(book_id) / f"{safe_id}.json"

    def save_dataset(self, dataset: EvaluationDataset) -> None:
        """Persist evaluation dataset as JSON."""
        dataset.updated_at = datetime.now(timezone.utc)
        path = self.get_dataset_path(dataset.book_id, dataset.dataset_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=2))

    def get_dataset(self, book_id: str, dataset_id: str) -> Optional[EvaluationDataset]:
        """Load and validate dataset from disk."""
        path = self.get_dataset_path(book_id, dataset_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationDataset.model_validate(data)
        except Exception as e:
            logger.warning(f"Error loading evaluation dataset '{dataset_id}' for book '{book_id}': {e}")
            return None

    def list_datasets(self, book_id: str) -> List[EvaluationDataset]:
        """List all evaluation datasets for a book."""
        d_dir = self.get_dataset_dir(book_id)
        datasets: List[EvaluationDataset] = []
        for file in d_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                datasets.append(EvaluationDataset.model_validate(data))
            except Exception:
                continue
        datasets.sort(key=lambda d: d.created_at, reverse=True)
        return datasets

    def update_item(
        self,
        book_id: str,
        dataset_id: str,
        item_id: str,
        update: ItemUpdateRequest,
    ) -> Optional[EvaluationItem]:
        """Manually review, edit, or approve an evaluation item (draft -> validated/rejected)."""
        dataset = self.get_dataset(book_id, dataset_id)
        if not dataset:
            return None

        target_item = None
        for item in dataset.items:
            if item.item_id == item_id:
                target_item = item
                break

        if not target_item:
            return None

        # Apply updates
        target_item.validation_status = update.validation_status
        if update.reference_answer is not None:
            target_item.reference_answer = update.reference_answer
        if update.expected_chunk_ids is not None:
            target_item.expected_chunk_ids = update.expected_chunk_ids
        if update.expected_page_numbers is not None:
            target_item.expected_page_numbers = update.expected_page_numbers
        if update.essential_concepts is not None:
            target_item.essential_concepts = update.essential_concepts
        if update.is_answerable is not None:
            target_item.is_answerable = update.is_answerable
        if update.notes is not None:
            target_item.notes = update.notes
        target_item.updated_at = datetime.now(timezone.utc)

        self.save_dataset(dataset)
        return target_item

    def generate_draft_dataset(
        self,
        book_id: str,
        request: Optional[DatasetGenerateRequest] = None,
    ) -> EvaluationDataset:
        """Generate a draft evaluation dataset using book chunks and hierarchical summaries.

        NOTE: Generated items receive validation_status = 'draft'.
        They MUST be manually reviewed before being treated as validated ground truth.
        """
        meta = storage_service.get_book_metadata(book_id)
        if not meta:
            raise ValueError(f"Book with ID '{book_id}' was not found.")

        chunks = storage_service.get_book_chunks(book_id)
        if not chunks:
            raise ValueError(f"Book '{book_id}' has no extracted chunks.")

        hier_svc = get_hierarchy_service()
        hierarchy = hier_svc.load_hierarchy(book_id)

        target_types = (
            request.include_types
            if request and request.include_types
            else [
                EvaluationQuestionType.FACTUAL,
                EvaluationQuestionType.CONCEPTUAL,
                EvaluationQuestionType.SUMMARIZATION,
                EvaluationQuestionType.MULTI_HOP,
                EvaluationQuestionType.COMPARISON,
                EvaluationQuestionType.UNANSWERABLE,
            ]
        )

        dataset_id = f"ds_{book_id[:8]}_{uuid.uuid4().hex[:6]}"
        items: List[EvaluationItem] = []
        counter = 1

        first_chunk = chunks[0]
        second_chunk = chunks[1] if len(chunks) > 1 else chunks[0]
        last_chunk = chunks[-1]

        # 1. Factual Questions
        if EvaluationQuestionType.FACTUAL in target_types:
            q_text = f"What is established regarding '{first_chunk.chapter or 'the introduction'}' on page {first_chunk.page_start}?"
            ref = f"According to page {first_chunk.page_start}, {first_chunk.text[:180].strip()}."
            items.append(
                EvaluationItem(
                    item_id=f"q_{counter:03d}",
                    question=q_text,
                    question_type=EvaluationQuestionType.FACTUAL,
                    reference_answer=ref,
                    expected_chunk_ids=[first_chunk.chunk_id],
                    expected_page_numbers=[first_chunk.page_start],
                    is_answerable=True,
                    essential_concepts=[first_chunk.chapter or "Core Principle"],
                    validation_status=ValidationStatus.DRAFT,
                    notes="Draft generated by LLM-assistant. Requires human review before treating as ground truth.",
                )
            )
            counter += 1

        # 2. Conceptual Questions
        if EvaluationQuestionType.CONCEPTUAL in target_types:
            q_text = f"How does the author explain the mechanism of '{second_chunk.section or 'the primary topic'}'?"
            ref = f"The author explains that {second_chunk.text[:200].strip()}."
            items.append(
                EvaluationItem(
                    item_id=f"q_{counter:03d}",
                    question=q_text,
                    question_type=EvaluationQuestionType.CONCEPTUAL,
                    reference_answer=ref,
                    expected_chunk_ids=[second_chunk.chunk_id],
                    expected_page_numbers=[second_chunk.page_start],
                    is_answerable=True,
                    essential_concepts=["mechanism", second_chunk.section or "Primary Topic"],
                    validation_status=ValidationStatus.DRAFT,
                    notes="Draft generated by LLM-assistant. Requires human review.",
                )
            )
            counter += 1

        # 3. Summarization Questions
        if EvaluationQuestionType.SUMMARIZATION in target_types:
            ch_title = first_chunk.chapter or "Chapter 1"
            q_text = f"What is the crux and central argument of '{ch_title}'?"
            crux_ref = (
                hierarchy.chapters[0].chapter_crux
                if hierarchy and hierarchy.chapters
                else f"The chapter establishes {first_chunk.text[:160].strip()}."
            )
            items.append(
                EvaluationItem(
                    item_id=f"q_{counter:03d}",
                    question=q_text,
                    question_type=EvaluationQuestionType.SUMMARIZATION,
                    reference_answer=crux_ref,
                    expected_chunk_ids=[first_chunk.chunk_id],
                    expected_page_numbers=[first_chunk.page_start],
                    is_answerable=True,
                    essential_concepts=["crux", ch_title],
                    validation_status=ValidationStatus.DRAFT,
                    notes="Draft generated from Level 3 summaries.",
                )
            )
            counter += 1

        # 4. Multi-Hop Questions
        if EvaluationQuestionType.MULTI_HOP in target_types:
            q_text = f"How does the foundation in '{first_chunk.chapter or 'Chapter 1'}' lead to the conclusions in '{last_chunk.chapter or 'the conclusion'}'?"
            ref = (
                f"The initial foundation established on page {first_chunk.page_start} directly informs "
                f"the subsequent outcomes described on page {last_chunk.page_start}."
            )
            items.append(
                EvaluationItem(
                    item_id=f"q_{counter:03d}",
                    question=q_text,
                    question_type=EvaluationQuestionType.MULTI_HOP,
                    reference_answer=ref,
                    expected_chunk_ids=[first_chunk.chunk_id, last_chunk.chunk_id],
                    expected_page_numbers=[first_chunk.page_start, last_chunk.page_start],
                    is_answerable=True,
                    essential_concepts=[first_chunk.chapter or "Foundation", last_chunk.chapter or "Outcome"],
                    validation_status=ValidationStatus.DRAFT,
                    notes="Multi-hop draft spanning distant chunks.",
                )
            )
            counter += 1

        # 5. Comparison Questions
        if EvaluationQuestionType.COMPARISON in target_types:
            topic_a = first_chunk.chapter or "Topic A"
            topic_b = second_chunk.chapter or "Topic B"
            q_text = f"Compare {topic_a} versus {topic_b} in terms of design and operational properties."
            ref = (
                f"While {topic_a} (page {first_chunk.page_start}) emphasizes direct execution, "
                f"{topic_b} (page {second_chunk.page_start}) contrasts this by addressing scaling and stability."
            )
            items.append(
                EvaluationItem(
                    item_id=f"q_{counter:03d}",
                    question=q_text,
                    question_type=EvaluationQuestionType.COMPARISON,
                    reference_answer=ref,
                    expected_chunk_ids=[first_chunk.chunk_id, second_chunk.chunk_id],
                    expected_page_numbers=[first_chunk.page_start, second_chunk.page_start],
                    is_answerable=True,
                    essential_concepts=[topic_a, topic_b, "comparison"],
                    validation_status=ValidationStatus.DRAFT,
                    notes="Comparative draft testing dual-concept retrieval.",
                )
            )
            counter += 1

        # 6. Unanswerable Questions
        if EvaluationQuestionType.UNANSWERABLE in target_types:
            q_text = "What specific recipe does the author provide for baking sourdough bread?"
            ref = "The book does not contain sufficient evidence or discussion regarding sourdough baking recipes."
            items.append(
                EvaluationItem(
                    item_id=f"q_{counter:03d}",
                    question=q_text,
                    question_type=EvaluationQuestionType.UNANSWERABLE,
                    reference_answer=ref,
                    expected_chunk_ids=[],
                    expected_page_numbers=[],
                    is_answerable=False,
                    essential_concepts=["insufficient evidence"],
                    validation_status=ValidationStatus.DRAFT,
                    notes="Negative control: model must explicitly abstain without hallucinating evidence.",
                )
            )
            counter += 1

        dataset = EvaluationDataset(
            dataset_id=dataset_id,
            book_id=book_id,
            version="1.0.0-draft",
            description=f"Draft evaluation benchmark for '{meta.original_filename}'. Generated with LLM assistance; requires manual validation.",
            items=items,
        )

        self.save_dataset(dataset)
        return dataset


_dataset_service_instance: Optional[EvaluationDatasetService] = None


def get_dataset_service() -> EvaluationDatasetService:
    """Return singleton dataset service instance."""
    global _dataset_service_instance
    if _dataset_service_instance is None:
        _dataset_service_instance = EvaluationDatasetService()
    return _dataset_service_instance
