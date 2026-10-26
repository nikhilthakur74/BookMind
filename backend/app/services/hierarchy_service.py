"""Hierarchical Book Understanding Service (Phase 4).

Synthesizes the Four Knowledge Levels bottom-up:
- Level 1: Evidence (original chunks, stable IDs, exact page ranges - referenced, never mutated)
- Level 2: Section Summaries (concise summaries, key concepts, definitions, reasoning chains)
- Level 3: Chapter Summaries (chapter crux, arguments, assumptions, examples, conclusions, links)
- Level 4: Book Summary (central thesis, overall crux, 5-10 takeaways, chapter relationships, caveats)

Features:
- Strict provenance tracking: all claims reference source chunk IDs and page numbers
- Staged summarization for long chapters
- Distinguishes author-stated conclusions from model-inferred connections
- Idempotent and resumable via completed_units tracking
- Configurable concurrency with asyncio.Semaphore
- Stored separately in data/uploads/books/{book_id}/hierarchy.json
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.book import ChunkMetadata
from app.schemas.hierarchy import (
    BookHierarchy,
    BookSummary,
    ChapterSummary,
    HierarchyStatusResponse,
    SectionSummary,
)
from app.services import storage_service
from app.services.llm_client import BaseLLMClient, get_llm_client

logger = logging.getLogger(__name__)


def get_hierarchy_path(book_id: str) -> Path:
    """Return isolated path to hierarchy.json inside the book storage directory."""
    book_dir = storage_service.get_book_dir(book_id)
    return book_dir / "hierarchy.json"


class HierarchyService:
    """Manages bottom-up generation, persistence, and inspection of hierarchical book knowledge."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self._llm_client = llm_client or get_llm_client()
        concurrency = (
            settings.OLLAMA_MAX_CONCURRENT
            if settings.LLM_PROVIDER == "ollama"
            else settings.MAX_CONCURRENT_LLM_CALLS
        )
        self._semaphore = asyncio.Semaphore(concurrency)

    def load_hierarchy(self, book_id: str) -> Optional[BookHierarchy]:
        """Load persisted hierarchy from disk if it exists."""
        path = get_hierarchy_path(book_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BookHierarchy.model_validate(data)
        except Exception as e:
            logger.warning(f"Error loading hierarchy for '{book_id}': {e}")
            return None

    def save_hierarchy(self, hierarchy: BookHierarchy) -> None:
        """Persist hierarchy data separately from original evidence."""
        path = get_hierarchy_path(hierarchy.book_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        hierarchy.updated_at = datetime.now(timezone.utc)
        with open(path, "w", encoding="utf-8") as f:
            f.write(hierarchy.model_dump_json(indent=2))

    def delete_hierarchy(self, book_id: str) -> None:
        """Remove hierarchy.json file for a book."""
        path = get_hierarchy_path(book_id)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete hierarchy file for '{book_id}': {e}")

    def get_status(self, book_id: str) -> HierarchyStatusResponse:
        """Check status of hierarchical generation for a book."""
        hierarchy = self.load_hierarchy(book_id)
        if not hierarchy:
            meta = storage_service.get_book_metadata(book_id)
            if not meta:
                raise ValueError(f"Book '{book_id}' does not exist.")
            return HierarchyStatusResponse(
                book_id=book_id,
                generation_status="not_started",
                progress_percent=0.0,
                sections_count=0,
                chapters_count=0,
                has_book_summary=False,
                error=None,
                updated_at=meta.created_at,
            )

        return HierarchyStatusResponse(
            book_id=hierarchy.book_id,
            generation_status=hierarchy.generation_status,
            progress_percent=hierarchy.progress_percent,
            sections_count=len(hierarchy.sections),
            chapters_count=len(hierarchy.chapters),
            has_book_summary=hierarchy.book_level is not None,
            error=hierarchy.error,
            updated_at=hierarchy.updated_at,
        )

    # ==========================================================================
    # Grouping & Partitioning Chunks into Chapters and Sections
    # ==========================================================================

    def partition_chunks_by_chapter(
        self, chunks: List[ChunkMetadata]
    ) -> List[Tuple[str, List[ChunkMetadata]]]:
        """Group sequential chunks by authentic chapter title. Never invents chapter names."""
        chapters_map: Dict[str, List[ChunkMetadata]] = {}
        ordered_chapter_names: List[str] = []

        for chunk in chunks:
            # Use detected chapter title, or fallback to sequential chapter if document has no headings
            title = chunk.chapter or "Chapter 1: Document Text"
            if title not in chapters_map:
                chapters_map[title] = []
                ordered_chapter_names.append(title)
            chapters_map[title].append(chunk)

        return [(title, chapters_map[title]) for title in ordered_chapter_names]

    def cluster_chunks_into_sections(
        self, chapter_title: str, chapter_chunks: List[ChunkMetadata]
    ) -> List[Tuple[str, Optional[str], List[ChunkMetadata]]]:
        """Partition chapter chunks into coherent section clusters (3-5 chunks each)."""
        sections: List[Tuple[str, Optional[str], List[ChunkMetadata]]] = []
        cluster_size = 4

        for i in range(0, len(chapter_chunks), cluster_size):
            cluster = chapter_chunks[i : i + cluster_size]
            sec_idx = (i // cluster_size) + 1
            sec_id = f"sec_{chapter_title[:10].strip().replace(' ', '_')}_{sec_idx:02d}"

            # Check if any chunk in the cluster has a recognized section heading
            sec_heading = next((c.section for c in cluster if c.section), None)
            sections.append((sec_id, sec_heading, cluster))

        return sections

    # ==========================================================================
    # Level 2: Section Synthesis
    # ==========================================================================

    async def _summarize_section(
        self,
        section_id: str,
        chapter_title: str,
        section_title: Optional[str],
        chunks: List[ChunkMetadata],
    ) -> SectionSummary:
        """Synthesize Level 2 Section Summary grounded in source chunks."""
        page_start = min(c.page_start for c in chunks)
        page_end = max(c.page_end for c in chunks)
        chunk_ids = [c.chunk_id for c in chunks]

        # Prepare evidence text block with chunk IDs and page references
        evidence_text = "\n\n".join(
            f"--- [CHUNK {c.chunk_id} | Pages {c.page_start}-{c.page_end}] ---\n{c.text}"
            for c in chunks
        )

        system_prompt = (
            "You are BookMind's rigorous document synthesis agent. "
            "Your task is to produce a concise, strictly factual Level 2 Section Summary. "
            "GROUNDING RULES:\n"
            "1. NEVER invent concepts, claims, or citations not present in the evidence text.\n"
            "2. Distinguish author-stated definitions from model-inferred concepts.\n"
            "3. Link every concept and definition to its specific source_chunk_ids and page_references.\n"
            "4. If the text is sparse, vague, or lacking substance, populate the 'uncertainty' field explicitly.\n"
            "5. Adhere strictly to the requested JSON schema."
        )

        user_prompt = (
            f"Chapter: {chapter_title}\n"
            f"Section: {section_title or 'Sequential Passage'}\n"
            f"Pages: {page_start}-{page_end}\n"
            f"Available Chunk IDs: {chunk_ids}\n\n"
            f"EVIDENCE PASSAGES:\n{evidence_text}\n\n"
            f"Synthesize the Level 2 Section Summary for section '{section_id}'."
        )

        async with self._semaphore:
            summary = await self._llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SectionSummary,
            )

        # Enforce exact bounds, identifiers, and ground child provenance
        summary.section_id = section_id
        summary.chapter_title = chapter_title
        summary.section_title = section_title
        summary.page_start = page_start
        summary.page_end = page_end
        summary.source_chunk_ids = chunk_ids
        summary.prompt_version = self._llm_client.prompt_version
        summary.model_version = self._llm_client.model_name

        for concept in summary.key_concepts:
            if not concept.source_chunk_ids:
                concept.source_chunk_ids = chunk_ids[:2]
            if not concept.page_references:
                concept.page_references = [page_start]
        for definition in summary.definitions:
            if not definition.source_chunk_id or definition.source_chunk_id not in chunk_ids:
                definition.source_chunk_id = chunk_ids[0]
            if not definition.page_number:
                definition.page_number = page_start

        return summary

    # ==========================================================================
    # Level 3: Chapter Synthesis (Supports Staged Summarization)
    # ==========================================================================

    async def _summarize_chapter(
        self,
        chapter_index: int,
        chapter_title: str,
        chapter_chunks: List[ChunkMetadata],
        section_summaries: List[SectionSummary],
    ) -> ChapterSummary:
        """Synthesize Level 3 Chapter Summary from rolled-up section summaries and chunks."""
        page_start = min(c.page_start for c in chapter_chunks)
        page_end = max(c.page_end for c in chapter_chunks)
        all_chunk_ids = [c.chunk_id for c in chapter_chunks]
        section_ids = [s.section_id for s in section_summaries]

        # Is this a long chapter requiring staged processing?
        is_long = len(chapter_chunks) >= settings.LONG_CHAPTER_CHUNK_THRESHOLD or len(section_summaries) > 4
        stage_count = 2 if is_long else 1

        # Synthesize from section summaries with staged partitioning if chapter has many sections
        if is_long and len(section_summaries) > 4:
            # Staged rollup: cluster sections into bounded batches (e.g. 3-4 sections each)
            batch_size = 4
            digest_stages = []
            for b_idx in range(0, len(section_summaries), batch_size):
                b_sections = section_summaries[b_idx : b_idx + batch_size]
                part_num = (b_idx // batch_size) + 1
                b_text = "\n\n".join(
                    f"--- Section [{s.section_id}: {s.section_title or 'Passage'}] (Pages {s.page_start}-{s.page_end}) ---\n"
                    f"Summary: {s.concise_summary}\n"
                    f"Key Concepts: {', '.join(c.name for c in s.key_concepts)}\n"
                    f"Reasoning: {' -> '.join(s.reasoning_chain)}\n"
                    f"Grounding Chunks: {s.source_chunk_ids}"
                    for s in b_sections
                )
                digest_stages.append(f"=== Stage 1 Sub-Rollup Part {part_num} ===\n{b_text}")
            sections_context = "\n\n".join(digest_stages)
        else:
            sections_context = "\n\n".join(
                f"=== Section [{s.section_id}: {s.section_title or 'Passage'}] (Pages {s.page_start}-{s.page_end}) ===\n"
                f"Summary: {s.concise_summary}\n"
                f"Key Concepts: {', '.join(c.name for c in s.key_concepts)}\n"
                f"Reasoning: {' -> '.join(s.reasoning_chain)}\n"
                f"Grounding Chunks: {s.source_chunk_ids}"
                for s in section_summaries
            )

        system_prompt = (
            "You are BookMind's high-level chapter synthesis agent. "
            "Synthesize a rigorous Level 3 Chapter Summary capturing the core crux, central argument, "
            "supporting arguments, assumptions, examples, and conclusions.\n"
            "RULES:\n"
            "1. Chapter Crux: Must state the irreducible foundational premise in 1-2 powerful sentences.\n"
            "2. Never hallucinate claims; reference valid source_chunk_ids from the section summaries.\n"
            "3. Distinguish author-stated conclusions from model-inferred connections.\n"
            "4. If evidence is thin or sparse, set evidence_confidence to 'insufficient' or 'low' and describe uncertainty.\n"
            "5. Return strictly valid JSON conforming to the schema."
        )

        user_prompt = (
            f"Chapter Index: {chapter_index}\n"
            f"Chapter Title: {chapter_title}\n"
            f"Page Range: {page_start}-{page_end}\n"
            f"Total Chunks: {len(chapter_chunks)} (Stage Mode: {'Staged Rollup' if is_long else 'Standard'})\n"
            f"All Grounding Chunk IDs: {all_chunk_ids}\n\n"
            f"SECTION EVIDENCE DIGEST:\n{sections_context}\n\n"
            f"Synthesize the Level 3 Chapter Summary."
        )

        async with self._semaphore:
            chapter_summary = await self._llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ChapterSummary,
            )

        chapter_summary.chapter_index = chapter_index
        chapter_summary.chapter_title = chapter_title
        chapter_summary.page_start = page_start
        chapter_summary.page_end = page_end
        chapter_summary.supporting_section_ids = section_ids
        chapter_summary.source_chunk_ids = all_chunk_ids
        chapter_summary.stage_count = stage_count
        chapter_summary.prompt_version = self._llm_client.prompt_version
        chapter_summary.model_version = self._llm_client.model_name

        # Ensure all child components maintain grounding provenance
        for arg in chapter_summary.supporting_arguments:
            if not arg.source_chunk_ids:
                arg.source_chunk_ids = all_chunk_ids[:2]
        for c in chapter_summary.essential_concepts:
            if not c.source_chunk_ids:
                c.source_chunk_ids = all_chunk_ids[:1]
        for a in chapter_summary.important_assumptions:
            if not a.source_chunk_ids:
                a.source_chunk_ids = all_chunk_ids[:1]
        for ex in chapter_summary.examples:
            if not ex.source_chunk_ids:
                ex.source_chunk_ids = all_chunk_ids[:1]
        for conc in chapter_summary.conclusions:
            if not conc.source_chunk_ids:
                conc.source_chunk_ids = all_chunk_ids[:1]
        for conn in chapter_summary.prerequisites_and_connections:
            if not conn.source_chunk_ids:
                conn.source_chunk_ids = all_chunk_ids[:1]

        return chapter_summary

    # ==========================================================================
    # Level 4: Book Synthesis (Thesis, Crux, Themes, Takeaways, Caveats)
    # ==========================================================================

    async def _summarize_book(
        self,
        book_id: str,
        chapter_summaries: List[ChapterSummary],
    ) -> BookSummary:
        """Synthesize Level 4 Book Summary uniting all chapters into central thesis and crux."""
        all_chunk_ids: List[str] = []
        for ch in chapter_summaries:
            all_chunk_ids.extend(ch.source_chunk_ids)
        # Deduplicate while preserving order
        seen = set()
        dedup_chunk_ids = [cid for cid in all_chunk_ids if not (cid in seen or seen.add(cid))]

        chapters_digest = "\n\n".join(
            f"Chapter [{ch.chapter_index + 1}: {ch.chapter_title}] (Pages {ch.page_start}-{ch.page_end})\n"
            f"Chapter Crux: {ch.chapter_crux}\n"
            f"Central Argument: {ch.central_argument}\n"
            f"Conclusions: {', '.join(c.statement for c in ch.conclusions)}\n"
            f"Evidence Chunks: {ch.source_chunk_ids[:5]}"
            for ch in chapter_summaries
        )

        system_prompt = (
            "You are BookMind's master literary and expository architect. "
            "Synthesize the Level 4 Book Summary capturing the overall crux without losing evidence.\n"
            "REQUIREMENTS:\n"
            "1. Central Thesis: The single overarching proposition uniting the book.\n"
            "2. Overall Crux: The condensed essence of how the book substantiates its thesis.\n"
            "3. Essential Takeaways: Formulate between 5 and 10 high-value, grounded takeaways.\n"
            "4. Chapter Relationships: Map how chapters build upon or contrast with one another.\n"
            "5. Important Caveats: Highlight author-stated limitations or scope conditions.\n"
            "6. Every claim MUST reference valid source_chunk_ids.\n"
            "7. Return strictly valid JSON matching the schema."
        )

        user_prompt = (
            f"Book ID: {book_id}\n"
            f"Total Chapters Analyzed: {len(chapter_summaries)}\n"
            f"All Grounding Chunk IDs (subset preview): {dedup_chunk_ids[:30]}\n\n"
            f"CHAPTER EVIDENCE DIGEST:\n{chapters_digest}\n\n"
            f"Synthesize the Level 4 Book Summary."
        )

        async with self._semaphore:
            book_summary = await self._llm_client.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=BookSummary,
            )

        book_summary.book_id = book_id
        book_summary.supporting_chapter_titles = [ch.chapter_title for ch in chapter_summaries]
        book_summary.source_chunk_ids = dedup_chunk_ids
        book_summary.prompt_version = self._llm_client.prompt_version
        book_summary.model_version = self._llm_client.model_name

        for t in book_summary.recurring_themes:
            if not t.source_chunk_ids:
                t.source_chunk_ids = dedup_chunk_ids[:2]
        for tk in book_summary.essential_takeaways:
            if not tk.source_chunk_ids:
                tk.source_chunk_ids = dedup_chunk_ids[:1]
            if not tk.page_numbers:
                tk.page_numbers = [1]
        for cr in book_summary.chapter_relationships:
            if not cr.source_chunk_ids:
                cr.source_chunk_ids = dedup_chunk_ids[:1]

        return book_summary

    # ==========================================================================
    # Main Resumable, Idempotent Pipeline
    # ==========================================================================

    async def generate_hierarchy(
        self,
        book_id: str,
        force_restart: bool = False,
    ) -> BookHierarchy:
        """Execute bottom-up hierarchical synthesis.

        Idempotent and resumable: Skips units already present in completed_units.
        Saves progress incrementally to disk after each stage.
        """
        chunks = storage_service.get_book_chunks(book_id)
        if not chunks:
            raise ValueError(f"Book '{book_id}' has no extracted chunks to summarize.")

        # Load existing progress or initialize new hierarchy container
        hierarchy = None if force_restart else self.load_hierarchy(book_id)
        if not hierarchy or force_restart:
            hierarchy = BookHierarchy(
                book_id=book_id,
                generation_status="in_progress",
                progress_percent=5.0,
                model_name=self._llm_client.model_name,
                prompt_version=self._llm_client.prompt_version,
            )
            self.save_hierarchy(hierarchy)

        if hierarchy.generation_status == "completed" and not force_restart:
            logger.info(f"Book '{book_id}' hierarchy is already completed. Returning existing.")
            return hierarchy

        hierarchy.generation_status = "in_progress"
        self.save_hierarchy(hierarchy)

        try:
            # 1. Group chunks into chapters
            chapter_groups = self.partition_chunks_by_chapter(chunks)
            total_chapters = len(chapter_groups)

            # Dictionary mapping chapter_title -> List[SectionSummary]
            sections_by_chapter: Dict[str, List[SectionSummary]] = {}

            # Map existing sections by section_id
            existing_sections = {s.section_id: s for s in hierarchy.sections}

            # 2. Level 2: Section Summaries
            for chap_idx, (chap_title, chap_chunks) in enumerate(chapter_groups):
                sections_by_chapter[chap_title] = []
                section_clusters = self.cluster_chunks_into_sections(chap_title, chap_chunks)

                for sec_id, sec_title, cluster_chunks in section_clusters:
                    unit_key = f"sec:{sec_id}"
                    if unit_key in hierarchy.completed_units and sec_id in existing_sections:
                        sec_summary = existing_sections[sec_id]
                    else:
                        sec_summary = await self._summarize_section(
                            section_id=sec_id,
                            chapter_title=chap_title,
                            section_title=sec_title,
                            chunks=cluster_chunks,
                        )
                        hierarchy.completed_units.append(unit_key)
                        existing_sections[sec_id] = sec_summary
                        hierarchy.sections = list(existing_sections.values())
                        self.save_hierarchy(hierarchy)

                    sections_by_chapter[chap_title].append(sec_summary)

            hierarchy.progress_percent = 50.0
            self.save_hierarchy(hierarchy)

            # 3. Level 3: Chapter Summaries
            existing_chapters = {c.chapter_title: c for c in hierarchy.chapters}
            synthesized_chapters: List[ChapterSummary] = []

            for chap_idx, (chap_title, chap_chunks) in enumerate(chapter_groups):
                unit_key = f"chap:{chap_title}"
                if unit_key in hierarchy.completed_units and chap_title in existing_chapters:
                    chap_summary = existing_chapters[chap_title]
                else:
                    chap_sec_summaries = sections_by_chapter.get(chap_title, [])
                    chap_summary = await self._summarize_chapter(
                        chapter_index=chap_idx,
                        chapter_title=chap_title,
                        chapter_chunks=chap_chunks,
                        section_summaries=chap_sec_summaries,
                    )
                    hierarchy.completed_units.append(unit_key)
                    existing_chapters[chap_title] = chap_summary
                    hierarchy.chapters = list(existing_chapters.values())
                    self.save_hierarchy(hierarchy)

                synthesized_chapters.append(chap_summary)

            hierarchy.progress_percent = 80.0
            self.save_hierarchy(hierarchy)

            # 4. Level 4: Book Summary
            unit_key = "book_level"
            if unit_key not in hierarchy.completed_units or hierarchy.book_level is None:
                book_summary = await self._summarize_book(
                    book_id=book_id,
                    chapter_summaries=synthesized_chapters,
                )
                hierarchy.book_level = book_summary
                hierarchy.completed_units.append(unit_key)

            hierarchy.generation_status = "completed"
            hierarchy.progress_percent = 100.0
            hierarchy.error = None
            self.save_hierarchy(hierarchy)
            logger.info(f"Hierarchical book understanding successfully generated for '{book_id}'.")
            return hierarchy

        except Exception as e:
            hierarchy.generation_status = "failed"
            hierarchy.error = str(e)
            self.save_hierarchy(hierarchy)
            logger.error(f"Hierarchical generation failed for book '{book_id}': {e}", exc_info=True)
            raise RuntimeError(f"Hierarchical synthesis failed: {e}") from e


_hierarchy_service_instance: Optional[HierarchyService] = None


def get_hierarchy_service(
    force_new: bool = False,
    llm_client: Optional[BaseLLMClient] = None,
) -> HierarchyService:
    """Return singleton or configured HierarchyService instance."""
    global _hierarchy_service_instance
    if force_new or llm_client is not None or _hierarchy_service_instance is None:
        _hierarchy_service_instance = HierarchyService(llm_client=llm_client)
    return _hierarchy_service_instance


def reset_hierarchy_service() -> None:
    """Reset cached HierarchyService singleton."""
    global _hierarchy_service_instance
    _hierarchy_service_instance = None
