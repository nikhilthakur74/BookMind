"""Indexing service orchestrating dense vector generation, Qdrant upserts, and BM25 index builds.

Ensures idempotent re-indexing, tracks indexing statuses ('not_indexed', 'indexing', 'indexed', 'failed'),
and provides comprehensive failure handling and status reporting.
"""

from datetime import datetime, timezone
import logging
from typing import Optional

from app.core.config import settings
from app.schemas.book import BookMetadata
from app.schemas.retrieval import IndexingStatusResponse
from app.services import storage_service
from app.services.bm25_service import BM25Service, get_bm25_service
from app.services.embedding_service import BaseEmbeddingService, get_embedding_service
from app.services.qdrant_service import QdrantService, get_qdrant_service

logger = logging.getLogger(__name__)


class IndexingService:
    """Manages the full indexing lifecycle for book chunks."""

    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        bm25_service: Optional[BM25Service] = None,
    ):
        self._qdrant_service = qdrant_service or get_qdrant_service()
        self._embedding_service = embedding_service or get_embedding_service()
        self._bm25_service = bm25_service or get_bm25_service()

    def get_status(self, book_id: str) -> IndexingStatusResponse:
        """Retrieve current indexing status and chunk count for a book."""
        meta = storage_service.get_book_metadata(book_id)
        if not meta:
            raise ValueError(f"Book '{book_id}' does not exist.")

        return IndexingStatusResponse(
            book_id=meta.book_id,
            indexing_status=meta.indexing_status,
            indexed_chunk_count=meta.indexed_chunk_count,
            total_chunks=meta.chunk_count,
            indexing_error=meta.indexing_error,
            indexed_at=meta.indexed_at,
        )

    def index_book(self, book_id: str, force: bool = False) -> BookMetadata:
        """Index a book into both Qdrant (dense vectors) and BM25 (lexical).

        Guarantees idempotency: any previous vectors or index entries for this book
        are cleanly purged before upserting the new entries.
        """
        meta = storage_service.get_book_metadata(book_id)
        if not meta:
            raise ValueError(f"Book with ID '{book_id}' was not found.")

        chunks = storage_service.get_book_chunks(book_id)
        if not chunks:
            raise ValueError(f"Book '{book_id}' has no extracted chunks to index.")

        if meta.indexing_status == "indexed" and not force:
            logger.info(f"Book '{book_id}' is already indexed. Skipping (force=False).")
            return meta

        # 1. Update status to 'indexing'
        meta.indexing_status = "indexing"
        meta.indexing_error = None
        storage_service.save_book_metadata(book_id, meta)

        try:
            # 2. Extract chunk texts
            texts = [chunk.text for chunk in chunks]
            logger.info(
                f"Generating embeddings for {len(texts)} chunks of book '{book_id}' "
                f"using model '{self._embedding_service.model_name}' on {self._embedding_service.device}..."
            )

            # 3. Compute dense embeddings in batches
            embeddings = self._embedding_service.embed_documents(
                texts=texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
            )

            # 4. Store in Qdrant with deterministic IDs (purging old vectors)
            upserted = self._qdrant_service.upsert_chunks(
                book_id=book_id,
                chunks=chunks,
                embeddings=embeddings,
                clear_existing=True,
            )

            # 5. Build and persist lexical BM25 index
            self._bm25_service.build_and_save_index(book_id=book_id, chunks=chunks)

            # 6. Mark as 'indexed'
            meta.indexing_status = "indexed"
            meta.indexed_chunk_count = upserted
            meta.indexing_error = None
            meta.indexed_at = datetime.now(timezone.utc)
            storage_service.save_book_metadata(book_id, meta)
            logger.info(f"Book '{book_id}' successfully indexed ({upserted} chunks).")
            return meta

        except Exception as e:
            # 7. Record failure gracefully
            meta.indexing_status = "failed"
            meta.indexing_error = str(e)
            storage_service.save_book_metadata(book_id, meta)
            logger.error(f"Failed to index book '{book_id}': {e}", exc_info=True)
            raise RuntimeError(f"Indexing failed for book '{book_id}': {e}") from e


_indexing_service_instance: Optional[IndexingService] = None


def get_indexing_service(
    force_new: bool = False,
    qdrant_service: Optional[QdrantService] = None,
    embedding_service: Optional[BaseEmbeddingService] = None,
    bm25_service: Optional[BM25Service] = None,
) -> IndexingService:
    """Return singleton or configured IndexingService instance."""
    global _indexing_service_instance
    if (
        force_new
        or qdrant_service is not None
        or embedding_service is not None
        or bm25_service is not None
    ):
        return IndexingService(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            bm25_service=bm25_service,
        )
    if _indexing_service_instance is None:
        _indexing_service_instance = IndexingService()
    return _indexing_service_instance


def reset_indexing_service() -> None:
    """Reset cached indexing service singleton."""
    global _indexing_service_instance
    _indexing_service_instance = None
