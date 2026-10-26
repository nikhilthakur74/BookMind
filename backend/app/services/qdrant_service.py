"""Qdrant vector database integration for BookMind.

Manages collection initialization, payload indexing, deterministic upserts,
strict book-level isolation filtering, and cosine similarity semantic retrieval.
"""

import logging
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.schemas.book import ChunkMetadata
from app.schemas.retrieval import SearchResultItem

logger = logging.getLogger(__name__)


def generate_point_id(book_id: str, chunk_id: str) -> str:
    """Generate deterministic UUIDv5 point ID to guarantee idempotency across indexing retries."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{book_id}:{chunk_id}"))


class QdrantService:
    """Service wrapping Qdrant client operations."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self._collection_name = collection_name or settings.QDRANT_COLLECTION
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._client = client or self._init_client()
        self._ensure_collection()

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client either via remote URL or local persistent directory."""
        if settings.QDRANT_URL:
            logger.info(f"Connecting to remote Qdrant server at {settings.QDRANT_URL}")
            return QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )

        storage_path = str(settings.resolved_qdrant_storage_dir)
        logger.info(f"Initializing local persistent Qdrant at {storage_path}")
        return QdrantClient(path=storage_path)

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _ensure_collection(self) -> None:
        """Create Qdrant collection and payload indexes if they do not exist."""
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self._collection_name for c in collections)
        except Exception as e:
            logger.warning(f"Error checking Qdrant collections: {e}. Attempting recreate check...")
            exists = False

        if not exists:
            logger.info(
                f"Creating Qdrant collection '{self._collection_name}' "
                f"(dim={self._dimension}, distance=Cosine)..."
            )
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )

        # Ensure keyword index on 'book_id' for fast, strict filtering
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="book_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        except Exception as e:
            # Index may already exist or embedded client may no-op
            logger.debug(f"Payload index creation note: {e}")

    def delete_book_vectors(self, book_id: str) -> None:
        """Delete all indexed vector points associated with a specific book."""
        try:
            book_filter = Filter(
                must=[
                    FieldCondition(
                        key="book_id",
                        match=MatchValue(value=book_id),
                    )
                ]
            )
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=book_filter),
            )
            logger.info(f"Deleted vector points for book '{book_id}' from Qdrant.")
        except Exception as e:
            logger.warning(f"Failed to delete vector points for '{book_id}': {e}")

    def upsert_chunks(
        self,
        book_id: str,
        chunks: List[ChunkMetadata],
        embeddings: List[List[float]],
        clear_existing: bool = True,
    ) -> int:
        """Store chunk vectors and complete metadata in Qdrant with deterministic IDs.

        Guarantees idempotency on retries:
        1. If clear_existing is True, any existing points for this book_id are purged.
        2. Point IDs are deterministically derived from book_id and chunk_id.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})"
            )

        if clear_existing:
            self.delete_book_vectors(book_id)

        points: List[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = generate_point_id(book_id, chunk.chunk_id)
            payload = chunk.model_dump(mode="json")
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
                wait=True,
            )
            total_upserted += len(batch)

        logger.info(f"Successfully upserted {total_upserted} vectors for book '{book_id}' into Qdrant.")
        return total_upserted

    def search_semantic(
        self,
        query_vector: List[float],
        book_id: str,
        top_k: int = 5,
    ) -> List[SearchResultItem]:
        """Execute dense cosine semantic similarity search strictly filtered by book_id.

        Preserves all chunk metadata and assigns 1-based ranks.
        """
        book_filter = Filter(
            must=[
                FieldCondition(
                    key="book_id",
                    match=MatchValue(value=book_id),
                )
            ]
        )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=book_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: List[SearchResultItem] = []
        for rank, point in enumerate(response.points, start=1):
            if not point.payload:
                continue
            chunk = ChunkMetadata.model_validate(point.payload)
            score = float(point.score) if point.score is not None else 0.0
            results.append(
                SearchResultItem(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    method="semantic",
                )
            )

        return results

    def count_book_vectors(self, book_id: str) -> int:
        """Count number of vector points currently stored for a book."""
        try:
            book_filter = Filter(
                must=[
                    FieldCondition(
                        key="book_id",
                        match=MatchValue(value=book_id),
                    )
                ]
            )
            result = self._client.count(
                collection_name=self._collection_name,
                count_filter=book_filter,
                exact=True,
            )
            return result.count
        except Exception as e:
            logger.warning(f"Error counting vectors for '{book_id}': {e}")
            return 0


_qdrant_service_instance: Optional[QdrantService] = None


def get_qdrant_service(force_new: bool = False, client: Optional[QdrantClient] = None) -> QdrantService:
    """Return singleton or new QdrantService instance."""
    global _qdrant_service_instance
    if client is not None or force_new:
        return QdrantService(client=client)
    if _qdrant_service_instance is None:
        _qdrant_service_instance = QdrantService()
    return _qdrant_service_instance


def reset_qdrant_service() -> None:
    """Reset cached Qdrant service singleton."""
    global _qdrant_service_instance
    _qdrant_service_instance = None
