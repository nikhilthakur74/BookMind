"""Lexical BM25 (Okapi) keyword retrieval service for BookMind.

Provides independent, deterministic lexical ranking over book chunks.
Indexes are isolated per book ID to enforce evidence boundaries and avoid cross-book leakage.
Implements standard Lucene/Okapi smoothed IDF and BM25 document length normalization.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.book import ChunkMetadata
from app.schemas.retrieval import SearchResultItem
from app.services import storage_service

logger = logging.getLogger(__name__)

# Tokenization regex for alphanumeric words
WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9_\u00C0-\u017F-]+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into lowercased lexical terms."""
    if not text:
        return []
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


class BM25Index:
    """In-memory or serialized BM25 index for a single book."""

    def __init__(
        self,
        book_id: str,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.book_id = book_id
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: Dict[str, int] = {}
        # term -> {chunk_id: frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        # term -> number of chunks containing term
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        # chunk_id -> ChunkMetadata
        self.chunks: Dict[str, ChunkMetadata] = {}

    def add_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Build or replace index entries from chunk metadata."""
        self.chunks.clear()
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.doc_frequencies.clear()

        total_length = 0
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            tokens = tokenize(chunk.text)
            doc_len = len(tokens)
            self.doc_lengths[chunk.chunk_id] = doc_len
            total_length += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][chunk.chunk_id] = count
                self.doc_frequencies[term] += 1

        self.doc_count = len(chunks)
        self.avg_doc_len = (total_length / self.doc_count) if self.doc_count > 0 else 0.0

    def compute_idf(self, term: str) -> float:
        """Compute Okapi BM25 smoothed IDF.

        IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
        Smoothed with +1.0 inside log to guarantee non-negative scores.
        """
        n = self.doc_frequencies.get(term, 0)
        if n == 0 or self.doc_count == 0:
            return 0.0
        val = (self.doc_count - n + 0.5) / (n + 0.5) + 1.0
        return math.log(val)

    def score(self, query: str) -> List[Tuple[str, float]]:
        """Compute BM25 scores for all documents matching any query term.

        Returns list of (chunk_id, score) tuples sorted by score descending.
        """
        tokens = tokenize(query)
        if not tokens or self.doc_count == 0:
            return []

        doc_scores: Dict[str, float] = defaultdict(float)

        for term in tokens:
            if term not in self.inverted_index:
                continue

            idf = self.compute_idf(term)
            postings = self.inverted_index[term]

            for chunk_id, freq in postings.items():
                doc_len = self.doc_lengths.get(chunk_id, 0)
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)
                tf_component = (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                doc_scores[chunk_id] += idf * tf_component

        results = list(doc_scores.items())
        # Sort descending by score, deterministic secondary sort by chunk_id
        results.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return results

    def to_dict(self) -> dict:
        """Serialize index for disk storage."""
        return {
            "book_id": self.book_id,
            "k1": self.k1,
            "b": self.b,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "inverted_index": {term: dict(postings) for term, postings in self.inverted_index.items()},
            "doc_frequencies": dict(self.doc_frequencies),
            "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        """Deserialize index from disk."""
        index = cls(
            book_id=data["book_id"],
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )
        index.doc_count = data.get("doc_count", 0)
        index.avg_doc_len = data.get("avg_doc_len", 0.0)
        index.doc_lengths = data.get("doc_lengths", {})
        index.inverted_index = defaultdict(dict, {k: dict(v) for k, v in data.get("inverted_index", {}).items()})
        index.doc_frequencies = defaultdict(int, data.get("doc_frequencies", {}))
        index.chunks = {
            item["chunk_id"]: ChunkMetadata.model_validate(item) for item in data.get("chunks", [])
        }
        return index


class BM25Service:
    """Manages lexical BM25 indexing and querying across book catalogs."""

    def __init__(self, k1: Optional[float] = None, b: Optional[float] = None):
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        self._cache: Dict[str, BM25Index] = {}

    def get_index_path(self, book_id: str) -> Path:
        """Return path to bm25_index.json inside the book storage directory."""
        book_dir = storage_service.get_book_dir(book_id)
        return book_dir / "bm25_index.json"

    def build_and_save_index(self, book_id: str, chunks: List[ChunkMetadata]) -> BM25Index:
        """Build BM25 index from chunks, save to disk, and update memory cache."""
        index = BM25Index(book_id=book_id, k1=self.k1, b=self.b)
        index.add_chunks(chunks)

        # Save to disk
        index_path = self.get_index_path(book_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, indent=2)

        # Update cache
        self._cache[book_id] = index
        logger.info(f"Built and persisted BM25 index for book '{book_id}' ({len(chunks)} chunks).")
        return index

    def load_or_build_index(self, book_id: str) -> Optional[BM25Index]:
        """Load index from cache or disk; if missing, build on-the-fly from stored chunks."""
        if book_id in self._cache:
            return self._cache[book_id]

        index_path = self.get_index_path(book_id)
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                index = BM25Index.from_dict(data)
                self._cache[book_id] = index
                return index
            except Exception as e:
                logger.warning(f"Failed to load cached BM25 index for '{book_id}': {e}. Rebuilding...")

        # If index file is missing, attempt to build from stored chunks
        chunks = storage_service.get_book_chunks(book_id)
        if chunks:
            return self.build_and_save_index(book_id, chunks)

        return None

    def search_bm25(
        self,
        query: str,
        book_id: str,
        top_k: int = 5,
    ) -> List[SearchResultItem]:
        """Execute BM25 keyword retrieval strictly isolated to the specified book_id."""
        index = self.load_or_build_index(book_id)
        if not index:
            logger.warning(f"Cannot search BM25: no index or chunks found for book '{book_id}'.")
            return []

        scored_pairs = index.score(query)
        top_pairs = scored_pairs[:top_k]

        results: List[SearchResultItem] = []
        for rank, (chunk_id, score) in enumerate(top_pairs, start=1):
            chunk = index.chunks.get(chunk_id)
            if not chunk:
                continue
            results.append(
                SearchResultItem(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    method="bm25",
                )
            )

        return results

    def delete_book_index(self, book_id: str) -> None:
        """Evict index from memory cache and remove bm25_index.json from disk."""
        self._cache.pop(book_id, None)
        index_path = self.get_index_path(book_id)
        if index_path.exists():
            try:
                index_path.unlink()
                logger.info(f"Removed BM25 index file for book '{book_id}'.")
            except Exception as e:
                logger.warning(f"Failed to delete BM25 index file for '{book_id}': {e}")


_bm25_service_instance: Optional[BM25Service] = None


def get_bm25_service(force_new: bool = False) -> BM25Service:
    """Return singleton or new BM25Service instance."""
    global _bm25_service_instance
    if force_new or _bm25_service_instance is None:
        _bm25_service_instance = BM25Service()
    return _bm25_service_instance


def reset_bm25_service() -> None:
    """Reset cached BM25 service singleton."""
    global _bm25_service_instance
    _bm25_service_instance = None
