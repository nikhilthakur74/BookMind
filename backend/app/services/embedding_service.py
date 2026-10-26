"""Embedding service for BookMind supporting BGE-M3 dense vectors.

Supports configurable model names, explicit device placement (CPU, CUDA, MPS, auto),
ONNX runtime acceleration, and a deterministic mock provider for zero-overhead unit testing.
"""

import abc
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(abc.ABC):
    """Abstract base class for text embedding providers."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Dense output vector dimension."""
        pass

    @property
    @abc.abstractmethod
    def device(self) -> str:
        """Device configured for inference ('cpu', 'cuda', etc.)."""
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate normalized dense embedding vectors for a list of document chunks."""
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate normalized dense embedding vector for a single search query."""
        pass


class DeterministicMockEmbeddingService(BaseEmbeddingService):
    """Deterministic embedding service producing reproducible 1024-d unit vectors.

    Ideal for fast, isolated unit and integration testing without downloading
    multi-gigabyte weights or requiring GPU / ONNX runtime initialization.
    """

    def __init__(self, model_name: str = "mock-bge-m3", dimension: int = 1024, device: str = "cpu"):
        self._model_name = model_name
        self._dimension = dimension
        self._device = device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> str:
        return self._device

    def _compute_vector(self, text: str) -> List[float]:
        """Generate deterministic normalized vector using token hashing and position encoding."""
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = text.lower().strip().split()
        if not tokens:
            # Fallback zero/fixed vector
            vec[0] = 1.0
            return vec.tolist()

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx1 = h % self._dimension
            idx2 = (h >> 16) % self._dimension
            idx3 = (h >> 32) % self._dimension
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx1] += 1.0 * sign
            vec[idx2] += 0.5 * sign
            vec[idx3] += 0.25 * sign

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        return [self._compute_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._compute_vector(text)


class BgeM3EmbeddingService(BaseEmbeddingService):
    """Production BGE-M3 embedding service.

    BGE-M3 provides dense 1024-dimensional multilingual embeddings.
    Uses ONNX Runtime with quantized weights for low memory footprint (~570MB)
    and fast CPU / CUDA inference.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        device: str = "cpu",
        repo_id: str = "hotchpotch/vespa-onnx-BAAI-bge-m3-only-dense",
        onnx_filename: str = "BAAI-bge-m3_quantized.onnx",
    ):
        self._model_name = model_name
        self._dimension = dimension
        self._repo_id = repo_id
        self._onnx_filename = onnx_filename
        self._session = None
        self._tokenizer = None
        self._device = self._resolve_device(device)
        self._providers = self._resolve_providers(self._device)

    def _resolve_device(self, requested: str) -> str:
        req = (requested or "cpu").lower().strip()
        if req in ("cuda", "gpu"):
            try:
                import onnxruntime as ort

                if "CUDAExecutionProvider" in ort.get_available_providers():
                    return "cuda"
            except Exception:
                pass
            logger.warning("CUDA requested for embeddings but CUDAExecutionProvider not available. Falling back to CPU.")
            return "cpu"
        elif req == "mps":
            logger.info("Apple Silicon MPS not directly supported by onnxruntime on Windows; using CPU.")
            return "cpu"
        return "cpu"

    def _resolve_providers(self, resolved_device: str) -> List[str]:
        if resolved_device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> str:
        return self._device

    def _ensure_loaded(self) -> None:
        """Lazy load tokenizer and ONNX inference session."""
        if self._session is not None and self._tokenizer is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(
                f"Required dependencies for BGE-M3 ONNX inference are missing: {e}. "
                "Ensure huggingface_hub, onnxruntime, and tokenizers are installed."
            ) from e

        logger.info(f"Loading BGE-M3 ONNX model ({self._repo_id}/{self._onnx_filename}) on {self._device}...")
        try:
            onnx_path = hf_hub_download(repo_id=self._repo_id, filename=self._onnx_filename)
            tokenizer_path = hf_hub_download(repo_id=self._repo_id, filename="tokenizer.json")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download/locate BGE-M3 ONNX assets from HuggingFace '{self._repo_id}': {e}. "
                "Check internet connection, or set EMBEDDING_PROVIDER=mock for offline/testing mode."
            ) from e

        try:
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=8192)
            self._tokenizer.enable_padding(length=None)

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=self._providers)
            logger.info(f"BGE-M3 ONNX session initialized successfully with providers: {self._session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BGE-M3 ONNX session: {e}") from e

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        encodings = self._tokenizer.encode_batch(batch_texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Check token_type_ids if required by model inputs
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, inputs)
        # BGE-M3 ONNX models usually output [batch_size, seq_len, 1024] or [batch_size, 1024]
        token_embeddings = outputs[0]

        if len(token_embeddings.shape) == 3:
            # Mean pooling with attention mask
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = token_embeddings

        # L2 Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = embeddings / norms

        return normalized.astype(np.float32).tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self._embed_batch([text])
        return results[0]


_cached_embedding_service: Optional[BaseEmbeddingService] = None


def get_embedding_service(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    provider: Optional[str] = None,
    force_new: bool = False,
) -> BaseEmbeddingService:
    """Factory creating or returning the configured embedding service instance.

    Priority:
    1. If provider == "mock" or settings.EMBEDDING_PROVIDER == "mock": DeterministicMockEmbeddingService
    2. If provider == "bge-m3" or settings.EMBEDDING_PROVIDER in ("auto", "onnx"): BgeM3EmbeddingService
    """
    global _cached_embedding_service
    if not force_new and _cached_embedding_service is not None:
        return _cached_embedding_service

    chosen_provider = (provider or settings.EMBEDDING_PROVIDER).lower().strip()
    chosen_device = (device or settings.EMBEDDING_DEVICE).lower().strip()
    chosen_model = model_name or settings.EMBEDDING_MODEL

    if chosen_provider == "mock":
        service = DeterministicMockEmbeddingService(
            model_name=chosen_model,
            dimension=settings.EMBEDDING_DIMENSION,
            device=chosen_device,
        )
    else:
        # Default production provider
        service = BgeM3EmbeddingService(
            model_name=chosen_model,
            dimension=settings.EMBEDDING_DIMENSION,
            device=chosen_device,
        )

    if not force_new:
        _cached_embedding_service = service

    return service


def reset_embedding_service() -> None:
    """Reset cached embedding service singleton (useful for testing)."""
    global _cached_embedding_service
    _cached_embedding_service = None
