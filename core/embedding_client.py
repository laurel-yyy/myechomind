"""Embedding abstraction layer.

Two backends:
    - MockEmbeddingClient : deterministic hash-derived vector (no download, no
      GPU, same input -> same vector). Good enough for wiring intent-recognition
      similarity fusion and RAG plumbing.
    - SentenceTransformerEmbeddingClient : real dense embeddings from a local
      Hugging Face model. Only imported when EMBEDDING_MODE=sentence-transformers.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# =====================================================================
# Mock: hash-based, deterministic, zero dependencies
# =====================================================================
class MockEmbeddingClient(EmbeddingClient):
    """Turn text into a fixed-length pseudo-vector via SHA-256 folding.

    Two identical inputs always yield the exact same vector, so downstream
    similarity math stays stable and repeatable across runs.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = int(dim or settings.embedding_dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t or "") for t in texts]

    def _hash_vector(self, text: str) -> list[float]:
        # Expand SHA-256 output by chained hashing until we have `dim` bytes.
        buf = bytearray()
        seed = text.encode("utf-8")
        counter = 0
        while len(buf) < self._dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            buf.extend(digest)
            counter += 1
        raw = buf[: self._dim]

        # Map bytes -> [-1, 1] and L2-normalize so cosine similarity is meaningful.
        vec = [(b / 127.5) - 1.0 for b in raw]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# =====================================================================
# Real: sentence-transformers (lazy import)
# =====================================================================
class SentenceTransformerEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "EMBEDDING_MODE=sentence-transformers requires the "
                "`sentence-transformers` package. Install with: "
                "pip install sentence-transformers"
            ) from exc

        self._model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self._model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


# =====================================================================
# Factory
# =====================================================================
_INSTANCE: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    """Return a cached singleton embedding client according to settings.embedding_mode."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    if settings.embedding_mode == "sentence-transformers":
        logger.info(
            "Instantiating SentenceTransformerEmbeddingClient (model=%s)",
            settings.embedding_model,
        )
        _INSTANCE = SentenceTransformerEmbeddingClient()
    else:
        logger.info("Instantiating MockEmbeddingClient (dim=%d)", settings.embedding_dim)
        _INSTANCE = MockEmbeddingClient()
    return _INSTANCE


def reset_embedding_client() -> None:
    """Testing helper: drop cached instance."""
    global _INSTANCE
    _INSTANCE = None


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two same-length vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
