"""ChromaDB client factory + a bridge embedding function.

Chroma normally computes embeddings via its own default model. We override that
by supplying a custom EmbeddingFunction that delegates to our project-wide
EmbeddingClient. That way every vector — for RAG, episodic memory, and user
profile — flows through the same abstraction and mock/real mode swap.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import settings
from core import get_embedding_client

logger = logging.getLogger(__name__)


class _LocalEmbeddingFunction(EmbeddingFunction):
    """Adapter: expose our EmbeddingClient as a Chroma EmbeddingFunction."""

    def __init__(self) -> None:
        self._client = get_embedding_client()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (chroma's signature)
        return self._client.embed(list(input))

    # Chroma introspects a `name()` method when persisting collection config;
    # provide a stable identifier so a rebuilt process can reopen the collection.
    @staticmethod
    def name() -> str:
        return "myechomind-local-embedding"


_client: Any = None
_embedding_fn: _LocalEmbeddingFunction | None = None


def get_chroma_client() -> Any:
    """Return a cached Chroma client (HTTP or PersistentClient per settings)."""
    global _client
    if _client is not None:
        return _client

    if settings.chroma_mode == "http":
        logger.info(
            "Connecting to ChromaDB HTTP at %s:%s",
            settings.chroma_host,
            settings.chroma_port,
        )
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    else:
        # Requires the full `chromadb` package (has native hnswlib).
        logger.info("Using ChromaDB PersistentClient at %s", settings.chroma_path)
        _client = chromadb.PersistentClient(path=settings.chroma_path)

    _client.heartbeat()  # fail fast if the server is not reachable
    return _client


def get_embedding_function() -> _LocalEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = _LocalEmbeddingFunction()
    return _embedding_fn


def get_or_create_collection(name: str):
    """Convenience wrapper: get_or_create a collection wired to our embedding fn."""
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
    )


def reset_chroma_client() -> None:
    """Testing helper: drop cached client and embedding function."""
    global _client, _embedding_fn
    _client = None
    _embedding_fn = None
