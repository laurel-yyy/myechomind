"""Singleton Redis client factory.

All modules that need Redis (working memory, tool cache, rate limits, etc.)
should call `get_redis_client()` rather than instantiate their own connection,
so we share one pool per process.
"""

from __future__ import annotations

import logging

import redis

from config import settings

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """Return a cached Redis client. Fails fast on first call if unreachable."""
    global _client
    if _client is not None:
        return _client
    logger.info(
        "Connecting to Redis at %s:%s db=%s",
        settings.redis_host,
        settings.redis_port,
        settings.redis_db,
    )
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        decode_responses=True,  # get str, not bytes
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    client.ping()  # fail loudly if Redis is not up
    _client = client
    return _client


def reset_redis_client() -> None:
    """Testing helper: drop cached connection."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
