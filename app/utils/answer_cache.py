"""Permission-scoped Redis cache for standalone RAG answers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

from app.configs.llm_config import GEMINI_MODEL
from app.memory_service.memory_team.stm.store import redis_client
from app.utils.retrieval_config import chunk_index_alias
from app.observability_metrics import dependency_operation, record_cache


logger = logging.getLogger(__name__)


def _cache_key(query: str, memory_scope_id: str) -> str | None:
    normalized_query = re.sub(r"\s+", " ", query).strip().casefold()
    if not normalized_query or not memory_scope_id:
        return None

    cache_identity = {
        "query": normalized_query,
        "permission_scope": memory_scope_id,
        "index": chunk_index_alias(),
        "gemini_model": os.getenv("GEMINI_MODEL", GEMINI_MODEL),
        "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "reranker_model": os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        "retrieval_threshold": os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"),
        "top_k": os.getenv("RETRIEVAL_TOP_K", "5"),
        "candidate_k": os.getenv("RETRIEVAL_CANDIDATE_K", "50"),
        # Bump this when prompts or answer behavior change.
        "cache_version": os.getenv("ANSWER_CACHE_VERSION", "1"),
    }
    digest = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"rag:answer:{digest}"


async def get_cached_answer(query: str, memory_scope_id: str) -> dict[str, Any] | None:
    key = _cache_key(query, memory_scope_id)
    if key is None:
        return None
    started = time.perf_counter()
    outcome = "miss"
    try:
        with dependency_operation("redis", "answer_cache_get"):
            value = await redis_client.get(key)
        if value:
            logger.info("RAG answer cache hit")
            outcome = "hit"
            return json.loads(value)
        logger.info("RAG answer cache miss")
        return None
    except Exception:
        outcome = "error"
        logger.warning("Answer cache read failed; continuing with retrieval", exc_info=True)
        return None
    finally:
        record_cache("lookup", outcome, time.perf_counter() - started)


async def cache_answer(
    query: str,
    memory_scope_id: str,
    result: dict[str, Any],
) -> None:
    key = _cache_key(query, memory_scope_id)
    if key is None:
        return
    started = time.perf_counter()
    outcome = "success"
    try:
        ttl_seconds = max(1, int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "900")))
        with dependency_operation("redis", "answer_cache_set"):
            await redis_client.set(key, json.dumps(result), ex=ttl_seconds)
    except Exception:
        outcome = "error"
        # Cache problems must not fail an otherwise successful chat request.
        logger.warning("Answer cache write failed", exc_info=True)
    finally:
        record_cache("write", outcome, time.perf_counter() - started)
