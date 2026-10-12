"""Low-cardinality application metrics for retrieval and model dependencies."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import metrics


_meter = metrics.get_meter("e2e-langgraph-rag")

CACHE_OPERATIONS = _meter.create_counter(
    "rag.cache.operations", unit="{operation}", description="Answer-cache lookups and writes"
)
CACHE_DURATION = _meter.create_histogram(
    "rag.cache.operation.duration", unit="s", description="Answer-cache operation duration"
)
RETRIEVAL_OPERATIONS = _meter.create_counter(
    "rag.retrieval.operations", unit="{operation}", description="Retrieval outcomes"
)
RETRIEVAL_DURATION = _meter.create_histogram(
    "rag.retrieval.duration", unit="s", description="End-to-end retrieval-agent duration"
)
RETRIEVAL_CHUNKS = _meter.create_histogram(
    "rag.retrieval.chunks", unit="{chunk}", description="Candidate and returned chunk counts"
)
RELEVANCE_SCORE = _meter.create_histogram(
    "rag.retrieval.relevance", unit="1", description="Top reranked relevance score"
)
RETRIEVAL_FALLBACKS = _meter.create_counter(
    "rag.retrieval.fallbacks", unit="{fallback}", description="Retrieval fallback outcomes"
)
DEPENDENCY_OPERATIONS = _meter.create_counter(
    "rag.dependency.operations", unit="{operation}", description="Dependency operation outcomes"
)
DEPENDENCY_DURATION = _meter.create_histogram(
    "rag.dependency.duration", unit="s", description="Dependency operation duration"
)
MODEL_TOKENS = _meter.create_counter(
    "rag.model.tokens", unit="{token}", description="Provider-reported model token usage"
)


def error_outcome(exc: BaseException) -> str:
    """Map exceptions to a small bounded set of metric outcomes."""
    name = type(exc).__name__.casefold()
    status_code = getattr(exc, "status_code", None)
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    if status_code == 429 or "ratelimit" in name or "resourceexhausted" in name:
        return "rate_limited"
    return "error"


def record_dependency(
    provider: str, operation: str, outcome: str, duration_seconds: float
) -> None:
    attributes = {"provider": provider, "operation": operation, "outcome": outcome}
    DEPENDENCY_OPERATIONS.add(1, attributes)
    DEPENDENCY_DURATION.record(max(0.0, duration_seconds), attributes)


@contextmanager
def dependency_operation(provider: str, operation: str) -> Iterator[None]:
    """Measure a dependency call while preserving its normal exception behavior."""
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        record_dependency(provider, operation, error_outcome(exc), time.perf_counter() - started)
        raise
    else:
        record_dependency(provider, operation, "success", time.perf_counter() - started)


def record_cache(operation: str, outcome: str, duration_seconds: float) -> None:
    attributes = {"operation": operation, "outcome": outcome}
    CACHE_OPERATIONS.add(1, attributes)
    CACHE_DURATION.record(max(0.0, duration_seconds), attributes)


def record_retrieval(
    outcome: str,
    duration_seconds: float,
    relevance_score: float | None = None,
) -> None:
    attributes = {"outcome": outcome}
    RETRIEVAL_OPERATIONS.add(1, attributes)
    RETRIEVAL_DURATION.record(max(0.0, duration_seconds), attributes)
    if relevance_score is not None:
        RELEVANCE_SCORE.record(min(1.0, max(0.0, relevance_score)))


def record_chunks(stage: str, count: int) -> None:
    RETRIEVAL_CHUNKS.record(max(0, count), {"stage": stage})


def record_fallback(reason: str) -> None:
    # Callers must use the finite reasons defined by the workflow.
    RETRIEVAL_FALLBACKS.add(1, {"reason": reason})


def record_model_tokens(operation: str, token_type: str, count: int) -> None:
    if count > 0:
        MODEL_TOKENS.add(
            count,
            {"provider": "gemini", "operation": operation, "token_type": token_type},
        )
