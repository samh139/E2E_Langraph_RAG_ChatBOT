from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import lru_cache
from typing import Any

import ollama

from app.utils.chunk_retriever import ChunkRetriever
from app.observability_metrics import (
    dependency_operation,
    record_fallback,
    record_retrieval,
)
from app.workflow.retrieval_state import RetrieverState


logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_ollama_client(host: str, timeout: float) -> ollama.Client:
    return ollama.Client(host=host, timeout=timeout)


class RetrievalAgent:
    """Embed a query with Ollama and retrieve/rerank raw Elasticsearch chunks."""

    def __init__(self, payload: RetrieverState, top_k: int | None = None):
        self.payload = payload
        self.top_k = top_k

    async def execute(self) -> dict[str, Any]:
        if not isinstance(self.payload, RetrieverState):
            raise TypeError("RetrievalAgent payload must be a RetrieverState")
        state = self.payload
        query = (state.refined_query or state.query).strip()
        threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("RETRIEVAL_SCORE_THRESHOLD must be between 0 and 1")
        if not query:
            return {
                "retrieved_chunks": [],
                "retrieval_score": 0.0,
                "retrieval_threshold": threshold,
                "retrieval_passed": False,
                "fallback_reason": "low_relevance",
            }
        if not state.allowed_roles:
            return {
                "retrieved_chunks": [],
                "retrieval_score": 0.0,
                "retrieval_threshold": threshold,
                "retrieval_passed": False,
                "fallback_reason": "retrieval_error",
                "retrieval_warning": "No authorized document roles were provided.",
                "error": "No authorized document roles were provided.",
            }

        model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        ollama_host = os.getenv(
            "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        )
        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        retriever: ChunkRetriever | None = None
        started = time.perf_counter()

        try:
            ollama_client = _get_ollama_client(ollama_host, ollama_timeout)
            with dependency_operation("ollama", "query_embedding"):
                response = await asyncio.to_thread(
                    ollama_client.embed,
                    model=model_name,
                    input=query,
                )
            vectors = [list(vector) for vector in response.embeddings]
            if len(vectors) != 1:
                raise RuntimeError(
                    f"Ollama returned {len(vectors)} vectors for one query"
                )
            query_vector = vectors[0]
            if len(query_vector) != 768:
                raise RuntimeError(
                    f"Ollama model {model_name!r} returned {len(query_vector)} dimensions; "
                    "the chunk index is configured for 768"
                )

            top_k = self.top_k or int(os.getenv("RETRIEVAL_TOP_K", "5"))
            candidate_k = int(os.getenv("RETRIEVAL_CANDIDATE_K", "50"))
            retriever = ChunkRetriever(top_k=top_k, candidate_k=candidate_k)
            candidates = await asyncio.to_thread(
                retriever.retrieve,
                query,
                query_vector,
                state.allowed_roles,
            )

            score = float(candidates[0]["relevance_score"]) if candidates else 0.0
            passed = bool(candidates) and score >= threshold
            outcome = "success" if passed else "low_relevance" if candidates else "no_results"
            record_retrieval(outcome, time.perf_counter() - started, score)
            if not passed:
                record_fallback("low_relevance")
            logger.info(
                "Retrieval completed: query=%r candidates=%d score=%.4f threshold=%.4f passed=%s",
                query,
                len(candidates),
                score,
                threshold,
                passed,
            )
            return {
                # Only accepted raw source chunks continue to answer synthesis.
                "retrieved_chunks": candidates if passed else [],
                "retrieval_score": score,
                "retrieval_threshold": threshold,
                "retrieval_passed": passed,
                "fallback_reason": None if passed else "low_relevance",
                "error": None,
            }
        except Exception as exc:
            record_retrieval("error", time.perf_counter() - started)
            record_fallback("retrieval_error")
            logger.exception("Retrieval failed")
            return {
                "retrieved_chunks": [],
                "retrieval_score": 0.0,
                "retrieval_threshold": threshold,
                "retrieval_passed": False,
                "fallback_reason": "retrieval_error",
                "retrieval_warning": str(exc),
                "error": str(exc),
            }
        finally:
            if retriever is not None:
                retriever.close()
