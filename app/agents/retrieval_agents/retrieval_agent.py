from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from app.embeddings import embed_texts, embedding_dimensions
from app.utils.chunk_retriever import ChunkRetriever
from app.observability_metrics import (
    dependency_operation,
    record_fallback,
    record_retrieval,
)
from app.workflow.retrieval_state import RetrieverState


logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Embed a query with the configured provider and retrieve raw chunks."""

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

        retriever: ChunkRetriever | None = None
        started = time.perf_counter()

        try:
            provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
            with dependency_operation(provider, "query_embedding"):
                response = await asyncio.to_thread(
                    embed_texts,
                    [query],
                    "search_query",
                )
            vectors = response
            if len(vectors) != 1:
                raise RuntimeError(
                    f"Embedding provider returned {len(vectors)} vectors for one query"
                )
            query_vector = vectors[0]
            if len(query_vector) != embedding_dimensions():
                raise RuntimeError(
                    f"Embedding provider returned {len(query_vector)} dimensions; "
                    f"the chunk index is configured for {embedding_dimensions()}"
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
            print(
                f"KB_RETRIEVAL query_id={state.query_id} "
                f"candidates={len(candidates)} score={score:.4f} "
                f"threshold={threshold:.4f} passed={passed}",
                flush=True,
            )
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
