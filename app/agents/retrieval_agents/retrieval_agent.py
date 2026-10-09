from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from typing import Any

import ollama

from app.utils.chunk_retriever import ChunkRetriever
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

        model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        ollama_host = os.getenv(
            "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        )
        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        retriever: ChunkRetriever | None = None

        try:
            ollama_client = _get_ollama_client(ollama_host, ollama_timeout)
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
            )

            score = float(candidates[0]["relevance_score"]) if candidates else 0.0
            passed = bool(candidates) and score >= threshold
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
