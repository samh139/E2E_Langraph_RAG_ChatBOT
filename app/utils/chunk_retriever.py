"""Hybrid chunk retrieval: Elasticsearch BM25 + ANN, RRF, then cross-encoder."""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Any

from elasticsearch import Elasticsearch

from app.utils.retrieval_config import chunk_index_alias, create_es_client
from app.utils.text_bm25 import BM25Retriever


logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device)


def rrf_merge(
    bm25_hits: list[dict[str, Any]],
    ann_hits: list[dict[str, Any]],
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Fuse BM25 and ANN ranks without comparing their incompatible raw scores."""
    merged: dict[str, dict[str, Any]] = {}

    for source_name, hits in (("bm25", bm25_hits), ("ann", ann_hits)):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.get("chunk_id")
            if not chunk_id:
                continue

            candidate = merged.setdefault(chunk_id, dict(hit))
            candidate["rrf_score"] = candidate.get("rrf_score", 0.0) + (
                1.0 / (rank_constant + rank)
            )
            if source_name == "bm25":
                candidate["bm25_score"] = hit.get("bm25_score", 0.0)
            else:
                candidate["ann_score"] = hit.get("ann_score", 0.0)
                # Keep content and document metadata regardless of which list
                # first contributed this chunk.
                for key in ("doc_id", "file_name", "file_type", "content"):
                    candidate.setdefault(key, hit.get(key))

    return sorted(
        merged.values(),
        key=lambda item: item.get("rrf_score", 0.0),
        reverse=True,
    )


def _sigmoid(value: float) -> float:
    # Clamp logits to avoid overflow on unusual model outputs.
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class ChunkRetriever:
    """Retrieve raw ingested chunks and rerank candidates for relevance."""

    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 50,
        bm25: BM25Retriever | None = None,
        es: Elasticsearch | None = None,
        cross_encoder: Any | None = None,
    ) -> None:
        self.top_k = max(1, top_k)
        self.candidate_k = max(self.top_k, candidate_k)
        self.index = chunk_index_alias()
        self._owns_es = es is None
        self.es = es or create_es_client()
        self.bm25 = bm25 or BM25Retriever(es=self.es, index=self.index)
        self._cross_encoder = cross_encoder

    def _ann(self, query_vector: list[float]) -> list[dict[str, Any]]:
        response = self.es.search(
            index=self.index,
            size=self.candidate_k,
            knn={
                "field": "embedding_vector",
                "query_vector": query_vector,
                "k": self.candidate_k,
                "num_candidates": max(self.candidate_k * 3, 50),
            },
            source={"excludes": ["embedding_vector"]},
        )

        hits = response.get("hits", {}).get("hits", [])
        results: list[dict[str, Any]] = []
        for hit in hits:
            source = hit.get("_source", {})
            content = source.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            results.append(
                {
                    "chunk_id": source.get("chunk_id") or hit.get("_id"),
                    "doc_id": source.get("doc_id"),
                    "file_name": source.get("file_name"),
                    "file_type": source.get("file_type"),
                    "content": content,
                    "content_hash": source.get("content_hash"),
                    "created_at": source.get("created_at"),
                    "ann_score": float(hit.get("_score") or 0.0),
                }
            )
        logger.info("ANN returned %d chunks from %s", len(results), self.index)
        return results

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            model_name = os.getenv(
                "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            device = os.getenv("CROSS_ENCODER_DEVICE", "cpu")
            self._cross_encoder = _load_cross_encoder(model_name, device)
        return self._cross_encoder

    def retrieve(
        self,
        query: str,
        query_vector: list[float],
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if len(query_vector) != 768:
            raise ValueError(
                f"Expected a 768-dimensional Ollama query vector; got {len(query_vector)}"
            )

        bm25_hits = self.bm25.search(query, top_k=self.candidate_k)
        ann_hits = self._ann(query_vector)
        candidates = rrf_merge(bm25_hits, ann_hits)[: self.candidate_k]
        if not candidates:
            return []

        cross_encoder = self._get_cross_encoder()
        pairs = [(query, candidate["content"]) for candidate in candidates]
        raw_scores = cross_encoder.predict(
            pairs,
            batch_size=int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16")),
            show_progress_bar=False,
        )
        raw_scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        if len(raw_scores) != len(candidates):
            raise RuntimeError(
                f"Reranker returned {len(raw_scores)} scores for {len(candidates)} chunks"
            )

        for candidate, raw_score in zip(candidates, raw_scores, strict=True):
            candidate["relevance_score"] = _sigmoid(float(raw_score))

        candidates.sort(key=lambda item: item["relevance_score"], reverse=True)
        # Return source chunks with scores for the gateway/logging; no cluster
        # summaries or synthetic documents are created here.
        return candidates[: self.top_k]

    def close(self) -> None:
        if self._owns_es:
            self.es.close()
