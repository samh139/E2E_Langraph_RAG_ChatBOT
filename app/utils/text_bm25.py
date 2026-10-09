"""BM25 search over the raw chunk index used by the ingestion pipeline."""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch

from app.utils.retrieval_config import chunk_index_alias, create_es_client


logger = logging.getLogger(__name__)


class BM25Retriever:
    """Search the live chunk alias using fields written by ingestion."""

    def __init__(
        self,
        es: Elasticsearch | None = None,
        index: str | None = None,
    ) -> None:
        self._owns_es = es is None
        self.es = es or create_es_client()
        self.index = index or chunk_index_alias()

    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        response = self.es.search(
            index=self.index,
            size=top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content^4", "file_name^2", "file_type"],
                    "type": "best_fields",
                }
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
                    "bm25_score": float(hit.get("_score") or 0.0),
                }
            )

        logger.info("BM25 returned %d chunks from %s", len(results), self.index)
        return results

    def close(self) -> None:
        if self._owns_es:
            self.es.close()
