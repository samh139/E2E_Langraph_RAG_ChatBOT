"""Document-level ranked retrieval metrics for golden evaluation cases."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


K = 5


def _document_key(value: str) -> str:
    # Golden data may contain a path while Elasticsearch stores only file_name.
    return PurePosixPath(value.replace("\\", "/")).name.strip().casefold()


def score_document_ranking(
    retrieved_chunks: list[dict[str, Any]], relevant_documents: list[str]
) -> list[dict[str, Any]]:
    """Score unique retrieved documents at the fixed cutoff K=5.

    Repeated chunks from one file count once, at that file's first rank.
    Precision@5 uses five as its denominator even when fewer results were returned.
    """
    relevant = {
        _document_key(name)
        for name in relevant_documents
        if isinstance(name, str) and _document_key(name)
    }
    ranked_documents: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        file_name = chunk.get("file_name")
        if not isinstance(file_name, str):
            continue
        key = _document_key(file_name)
        if key and key not in seen:
            seen.add(key)
            ranked_documents.append(key)

    top_k = ranked_documents[:K]
    hits = sum(document in relevant for document in top_k)
    reciprocal_rank = next(
        (1.0 / rank for rank, document in enumerate(top_k, start=1) if document in relevant),
        0.0,
    )
    recall = hits / len(relevant) if relevant else 0.0
    return [
        {"name": "Precision@5", "status": "completed", "score": hits / K},
        {"name": "Recall@5", "status": "completed", "score": recall},
        {"name": "MRR@5", "status": "completed", "score": reciprocal_rank},
    ]
