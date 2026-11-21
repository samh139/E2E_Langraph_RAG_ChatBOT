"""Shared document/query embedding client.

Set EMBEDDING_PROVIDER=vertex to call a deployed Vertex AI custom prediction
endpoint or publisher model, or retain EMBEDDING_PROVIDER=ollama for local development.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from functools import lru_cache
from typing import Literal

import ollama


EmbeddingTask = Literal["search_document", "search_query"]


def embedding_dimensions() -> int:
    return int(os.getenv("EMBEDDING_DIMENSIONS", "768"))


@lru_cache(maxsize=1)
def _ollama_client() -> ollama.Client:
    host = os.getenv(
        "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://docker.internal")
    )
    return ollama.Client(
        host=host,
        timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
    )


@lru_cache(maxsize=1)
def _vertex_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return AuthorizedSession(credentials)


def _embed_vertex(texts: list[str], task: EmbeddingTask) -> list[list[float]]:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "e2e-langragh-chatbot-510511")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    endpoint_id = os.getenv("VERTEX_EMBEDDING_ENDPOINT_ID", "").strip()
    model_name = os.getenv("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001")
    timeout = float(os.getenv("VERTEX_EMBEDDING_TIMEOUT_SECONDS", "120"))
    max_workers = int(os.getenv("VERTEX_EMBEDDING_MAX_WORKERS", "8"))

    # gemini-embedding-001 accepts 1 instance per :predict request on Vertex AI
    batch_size = 1 if model_name.startswith("gemini-embedding") else 16
    sanitized_texts = [t if (t and t.strip()) else " " for t in texts]
    expected_dims = embedding_dimensions()

    if endpoint_id:
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/{endpoint_id}:predict"
        )
    else:
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_name}:predict"
        )
        vertex_task = "RETRIEVAL_DOCUMENT" if task == "search_document" else "RETRIEVAL_QUERY"

    batches = [
        (start, sanitized_texts[start : start + batch_size])
        for start in range(0, len(sanitized_texts), batch_size)
    ]

    def _embed_batch(item: tuple[int, list[str]]) -> list[list[float]]:
        start, batch = item
        if endpoint_id:
            body = {"instances": [{"content": text, "task_type": task} for text in batch]}
        else:
            body = {
                "instances": [{"content": text, "task_type": vertex_task} for text in batch],
                "parameters": {"outputDimensionality": expected_dims},
            }

        response = _vertex_session().post(url, json=body, timeout=timeout)
        if not response.ok:
            raise RuntimeError(
                f"Vertex embedding failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(batch):
            raise RuntimeError("Vertex embedding returned an invalid prediction count")

        batch_vectors: list[list[float]] = []
        for index, prediction in enumerate(predictions):
            vector = prediction.get("embeddings") if isinstance(prediction, dict) else prediction
            if isinstance(vector, dict):
                vector = vector.get("values")
            if not isinstance(vector, list) or len(vector) != expected_dims:
                actual = len(vector) if isinstance(vector, list) else "invalid"
                raise RuntimeError(
                    f"Vertex embedding {start + index} has {actual} dimensions; "
                    f"expected {expected_dims}"
                )
            batch_vectors.append([float(value) for value in vector])
        return batch_vectors

    # executor.map preserves the exact input order of batches
    vectors: list[list[float]] = []
    worker_count = min(max_workers, len(batches))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for batch_vectors in executor.map(_embed_batch, batches):
            vectors.extend(batch_vectors)

    return vectors


def embed_texts(
    texts: list[str], task: EmbeddingTask = "search_document"
) -> list[list[float]]:
    """Embed text with the configured provider, validating its output shape."""
    if not texts:
        return []
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
    if provider == "vertex":
        vectors = _embed_vertex(texts, task)
    elif provider == "ollama":
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        result = _ollama_client().embed(model=model, input=texts)
        vectors = [list(vector) for vector in result.embeddings]
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider!r}")

    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding provider returned {len(vectors)} vectors for {len(texts)} inputs")
    expected = embedding_dimensions()
    for index, vector in enumerate(vectors):
        if len(vector) != expected:
            raise RuntimeError(
                f"Embedding {index} has {len(vector)} dimensions; expected {expected}"
            )
    return vectors


async def embed_texts_async(
    texts: list[str], task: EmbeddingTask = "search_document"
) -> list[list[float]]:
    """Async adapter for services whose surrounding code is asynchronous."""
    return await asyncio.to_thread(embed_texts, texts, task)
