"""Vertex AI custom prediction server for a packaged Nomic Embed model."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


MODEL_DIR = os.getenv("AIP_MODEL_DIR", "/model")
EXPECTED_DIMS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
app = Flask(__name__)


def _resolve_model_dir() -> str:
    """Materialize Vertex's managed copy of artifactUri to local disk."""
    storage_uri = os.getenv("AIP_STORAGE_URI")
    if not storage_uri:
        return MODEL_DIR
    from google.cloud import storage

    if not storage_uri.startswith("gs://"):
        raise RuntimeError("AIP_STORAGE_URI must be a gs:// URI")
    bucket_name, _, prefix = storage_uri[5:].partition("/")
    target = Path(tempfile.gettempdir()) / "nomic-model"
    target.mkdir(parents=True, exist_ok=True)
    bucket = storage.Client().bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
        relative = blob.name[len(prefix.rstrip("/")) + 1 :]
        if relative:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination))
    return str(target)


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    model = SentenceTransformer(_resolve_model_dir(), trust_remote_code=True, device="cpu")
    return model


@app.get("/health")
def health():
    try:
        return ("ok", 200) if load_model() else ("model unavailable", 503)
    except Exception as exc:
        app.logger.exception("Model health check failed")
        return (str(exc), 503)


@app.post("/predict")
def predict():
    body = request.get_json(force=True, silent=False)
    instances = body.get("instances") if isinstance(body, dict) else None
    if not isinstance(instances, list) or not instances:
        return jsonify(error="instances must be a non-empty array"), 400

    texts = []
    for instance in instances:
        if not isinstance(instance, dict):
            return jsonify(error="each instance must be an object"), 400
        content = instance.get("content")
        task = instance.get("task_type")
        if not isinstance(content, str) or not content.strip():
            return jsonify(error="instance content must be a non-empty string"), 400
        if task not in {"search_document", "search_query"}:
            return jsonify(error="task_type must be search_document or search_query"), 400
        texts.append(f"{task}: {content}")

    vectors = load_model().encode(
        texts,
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vectors.shape[1] != EXPECTED_DIMS:
        return jsonify(error=f"model returned {vectors.shape[1]} dimensions; expected {EXPECTED_DIMS}"), 500
    return jsonify(predictions=[{"embeddings": vector.tolist()} for vector in vectors])
