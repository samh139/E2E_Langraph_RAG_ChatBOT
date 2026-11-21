"""Embedding providers used by ingestion, retrieval, and long-term memory."""

from app.embeddings.client import embed_texts, embed_texts_async, embedding_dimensions

__all__ = ["embed_texts", "embed_texts_async", "embedding_dimensions"]
