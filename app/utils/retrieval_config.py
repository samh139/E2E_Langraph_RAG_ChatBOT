"""Shared Elasticsearch settings for chunk retrieval utilities."""

import os
import re

from elasticsearch import Elasticsearch


def chunk_index_alias() -> str:
    """Return the write/read alias configured by the ingestion AliasManager."""
    explicit_alias = os.getenv("ES_INDEX_ALIAS", "").strip()
    if explicit_alias:
        return explicit_alias

    base = os.getenv("ES_INDEX_NAME", "es_documents").strip() or "es_documents"
    normalized_base = re.sub(r"[^a-z0-9_-]", "_", base.lower())
    return f"{normalized_base}_live"


def create_es_client() -> Elasticsearch:
    """Create the same Elasticsearch client configuration used by ingestion."""
    options: dict[str, object] = {}
    password = os.getenv("ES_PASSWORD")
    if password:
        options["basic_auth"] = (os.getenv("ES_USERNAME", "elastic"), password)
    return Elasticsearch(os.getenv("ES_HOST", "http://elasticsearch:9200"), **options)
