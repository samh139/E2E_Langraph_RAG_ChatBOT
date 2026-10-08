"""Create chunk-index generations and atomically move the live write alias."""

import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from elasticsearch import Elasticsearch


EMBEDDING_DIMS = 768


def chunk_index_mappings() -> dict:
    """Mappings shared by the normal and staged chunk indices."""
    return {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "file_type": {"type": "keyword"},
            "content": {"type": "text"},
            "content_hash": {"type": "keyword"},
            "created_at": {"type": "date"},
            "embedding_vector": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine",
            },
        }
    }


class AliasManager:
    """Manage a stable write alias over versioned physical chunk indices."""

    def __init__(self, es: Elasticsearch | None = None):
        base = os.getenv("ES_INDEX_NAME", "es_documents").strip() or "es_documents"
        self.base_index = re.sub(r"[^a-z0-9_-]", "_", base.lower())
        self.alias = (
            os.getenv("ES_INDEX_ALIAS", "").strip()
            or f"{self.base_index}_live"
        )

        if es is not None:
            self.es = es
            self._owns_client = False
        else:
            es_options = {}
            password = os.getenv("ES_PASSWORD")
            if password:
                es_options["basic_auth"] = (
                    os.getenv("ES_USERNAME", "elastic"),
                    password,
                )
            self.es = Elasticsearch(
                os.getenv("ES_HOST", "http://elasticsearch:9200"),
                **es_options,
            )
            self._owns_client = True

    def close(self) -> None:
        if self._owns_client:
            self.es.close()

    def make_temp_index_name(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{self.base_index}-rebuild-{timestamp}-{uuid4().hex[:8]}"

    def create_index(self, index_name: str) -> None:
        self.es.indices.create(index=index_name, mappings=chunk_index_mappings())

    def _current_targets(self) -> dict:
        if not self.es.indices.exists_alias(name=self.alias):
            return {}
        return self.es.indices.get_alias(name=self.alias)

    def ensure_live_alias(self) -> str:
        """Ensure the alias has one write target, preserving a legacy base index."""
        targets = self._current_targets()
        if targets:
            return next(iter(targets))

        if self.es.indices.exists(index=self.alias):
            raise RuntimeError(
                f"'{self.alias}' already exists as a concrete index, so it cannot "
                "also be used as an alias. Set ES_INDEX_ALIAS to a different name."
            )

        # Reuse an existing concrete index on first adoption. This keeps current
        # ingested data available while adding the new, non-conflicting alias.
        if self.es.indices.exists_alias(name=self.base_index):
            base_targets = self.es.indices.get_alias(name=self.base_index)
            if len(base_targets) != 1:
                raise RuntimeError(
                    f"Base name '{self.base_index}' points to multiple indices; "
                    "set ES_INDEX_ALIAS explicitly and resolve the base alias first."
                )
            target = next(iter(base_targets))
        elif self.es.indices.exists(index=self.base_index):
            target = self.base_index
        else:
            target = f"{self.base_index}-000001"
            if not self.es.indices.exists(index=target):
                self.create_index(target)

        self.es.indices.update_aliases(
            actions=[
                {
                    "add": {
                        "index": target,
                        "alias": self.alias,
                        "is_write_index": True,
                    }
                }
            ]
        )
        return target

    def swap_alias(self, new_index: str) -> list[str]:
        """Atomically point the live alias at a completed generation.

        Returns the old physical indices. They are intentionally retained for
        rollback and are not deleted here.
        """
        if not self.es.indices.exists(index=new_index):
            raise ValueError(f"Cannot move alias to missing index '{new_index}'.")

        targets = self._current_targets()
        actions = [
            {"remove": {"index": old_index, "alias": self.alias}}
            for old_index in targets
        ]
        actions.append(
            {
                "add": {
                    "index": new_index,
                    "alias": self.alias,
                    "is_write_index": True,
                }
            }
        )
        self.es.indices.update_aliases(actions=actions)
        return list(targets)
