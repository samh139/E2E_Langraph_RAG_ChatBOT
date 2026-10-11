"""MongoDB persistence for explicit offline DeepEval benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient


class EvaluationResultStore:
    """Persist one result per benchmark run and query, with retry-safe upserts."""

    def __init__(
        self,
        uri: str,
        database: str = "rag_evaluation",
        collection: str = "deepeval_results",
    ) -> None:
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.client.admin.command("ping")
        self.collection = self.client[database][collection]
        self.collection.create_index(
            [("run_id", ASCENDING), ("query_id", ASCENDING)], unique=True
        )
        self.collection.create_index([("created_at", DESCENDING)])

    def save(self, record: dict[str, Any]) -> None:
        document = {**record, "created_at": datetime.now(timezone.utc)}
        self.collection.replace_one(
            {"run_id": document["run_id"], "query_id": document["query_id"]},
            document,
            upsert=True,
        )

    def close(self) -> None:
        self.client.close()
