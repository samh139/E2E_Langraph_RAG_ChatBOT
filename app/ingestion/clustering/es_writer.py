# app/ingestion/clustering/es_writer.py

import os
from datetime import datetime, timezone
from elasticsearch import Elasticsearch
from app.ingestion.clustering.config import ES_HOST, EMBEDDING_DIM

class ESWriter:
    """
    Handles all Elasticsearch writes for cluster documents, fully optimized for the 9.x SDK.
    """

    def __init__(self, index_name: str):
        self.index_name = index_name
        self.active_index = index_name

        # Grab secure cluster passwords
        es_password = os.getenv("ES_PASSWORD", "strongpassword123")

        # Initialize the secure 9.x client with Basic Authentication
        self.es = Elasticsearch(
            ES_HOST,
            basic_auth=("elastic", es_password)
        )

    def create_index(self, index_name: str):
        """Creates target index passing explicit mappings parameters matching 9.x rules."""
        print(f"[ES] Verifying/Creating cluster index: {index_name} (dims={EMBEDDING_DIM})")

        if self.es.indices.exists(index=index_name):
            print(f"[✓] Cluster index '{index_name}' verified active.")
            return

        # Properties mapping wrapper floating at the top-level
        target_mappings = {
            "properties": {
                "cluster_id": {"type": "keyword"},
                "level": {"type": "integer"},
                "parent_cluster": {"type": "keyword"},
                "summary": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM, # Will resolve to 384 via corrected config
                    "index": True,
                    "similarity": "cosine",
                },
                "chunk_ids": {"type": "keyword"},
                "subclusters": {"type": "keyword"},
                "meta_stats": {"type": "object"},
                "last_updated": {"type": "date"}
            }
        }

        try:
            # Native 9.x parameters call map
            self.es.indices.create(index=index_name, mappings=target_mappings)
            print(f"[✓] Cluster Index '{index_name}' successfully initialized.")
        except Exception as e:
            print(f"[X] Cluster index instantiation block failed: {e}")
            raise e
    
    def write_cluster_doc(self, index: str, doc_id: str, body: dict):
        payload = dict(body)
        payload["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.es.index(index=index, id=doc_id, document=payload)

    def update_cluster_field(self, index: str, cluster_id: str, field_name: str, field_value):
        # Native 9.x update body schema execution context mapping
        self.es.update(
            index=index,
            id=cluster_id,
            body={"doc": {field_name: field_value}},
            retry_on_conflict=5
        )

    def safe_delete_index(self):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
