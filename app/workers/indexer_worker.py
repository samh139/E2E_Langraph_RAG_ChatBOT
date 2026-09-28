# app/workers/indexer_worker.py

import os
from pathlib import Path
import json
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(Path(__file__).resolve().parents[1] / "configs" / ".env")

class IndexerWorker:
    def __init__(self, kafka_bootstrap=None):
        # Synchronized bootstrap mapping across all components
        self.kafka_bootstrap = (kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP")
                                or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
                                or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")
                                or "127.0.0.1:9094")
        self.input_topic = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")

        es_host = os.getenv("ES_HOST", "http://elasticsearch:9200")
        es_password = os.getenv("ES_PASSWORD") # 🎯 Grab our 9.x credential string

        # Enforce a guaranteed string fallback and clean empty env spaces
        env_index = os.getenv("ES_INDEX_NAME", "").strip()
        self.index_name = env_index if env_index else "es_documents"

        es_options = {}
        if es_password:
            es_options["basic_auth"] = (os.getenv("ES_USERNAME", "elastic"), es_password)
        self.es = Elasticsearch(es_host, **es_options)

        # Kafka consumer setup
        self.consumer = Consumer({
            "bootstrap.servers": self.kafka_bootstrap,
            "group.id": "indexer-worker",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False
        })

        self.create_index_if_not_exists()

    def create_index_if_not_exists(self):
        # ✅ Fixed Indentation: Now properly nested directly under the class scope
        if not self.index_name or not isinstance(self.index_name, str):
            self.index_name = "es_documents"

        try:
            if self.es.indices.exists(index=self.index_name):
                print(f"[✓] Elasticsearch index '{self.index_name}' already exists.")
                return

            print(f"[*] Creating Elasticsearch index '{self.index_name}' with modern 9.x structures...")

            # Properties structure floats cleanly at the top-level under the mappings keyword parameter
            target_mappings = {
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
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }

            # Explicit named call mapping to fit the new SDK interface specification
            self.es.indices.create(index=self.index_name, mappings=target_mappings)
            print(f"[✓] Elasticsearch index '{self.index_name}' successfully initialized.")
        except Exception as e:
            print(f"[X] Index verification failed against target '{self.index_name}': {repr(e)}")
            raise e

    def index_chunk(self, chunk):
        embedding = chunk.get("embedding_vector")
        if not embedding:
            raise ValueError(f"Missing embedding for chunk: {chunk.get('chunk_id')}")

        document = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "file_name": chunk.get("file_name"),
            "file_type": chunk.get("file_type"),
            "content": chunk["content"],
            "content_hash": chunk.get("content_hash"),
            "created_at": chunk.get("created_at"),
            "embedding_vector": embedding
        }
        self.es.index(index=self.index_name, id=chunk["chunk_id"], document=document)

    def run(self):
        self.consumer.subscribe([self.input_topic])
        print(f"[*] IndexerWorker listening on '{self.input_topic}'...")

        try:
            empty_polls = 0
            while empty_polls < 40:
                msg = self.consumer.poll(timeout=1.5)
                if msg is None:
                    empty_polls += 1
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise RuntimeError(f"Kafka error: {msg.error()}")

                empty_polls = 0
                try:
                    chunk = json.loads(msg.value().decode("utf-8"))
                    self.index_chunk(chunk)
                    self.consumer.commit(msg, asynchronous=False)
                    #print(f"[✓] Indexed chunk: {chunk['chunk_id']} | Source: {chunk.get('file_name')}")
                except Exception as e:
                    print(f"[X] Indexing failed: {repr(e)}")
                    raise

        finally:
            self.consumer.close()
            self.es.close()
        print("[*] IndexerWorker finished.")

if __name__ == "__main__":
    worker = IndexerWorker()
    worker.run()
