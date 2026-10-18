# app/workers/indexer_worker.py

import os
import time
from pathlib import Path
import json
from confluent_kafka import Consumer, KafkaError, Producer
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from app.utils.alias_manager import AliasManager
from app.utils.kafka_delivery import publish_dead_letter, retry_settings

load_dotenv(Path(__file__).resolve().parents[1] / "configs" / ".env")

class IndexerWorker:
    def __init__(self, kafka_bootstrap=None, target_index=None):
        # Synchronized bootstrap mapping across all components
        self.kafka_bootstrap = (kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP")
                                or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
                                or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")
                                or "127.0.0.1:9094")
        self.input_topic = os.getenv("KAFKA_EMBEDDED_TOPIC", "es.embedded_chunks")
        self.dead_letter_topic = os.getenv(
            "KAFKA_EMBEDDED_DLQ_TOPIC", f"{self.input_topic}.dlq"
        )

        es_host = os.getenv("ES_HOST", "http://elasticsearch:9200")
        es_password = os.getenv("ES_PASSWORD") # 🎯 Grab our 9.x credential string

        es_options = {}
        if es_password:
            es_options["basic_auth"] = (os.getenv("ES_USERNAME", "elastic"), es_password)
        self.es = Elasticsearch(es_host, **es_options)
        self.alias_manager = AliasManager(es=self.es)
        self.target_index = target_index

        if target_index:
            self.index_name = target_index
            if not self.es.indices.exists(index=target_index):
                self.alias_manager.create_index(target_index)
        else:
            self.index_name = self.alias_manager.alias
            self.alias_manager.ensure_live_alias()

        # Kafka consumer setup
        self.consumer = Consumer({
            "bootstrap.servers": self.kafka_bootstrap,
            "group.id": "indexer-worker",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False
        })
        self.producer = Producer({
            "bootstrap.servers": self.kafka_bootstrap,
            "acks": "all",
            "linger.ms": 5,
            "enable.idempotence": True,
        })

        print(f"[*] Elasticsearch write target: {self.index_name}")

    def index_chunk(self, chunk):
        embedding = chunk.get("embedding_vector")
        if not embedding:
            raise ValueError(f"Missing embedding for chunk: {chunk.get('chunk_id')}")

        document = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "file_name": chunk.get("file_name"),
            "file_type": chunk.get("file_type"),
            "allowed_role": chunk.get("allowed_role"),
            "content": chunk["content"],
            "content_hash": chunk.get("content_hash"),
            "created_at": chunk.get("created_at"),
            "embedding_vector": embedding
        }
        self.es.index(index=self.index_name, id=chunk["chunk_id"], document=document)

    @staticmethod
    def parse_chunk(message):
        try:
            chunk = json.loads(message.value().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid embedded chunk JSON: {exc}") from exc
        if not isinstance(chunk, dict):
            raise ValueError("Embedded chunk payload must be a JSON object")
        if not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"]:
            raise ValueError("Embedded chunk is missing a non-empty chunk_id")
        if not isinstance(chunk.get("doc_id"), str) or not chunk["doc_id"]:
            raise ValueError(f"Chunk {chunk['chunk_id']!r} is missing doc_id")
        if not isinstance(chunk.get("content"), str):
            raise ValueError(f"Chunk {chunk['chunk_id']!r} is missing text content")
        embedding = chunk.get("embedding_vector")
        if not isinstance(embedding, list) or not embedding or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise ValueError(f"Chunk {chunk['chunk_id']!r} has an invalid embedding")
        return chunk

    def index_with_retries(self, chunk):
        attempts, delay, max_delay = retry_settings()
        for attempt in range(1, attempts + 1):
            try:
                self.index_chunk(chunk)
                return
            except Exception as exc:
                meta = getattr(exc, "meta", None)
                status = getattr(exc, "status_code", None) or getattr(meta, "status", None)
                if status in {400, 404, 409}:
                    raise ValueError(
                        f"Elasticsearch permanently rejected chunk "
                        f"{chunk['chunk_id']!r} (HTTP {status}): {exc}"
                    ) from exc
                if attempt == attempts:
                    raise RuntimeError(
                        f"Elasticsearch write failed after {attempts} attempts; "
                        "source offset remains uncommitted for replay"
                    ) from exc
                print(
                    f"[!] Elasticsearch attempt {attempt}/{attempts} failed: "
                    f"{exc}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
                delay = min(max_delay, max(0.1, delay * 2))

    def run(self):
        self.consumer.subscribe([self.input_topic])
        print(f"[*] IndexerWorker listening on '{self.input_topic}'...")

        indexed_count = 0
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
                    try:
                        chunk = self.parse_chunk(msg)
                    except ValueError as exc:
                        publish_dead_letter(
                            self.producer,
                            topic=self.dead_letter_topic,
                            message=msg,
                            stage="indexing_input_validation",
                            error=exc,
                        )
                        self.consumer.commit(msg, asynchronous=False)
                        continue
                    try:
                        self.index_with_retries(chunk)
                    except ValueError as exc:
                        publish_dead_letter(
                            self.producer,
                            topic=self.dead_letter_topic,
                            message=msg,
                            stage="elasticsearch_permanent_rejection",
                            error=exc,
                        )
                        self.consumer.commit(msg, asynchronous=False)
                        continue
                    self.consumer.commit(msg, asynchronous=False)
                    indexed_count += 1
                    #print(f"[✓] Indexed chunk: {chunk['chunk_id']} | Source: {chunk.get('file_name')}")
                except Exception as e:
                    print(f"[X] Indexing failed: {repr(e)}")
                    raise

        finally:
            self.consumer.close()
            self.producer.flush()
            self.es.close()
        print("[*] IndexerWorker finished.")
        return indexed_count

if __name__ == "__main__":
    worker = IndexerWorker()
    worker.run()
