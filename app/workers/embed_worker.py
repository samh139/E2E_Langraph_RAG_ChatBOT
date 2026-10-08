import json
import os
from pathlib import Path

import ollama
from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from dotenv import load_dotenv

from app.utils.kafka_delivery import produce_confirmed


load_dotenv(Path(__file__).resolve().parents[1] / "configs" / ".env")


class EmbedWorker:
    """Consume chunks, embed them with Ollama, and publish vectors to Kafka."""

    def __init__(self, kafka_bootstrap: str | None = None):
        self.kafka_bootstrap = (
            kafka_bootstrap
            or os.getenv("KAFKA_BOOTSTRAP")
            or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
            or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")
            or "127.0.0.1:9094"
        )
        self.input_topic = os.getenv("KAFKA_CHUNKS_TOPIC", "es.chunks")
        self.output_topic = os.getenv(
            "KAFKA_EMBEDDED_TOPIC", "es.embedded_chunks"
        )

        self.model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.embedding_dimensions = 768
        self.batch_size = int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "32"))
        if self.batch_size < 1:
            raise ValueError("OLLAMA_EMBED_BATCH_SIZE must be at least 1")

        ollama_host = os.getenv(
            "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        )
        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        self.ollama_client = ollama.Client(host=ollama_host, timeout=ollama_timeout)

        self.consumer = Consumer(
            {
                "bootstrap.servers": self.kafka_bootstrap,
                "group.id": os.getenv("KAFKA_EMBED_GROUP_ID", "embed-worker"),
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        self.producer = Producer(
            {
                "bootstrap.servers": self.kafka_bootstrap,
                "acks": "all",
                "linger.ms": 5,
            }
        )

        print(
            f"[*] Ollama embedding worker configured: model={self.model_name}, "
            f"host={ollama_host}, dimensions={self.embedding_dimensions}"
        )

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.ollama_client.embed(model=self.model_name, input=texts)
        embeddings = [list(vector) for vector in response.embeddings]

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        for index, vector in enumerate(embeddings):
            if len(vector) != self.embedding_dimensions:
                raise RuntimeError(
                    f"Ollama model {self.model_name!r} returned vector {index} with "
                    f"{len(vector)} dimensions; expected {self.embedding_dimensions}"
                )

        return embeddings

    @staticmethod
    def _parse_message(message) -> dict | None:
        if message.error():
            if message.error().code() == KafkaError._PARTITION_EOF:
                return None
            raise RuntimeError(f"Kafka consumer error: {message.error()}")

        try:
            chunk = json.loads(message.value().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid JSON in {message.topic()} partition {message.partition()} "
                f"offset {message.offset()}: {exc}"
            ) from exc

        if not isinstance(chunk, dict):
            raise ValueError("Kafka chunk payload must be a JSON object")
        if not isinstance(chunk.get("content"), str) or not chunk["content"].strip():
            raise ValueError(f"Chunk {chunk.get('chunk_id')!r} has no text content")
        if not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"]:
            raise ValueError("Chunk payload is missing a non-empty chunk_id")
        return chunk

    def process_batch(self, messages: list) -> int:
        valid_messages = []
        chunks = []
        for message in messages:
            chunk = self._parse_message(message)
            if chunk is not None:
                valid_messages.append(message)
                chunks.append(chunk)

        if not chunks:
            return 0

        embeddings = self.get_embeddings_batch([chunk["content"] for chunk in chunks])
        if len(embeddings) != len(chunks):
            # Defensive check: never commit inputs if any embedding is missing.
            raise RuntimeError(
                f"Got {len(embeddings)} vectors for {len(chunks)} chunks; "
                "input offsets were not committed"
            )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk["embedding_vector"] = embedding
            produce_confirmed(
                self.producer,
                topic=self.output_topic,
                key=chunk["chunk_id"].encode("utf-8"),
                value=json.dumps(chunk).encode("utf-8"),
            )

        # Commit each partition's highest processed offset only after all of its
        # output records have been confirmed. A crash before this can cause
        # duplicate output, so downstream indexing should remain idempotent by chunk_id.
        highest_offsets: dict[tuple[str, int], int] = {}
        for message in valid_messages:
            partition_key = (message.topic(), message.partition())
            highest_offsets[partition_key] = max(
                highest_offsets.get(partition_key, -1), message.offset()
            )

        offsets = [
            TopicPartition(topic, partition, offset + 1)
            for (topic, partition), offset in highest_offsets.items()
        ]
        self.consumer.commit(offsets=offsets, asynchronous=False)
        return len(chunks)

    def run(self) -> None:
        self.consumer.subscribe([self.input_topic])
        print(
            f"[*] EmbedWorker listening on {self.input_topic!r}; "
            f"embedding with {self.model_name} in batches of {self.batch_size}."
        )

        try:
            empty_polls = 0
            while empty_polls < 40:
                messages = self.consumer.consume(
                    num_messages=self.batch_size,
                    timeout=1.5,
                )
                if not messages:
                    empty_polls += 1
                    continue

                empty_polls = 0
                processed = self.process_batch(messages)
                if processed:
                    print(f"[*] Embedded and published {processed} chunks.")
        finally:
            self.consumer.close()
            self.producer.flush()

        print("[*] EmbedWorker cycle finished.")
