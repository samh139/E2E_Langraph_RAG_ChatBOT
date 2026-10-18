"""Reliable Kafka delivery helpers used by ingestion workers."""

import json
import os
from datetime import datetime, timezone


def produce_confirmed(producer, *, topic, key, value, timeout=30):
    errors = []
    delivered = []

    def on_delivery(error, message):
        delivered.append(True)
        if error is not None:
            errors.append(str(error))

    producer.produce(topic=topic, key=key, value=value, on_delivery=on_delivery)
    remaining = producer.flush(timeout)
    if errors:
        raise RuntimeError(f"Kafka delivery to {topic} failed: {errors[0]}")
    if remaining or not delivered:
        raise TimeoutError(f"Kafka delivery to {topic} was not confirmed within {timeout}s")


def publish_dead_letter(producer, *, topic, message, stage, error, timeout=30):
    """Publish the original record and failure context before its source is acked."""
    raw_value = message.value()
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="replace")
    raw_key = message.key()
    if isinstance(raw_key, bytes):
        raw_key = raw_key.decode("utf-8", errors="replace")

    record = {
        "source_topic": message.topic(),
        "source_partition": message.partition(),
        "source_offset": message.offset(),
        "source_key": raw_key,
        "payload": raw_value,
        "failure_stage": stage,
        "error": str(error),
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    record_key = f"{message.topic()}:{message.partition()}:{message.offset()}"
    produce_confirmed(
        producer,
        topic=topic,
        key=record_key.encode("utf-8"),
        value=json.dumps(record).encode("utf-8"),
        timeout=timeout,
    )


def retry_settings():
    attempts = int(os.getenv("KAFKA_PROCESSING_MAX_ATTEMPTS", "5"))
    initial_delay = float(os.getenv("KAFKA_RETRY_INITIAL_SECONDS", "1"))
    max_delay = float(os.getenv("KAFKA_RETRY_MAX_SECONDS", "30"))
    if attempts < 1 or initial_delay < 0 or max_delay < initial_delay:
        raise ValueError("Invalid Kafka retry settings")
    return attempts, initial_delay, max_delay
