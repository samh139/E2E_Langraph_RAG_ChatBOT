"""Confirm broker delivery before acknowledging upstream work."""


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
