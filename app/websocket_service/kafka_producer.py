from confluent_kafka import Producer
import json
import os

producer = Producer({
    'bootstrap.servers': os.getenv(
        "KAFKA_BOOTSTRAP", os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    )
})

def send_message(topic, key, value, headers=None):
    producer.produce(topic, key=key, value=json.dumps(value), headers=headers)
    producer.poll(0)  # Trigger delivery of messages
