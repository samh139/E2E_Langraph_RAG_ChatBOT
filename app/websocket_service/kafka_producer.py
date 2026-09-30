from confluent_kafka import Producer
import json

producer = Producer({'bootstrap.servers': 'kafka:9092'})

def send_message(topic, key , value):
    producer.produce(topic, key=key, value=json.dumps(value))
    producer.poll(0)  # Trigger delivery of messages