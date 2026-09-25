import os
import json
import time
import requests
from confluent_kafka import Consumer, Producer, KafkaError
from dotenv import load_dotenv

load_dotenv("app/config/.env")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST", "127.0.0.1:9094")
INPUT_TOPIC = os.getenv("KAFKA_CHUNKS_TOPIC", "dsprawl.chunks")
OUTPUT_TOPIC = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434") 

class EmbedWorker:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'group.id': 'embed-worker',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })
        self.producer = Producer({
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'acks': 'all',
            'linger.ms': 5
        })

    def get_embedding(self, text: str):
        print(f"Embedding request started for text length: {len(text)} characters")
        body = {
            "model": "nomic-embed-text",
            "input": [text]  
        }
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json=body, timeout=120)
        resp.raise_for_status()
        out = resp.json()
        return out["embeddings"]  

    def run(self):
        self.consumer.subscribe([INPUT_TOPIC])
        print(f"[*] EmbedWorker active. Consuming from '{INPUT_TOPIC}'...")
        
        empty_polls = 0
        while empty_polls < 4:
            msg = self.consumer.poll(timeout=1.5)
            if msg is None:
                empty_polls += 1
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"[X] Embed consumer error: {msg.error()}")
                break

            print("Message received in consumer :", msg.value().decode('utf-8')[:100] + "...")
            empty_polls = 0
            
            try:
                chunk = json.loads(msg.value().decode('utf-8'))
                vec = self.get_embedding(chunk["content"])
                chunk["embedding_vector"] = vec
                
                self.producer.produce(
                    topic=OUTPUT_TOPIC,
                    key=chunk["chunk_id"].encode('utf-8'),
                    value=json.dumps(chunk).encode('utf-8')
                )
                self.consumer.commit(msg, asynchronous=False)
            except Exception as e:
                print("embed error:", e)
                time.sleep(1)
        
        self.producer.flush()
        self.consumer.close()
