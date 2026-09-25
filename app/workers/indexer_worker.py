import os
import json
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

load_dotenv("app/config/.env")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST", "127.0.0.1:9094")
INPUT_TOPIC = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")

class IndexerWorker:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'group.id': 'indexer-worker',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })

    def run(self):
        self.consumer.subscribe([INPUT_TOPIC])
        print(f"[*] IndexerWorker active. Consuming from '{INPUT_TOPIC}'...")
        
        empty_polls = 0
        while empty_polls < 4:
            msg = self.consumer.poll(timeout=1.5)
            if msg is None:
                empty_polls += 1
                continue
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"[X] Indexer consumer error: {msg.error()}")
                break

            print("Message received in indexer consumer :", msg.value().decode('utf-8')[:100] + "...")
            empty_polls = 0
            
            try:
                payload = json.loads(msg.value().decode('utf-8'))
                # Your Elasticsearch indexing logic goes here
                print(f"[✓] Indexed Document in DB -> ID: {payload.get('chunk_id', 'unknown')[:12]}")
                self.consumer.commit(msg, asynchronous=False)
            except Exception as e:
                print("indexing error:", e)
                
        self.consumer.close()
