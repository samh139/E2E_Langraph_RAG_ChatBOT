import os
from pathlib import Path
import json
import torch
from confluent_kafka import Consumer, Producer, KafkaError
from app.utils.kafka_delivery import produce_confirmed
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[1] / "configs" / ".env")

class EmbedWorker:
    def __init__(self, kafka_bootstrap=None):
        # Fallback order: Explicit argument override -> Env configuration -> Default Fallback
        self.kafka_bootstrap = (kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP")
                                or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
                                or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")
                                or "127.0.0.1:9094")
        self.input_topic = os.getenv("KAFKA_CHUNKS_TOPIC", "dsprawl.chunks")
        self.output_topic = os.getenv("KAFKA_EMBEDDED_TOPIC", "dsprawl.embedded_chunks")

        self.consumer = Consumer({
            'bootstrap.servers': self.kafka_bootstrap,
            'group.id': 'embed-worker',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })
        self.producer = Producer({
            'bootstrap.servers': self.kafka_bootstrap,
            'acks': 'all',
            'linger.ms': 5
        })

        # 🎯 HARDWARE ENGINE ACCELERATOR RESOLUTION (TAILORED FOR M5 MAC & DOCKER CORES)
        if torch.backends.mps.is_available():
            self.device = "mps"  # Full Apple Silicon GPU/Neural Engine compilation
        elif torch.cuda.is_available():
            self.device = "cuda" # Nvidia cluster framework
        else:
            self.device = "cpu"  # Isolated docker emulation layer fallback

        print(f"[*] Initializing local embedding engine on compute device: {self.device.upper()}")

        self.model_name = "BAAI/bge-small-en-v1.5"
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def get_embedding(self, text: str) -> list:
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def run(self):
        self.consumer.subscribe([self.input_topic])
        print(f"[*] EmbedWorker active. Processing chunks with {self.model_name}...")

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
                    raise RuntimeError(f"Embed consumer error: {msg.error()}")

                #print(f"[*] Message received in consumer: {msg.value().decode('utf-8')[:60]}...")
                empty_polls = 0

                try:
                    chunk = json.loads(msg.value().decode('utf-8'))
                    vec = self.get_embedding(chunk["content"])
                    chunk["embedding_vector"] = vec

                    produce_confirmed(
                        self.producer,
                        topic=self.output_topic,
                        key=chunk["chunk_id"].encode('utf-8'),
                        value=json.dumps(chunk).encode('utf-8')
                    )
                    self.consumer.commit(msg, asynchronous=False)
                except Exception as e:
                    print(f"[X] Embedded processing error: {e}")
                    raise

        finally:
            self.consumer.close()
        self.producer.flush()
        print("[*] EmbedWorker cycle executed successfully. Closed connections.")
