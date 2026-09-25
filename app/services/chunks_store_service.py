# app/ingestion/services/chunks_store_service.py
import os
import json
import uuid
from datetime import datetime
from kafka import KafkaProducer
from pathlib import Path
from app.core.utils.file_utils import read_text_file, extract_text_from_docx, extract_text_from_pdf
from app.ingestion.chunkers.splitters import get_splitters
from dotenv import load_dotenv
import hashlib


load_dotenv("app/config/secrets.env")


#KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST")

CHUNKS_TOPIC = os.getenv("KAFKA_CHUNKS_TOPIC", "dsprawl.chunks")

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=5,
)


class Job:
    def __init__(self, file_path: str, owner: str = "owner@example.com", acl=None):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.file_id = str(uuid.uuid4())
        self.owner = owner
        self.acl = acl or ["ROLE_USER"]
        self.etag = str(uuid.uuid4())

def make_chunks_from_text(text: str, chunk_size=800, overlap=100):
    splitter = get_splitters(chunk_size, overlap)
    return [c.page_content for c in splitter.create_documents([text])]

def ingest_file(job: Job):
    path = job.file_path.lower()
    if path.endswith(".pdf"):
        text = extract_text_from_pdf(job.file_path)
    elif path.endswith(".docx"):
        text = extract_text_from_docx(job.file_path)
    else:
        text = read_text_file(job.file_path)

    chunks = make_chunks_from_text(text)
    for i, chunk_text in enumerate(chunks):
        chunk = {
            "doc_id": job.file_id,
            "file_name": job.file_name,
            "chunk_id": f"{job.file_id}_chunk_{i}",
            "content": chunk_text,
            "content_hash": hashlib.sha1(chunk_text.strip().encode("utf-8")).hexdigest(),
            "page": None,
            "owner": job.owner,
            "acl": job.acl,
            "etag": job.etag,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "file_name": job.file_name,
            }
        }
        producer.send(CHUNKS_TOPIC, chunk)
    producer.flush()
    return len(chunks)