import os
import sys
import json
import uuid
import hashlib
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from confluent_kafka import Producer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from app.ingestion.minio_client import MinioStorageClient
from app.workflow.ingestion_workflow import IngestionWorkflow as IngestionLangGraph
from app.workflow.state import IngestionState
from workers.embed_worker import EmbedWorker
from workers.indexer_worker import IndexerWorker

class IngestionService:
    def __init__(self):
        self.storage = MinioStorageClient()
        self.kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9094")
        
        self.producer_config = {
            'bootstrap.servers': self.kafka_bootstrap,
            'acks': 'all',
            'linger.ms': 5,
            'max.in.flight.requests.per.connection': 1
        }
        self.producer = Producer(self.producer_config)
        self.chunks_topic = "dsprawl.chunks"

    async def bulk_ingest(self, target_folder_path: str) -> int:
        print("\n=====================================================================")
        print(f"🚀 [PHASE 1] Starting Bulk Ingest Scanner & LangGraph Extraction")
        print("=====================================================================")
        
        dir_path = Path(target_folder_path)
        if not dir_path.exists():
            print(f"[X] Directory error: Path not found at {target_folder_path}")
            return 0

        supported_exts = {'.pdf', '.docx', '.pptx'}
        files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]
        
        if not files:
            print("[*] No target documents found.")
            return 0

        print(f"[*] Found {len(files)} items. Running LangGraph workflow pipeline...")

        for file_path in files:
            run_id = str(uuid.uuid4())
            print(f"\n[*] Processing document: {file_path.name} | Run ID: {run_id[:8]}")
            
            try:
                minio_key = f"raw_vault/{run_id}_{file_path.name}"
                self.storage.upload_document(str(file_path), minio_key)
                
                local_worker_path = f"/tmp/rag_ingest/{run_id}_{file_path.name}"
                os.makedirs(os.path.dirname(local_worker_path), exist_ok=True)
                self.storage.client.fget_object(self.storage.bucket_name, minio_key, local_worker_path)

                state_input = IngestionState(
                    run_id=run_id,
                    file_path=local_worker_path,
                    file_name=file_path.name,
                    file_extension=file_path.suffix.lower(),
                    mime_type=f"application/{file_path.suffix.lower()[1:]}",
                    file_type=file_path.suffix.lower()[1:],
                    status="pending",
                    stage_timings_ms={},
                    extracted_text=""
                )

                print(f"[*] Dispatching to LangGraph workflow architecture...")
                final_state = await IngestionLangGraph.ainvoke(state_input)
                
                if hasattr(final_state, 'error') and getattr(final_state, 'error'):
                    print(f"[X] Workflow failed on file {file_path.name}: {getattr(final_state, 'error')}")
                    continue

                # The LangGraph workflow automatically maps text items to ChunkingAgent internally.
                # It appends structured document fragments into your state object attributes.
                chunks = getattr(final_state, "chunks", [])
                print(f"[✓] Graph extraction and chunking complete. Pushing to Kafka topic: {self.chunks_topic}")
                
                for idx, chunk_item in enumerate(chunks):
                    # Adapts to LangGraph Chunk output model keys dynamically
                    content_str = chunk_item.get("content", chunk_item.get("text_content", ""))
                    
                    chunk_payload = {
                        "doc_id": run_id,
                        "file_name": file_path.name,
                        "chunk_id": chunk_item.get("id", chunk_item.get("chunk_id", f"{run_id}_c{idx}")),
                        "content": content_str,
                        "content_hash": hashlib.sha1(content_str.strip().encode("utf-8")).hexdigest(),
                        "pages": chunk_item.get("pages", chunk_item.get("page", [])),
                        "heading": chunk_item.get("heading"),
                        "subheading": chunk_item.get("subheading"),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self.producer.produce(
                        topic=self.chunks_topic,
                        key=chunk_payload["chunk_id"].encode('utf-8'),
                        value=json.dumps(chunk_payload).encode('utf-8')
                    )
                
                self.producer.flush()
                
                if os.path.exists(local_worker_path):
                    os.remove(local_worker_path)

                print(f"[✓] Document ingestion complete for: {file_path.name}")

            except Exception as e:
                print(f"[X] Critical pipeline issue on asset {file_path.name}: {e}")

        return len(files)

    def run_embedding_phase(self):
        print("\n=====================================================================")
        print(f"🧠 [PHASE 2] Handing Execution Over to EmbedWorker Module")
        print("=====================================================================")
        # Instantiates and executes your actual class component code file logic cleanly
        embed_worker = EmbedWorker()
        embed_worker.start() if hasattr(embed_worker, 'start') else embed_worker.run()

    def run_indexing_phase(self):
        print("\n=====================================================================")
        print(f"🔍 [PHASE 3] Handing Execution Over to IndexerWorker Module")
        print("=====================================================================")
        # Instantiates and executes your actual class component code file logic cleanly
        indexer_worker = IndexerWorker()
        indexer_worker.start() if hasattr(indexer_worker, 'start') else indexer_worker.run()

    # def run_clustering_service(self):
    #     print("\n=====================================================================")
    #     print(f"📊 [PHASE 4] Executing Out-Of-Band Topic Clustering Service")
    #     print("=====================================================================")
    #     print("[*] Performing spatial clustering computation maps against index endpoints...")
    #     time.sleep(1.0)
    #     print("[SUCCESS] Spatial topology calculated. Clustering state sync complete.")

    async def execute_complete_pipeline(self, folder_path: str):
        start_time = time.perf_counter()
        print("=====================================================================")
        print(f"✨ UNIFIED INGESTION ENGINE LIFECYCLE INITIALIZED AT: {time.strftime('%X')}")
        print("=====================================================================")
        
        total_files = await self.bulk_ingest(folder_path)
        if total_files == 0:
            print("[X] Ingestion lifecycle aborted: Processing folder source path is empty.")
            return

        self.run_embedding_phase()
        self.run_indexing_phase()
        # self.run_clustering_service()

        duration = time.perf_counter() - start_time
        print("\n=====================================================================")
        print(f"🏆 SUCCESS: END-TO-END BATCH LIFECYCLE COMPLETE! Total Time: {duration:.3f}s")
        print("=====================================================================")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("app/configs/.env")
    
    service = IngestionService()
    TARGET_DATA_DIR = "./data_staging"
    
    asyncio.run(service.execute_complete_pipeline(TARGET_DATA_DIR))
