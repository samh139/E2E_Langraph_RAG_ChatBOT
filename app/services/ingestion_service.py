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
from app.workflow.ingestion_workflow import graph as IngestionLangGraph
from app.workflow.state import IngestionState
from app.utils.kafka_delivery import produce_confirmed
from app.workers.embed_worker import EmbedWorker
from app.workers.indexer_worker import IndexerWorker
from app.ingestion.clustering.cluster_job import run_clustering_job


class IngestionService:
    def __init__(self):
        self.storage = MinioStorageClient()
        # Fallback cascade logic to ensure same port is mapped across all sub-workers
        self.kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST") or "127.0.0.1:9094"
        
        self.producer_config = {
            'bootstrap.servers': self.kafka_bootstrap,
            'acks': 'all',
            'linger.ms': 5,
            'max.in.flight.requests.per.connection': 1
        }
        self.producer = Producer(self.producer_config)
        self.chunks_topic = os.getenv("KAFKA_CHUNKS_TOPIC", "dsprawl.chunks")

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
        processed_count = 0

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
                
                # Compiled StateGraph returns a mapping, even with a Pydantic input.
                if not isinstance(final_state, dict):
                    final_state = final_state.model_dump()
                if final_state.get("error"):
                    raise RuntimeError(final_state["error"])
                chunks = final_state.get("chunks") or []
                if not chunks:
                    raise ValueError("Workflow produced no chunks")
                print(f"[✓] Graph extraction complete. Pushing to Kafka topic: {self.chunks_topic}")
                
                for chunk_item in chunks:
                    content_str = chunk_item["content"]
                    
                    chunk_payload = {
                        "chunk_id": chunk_item["chunk_id"],
                        "doc_id": chunk_item["doc_id"],
                        "file_name": chunk_item["file_name"],
                        "file_type": chunk_item["file_type"],
                        "content": content_str,
                        "content_hash": hashlib.sha1(content_str.strip().encode("utf-8")).hexdigest(),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    
                    produce_confirmed(
                        self.producer,
                        topic=self.chunks_topic,
                        key=chunk_payload["chunk_id"].encode('utf-8'),
                        value=json.dumps(chunk_payload).encode('utf-8')
                    )
                
                self.producer.flush()
                processed_count += 1
                
                if os.path.exists(local_worker_path):
                    os.remove(local_worker_path)

                print(f"[✓] Document ingestion complete for: {file_path.name}")
            except Exception as e:
                print(f"[X] Error processing file {file_path.name}: {e}")
                continue
                
        return processed_count # FIXED: Crucial count returned to caller

    def run_embedding_phase(self):
        print("\n=====================================================================")
        print(f"🧠 [PHASE 2] Handing Execution Over to EmbedWorker Module")
        print("=====================================================================")
        # Explicit bootstrap config passed to match running engine context
        embed_worker = EmbedWorker(kafka_bootstrap=self.kafka_bootstrap)
        embed_worker.run()

    def run_indexing_phase(self):
        print("\n=====================================================================")
        print(f"🔍 [PHASE 3] Handing Execution Over to IndexerWorker Module")
        print("=====================================================================")
        # Explicit bootstrap config passed to match running engine context
        indexer_worker = IndexerWorker(kafka_bootstrap=self.kafka_bootstrap)
        indexer_worker.run()

    async def execute_complete_pipeline(self, folder_path: str):
        start_time = time.perf_counter()
        print("=====================================================================")
        print(f"✨ UNIFIED INGESTION ENGINE LIFECYCLE INITIALIZED AT: {time.strftime('%X')}")
        print("=====================================================================")
        
        total_files = await self.bulk_ingest(folder_path)
        if total_files == 0:
            print("[X] Ingestion lifecycle aborted: Processing folder source path is empty or files failed.")
            return

        # This entry point is a finite batch: finish publishing embeddings before
        # starting the indexer's idle timer (model loading can take over a minute).
        await asyncio.to_thread(self.run_embedding_phase)
        await asyncio.to_thread(self.run_indexing_phase)

                # 🎯 [PHASE 4]: HIERARCHICAL TOPIC CLUSTERING & SUMMARY MAP GENERATION
        print("\n=====================================================================")
        print(f"🗂️ [PHASE 4] Rebuilding Hierarchical Topic Clusters & Summaries")
        print("=====================================================================")
        try:
            # Executes text grouping dynamically over all chunks now safe in ES
            await asyncio.to_thread(run_clustering_job, dry_run=False)
            print("[✓] Hierarchical document clustering schema map complete.")
        except Exception as cluster_err:
            print(f"[X] Clustering optimization execution pass failed: {cluster_err}")

        duration = time.perf_counter() - start_time
        print("\n=====================================================================")
        print(f"🏆 SUCCESS: END-TO-END BATCH LIFECYCLE COMPLETE! Total Time: {duration:.3f}s")
        print("=====================================================================")
        
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("app/configs/.env")
    
    service = IngestionService()
    TARGET_DATA_DIR = "./data"
    
    asyncio.run(service.execute_complete_pipeline(TARGET_DATA_DIR))
