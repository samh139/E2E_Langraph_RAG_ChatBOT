import os
import sys
import json
import uuid
import hashlib
import time
import yaml  # Added for production roles mapping resolution
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from confluent_kafka import Producer

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from app.ingestion.minio_client import MinioStorageClient
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.docx_extractor import DocxExtractor
from app.utils.chunking_engine import ChunkingEngine
from app.utils.file_utils import get_file_type
from app.utils.kafka_delivery import produce_confirmed
from app.workers.embed_worker import EmbedWorker
from app.workers.indexer_worker import IndexerWorker
from app.utils.alias_manager import AliasManager


class IngestionService:
    def __init__(self):
        self.storage = None
        self.kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP") or os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("KAFKA_BOOTSTRAP_LOCALHOST") or "127.0.0.1:9094"
        
        self.producer_config = {
            'bootstrap.servers': self.kafka_bootstrap,
            'acks': 'all',
            'linger.ms': 5,
            'max.in.flight.requests.per.connection': 1
        }
        self.producer = Producer(self.producer_config)
        self.chunks_topic = os.getenv("KAFKA_CHUNKS_TOPIC", "es.chunks")
        self._last_chunk_count = 0
        self._failed_files = []
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        
        # Load the document-to-role map used to tag each ingested chunk.
        self.roles_config_path = ROOT_DIR / "app" / "configs" / "rbac_roles.yaml"
        self._load_roles_mapping()

    def _load_roles_mapping(self):
        """Load document-to-role assignments from the RBAC YAML file."""
        try:
            with open(self.roles_config_path, "r", encoding="utf-8") as roles_file:
                config = yaml.safe_load(roles_file) or {}
            self.roles_mapping = config.get("roles_mapping", {})
            self.fallback_role = config.get("security_defaults", {}).get(
                "fallback_role", "bank_employee"
            )
            if not isinstance(self.roles_mapping, dict):
                raise ValueError("roles_mapping must be a mapping of roles to filenames")
        except Exception as e:
            print(
                f"[⚠️ WARNING] Could not load {self.roles_config_path}: {e}. "
                "Using the bank_employee fallback role."
            )
            self.roles_mapping = {}
            self.fallback_role = "bank_employee"

    def _resolve_allowed_role(self, file_name: str) -> str:
        """Production explicit match checker for incoming banking assets."""
        for role, files_list in self.roles_mapping.items():
            if file_name in files_list:
                return role
        return self.fallback_role

    async def bulk_ingest(self, target_folder_path: str, dry_run: bool = False) -> int:
        print("\n=====================================================================")
        print("🚀 [PHASE 1] Starting content-based document scan, extraction, and chunking")
        print("=====================================================================")
        
        dir_path = Path(target_folder_path)
        if not dir_path.exists():
            print(f"[X] Directory error: Path not found at {target_folder_path}")
            return 0

        files = []
        for candidate in dir_path.iterdir():
            if not candidate.is_file():
                continue
            try:
                detected_type = get_file_type(str(candidate))
            except ValueError:
                continue
            files.append((candidate, detected_type))
        
        if not files:
            print("[*] No target documents found.")
            return 0

        print(f"[*] Found {len(files)} PDF/DOCX documents. Running extraction and chunking...")
        processed_count = 0
        self._last_chunk_count = 0
        self._failed_files = []

        for file_path, detected_type in files:
            run_id = str(uuid.uuid4())
            print(f"\n[*] Processing document: {file_path.name} | Run ID: {run_id[:8]}")
            
            # 🔒 Explicitly find out which target role owns this bank document
            resolved_role = self._resolve_allowed_role(file_path.name)
            print(
                f"🔒 Security Role Assigned: [{resolved_role}] "
                f"for document {file_path.name}"
            )
            
            try:
                if dry_run:
                    local_worker_path = str(file_path)
                    print(f"[DRY-RUN] Extracting locally; no MinIO upload or Kafka publish.")
                else:
                    if self.storage is None:
                        self.storage = MinioStorageClient()
                    minio_key = f"raw_vault/{run_id}_{file_path.name}"
                    self.storage.upload_document(str(file_path), minio_key)

                    local_worker_path = f"/tmp/rag_ingest/{run_id}_{file_path.name}"
                    os.makedirs(os.path.dirname(local_worker_path), exist_ok=True)
                    self.storage.client.fget_object(
                        self.storage.bucket_name, minio_key, local_worker_path
                    )

                extractor = self.pdf_extractor if detected_type == "pdf" else self.docx_extractor
                extraction = await asyncio.to_thread(extractor.extract, local_worker_path)
                elements = (extraction.get("extraction_metadata") or {}).get("elements", [])
                if not elements:
                    raise ValueError("Extractor produced no document elements")
                raw_chunks = ChunkingEngine(max_chars=1000, overlap=250).chunk(
                    elements=elements,
                    file_name=file_path.name,
                )
                chunks = []
                for idx, chunk_item in enumerate(raw_chunks):
                    content = chunk_item.get("content", chunk_item.get("text_content", "")) if isinstance(chunk_item, dict) else getattr(chunk_item, "content", str(chunk_item))
                    raw_chunk_id = chunk_item.get("id", chunk_item.get("chunk_id")) if isinstance(chunk_item, dict) else getattr(chunk_item, "id", None)
                    chunks.append({
                        "chunk_id": raw_chunk_id or f"{run_id}_c{idx}",
                        "doc_id": run_id,
                        "file_name": file_path.name,
                        "file_type": detected_type,
                        "allowed_role": resolved_role,
                        "content": content.strip(),
                    })
                if not chunks:
                    raise ValueError("Chunker produced no chunks")
                self._last_chunk_count += len(chunks)

                if dry_run:
                    print(f"[DRY-RUN] {file_path.name}: extraction would produce {len(chunks)} chunks.")
                    processed_count += 1
                    continue

                print(f"[✓] Extraction complete. Pushing to Kafka topic: {self.chunks_topic}")
                
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
                        "allowed_role": chunk_item.get("allowed_role", resolved_role),
                    }
                    
                    produce_confirmed(
                        self.producer,
                        topic=self.chunks_topic,
                        key=chunk_payload["chunk_id"].encode('utf-8'),
                        value=json.dumps(chunk_payload).encode('utf-8')
                    )
                
                self.producer.flush()
                processed_count += 1
                
                if not dry_run and os.path.exists(local_worker_path):
                    os.remove(local_worker_path)

                print(f"[✓] Document ingestion complete for: {file_path.name}")
            except Exception as e:
                print(f"[X] Error processing file {file_path.name}: {e}")
                self._failed_files.append(file_path.name)
                continue
                
        return processed_count

    def run_embedding_phase(self):
        print("\n=====================================================================")
        print(f"🧠 [PHASE 2] Handing Execution Over to EmbedWorker Module")
        print("=====================================================================")
        embed_worker = EmbedWorker(kafka_bootstrap=self.kafka_bootstrap)
        embed_worker.run()

    def run_indexing_phase(self, target_index: str | None = None) -> int:
        print("\n=====================================================================")
        print(f"🔍 [PHASE 3] Handing Execution Over to IndexerWorker Module")
        print("=====================================================================")
        indexer_worker = IndexerWorker(
            kafka_bootstrap=self.kafka_bootstrap,
            target_index=target_index,
        )
        return indexer_worker.run()

    async def execute_complete_pipeline(self, folder_path: str, dry_run: bool = False):
        start_time = time.perf_counter()
        print("=====================================================================")
        print(f"✨ UNIFIED INGESTION ENGINE LIFECYCLE INITIALIZED AT: {time.strftime('%X')}")
        print("=====================================================================")
        
        total_files = await self.bulk_ingest(folder_path, dry_run=dry_run)
        if total_files == 0:
            print("[X] Ingestion lifecycle aborted: Processing folder source path is empty or files failed.")
            return {"status": "NO_FILES", "processed_files": 0}

        if dry_run:
            print(
                f"[DRY-RUN] Preview complete: {total_files} files and "
                f"{self._last_chunk_count} chunks; failed files: "
                f"{self._failed_files}. No MinIO, Kafka, or Elasticsearch writes "
                "or alias changes were made."
            )
            return {
                "status": "DRY_RUN_PARTIAL" if self._failed_files else "DRY_RUN",
                "processed_files": total_files,
                "chunks": self._last_chunk_count,
                "failed_files": self._failed_files,
                }
        if self._failed_files:
            raise RuntimeError(
            "Full index rebuild cancelled because some source files failed: "
            + ", ".join(self._failed_files)
            + ". The current live alias was not changed."
            )
        alias_manager = AliasManager()
        staging_index = None
        try:
            old_live_index = alias_manager.ensure_live_alias()
            staging_index = alias_manager.make_temp_index_name()
            alias_manager.create_index(staging_index)
            print(
                f"[*] Building replacement index '{staging_index}' while alias "
                f"'{alias_manager.alias}' remains on '{old_live_index}'."
            )
            await asyncio.to_thread(self.run_embedding_phase)
            indexed_count = await asyncio.to_thread(
                self.run_indexing_phase, staging_index
            )
            if indexed_count != self._last_chunk_count:
                raise RuntimeError(
                    f"Staging index received {indexed_count} chunks; "
                    f"expected {self._last_chunk_count}. Live alias was not changed."
                )
            alias_manager.es.indices.refresh(index=staging_index)
            stored_count = alias_manager.es.count(index=staging_index)["count"]
            if stored_count != self._last_chunk_count:
                raise RuntimeError(
                    f"Staging index contains {stored_count} documents; "
                    f"expected {self._last_chunk_count}. Live alias was not changed."
                )
            previous_indices = alias_manager.swap_alias(staging_index)
            print(
                f"[✓] Alias '{alias_manager.alias}' now points to "
                f"'{staging_index}'. Previous indices retained for rollback: "
                f"{previous_indices}."
            )
        finally:
            alias_manager.close()
        
        duration = time.perf_counter() - start_time
        print("\n=====================================================================")
        print(f"🏆 SUCCESS: END-TO-END BATCH LIFECYCLE COMPLETE! Total Time: {duration:.3f}s")
        print("=====================================================================")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("app/configs/.env")
    service = IngestionService()
    TARGET_DATA_DIR = "./data"
    dry_run = os.getenv("INGESTION_DRY_RUN", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    asyncio.run(service.execute_complete_pipeline(TARGET_DATA_DIR, dry_run=dry_run))
