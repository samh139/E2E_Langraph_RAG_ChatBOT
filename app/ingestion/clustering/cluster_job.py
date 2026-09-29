# clustering/cluster_job.py
# ============================================================
# 🚀 Production Cluster Job Orchestrator
# ============================================================
# Responsibilities:
#   • Prepare temp index
#   • Run full clustering (L1 + L2)
#   • Write clusters → temp index
#   • Atomically swap alias after success
#   • Delete old index
#   • Publish cluster alias only after a successful build
#   • Chunk tag enrichment persists independently and resumes on retry
#   • Supports dry_run=True (no ES ops)
# ============================================================

from datetime import datetime
import traceback
import time

from app.ingestion.clustering.alias_manager import AliasManager
from app.ingestion.clustering.es_writer import ESWriter
from app.ingestion.clustering.cluster_builder import build_all_clusters
from app.ingestion.clustering.tag_enrichment import enrich_chunks
from app.ingestion.clustering.config import (
    CLUSTERS_ALIAS,
    DRY_RUN,
)
import argparse

from app.app_logger import LoggerFactory
logger = LoggerFactory.get_logger("cluster_job")


# ------------------------------------------------------------
# Helper: timestamp
# ------------------------------------------------------------
def ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------
# PRODUCTION RUNNER
# ------------------------------------------------------------
def run_clustering_job(dry_run: bool = DRY_RUN) -> dict:
    """
    Runs the full production cluster build:
        1) Create temp index
        2) Build clusters → write to temp index
        3) Atomic alias swap
        4) Delete old index

    Returns structured result object.
    """

    print("============================================================")
    print(f"[CLUSTER JOB] Starting clustering job  ({ts()})")
    print("============================================================")

    alias_mgr = AliasManager()

    # Generate temp index name
    temp_index = alias_mgr.make_temp_index_name()
    print(f"[CLUSTER JOB] Temp index → {temp_index}")

    # Create ESWriter (but ES operations may be skipped if dry_run=True)
    writer = ESWriter(temp_index)

    start_time = time.time()

    try:
        # ------------------------------------------------------------
        # STEP 1 — create new temp index
        # ------------------------------------------------------------
        if dry_run:
            logger.warning("[DRY-RUN] Skipping index creation")
        else:
            writer.create_index(temp_index)
            logger.info("[CLUSTER JOB] Temp index created successfully")

        # ------------------------------------------------------------
        # STEP 2 — run full cluster build
        # ------------------------------------------------------------
        logger.info("[CLUSTER JOB] Starting cluster build pipeline...")
        if not dry_run:
            logger.info("[CLUSTER JOB] Enriching chunk tags...")
            enrich_chunks()
        docs = build_all_clusters(writer)
        logger.info(f"[CLUSTER JOB] Cluster build complete → {len(docs)} docs")

        # ------------------------------------------------------------
        # STEP 3 — alias swap (production only)
        # ------------------------------------------------------------
        if dry_run:
            logger.warning("[DRY-RUN] Skipping alias swap + cleanup")
            return {
                "status": "DRY_RUN_OK",
                "temp_index": temp_index,
                "clusters_built": len(docs),
                "elapsed_sec": round(time.time() - start_time, 2),
            }

        logger.info("[CLUSTER JOB] Performing alias swap...")
        alias_mgr.finalize_new_index(temp_index, delete_old=True)

        logger.info("[CLUSTER JOB] Alias now points to new clusters")
        logger.info(f"[CLUSTER JOB] COMPLETED SUCCESSFULLY ({ts()})")

        return {
            "status": "SUCCESS",
            "index": temp_index,
            "clusters_built": len(docs),
            "elapsed_sec": round(time.time() - start_time, 2),
        }

    except Exception as e:
        # ------------------------------------------------------------
        # FAILURE HANDLING
        # ------------------------------------------------------------
        logger.exception(f"❌ [CLUSTER JOB] FAILURE! {e}")

        if not dry_run:
            # Cleanup temp index
            try:
                writer.safe_delete_index()
                logger.info("[CLUSTER JOB] Temp index deleted after failure")
            except Exception:
                logger.exception("⚠️ Failed to delete temp index during cleanup")

        return {
            "status": "FAILED",
            "error": str(e),
            "elapsed_sec": round(time.time() - start_time, 2),
        }
