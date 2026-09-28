# clustering/cluster_builder.py
# ---------------------------------------------------------
# Cluster Builder Orchestrator (Level 1 + Level 2)
# ---------------------------------------------------------
# Responsibilities:
#   • Load vectors + chunk_ids from vector_loader
#   • Run Level-1 Agglomerative clustering
#   • Create parent cluster docs (level = 1)
#       - initialize: subclusters = []
#   • Run Level-2 KMeans subclustering for large clusters
#   • Write clusters to ES via ESWriter
#   • Return the full cluster doc list to cluster_job
# ---------------------------------------------------------

from datetime import datetime
import numpy as np

from app.ingestion.clustering.config import (
    CHUNKS_INDEX,
    AGGLOMERATIVE_MIN_CLUSTER_SIZE,
    SUBCLUSTER_MIN_SIZE,
    CLUSTERS_ALIAS,
    EMBEDDING_DIM
)

from app.ingestion.clustering.vector_loader import load_all_vectors_and_ids , get_es_connection
from app.ingestion.clustering.agglomerative import run_agglomerative_clustering
from app.ingestion.clustering.kmeans_subclusters import build_subclusters_for_root_cluster
from app.ingestion.clustering.es_writer import ESWriter
from app.app_logger import LoggerFactory
from app.ingestion.clustering.cluster_summarizer import summarize_cluster
from app.ingestion.clustering.chunk_fetcher import fetch_chunk_texts

logger = LoggerFactory.get_logger("cluster_builder")



# ---------------------------------------------------------
# Build Level-1 Cluster Document
# ---------------------------------------------------------
def make_root_cluster_doc(cluster_id: str, centroid: np.ndarray, chunk_ids: list):
    """
    level=1 cluster document with an initialized empty subclusters list.
    """
    return {
        "cluster_id": cluster_id,
        "level": 1,
        "parent_cluster": None,
        "vector": centroid.tolist(),
        "chunk_ids": chunk_ids,
        "subclusters": [],       # IMPORTANT per spec
        "meta_stats": {
            "chunk_count": len(chunk_ids),
            "avg_length": None,
        },
        "last_updated": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------
# Master Build Function (Called by cluster_job)
# ---------------------------------------------------------
def build_all_clusters(es_writer: ESWriter):
    """
    Main orchestrator that produces:
        - Level-1 clusters via Agglomerative
        - Level-2 clusters via KMeans inside those clusters

    Writes everything into a temporary ES index via ESWriter.
    Returns:
        list of ALL cluster docs (level1 + level2)
    """
    logger.info("[BUILDER] Loading vectors from chunks_es...")
    vectors, chunk_ids = load_all_vectors_and_ids()

    assert vectors.shape[1] == EMBEDDING_DIM, (
    f"Embedding dim mismatch: {vectors.shape[1]} != {EMBEDDING_DIM}"
)

    N = len(chunk_ids)
    logger.info(f"[BUILDER] Loaded {N} vectors for clustering")

    # -----------------------------------------------------
    # Level-1: Agglomerative
    # -----------------------------------------------------
    logger.info("[BUILDER] Running Level-1 Agglomerative clustering...")

    '''
    # -----------------------------------------------------
    # Level-1A: Agglomerative-Silhouette score for clustering
    # -----------------------------------------------------
    best_threshold = find_best_threshold(vectors)
    logger.info(f"[BUILDER] Best threshold found: {best_threshold}")
    '''
    root_clusters = run_agglomerative_clustering(vectors, chunk_ids)
    logger.info(f"[BUILDER] Created {len(root_clusters)} root clusters")

    all_cluster_docs = []

    # -----------------------------------------------------
    # Process each Level-1 cluster
    # -----------------------------------------------------
    for root_id, member_indices in root_clusters.items():
        # vectors of members
        member_vecs = vectors[member_indices]
        member_ids = [chunk_ids[i] for i in member_indices]
        logger.info(f"Member IDs Generated: {len(member_ids)}")

        centroid = np.mean(member_vecs, axis=0)

        # Build parent doc
        root_doc = make_root_cluster_doc(root_id, centroid, member_ids)

        texts = fetch_chunk_texts(member_ids)
        root_doc["summary"], _ = summarize_cluster(texts)

        # Write root cluster doc
        es_writer.write_cluster_doc(
            index=es_writer.active_index,
            doc_id=root_id,
            body=root_doc
        )
        all_cluster_docs.append(root_doc)

        logger.info(f"[BUILDER] Root cluster {root_id} written "
                    f"({len(member_ids)} chunks)")

        # -------------------------------------------------
        # Level-2: K-Means Subclustering (only large clusters)
        # -------------------------------------------------
        if len(member_ids) >= SUBCLUSTER_MIN_SIZE:
            logger.info(f"[BUILDER] Subclustering {root_id} "
                        f"(size={len(member_ids)})...")
            

            sub_docs = build_subclusters_for_root_cluster(
                es_writer=es_writer,
                parent_cluster_id=root_id,
                vectors=member_vecs,
                chunk_ids=member_ids
            )

            all_cluster_docs.extend(sub_docs)
        else:
            logger.info(f"[BUILDER] Root cluster {root_id} too small "
                        f"for subclustering; skipping.")

    logger.info("[BUILDER] Completed full cluster build.")
    return all_cluster_docs
