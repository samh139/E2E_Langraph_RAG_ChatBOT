# clustering/kmeans_subclusters.py
# ---------------------------------------------------------
# ES-Integrated K-Means Subclustering Module (Level = 2)
# Pointer model: store chunk_ids only
# ---------------------------------------------------------

import uuid
import numpy as np
from sklearn.cluster import KMeans

from app.app_logger import LoggerFactory
from app.ingestion.clustering.es_writer import ESWriter
from app.ingestion.clustering.cluster_summarizer import summarize_cluster
from app.ingestion.clustering.chunk_fetcher import fetch_chunk_texts

logger = LoggerFactory.get_logger("kmeans_subclusters")


# ---------------------------------------------------------
# Balanced Adaptive K Policy (FINAL)
# ---------------------------------------------------------
def choose_k_balanced(cluster_size: int) -> int:
    """
    FINAL policy:
        <20     → K = 1
        20–200  → K = 3
        200–1000 → round(N/150), capped at 10
        >1000   → K = 10
    """
    if cluster_size < 5:
        return 1
    if 5 <= cluster_size <= 10:
        return 3
    if 10 < cluster_size <= 50:
        return min(max(2, round(cluster_size / 30)), 10)
    return 5


# ---------------------------------------------------------
# Generate Subcluster ID
# ---------------------------------------------------------
def make_subcluster_id(parent_cluster_id: str, numeric_id: int) -> str:
    return f"SC_{parent_cluster_id}_{numeric_id:02d}"


# ---------------------------------------------------------
# Compute KMeans Subclusters
# ---------------------------------------------------------
def compute_kmeans_subclusters(parent_cluster_id: str, vectors: np.ndarray, chunk_ids: list):
    N = len(chunk_ids)
    logger.info(f"[SUBCLUSTER] Parent={parent_cluster_id}  N={N}")

    K = choose_k_balanced(N)

    # Case: small cluster → no KMeans
    if K == 1:
        logger.info(f"[SUBCLUSTER] Skipping KMeans (K=1)")
        return [
            {
                "cluster_id": make_subcluster_id(parent_cluster_id, 1),
                "level": 2,
                "parent_cluster": parent_cluster_id,
                "vector": np.mean(vectors, axis=0).tolist(),
                "chunk_ids": chunk_ids,
                "meta_stats": {
                    "chunk_count": N,
                    "avg_length": None,
                }
            }
        ]

    logger.info(f"[SUBCLUSTER] Running KMeans K={K}")

    model = KMeans(n_clusters=K, random_state=42, n_init="auto")
    labels = model.fit_predict(vectors)
    centers = model.cluster_centers_

    sub_docs = []

    for k in range(K):
        member_idxs = np.where(labels == k)[0]
        member_ids = [chunk_ids[i] for i in member_idxs]

        sc_id = make_subcluster_id(parent_cluster_id, k + 1)

        sub_docs.append({
            "cluster_id": sc_id,
            "level": 2,
            "parent_cluster": parent_cluster_id,
            "vector": centers[k].tolist(),
            "chunk_ids": member_ids,
            "meta_stats": {
                "chunk_count": len(member_ids),
                "avg_length": None,
            }
        })

        logger.info(f"[SUBCLUSTER] {sc_id} → {len(member_ids)} chunks")

    return sub_docs


# ---------------------------------------------------------
# Write Subclusters to ES
# ---------------------------------------------------------
def write_subclusters_to_es(es_writer: ESWriter, root_cluster_id: str, subcluster_docs: list):
    """
    Writes all subclusters into es_writer.active_index
    and updates parent cluster with list of subcluster IDs.
    """
    index = es_writer.active_index
    logger.info(f"[ES] Writing {len(subcluster_docs)} subclusters → index={index}")

    for doc in subcluster_docs:
        texts = fetch_chunk_texts(doc["chunk_ids"])
        doc["summary"], _ = summarize_cluster(texts)
        es_writer.write_cluster_doc(index=index, doc_id=doc["cluster_id"], body=doc)

    sub_ids = [doc["cluster_id"] for doc in subcluster_docs]

    es_writer.update_cluster_field(
        index=index,
        cluster_id=root_cluster_id,
        field_name="subclusters",
        field_value=sub_ids
    )

    logger.info(f"[ES] Updated parent {root_cluster_id} with {len(sub_ids)} subclusters")


# ---------------------------------------------------------
# Public API Entry Point
# ---------------------------------------------------------
def build_subclusters_for_root_cluster(
    es_writer: ESWriter,
    parent_cluster_id: str,
    vectors: np.ndarray,
    chunk_ids: list
):
    sub_docs = compute_kmeans_subclusters(
        parent_cluster_id=parent_cluster_id,
        vectors=vectors,
        chunk_ids=chunk_ids,
    )
    
    write_subclusters_to_es(
        es_writer=es_writer,
        root_cluster_id=parent_cluster_id,
        subcluster_docs=sub_docs
    )

    return sub_docs
