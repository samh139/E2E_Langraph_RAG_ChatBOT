# clustering/agglomerative.py
"""
Level-1 Agglomerative Clustering (Pointer Model)
Produces only:
    { cluster_id: [member_indices] }

cluster_builder will build the actual ES docs.
"""

import numpy as np
import uuid
from typing import Dict, List

from app.ingestion.clustering.config import (
    AGGLOMERATIVE_DISTANCE_THRESHOLD,
    AGGLOMERATIVE_MIN_CLUSTER_SIZE,
    VERBOSE,
)

from sklearn.cluster import AgglomerativeClustering
from app.app_logger import LoggerFactory

logger=LoggerFactory.get_logger("agglomerative")

from sklearn.metrics.pairwise import cosine_distances

# ---------------------------------------------------------
# Main API — returns dict: {cluster_id: [indices]}
# ---------------------------------------------------------

def run_agglomerative_clustering(vectors: np.ndarray, chunk_ids: List[str]) -> Dict[str, List[int]]:
    """
    vectors: np.ndarray (N x 384)
    chunk_ids: parallel list of IDs

    Returns:
        dict: { cluster_id: [list_of_indices] }
    """

    N = len(chunk_ids)
    if VERBOSE:
        logger.info(f"🔹 Running Agglomerative on {N} vectors")

    if N == 0:
        return {}

    # -----------------------------------------------------
    # Compute clustering labels
    # -----------------------------------------------------

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=AGGLOMERATIVE_DISTANCE_THRESHOLD,
    )

    labels = model.fit_predict(vectors)
    unique_labels = set(labels)

    if VERBOSE:
        logger.info(f"🔹 Produced {len(unique_labels)} raw clusters")

    # -----------------------------------------------------
    # Build raw buckets: {label → [indices]}
    # -----------------------------------------------------
    buckets = {}
    for idx, label in enumerate(labels):
        buckets.setdefault(label, []).append(idx)

    # -----------------------------------------------------
    # Separate large vs small clusters
    # -----------------------------------------------------
    large = {}
    small = {}

    for label, idxs in buckets.items():
        if len(idxs) < AGGLOMERATIVE_MIN_CLUSTER_SIZE:
            small[label] = idxs
        else:
            large[label] = idxs

    if VERBOSE:
        logger.info(f"🔹 Kept {len(large)} clusters, {len(small)} small clusters found")

    # Edge case: all clusters small → collapse into one big cluster
    if len(large) == 0:
        if VERBOSE:
            logger.info("⚠️ All clusters < MIN_SIZE → collapsing into one")
        return {
            f"lvl1-{uuid.uuid4().hex[:12]}": list(range(N))
        }

    # -----------------------------------------------------
    # Merge small clusters into the largest cluster
    # -----------------------------------------------------
    largest_label = max(large, key=lambda k: len(large[k]))

    for _, tiny_idxs in small.items():
        large[largest_label].extend(tiny_idxs)
    logger.info(f"🔹 Merged {len(small)} small clusters into: {largest_label}")

    # -----------------------------------------------------
    # Return clean dict: { cluster_id: [member_indices] }
    # -----------------------------------------------------
    result = {}
    for label, idxs in large.items():
        cid = f"lvl1-{uuid.uuid4().hex[:12]}"
        result[cid] = idxs

        if VERBOSE:
            logger.info(f"   • {cid}: {len(idxs)} members")

    if VERBOSE:
        logger.info(f"✅ Final Level-1 clusters: {len(result)}")

    return result

 