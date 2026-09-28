# clustering/vector_loader.py
# ---------------------------------------------------------
# Vector Loader (Scroll-based)
# ---------------------------------------------------------
# Provides:
#   • VectorLoader class
#   • load_all_vectors(...)
#   • load_all_vectors_and_ids()  <-- required by cluster_builder
# ---------------------------------------------------------

import time
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from elasticsearch import Elasticsearch

from app.app_logger import LoggerFactory
from app.ingestion.clustering.config import CHUNKS_INDEX, ES_HOST

logger = LoggerFactory.get_logger("vector_loader")

def es_connect(es_host: str):
    options = {}
    password = os.getenv("ES_PASSWORD")
    if password:
        options["basic_auth"] = (os.getenv("ES_USERNAME", "elastic"), password)
    return Elasticsearch(es_host, **options)


# Initialize lazily; all clustering readers use the configured Docker endpoint.
es_connection = None


def get_es_connection() -> Elasticsearch:
    global es_connection
    if es_connection is None:
        es_connection = es_connect(es_host=ES_HOST)
    return es_connection


class VectorLoader:
    """
    Loads chunk vectors + metadata from chunks_es in safe batches.
    Uses ES Scroll API → safe for millions of documents.
    """

    def __init__(
        self,
        index: str,
        vector_field: str = "embedding_vector",
        length_field: Optional[str] = None,
        batch_size: int = 2000,
        scroll: str = "2m"
    ):
        self.index = index
        self.vector_field = vector_field
        self.length_field = length_field
        self.batch_size = batch_size
        self.scroll = scroll
        self.es: Elasticsearch = get_es_connection()

    # ---------------------------------------------------------
    def load_vectors(
        self,
        filter_query: Optional[Dict] = None
    ) -> Tuple[List[str], np.ndarray, List[int], List[Dict]]:
        """
        Returns:
            chunk_ids: List[str]
            vectors: np.ndarray NxD
            lengths: List[int]
            metadata: List[dict]
        """

        logger.info(f"[VEC] Loading vectors from index: {self.index}")

        chunk_ids: List[str] = []
        vectors: List[np.ndarray] = []
        lengths: List[int] = []
        metadata: List[Dict] = []

        query = filter_query if filter_query else {"query": {"match_all": {}}}

        # -----------------------------------------
        # Scroll initialization
        # -----------------------------------------
        t0 = time.time()
        resp = self.es.search(
            index=self.index,
            body=query,
            size=self.batch_size,
            scroll=self.scroll
        )

        scroll_id = resp.get("_scroll_id")
        hits = resp["hits"]["hits"]
        total_loaded = 0

        # -----------------------------------------
        # Scroll loop
        # -----------------------------------------
        while hits:
            for h in hits:
                _id = h["_id"]
                src = h["_source"]

                vec = src.get(self.vector_field)
                if vec is None:
                    continue

                chunk_ids.append(_id)
                vectors.append(np.array(vec, dtype=np.float32))

                if self.length_field:
                    lengths.append(len(src.get(self.length_field, "")))
                else:
                    lengths.append(len(src.get("content", "")))

                metadata.append(src)

            total_loaded += len(hits)
            resp = self.es.scroll(scroll_id=scroll_id, scroll=self.scroll)
            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]

        # Cleanup scroll
        try:
            self.es.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass

        #logger.info(f"[VEC] Loaded {total_loaded:,} vectors in {time.time()-t0:.2f}s")
        logger.info(
            f"[VEC] Scanned {total_loaded:,} docs | "
            f"Loaded {len(vectors):,} vectors in {time.time()-t0:.2f}s"
        )

        if total_loaded == 0:
            logger.error("[VEC] ❌ No vectors found! Check vector_field mapping.")
            return [], np.zeros((0, 384)), [], []

        matrix = np.vstack(vectors)
        return chunk_ids, matrix, lengths, metadata


# ---------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------
def load_all_vectors(
    chunks_index: str = CHUNKS_INDEX,
    vector_field: str = "embedding_vector",
    length_field: Optional[str] = None
):
    loader = VectorLoader(
        index=chunks_index,
        vector_field=vector_field,
        length_field=length_field
    )
    return loader.load_vectors()


# ---------------------------------------------------------
# REQUIRED BY cluster_builder.py
# ---------------------------------------------------------
def load_all_vectors_and_ids():
    """
    Returns exactly:
        vectors: np.ndarray
        chunk_ids: List[str]

    (Lengths & metadata are ignored because cluster_builder
     does not use them.)
    """
    chunk_ids, vectors, _, _ = load_all_vectors()
    return vectors, chunk_ids
