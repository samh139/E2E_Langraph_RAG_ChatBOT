from typing import List, Optional
from app.ingestion.clustering.vector_loader import get_es_connection

def resolve_cluster_ids_to_chunk_ids(
    cluster_ids: List[str],
) -> Optional[List[str]]:
    """
    Resolves lvl1 cluster_ids → chunk_ids by expanding
    into level-2 subclusters using parent_cluster.
    """

    if not cluster_ids:
        return None

    es = get_es_connection()
    chunk_ids = set()

    # 🔹 Fetch ALL level-2 subclusters under given lvl1 clusters
    res = es.search(
        index="clusters_v2",
        body={
            "size": 1000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"level": 2}},
                        {"terms": {"parent_cluster": cluster_ids}},
                    ]
                }
            },
            "_source": ["chunk_ids"],
        },
    )

    hits = res.get("hits", {}).get("hits", [])

    for h in hits:
        chunk_ids.update(h["_source"].get("chunk_ids", []))

    return list(chunk_ids) if chunk_ids else None

'''

def resolve_cluster_ids_to_chunk_ids(
    cluster_ids: List[str],
) -> Optional[List[str]]:
    """
    Resolves lvl1 cluster_ids → chunk_ids by expanding
    into level-2 subclusters using parent_cluster.
    """

    if not cluster_ids:
        return None

    es = get_es_connection()
    chunk_ids = set()

    # 🔹 Fetch ALL level-2 subclusters under given lvl1 clusters
    res = es.search(
        index="clusters_v2",
        body={
            "size": 1000,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"level": 2}},
                        {"terms": {"parent_cluster": cluster_ids}},
                    ]
                }
            },
            "_source": ["chunk_ids"],
        },
    )

    hits = res.get("hits", {}).get("hits", [])

    for h in hits:
        chunk_ids.update(h["_source"].get("chunk_ids", []))

    return list(chunk_ids) if chunk_ids else None


def resolve_cluster_ids_to_chunk_ids(
    cluster_ids: List[str],
    expand_level2: bool = True,
) -> Optional[List[str]]:
    """
    Resolves cluster_ids → chunk_ids
    Optionally expands into level-2 subclusters.
    """

    if not cluster_ids:
        return None

    es = get_es_connection()
    chunk_ids = set()

    # 1️⃣ Fetch level-1 clusters
    res = es.search(
        index="clusters_v2",  # ideally CLUSTERS_ALIAS
        body={
            "size": len(cluster_ids),
            "query": {
                "terms": {
                    "cluster_id": cluster_ids
                }
            }
        }
    )

    hits = res.get("hits", {}).get("hits", [])

    for h in hits:
        src = h["_source"]
        chunk_ids.update(src.get("chunk_ids", []))

        # 2️⃣ Expand level-2 subclusters if enabled
        if expand_level2 and src.get("subclusters"):
            sub_res = es.search(
                index="clusters_v2",
                body={
                    "size": len(src["subclusters"]),
                    "query": {
                        "terms": {
                            "cluster_id": src["subclusters"]
                        }
                    }
                }
            )
            for sh in sub_res.get("hits", {}).get("hits", []):
                chunk_ids.update(sh["_source"].get("chunk_ids", []))

    return list(chunk_ids) if chunk_ids else None

    '''