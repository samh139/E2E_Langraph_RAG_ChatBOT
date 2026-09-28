"""
Utility: Nightly Cluster Refiner (Qwen-generated CHUNK tags + Vectors)
---------------------------------------------------------------------
Responsibilities:
  • Agglomerative reclustering (cosine distance)
  • Qwen generates chunk-level tags
  • Embeds those tags using nomic-embed-text
  • Aggregates tags per cluster
  • Generates cluster summary
  • Upserts cluster + enriched chunk docs
  • Designed for nohup overnight runs
"""

import datetime
import time
import numpy as np
import requests

from elasticsearch import Elasticsearch, helpers
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

ES_URL = "http://localhost:9200"
CHUNKS_INDEX = "dsprawl_documents"
CLUSTERS_INDEX = "clusters_v2"

ES = Elasticsearch(ES_URL)

OLLAMA_URL = "http://localhost:11434"

MAX_RETRIES = 3


# ----------------------------------------------------------------------
# Ollama helpers
# ----------------------------------------------------------------------

def fire_fast_modal_request(prompt: str):

    for attempt in range(MAX_RETRIES):

        try:

            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "gemma3:12b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )

            r.raise_for_status()

            return r.json()["response"]

        except Exception as e:

            print(f"⚠️ Ollama error (attempt {attempt+1}): {e}")
            time.sleep(3)

    return ""


# ----------------------------------------------------------------------
# CHANGED FUNCTION 1 (BATCHED EMBEDDINGS)
# ----------------------------------------------------------------------

def embed_texts(texts):
    vectors = []
    # Nomic usually returns 768 dimensions. 
    # Change this if you use a different model.
    EMBED_DIM = 768 

    for i, text in enumerate(texts):
        # Handle empty/none tags immediately without hitting the API
        if not text or not str(text).strip():
            vectors.append(np.zeros(EMBED_DIM))
            continue

        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=60
            )
            r.raise_for_status()
            
            vec = r.json().get("embedding")
            
            # Check if we actually got a list back and it's the right size
            if vec and len(vec) == EMBED_DIM:
                vec = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(vec)
                vectors.append(vec / norm if norm > 0 else vec)
            else:
                # API returned something weird or empty
                vectors.append(np.zeros(EMBED_DIM))
                
        except Exception as e:
            print(f"⚠️ embedding failed for chunk {i}")
            vectors.append(np.zeros(EMBED_DIM))

    # Now this will NEVER fail because every element is exactly (768,)
    return np.array(vectors)

# ----------------------------------------------------------------------
# CHANGED FUNCTION 2 (BATCH TAGGING)
# ----------------------------------------------------------------------

def tags_for_chunk(texts):

    numbered = "\n\n".join(
        [f"{i+1}. {t[:1500]}" for i, t in enumerate(texts)]
    )

    prompt = f"""
Extract 5 short semantic tags for EACH numbered paragraph.

Return format strictly:

1: tag1, tag2, tag3
2: tag1, tag2, tag3

Paragraphs:
{numbered}
"""

    raw = fire_fast_modal_request(prompt)

    results = []

    for line in raw.split("\n"):

        if ":" not in line:
            continue

        parts = line.split(":", 1)[1]

        tags = [
            t.strip().strip('"').strip("'").title()
            for t in parts.split(",")
            if len(t.strip()) > 1
        ]

        results.append(tags[:6])

    return results


# ----------------------------------------------------------------------
# Summary helper (unchanged)
# ----------------------------------------------------------------------

def summary_for_cluster(title, texts):

    sample = "\n\n".join(texts[:5])

    prompt = f"""
Summarize this document cluster in 3–4 sentences.

Cluster ID: {title}

Sample content:
{sample}
"""

    return fire_fast_modal_request(prompt)


# ----------------------------------------------------------------------
# Elasticsearch helpers
# ----------------------------------------------------------------------

def es_search(index: str, query: dict, size: int = 10000):

    res = ES.search(
        index=index,
        query=query["query"],
        size=size
    )

    return res["hits"]["hits"]


def es_update_or_create(index: str, doc_id: str, body: dict):

    try:

        ES.update(
            index=index,
            id=doc_id,
            doc=body,
            doc_as_upsert=True
        )

    except Exception as e:

        print(f"⚠️ ES update failed for {doc_id}: {e}")


# ----------------------------------------------------------------------
# Core refinement
# ----------------------------------------------------------------------

def refine_clusters(refresh_summary=True, refresh_tags=True, cluster_count=None):

    job_start = time.time()

    print(f"🕛 Starting cluster refinement — {datetime.datetime.utcnow().isoformat()}")

    # ------------------------------------------------------------------
    # Fetch chunks
    # ------------------------------------------------------------------

    t_fetch = time.time()

    chunks = es_search(CHUNKS_INDEX, {"query": {"match_all": {}}})

    if not chunks:

        print("⚠️ No chunks found")
        return

    total_chunks = len(chunks)

    print(f"✅ Retrieved {total_chunks} chunks in {time.time()-t_fetch:.2f}s")

    texts = [c["_source"]["content"] for c in chunks]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    print("🧠 Generating embeddings...")

    t_embed = time.time()

    vecs = embed_texts(texts)

    print(f"✅ Embedding complete in {time.time()-t_embed:.2f}s")

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    print("🔗 Performing clustering...")

    t_cluster = time.time()

    n_clusters = cluster_count or max(5, total_chunks // 50)

    dist = cosine_distances(vecs)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )

    labels = model.fit_predict(dist)

    print(f"✅ {n_clusters} clusters formed in {time.time()-t_cluster:.2f}s")

    clusters = {}

    for lab, doc, v in zip(labels, chunks, vecs):

        clusters.setdefault(lab, {"docs": [], "vecs": []})

        clusters[lab]["docs"].append(doc)

        clusters[lab]["vecs"].append(v)

    now = datetime.datetime.utcnow().isoformat()

    # ------------------------------------------------------------------
    # Process clusters
    # ------------------------------------------------------------------

    for idx, (cid, group) in enumerate(clusters.items(), start=1):

        c_start = time.time()

        docs = group["docs"]

        vset = np.vstack(group["vecs"])

        centroid = np.mean(vset, axis=0)

        cluster_texts = [d["_source"]["content"] for d in docs]

        cluster_id = f"CL_{cid}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        print(f"\n▶️ Cluster {idx}/{len(clusters)} — {len(docs)} chunks")

        all_tags = []

        chunk_actions = []

        BATCH_SIZE = 10

        for i in range(0, len(docs), BATCH_SIZE):

            batch_docs = docs[i:i+BATCH_SIZE]

            batch_texts = [d["_source"]["content"] for d in batch_docs]

            batch_tags = tags_for_chunk(batch_texts)

            tag_strings = [" ".join(tags) for tags in batch_tags]

            tag_vecs = embed_texts(tag_strings)

            for d, chunk_tags, tag_vec in zip(batch_docs, batch_tags, tag_vecs):

                chunk_actions.append({
                    "_op_type": "update",
                    "_index": CHUNKS_INDEX,
                    "_id": d["_id"],
                    "doc": {
                        "chunk_metadata": {"tags": chunk_tags},
                        "tag_vector": tag_vec.tolist()
                    },
                    "doc_as_upsert": True
                })

                all_tags.extend(chunk_tags)

            print(f"   processed {min(i+BATCH_SIZE,len(docs))}/{len(docs)}")

        if chunk_actions:

            helpers.bulk(ES, chunk_actions, raise_on_error=False)

            print(f"   ✅ {len(chunk_actions)} chunks updated")

        unique_tags = sorted(set(all_tags))

        print(f"   🧩 {len(unique_tags)} unique tags")

        summary = ""

        if refresh_summary:

            summary = summary_for_cluster(cluster_id, cluster_texts)

        cluster_doc = {

            "cluster_id": cluster_id,

            "summary": summary,

            "tags": unique_tags,

            "vector": centroid.tolist(),

            "meta_stats": {
                "chunk_count": len(docs),
                "avg_length": int(np.mean([len(t) for t in cluster_texts]))
            },

            "last_updated": now
        }

        es_update_or_create(CLUSTERS_INDEX, cluster_id, cluster_doc)

        print(f"✅ Cluster {cluster_id} upserted — {time.time()-c_start:.2f}s")

    total_dur = time.time() - job_start

    print(f"\n🏁 Refinement finished in {total_dur/60:.2f} min")

    print(f"✅ Completed at {datetime.datetime.utcnow().isoformat()}")


# ----------------------------------------------------------------------

if __name__ == "__main__":

    print("******* Start time *******", datetime.datetime.utcnow())

    refine_clusters()

    print("******* End time *******", datetime.datetime.utcnow())