# clustering/config.py

import os

# ----------------------------------------------------------------------
# Elasticsearch Infrastructure Settings
# ----------------------------------------------------------------------
# Container-network endpoint routing hook
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")

# Elasticsearch 9.x System Security Authentication Credentials
ES_PASSWORD = os.getenv("ES_PASSWORD", "strongpassword123")

# ----------------------------------------------------------------------
# Synchronized Index Targets & Specifications
# ----------------------------------------------------------------------
# FIXED: Synchronized cleanly with your current active database index string
CHUNKS_INDEX = os.getenv("ES_INDEX_NAME", "es_documents")

# FIXED: Standardized to 384 dimensions to perfectly match BAAI/bge-small-en-v1.5
EMBEDDING_DIM = 384   

# Group mapping target schemas
CLUSTERS_ALIAS = "clusters_v3"
TEMP_INDEX_PREFIX = "clusters_v3_temp_"

# ----------------------------------------------------------------------
# LEVEL-1 CLUSTERING PARAMETERS — AGGLOMERATIVE
# ----------------------------------------------------------------------
# Optimal semantic cosine metric separation threshold for 384-d vectors
AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.35
AGGLOMERATIVE_MIN_CLUSTER_SIZE = 5

# ----------------------------------------------------------------------
# LEVEL-2 CLUSTERING PARAMETERS — ADAPTIVE K-MEANS
# ----------------------------------------------------------------------
# Control metrics to handle split distribution ceilings safely
SUBCLUSTER_MIN_SIZE = 10
SUBCLUSTER_MAX_SIZE = 50 

# Adaptive cluster partition targets
SUBCLUSTER_MIN_K = 3
SUBCLUSTER_MAX_K = 8

# ----------------------------------------------------------------------
# Batch Operational Throughput Settings
# ----------------------------------------------------------------------
BATCH_SIZE = 3000
MAX_CHUNKS = 2_000_000     
VERBOSE = True
DRY_RUN = False
