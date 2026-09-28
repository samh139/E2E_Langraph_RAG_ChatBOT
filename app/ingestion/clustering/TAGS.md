# Tag enrichment

The normal clustering job enriches chunks before building the hierarchy. It writes
`chunk_metadata.tags` and `tags_vector` to the configured chunk index, and `tags`
and `tags_vector` to both root clusters and subclusters. Cluster tags are the 30
most frequent normalized member tags, with alphabetical tie-breaking.

Tag generation uses `TAG_MODEL` when set, otherwise `SUMMARY_MODEL` (default
`gemma3:1b`), at `OLLAMA_URL`. Tag vectors use the existing
`BAAI/bge-small-en-v1.5` embedding model and have 384 dimensions. Retrieval should
use this same embedding model when querying `tags_vector`.

Backfill the existing hierarchy without re-ingesting or reclustering:

```sh
docker compose run --rm --no-deps app-worker python -u -m app.ingestion.clustering.tag_enrichment
```

This command adds mappings to existing indexes and partially updates documents.
Completed chunks are skipped on a retry; existing cluster summaries and member
IDs are preserved. It does not create a new cluster hierarchy when the cluster
alias is absent. Do not run it concurrently with a clustering rebuild.

Failures propagate after three generation attempts. Updates already saved remain
in Elasticsearch and can be reused by the next run. Chunk tag updates are not
rolled back if a later cluster build fails. The older standalone refiner scripts
are not needed for this path.
