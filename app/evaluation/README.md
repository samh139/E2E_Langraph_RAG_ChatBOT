# Offline DeepEval benchmark

This is an explicitly run, one-shot benchmark. It does not run for normal chat requests. Inputs can be plain text questions or JSONL golden cases with a reference answer:

```text
What are the NEFT charges for SBI Bank?
How can I replace a damaged debit card?
Are ATM fees different for other banks?
```

The repository includes three practice cases in `app/evaluation/golden_dataset.jsonl`. Each JSONL row contains an `id`, `input`, `expected_output`, `source_documents`, and official SBI `source_url`.

For golden cases with a non-empty `expected_output`, the runner scores Faithfulness, Answer Relevancy, Contextual Relevancy, Contextual Precision, Contextual Recall, and Answer Correctness (implemented with DeepEval `GEval`). Plain-text query files do not include references, so the three reference-based metrics are recorded as skipped for those cases.

Run the practice dataset from the directory containing `docker-compose.yml`:

```bash
docker compose --profile evaluation run --build --rm evaluation-runner \
  --queries-file app/evaluation/golden_dataset.jsonl
```

For your own questions, save one question per line in `app/evaluation/queries.txt` and run:

```bash
docker compose --profile evaluation run --build --rm evaluation-runner \
  --queries-file app/evaluation/queries.txt
```

For a single question:

```bash
docker compose --profile evaluation run --rm evaluation-runner \
  --query "What are the NEFT charges for SBI Bank?"
```

The runner starts Elasticsearch, Redis, and MongoDB as dependencies. MongoDB is only a dependency of this opt-in Compose profile; it does not gate the normal retrieval-router. Results go to database `rag_evaluation`, collection `deepeval_results`.

Each record includes a benchmark run ID, query ID, question, expected answer (if supplied), refined query, response, retrieved chunk IDs/names/text, retrieval score, judge model, and metric scores/reasons. Failures are saved with their query and error before the run stops. The retrieved text and answer are stored to make a score reviewable, so keep the local MongoDB volume and query file within your intended data-handling boundary.

Inspect recent results with:

```bash
docker compose exec mongo mongosh rag_evaluation --quiet --eval \
  'db.deepeval_results.find({}, {query:1, status:1, metrics:1, error:1, created_at:1}).sort({created_at:-1}).limit(10).forEach(printjson)'
```
