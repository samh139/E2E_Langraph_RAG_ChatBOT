# On-demand DeepEval

Run the RAG workflow and score one answer when you explicitly want to inspect it. This command is separate from the WebSocket/Kafka request handler, so DeepEval does not run for normal chat responses. It does not create or depend on a golden dataset.

From the project directory, with its services running:

```bash
docker compose exec retrieval-router python -m app.evaluation.run \
  --query "What are the NEFT charges for SBI Bank?"
```

The command directly invokes `RetrievalWorkflow`, then evaluates:

- **Faithfulness**: whether the answer is supported by retrieved chunk text.
- **Answer relevancy**: whether the answer addresses the refined query.
- **Contextual relevancy**: whether retrieved chunks are relevant to the refined query.

It uses Gemini on Vertex AI as the judge, following `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS`. Set `DEEPEVAL_MODEL` or pass `--model` to choose a judge model. The default is the app's `GEMINI_MODEL` (currently `gemini-2.5-flash`).

Each run incurs judge-model calls and prints scores/reasons to the terminal. These metrics run in score-only mode; they do not define pass/fail thresholds. The workflow may read memory for query refinement, but this command does not use the request router and therefore does not store the query/answer in STM or LTM.

If the workflow does not return readable retrieved chunks, the evaluator stops with an explanatory error because faithfulness and contextual relevance need retrieval context. Inspect the retrieval output and score threshold before evaluating that query.
