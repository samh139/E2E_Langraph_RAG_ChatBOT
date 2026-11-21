# Production model tracking and Vertex AI Model Registry

This repository uses Gemini as a managed Vertex AI publisher model and Nomic as
the self-managed embedding model. Gemini is tracked by its model ID in each app
release manifest. Nomic is packaged and versioned in Vertex AI Model Registry,
then served from a Vertex endpoint.

## Release tracking

Set `APP_RELEASE` and `APP_GIT_SHA` in the deployment environment. Generate and
archive the release manifest with the app image or deployment metadata:

```sh
python scripts/write_model_manifest.py
```

The manifest records the Gemini model ID, Nomic provider/model/version/endpoint,
embedding dimensions, release, and app revision. The app also logs the active
generation and embedding identifiers at startup. Gemini remains configured by
`GEMINI_MODEL` (currently `gemini-2.5-flash`); it is a managed publisher model,
so its weights are not imported into your project's Model Registry.
The GitHub deployment workflow archives this manifest. It reads optional model
settings from repository or production-environment variables named
`GCP_LOCATION`, `GEMINI_MODEL`, `EMBEDDING_PROVIDER`,
`VERTEX_EMBEDDING_MODEL`, `VERTEX_EMBEDDING_VERSION`, and
`VERTEX_EMBEDDING_ENDPOINT_ID`.

## Package and register Nomic

The serving container expects a pinned Hugging Face revision of
`nomic-ai/nomic-embed-text-v1.5`. It applies Nomic's required `search_document:`
and `search_query:` task prefixes and returns normalized 768-dimensional
vectors. Review the model license and pin an immutable Hugging Face commit SHA.

Prerequisites: enable Vertex AI, Cloud Storage, and Artifact Registry APIs;
create a regional bucket and Docker repository in the same region as the Vertex
endpoint; grant the deploying identity permissions to upload models and use the
bucket/repository; and authenticate with `gcloud` and Application Default
Credentials. The application runtime identity needs permission to call the
Vertex endpoint (`aiplatform.endpoints.predict`). Do not put service account key
files in the image or deployment environment.

```sh
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export NOMIC_MODEL_REVISION=the-immutable-hugging-face-commit-sha
export NOMIC_ARTIFACT_URI=gs://your-us-central1-bucket/models/nomic-embed-text-v1-5/revision-id
export NOMIC_SERVING_IMAGE=us-central1-docker.pkg.dev/your-project-id/models/nomic-serving:revision-id

python -m pip install huggingface-hub
python scripts/prepare_nomic_artifacts.py
gcloud storage rsync --recursive build/nomic-model "$NOMIC_ARTIFACT_URI"
gcloud builds submit model-serving/nomic --tag "$NOMIC_SERVING_IMAGE"
bash scripts/register_nomic_model.sh
```

The first registration creates model ID `nomic-embed-text` with alias `staging`.
For later versions, set `NOMIC_PARENT_MODEL` to the full existing Registry model
resource name, such as
`projects/PROJECT/locations/us-central1/models/MODEL_ID`, before running the
registration script. The script creates another version with alias `staging`;
it does not promote a version to production automatically.
Use a unique serving image tag for every runtime build and enable immutable tags
on the Artifact Registry repository so each Registry version keeps the same
inference code.

Create a Vertex endpoint once, then deploy the evaluated Registry version to
it. Use the actual numeric Registry version in the deployment command and record
that version in the app environment:

```sh
gcloud ai endpoints create \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --display-name=nomic-embed-text

# Find the numeric version created by the upload script.
gcloud ai models list-version nomic-embed-text \
  --project="$GOOGLE_CLOUD_PROJECT" --region="$GOOGLE_CLOUD_LOCATION"

# Create a separate staging endpoint, or deploy to an endpoint with no production
# traffic. Replace ENDPOINT_ID and VERSION_ID with the resources you created.
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --region="$GOOGLE_CLOUD_LOCATION" \
  --model="nomic-embed-text@VERSION_ID" \
  --display-name="nomic-embed-text-vVERSION_ID" \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=2 \
  --traffic-split=0=100
```

Evaluate the `staging` version against the retrieval evaluation set and latency
target on a staging endpoint before promotion. In Model Registry, move the `production` alias to the
accepted version, then deploy that exact version to the production endpoint (or
update traffic on the existing endpoint). Registry aliases are mutable release
labels; a running endpoint continues serving the version actually deployed.
Keep `VERTEX_EMBEDDING_VERSION` aligned with that deployed version.

Configure the app runtime (all app services that embed content or queries must
receive the same values):

```dotenv
EMBEDDING_PROVIDER=vertex
VERTEX_EMBEDDING_MODEL=nomic-embed-text
VERTEX_EMBEDDING_VERSION=VERSION_ID
VERTEX_EMBEDDING_ENDPOINT_ID=ENDPOINT_ID
EMBEDDING_DIMENSIONS=768
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
APP_RELEASE=release-name
APP_GIT_SHA=git-commit-sha
```

The code uses the shared embedding adapter for ingestion, query retrieval, and
long-term memory. The current Ollama provider remains available for local use.

## Index migration requirement

Do not mix vectors produced by the old Ollama model and the new packaged
Sentence Transformers model in one live index, even though both are 768
dimensions. Their vector spaces and Nomic prompt behavior may differ. Build a
new chunk index by re-embedding the source chunks with the selected Vertex model,
validate retrieval, then switch the Elasticsearch alias using the existing
staged-index workflow. Rebuild long-term memory vectors as well, or clear/rebuild
that memory index before using semantic memory search. Keep the previous index
available for rollback until validation is complete.
