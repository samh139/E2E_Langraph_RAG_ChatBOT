#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${GOOGLE_CLOUD_LOCATION:=us-central1}"
: "${NOMIC_ARTIFACT_URI:?Set NOMIC_ARTIFACT_URI to the gs:// model artifact directory}"
: "${NOMIC_SERVING_IMAGE:?Set NOMIC_SERVING_IMAGE to the Artifact Registry image URI}"
: "${NOMIC_MODEL_REVISION:?Set the pinned Hugging Face commit SHA}"

model_args=()
if [[ -n "${NOMIC_PARENT_MODEL:-}" ]]; then
  model_args+=("--parent-model=${NOMIC_PARENT_MODEL}")
else
  model_args+=("--model-id=${NOMIC_MODEL_ID:-nomic-embed-text}")
fi

gcloud ai models upload \
  --project="${GOOGLE_CLOUD_PROJECT}" \
  --region="${GOOGLE_CLOUD_LOCATION}" \
  --display-name="${NOMIC_DISPLAY_NAME:-nomic-embed-text}" \
  --artifact-uri="${NOMIC_ARTIFACT_URI}" \
  --container-image-uri="${NOMIC_SERVING_IMAGE}" \
  --container-ports=8080 \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-env-vars="EMBEDDING_DIMENSIONS=768" \
  --labels=provider=nomic,task=embeddings \
  --version-aliases=staging \
  --version-description="Nomic ${NOMIC_MODEL_REVISION}; app revision ${APP_GIT_SHA:-unknown}" \
  "${model_args[@]}"
