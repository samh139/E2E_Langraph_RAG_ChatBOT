#!/usr/bin/env bash
set -euo pipefail

image_uri="${1:?Usage: deploy_vm.sh IMAGE_URI GAR_LOCATION}"
gar_location="${2:?Usage: deploy_vm.sh IMAGE_URI GAR_LOCATION}"
env_file="${RAG_APP_ENV_FILE:-/etc/rag-chatbot/app.env}"

if [[ ! -r "$env_file" ]]; then
  echo "Runtime environment file is missing or unreadable: $env_file" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker must be installed on the VM." >&2
  exit 1
fi
if ! command -v gcloud >/dev/null 2>&1; then
  echo "The Google Cloud CLI must be installed on the VM for Artifact Registry auth." >&2
  exit 1
fi

exec 9>/var/lock/rag-chatbot-deploy.lock
if ! flock -n 9; then
  echo "Another chatbot deployment is already running." >&2
  exit 1
fi

gcloud auth configure-docker "${gar_location}-docker.pkg.dev" --quiet
docker pull "$image_uri"

docker rm -f rag-websocket rag-retrieval-router >/dev/null 2>&1 || true

docker run -d \
  --name rag-websocket \
  --restart unless-stopped \
  --network host \
  --env-file "$env_file" \
  "$image_uri" \
  python -u -m app.websocket_service.main

docker run -d \
  --name rag-retrieval-router \
  --restart unless-stopped \
  --network host \
  --env-file "$env_file" \
  "$image_uri" \
  python -u -c "import asyncio; from app.router_service.request_response_router import RequestResponseRouter; asyncio.run(RequestResponseRouter().start())"

sleep 5
for container in rag-websocket rag-retrieval-router; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
    docker logs --tail 100 "$container" >&2 || true
    exit 1
  fi
done

echo "Deployed $image_uri"
