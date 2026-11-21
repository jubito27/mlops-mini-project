#!/bin/bash
set -Eeuo pipefail

AWS_REGION="ap-southeast-2"
ECR_REGISTRY="970734043001.dkr.ecr.ap-southeast-2.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/jubito27_ecr:latest"
CONTAINER_NAME="my-app"
ENV_FILE="/etc/mlops-mini-project.env"

command -v aws >/dev/null 2>&1 || { echo "AWS CLI is required on the EC2 instance." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is not installed." >&2; exit 1; }

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker pull "$IMAGE_URI"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

RUN_ARGS=(--detach --publish 80:5000 --name "$CONTAINER_NAME" --restart unless-stopped)
if [[ -f "$ENV_FILE" ]]; then
  RUN_ARGS+=(--env-file "$ENV_FILE")
fi

docker run "${RUN_ARGS[@]}" "$IMAGE_URI"

for attempt in {1..30}; do
  if docker ps --filter "name=^/${CONTAINER_NAME}$" --filter status=running --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container $CONTAINER_NAME is running."
    exit 0
  fi
  sleep 2
done

docker logs "$CONTAINER_NAME" || true
echo "Container did not remain running." >&2
exit 1
