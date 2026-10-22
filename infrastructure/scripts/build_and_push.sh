#!/usr/bin/env bash
# Build the ShelfSentry AI container image (arm64) and push it to Amazon ECR.
#
# Run this ON YOUR MACHINE (Docker required). The Kiro sandbox cannot build or
# push container images, so this is a manual, reproducible step. After it
# completes it prints the EcrImageUri to pass to the CloudFormation deploy.
#
# The image is arm64 (Graviton). On Apple Silicon this is native; on x86,
# Docker Buildx + QEMU builds arm64 (slower).
#
# Usage:
#   ./build_and_push.sh [region] [repo] [tag]
# Defaults: region=us-east-1 repo=shelfsentry tag=<UTC timestamp>

set -euo pipefail

REGION="${1:-us-east-1}"
REPO="${2:-shelfsentry}"
TAG="${3:-$(date -u +%Y%m%d-%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND="$ROOT/backend"
DOCKERFILE="$ROOT/infrastructure/docker/Dockerfile"

echo "==> Account / region"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO}:${TAG}"
echo "    account=${ACCOUNT_ID} region=${REGION}"
echo "    image=${IMAGE_URI}"

echo "==> Ensure ECR repository exists (idempotent)"
if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository \
    --repository-name "$REPO" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability IMMUTABLE \
    --region "$REGION" >/dev/null
  echo "    created repository $REPO"
else
  echo "    repository $REPO already exists"
fi

echo "==> Docker login to ECR"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

echo "==> Build arm64 image (context: $BACKEND)"
# --provenance=false keeps the pushed artifact a single-arch manifest Lambda
# accepts. --platform linux/arm64 matches the function Architecture.
docker buildx build \
  --platform linux/arm64 \
  --provenance=false \
  -f "$DOCKERFILE" \
  -t "$IMAGE_URI" \
  --push \
  "$BACKEND"

echo "==> Resolve the immutable digest (recommended for CloudFormation)"
DIGEST="$(aws ecr describe-images \
  --repository-name "$REPO" \
  --image-ids imageTag="$TAG" \
  --region "$REGION" \
  --query 'imageDetails[0].imageDigest' --output text)"
IMAGE_BY_DIGEST="${REGISTRY}/${REPO}@${DIGEST}"

cat <<EOF

================ IMAGE PUBLISHED ================
By tag:    ${IMAGE_URI}
By digest: ${IMAGE_BY_DIGEST}   <-- prefer this for EcrImageUri
================================================

Next: deploy with
  aws cloudformation deploy \\
    --template-file infrastructure/template.yaml \\
    --stack-name shelfsentry-dev \\
    --capabilities CAPABILITY_NAMED_IAM \\
    --parameter-overrides EcrImageUri=${IMAGE_BY_DIGEST} \\
    --region ${REGION}
EOF
