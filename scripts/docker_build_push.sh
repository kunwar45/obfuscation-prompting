#!/usr/bin/env bash

set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-iman1121/iman-kunwar-genai}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-Dockerfile}"
CONTEXT_DIR="${CONTEXT_DIR:-.}"
STAGE_CONTEXT="true"
REMOVE_LOCAL_AFTER_PUSH="true"

usage() {
  cat <<EOF
Usage:
  scripts/docker_build_push.sh <tag>
  scripts/docker_build_push.sh <tag> --push
  scripts/docker_build_push.sh <tag> --platform linux/amd64 --push
  scripts/docker_build_push.sh <tag> --no-stage
  scripts/docker_build_push.sh <tag> --push --keep-local

Environment overrides:
  IMAGE_REPO=iman1121/iman-kunwar-genai
  DOCKERFILE_PATH=Dockerfile
  CONTEXT_DIR=.
  PLATFORM=linux/amd64
  STAGE_CONTEXT=true
  REMOVE_LOCAL_AFTER_PUSH=true

Examples:
  scripts/docker_build_push.sh v1
  scripts/docker_build_push.sh 2026-04-01 --push
  scripts/docker_build_push.sh latest --no-stage
  scripts/docker_build_push.sh latest --push --keep-local
  IMAGE_REPO=iman1121/iman-kunwar-genai scripts/docker_build_push.sh latest --platform linux/amd64 --push
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

TAG=""
PUSH_IMAGE="false"
PLATFORM="${PLATFORM:-linux/amd64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-local)
      REMOVE_LOCAL_AFTER_PUSH="false"
      shift
      ;;
    --no-stage)
      STAGE_CONTEXT="false"
      shift
      ;;
    --push)
      PUSH_IMAGE="true"
      shift
      ;;
    --platform)
      if [[ $# -lt 2 ]]; then
        echo "Error: --platform requires a value." >&2
        exit 1
      fi
      PLATFORM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$TAG" ]]; then
        echo "Error: multiple tags provided ('$TAG' and '$1')." >&2
        exit 1
      fi
      TAG="$1"
      shift
      ;;
  esac
done

if [[ -z "$TAG" ]]; then
  echo "Error: tag is required." >&2
  exit 1
fi

FULL_IMAGE_NAME="${IMAGE_REPO}:${TAG}"

BUILD_CONTEXT="$CONTEXT_DIR"
BUILD_DOCKERFILE="$DOCKERFILE_PATH"
TEMP_CONTEXT_DIR=""

cleanup() {
  if [[ -n "$TEMP_CONTEXT_DIR" && -d "$TEMP_CONTEXT_DIR" ]]; then
    rm -rf "$TEMP_CONTEXT_DIR"
  fi
}
trap cleanup EXIT

if [[ "$STAGE_CONTEXT" == "true" ]]; then
  TEMP_CONTEXT_DIR="$(mktemp -d)"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git/' \
      --exclude '.env' \
      --exclude 'venv/' \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      --exclude '.pytest_cache/' \
      --exclude '.mypy_cache/' \
      --exclude '._*' \
      --exclude '.DS_Store' \
      --exclude 'data/' \
      --exclude 'results/' \
      --exclude 'activations/' \
      "${CONTEXT_DIR}/" "${TEMP_CONTEXT_DIR}/"
  else
    echo "Error: rsync is required when staging a clean Docker context." >&2
    echo "Either install rsync or rerun with --no-stage." >&2
    exit 1
  fi
  BUILD_CONTEXT="$TEMP_CONTEXT_DIR"
  BUILD_DOCKERFILE="$TEMP_CONTEXT_DIR/$DOCKERFILE_PATH"
fi

BUILD_CMD=(docker build -f "$BUILD_DOCKERFILE" -t "$FULL_IMAGE_NAME")
if [[ -n "$PLATFORM" ]]; then
  BUILD_CMD+=(--platform "$PLATFORM")
fi
BUILD_CMD+=("$BUILD_CONTEXT")

echo "Building ${FULL_IMAGE_NAME}"
"${BUILD_CMD[@]}"

echo "Built ${FULL_IMAGE_NAME}"

if [[ "$PUSH_IMAGE" == "true" ]]; then
  echo "Pushing ${FULL_IMAGE_NAME}"
  docker push "$FULL_IMAGE_NAME"
  echo "Pushed ${FULL_IMAGE_NAME}"
  if [[ "$REMOVE_LOCAL_AFTER_PUSH" == "true" ]]; then
    echo "Removing local image ${FULL_IMAGE_NAME}"
    docker image rm "$FULL_IMAGE_NAME"
    echo "Removed local image ${FULL_IMAGE_NAME}"
  fi
else
  echo "Push skipped. Run this to publish:"
  echo "  docker push ${FULL_IMAGE_NAME}"
fi
