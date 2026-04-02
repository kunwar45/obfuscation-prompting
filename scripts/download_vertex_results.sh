#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${VERTEX_PROJECT:-}"
DEST_DIR="${VERTEX_DOWNLOAD_DIR:-./vertex_downloads}"
GCS_URI=""
RUN_PATH=""
DRY_RUN="false"

usage() {
  cat <<EOF
Usage:
  scripts/download_vertex_results.sh [options]

Options:
  --gcs-uri URI        Full GCS URI to copy from.
  --run-path PATH      GCS path relative to the bucket root or relative to the
                       configured experiment prefix, for example:
                       obfuscation-prompting/vertex-smoke-test/vertex-smoke-test-l4/<run_id>
  --dest-dir PATH      Local destination directory. Default: ./vertex_downloads
  --project PROJECT    Google Cloud project ID. Optional if gcloud default is set.
  --dry-run            Print the gcloud command without downloading files.
  -h, --help           Show this help message.

Environment overrides:
  VERTEX_PROJECT=my-gcp-project
  VERTEX_DOWNLOAD_DIR=./vertex_downloads

Examples:
  scripts/download_vertex_results.sh \\
    --gcs-uri gs://my-bucket/obfuscation-prompting/vertex-smoke-test

  scripts/download_vertex_results.sh \\
    --gcs-uri gs://my-bucket/obfuscation-prompting/vertex-smoke-test/vertex-smoke-test-l4/20260402_120000_deadbeef

  scripts/download_vertex_results.sh \\
    --run-path obfuscation-prompting/framing-experiment/framing-mini-smoke/20260402_120000_deadbeef \\
    --dest-dir ./downloads
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gcs-uri)
      GCS_URI="${2:-}"
      shift 2
      ;;
    --run-path)
      RUN_PATH="${2:-}"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="${2:-}"
      shift 2
      ;;
    --project)
      PROJECT_ID="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$GCS_URI" && -n "$RUN_PATH" ]]; then
  echo "Error: use either --gcs-uri or --run-path, not both." >&2
  exit 1
fi

if [[ -z "$GCS_URI" && -z "$RUN_PATH" ]]; then
  echo "Error: one of --gcs-uri or --run-path is required." >&2
  exit 1
fi

if [[ -n "$RUN_PATH" ]]; then
  if [[ "$RUN_PATH" == gs://* ]]; then
    GCS_URI="$RUN_PATH"
  else
    echo "Error: --run-path must be a full gs:// URI or use --gcs-uri directly." >&2
    exit 1
  fi
fi

if [[ "$GCS_URI" != gs://* ]]; then
  echo "Error: --gcs-uri must start with gs://." >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud is not installed or not on PATH." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

CMD=(
  gcloud storage cp
  --recursive
  "$GCS_URI"
  "$DEST_DIR"
)

if [[ -n "$PROJECT_ID" ]]; then
  CMD=(
    gcloud
    "--project=${PROJECT_ID}"
    storage
    cp
    --recursive
    "$GCS_URI"
    "$DEST_DIR"
  )
fi

echo "Downloading Vertex artifacts"
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run only. No files downloaded."
  exit 0
fi

"${CMD[@]}"
