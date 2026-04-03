#!/usr/bin/env bash

set -euo pipefail

REGION="${VERTEX_REGION:-us-central1}"
PROJECT_ID="${VERTEX_PROJECT:-}"
DISPLAY_NAME=""
CONFIG_PATH=""
DRY_RUN="false"

usage() {
  cat <<EOF
Usage:
  scripts/submit_vertex_job.sh --display-name NAME --config PATH [options]

Options:
  --display-name NAME   Vertex AI custom job display name.
  --config PATH         Path to the Vertex AI custom job YAML config.
  --region REGION       Vertex AI region. Default: us-central1
  --project PROJECT_ID  Google Cloud project ID. Optional if gcloud default is set.
  --dry-run             Print the gcloud command without submitting the job.
  -h, --help            Show this help message.

Environment overrides:
  VERTEX_REGION=us-central1
  VERTEX_PROJECT=my-gcp-project

Examples:
  scripts/submit_vertex_job.sh \\
    --display-name vertex-smoke-test-l4 \\
    --config vertex_jobs/smoke_l4.yaml

  scripts/submit_vertex_job.sh \\
    --project my-project \\
    --region us-central1 \\
    --display-name framing-mini-smoke \\
    --config vertex_jobs/experiment_l4.yaml \\
    --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --display-name)
      DISPLAY_NAME="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --region)
      REGION="${2:-}"
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

if [[ -z "$DISPLAY_NAME" ]]; then
  echo "Error: --display-name is required." >&2
  exit 1
fi

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Error: --config is required." >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file '$CONFIG_PATH' does not exist." >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud is not installed or not on PATH." >&2
  exit 1
fi

CMD=(
  gcloud ai custom-jobs create
  "--region=${REGION}"
  "--display-name=${DISPLAY_NAME}"
  "--config=${CONFIG_PATH}"
)

if [[ -n "$PROJECT_ID" ]]; then
  CMD+=("--project=${PROJECT_ID}")
fi

echo "Submitting Vertex AI custom job"
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run only. No job submitted."
  exit 0
fi

"${CMD[@]}"
