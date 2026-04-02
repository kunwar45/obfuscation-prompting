# Vertex AI Job Presets

This directory holds lightweight YAML configs for submitting Vertex AI custom
jobs with:

```bash
scripts/submit_vertex_job.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --region us-central1 \
  --display-name vertex-smoke-test-l4 \
  --config vertex_jobs/smoke_l4.yaml
```

## Files

- `smoke_l4.yaml`: minimal container smoke test on a single L4 GPU.
- `experiment_l4.yaml`: starter config for a real research run.
- `runs/`: generated per-run YAML configs. Do not reuse one file across multiple runs.

## Results Flow

Vertex runs use `scripts/vertex_job_runner.py` inside the container. The wrapper:

- runs your experiment command unchanged
- writes a metadata manifest to `.vertex_job/metadata.json`
- uploads `results/`, `data/`, and `activations/` to GCS
- uploads on both success and failure when possible

Recommended layout:

```text
gs://<bucket>/obfuscation-prompting/<experiment>/<display-name>/<timestamp>_<id>/
```

Example artifact paths:

```text
gs://<bucket>/obfuscation-prompting/framing-experiment/framing-mini-smoke/<run>/results/
gs://<bucket>/obfuscation-prompting/framing-experiment/framing-mini-smoke/<run>/data/
gs://<bucket>/obfuscation-prompting/framing-experiment/framing-mini-smoke/<run>/activations/
gs://<bucket>/obfuscation-prompting/framing-experiment/framing-mini-smoke/<run>/metadata.json
```

## Quick Start

1. Generate a per-run YAML from a shared template.
2. Build and push the Docker image you want to run.
3. Submit the generated run config.
4. After the job finishes, list and download the artifacts you want.

Generate a run-specific config:

```bash
python scripts/create_vertex_run_config.py \
  --template experiment_l4.yaml \
  --run-name framing-30-scenarios \
  --display-name framing-30-scenarios \
  --experiment-name framing-experiment \
  --image-uri docker.io/iman1121/iman-kunwar-genai:latest
```

This prints a path like:

```text
vertex_jobs/runs/20260402_123456_framing-30-scenarios.yaml
```

Smoke test:

```bash
scripts/submit_vertex_job.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --region us-central1 \
  --display-name vertex-smoke-test-l4 \
  --config vertex_jobs/runs/<generated-run-config>.yaml
```

List uploaded runs:

```bash
gcloud storage ls gs://YOUR_BUCKET/obfuscation-prompting/vertex-smoke-test/
```

Download one run:

```bash
scripts/download_vertex_results.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --gcs-uri gs://YOUR_BUCKET/obfuscation-prompting/vertex-smoke-test/vertex-smoke-test-l4/<run_id> \
  --dest-dir ./vertex_downloads
```

Download an entire experiment prefix:

```bash
scripts/download_vertex_results.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --gcs-uri gs://YOUR_BUCKET/obfuscation-prompting/framing-experiment \
  --dest-dir ./vertex_downloads
```

## Suggested Workflow

1. Generate a new per-run YAML under `vertex_jobs/runs/`.
2. Build and push the image for that run.
3. Submit that exact run config.
4. Confirm the Vertex job was accepted successfully.
5. Stop there unless the user explicitly asks for post-run work.
6. After the run finishes, download artifacts with `scripts/download_vertex_results.sh`.

## Notes

- `gcloud ai custom-jobs create --config` expects YAML fields like `workerPoolSpecs` and `scheduling` at the top level.
- `displayName`, `region`, and optionally `project` are passed by the submit script rather than stored in the YAML.
- The YAML presets use env vars like `VERTEX_RESULTS_GCS_URI`, `VERTEX_EXPERIMENT_NAME`, and `VERTEX_IMAGE_URI` to drive artifact uploads.
- Treat `smoke_l4.yaml` and `experiment_l4.yaml` as templates, not as files to edit in place for actual runs.
