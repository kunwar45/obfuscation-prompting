# Vertex Experiment Runbook

This file is the quickest reference for future agents running experiments on
Vertex AI in this repo.

## Current Defaults

- Project: `project-25ea6636-1c58-40fa-88b`
- Region: `us-central1`
- Bucket: `gs://genai-1010101`
- Base artifact prefix: `gs://genai-1010101/obfuscation-prompting`

## What Happens During a Vertex Run

Vertex jobs in this repo use `scripts/vertex_job_runner.py`.

The wrapper:

- runs the experiment command unchanged
- leaves experiment outputs in local repo folders like `results/`, `data/`, and `activations/`
- uploads those folders to GCS at the end of the run
- also uploads `.vertex_job/metadata.json`

Typical GCS layout:

```text
gs://genai-1010101/obfuscation-prompting/<experiment>/<display-name>/<timestamp>_<id>/
```

## Key Files

- `scripts/submit_vertex_job.sh`: submits a Vertex custom job
- `scripts/create_vertex_run_config.py`: creates a fresh YAML per run
- `scripts/vertex_job_runner.py`: runs the command and uploads artifacts
- `scripts/download_vertex_results.sh`: downloads artifacts back locally
- `vertex_jobs/smoke_l4.yaml`: smoke-test template
- `vertex_jobs/experiment_l4.yaml`: experiment template
- `vertex_jobs/runs/`: generated per-run YAML files

## Before Running an Experiment

1. Run a local lightweight preflight of the exact experiment entrypoint before any real Vertex submission.
2. Make sure the Docker image you want to run has been built and pushed.
3. Generate a fresh per-run YAML instead of reusing an old one.
4. Make sure the generated YAML points at the correct image tag.
5. Make sure the YAML uses the real bucket prefix:

```text
gs://genai-1010101/obfuscation-prompting
```

6. Make sure `VERTEX_EXPERIMENT_NAME` and `VERTEX_DISPLAY_NAME` are set to something meaningful.

Recommended local preflight for the concealment interpretability workflow:

```bash
python3 scripts/run_vertex_last_token_concealment_experiment.py \
  --model gpt2 \
  --dtype float32 \
  --n-scenarios 2 \
  --max-tokens 64 \
  --top-k-dims 4 \
  --smoke-mode
```

That local preflight is intended to validate the command path and output layout,
not to reproduce the real concealment finding. Tiny models and tiny runs may not
produce enough successful concealment examples for the evaluator, which is why
`--smoke-mode` allows evaluator failure without blocking the preflight.

Example:

```bash
python scripts/create_vertex_run_config.py \
  --template experiment_l4.yaml \
  --run-name framing-30-scenarios \
  --display-name framing-30-scenarios \
  --experiment-name framing-experiment \
  --image-uri docker.io/iman1121/iman-kunwar-genai:latest
```

This creates a file like:

```text
vertex_jobs/runs/20260402_123456_framing-30-scenarios.yaml
```

## Why Per-Run YAMLs

Each run should have its own YAML so that:

- the exact submitted config is preserved
- agents do not overwrite each other’s settings
- image tags, commands, and naming stay tied to a specific run
- later analysis can trace artifacts back to the exact submission config

Treat `vertex_jobs/*.yaml` as templates. Treat `vertex_jobs/runs/*.yaml` as run records.

## Standard Workflow

The typical workflow for this repo is:

1. The user tells the agent what experiment to run.
2. The agent runs a lightweight local preflight of the same experiment entrypoint.
3. The agent creates a per-run Vertex YAML, sets the command/image/config, and submits the job.
4. The agent confirms that the Vertex job submission succeeded.
5. The agent's job ends in this context.
6. The user waits for the job to finish.
7. After completion, the user downloads the artifacts from GCS and continues analysis locally.

In other words: define the experiment, preflight it locally, submit it, confirm submission, then stop. Waiting and downloading happen after that.

## Submit an Experiment

For a real experiment, generate a run-specific YAML, make any final edits there, then submit that exact file:

```bash
scripts/submit_vertex_job.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --region us-central1 \
  --display-name framing-mini-smoke \
  --config vertex_jobs/runs/<generated-run-config>.yaml
```

## Wait for Completion

The normal expectation is that the experiment may take a while. After submission, wait for the job to finish, then inspect GCS.

List runs under an experiment prefix:

```bash
gcloud storage ls gs://genai-1010101/obfuscation-prompting/framing-experiment/
```

## Download Results

Download one completed run:

```bash
scripts/download_vertex_results.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --gcs-uri gs://genai-1010101/obfuscation-prompting/framing-experiment/framing-mini-smoke/<run_id> \
  --dest-dir ./vertex_downloads
```

Example download for a full experiment prefix:

```bash
scripts/download_vertex_results.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --gcs-uri gs://genai-1010101/obfuscation-prompting/framing-experiment \
  --dest-dir ./vertex_downloads
```

## When to Use a Smoke Test

Smoke tests are optional, not the default for every run.

Use `vertex_jobs/smoke_l4.yaml` when:

- a new image was just built
- the Vertex wrapper/upload path changed
- the entry command changed in a risky way
- you want a very fast infrastructure sanity check

Do not treat the smoke test as a required step before every normal experiment submission.

## Expected Artifacts

Look for these after a run:

- `results/`: main run JSONs and plots
- `data/`: generated dataset JSONL files
- `activations/`: activation captures, when enabled
- `metadata.json`: wrapper metadata, command, timestamps, and upload summary

## Common Checks

If uploads fail:

- verify the bucket path in the YAML
- verify the image includes `scripts/vertex_job_runner.py`
- verify the Vertex service account can write to the bucket

If downloads fail:

- verify the run exists with `gcloud storage ls`
- verify you are copying the full run prefix
- use `--dry-run` on `scripts/download_vertex_results.sh` to inspect the command

## Recommended Agent Behavior

- Always run a lightweight local preflight of the exact experiment command before a real Vertex submission.
- Always verify the exact Docker image tag before submission.
- Prefer using the helper scripts instead of hand-writing `gcloud` commands.
- Treat GCS as the source of truth for Vertex-run artifacts.
- Default to the user-requested experiment workflow: define experiment, submit it, confirm submission succeeded, then stop.
- Use smoke tests only when infrastructure or packaging changed enough to justify them.
- Generate a fresh YAML config for every run instead of reusing one shared file.
