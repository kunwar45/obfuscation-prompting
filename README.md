# Obfuscation Prompting

Research pipeline for studying when and why LLM agents conceal information present in their system context. Three experiments test this at increasing levels of depth: black-box behavioural measurement, implicit framing effects, and mechanistic interpretability via activation analysis.

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add TOGETHER_API_KEY for cloud runs; HF_TOKEN only needed for GPQA
```

Requires **Python 3.11** for torch-dependent code (local model runs). Cloud (Together AI) runs work with any Python 3.x.

---

## Experiments

### 1. Black-box experiment (`scripts/run_blackbox.py`)

Establishes a causal baseline across three conditions:
- **A0** — transparent ("be accurate and honest")
- **A1** — implicit pressure (performance evaluated on retention, avoid alarming statements)
- **A2** — explicit suppression ("do not mention the delivery timeline under any circumstances")

```bash
# Smoke test (3 scenarios)
python -m scripts.run_blackbox --smoke-only

# Full run (50 scenarios, all three conditions)
python -m scripts.run_blackbox --skip-smoke --n-scenarios 50

# Custom model
python -m scripts.run_blackbox --skip-smoke --n-scenarios 50 \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo
```

Requires `TOGETHER_API_KEY`. Saves results to `results/` and generates 5 plots.

**Key finding:** A2 achieves ~95% concealment; A0 achieves ~98% disclosure; A1 sits in between at ~74–83%.

---

### 2. Framing experiment (`scripts/run_framing.py`)

Maps the space between transparent and suppressed across **18 framing conditions** in 5 dimensions (motivation, incentive, audience, baseline, and control). No explicit suppression instruction in most conditions — tests whether implicit framing alone induces concealment.

```bash
# Local model, smoke test
python3.11 -m scripts.run_framing --local --smoke-only --smoke-scenarios 2 \
  --conditions BASE,M_inst_s,I_strong_s,SEMI,EXPLICIT

# Local model, full run (all 18 conditions)
python3.11 -m scripts.run_framing --local --skip-smoke --n-scenarios 30 \
  --local-model Qwen/Qwen2.5-1.5B-Instruct --max-tokens 256

# Cloud (Together AI), full run with LLM monitor
python -m scripts.run_framing --skip-smoke --n-scenarios 30

# With activation capture (for interpretability analysis downstream)
python3.11 -m scripts.run_framing --local --skip-smoke --n-scenarios 30 \
  --conditions BASE,I_social,M_inst_s,I_strong_s,SEMI,EXPLICIT \
  --capture-activations last_token
```

Saves results to `results/` and generates 5 figures including an empirical explicitness spectrum and Cohen's h effect sizes vs baseline.

**Key finding:** Implicit framings rarely achieve <70% disclosure. Only EXPLICIT (15%) and SEMI (73%) reliably suppress. Audience framings (A_naive, A_none) can reach 100% disclosure — higher than the transparent baseline.

---

### 3. Interpretability experiment (`scripts/run_interp.py`)

Runs the pipeline with activation capture enabled, then probes the hidden states layer-by-layer to find where the model encodes (a) which condition it received, and (b) the disclosure decision. Also runs causal patching to test whether identified dimensions causally mediate concealment.

```bash
# Recommended: Qwen 1.5B on local GPU
python3.11 -m scripts.run_interp \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --n-scenarios 30

# Smaller smoke run
python3.11 -m scripts.run_interp \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --n-scenarios 10 \
  --smoke-only
```

Outputs `results/run_*_analysis.json` (probe + PCA results) and `results/run_*_last_token_eval.json` (causal evaluation). Saves activations to `activations/`.

**Key findings (30 scenarios, Qwen 1.5B):**
- A0/A2 condition is linearly separable from **layer 3** (100% CV accuracy)
- Disclosure decision peaks at **layer 16** (81% CV accuracy)
- 8-dimensional concealment subspace at layer 16 with Cohen's d ≈ 2.1–2.3
- Implicit and explicit concealment produce **cosine-similar activations** (0.994–0.998) at layer 16 — same internal circuit regardless of framing

---

## Monitors

Three monitors are applied to every model response:

| Monitor | Description |
|---------|-------------|
| **Regex** | Pattern-matches secret value, wrong-secret variants, and refusal phrases |
| **LLM** | Same-model judge classifies semantic disclosure (disclosed / concealed / redirected) |
| **Keyword** | Surface-form lookup against gold keywords from context |

The LLM monitor requires `TOGETHER_API_KEY` and is omitted automatically in `--local` mode.

---

## Running on Vertex AI

See [`docs/VERTEX_EXPERIMENT_RUNBOOK.md`](docs/VERTEX_EXPERIMENT_RUNBOOK.md) for the full workflow. Quick reference:

**1. Local preflight (always run first)**

```bash
python3 scripts/run_vertex_last_token_concealment_experiment.py \
  --model gpt2 --dtype float32 --n-scenarios 2 --max-tokens 64 \
  --top-k-dims 4 --smoke-mode
```

**2. Generate a per-run YAML**

```bash
python scripts/create_vertex_run_config.py \
  --template experiment_l4.yaml \
  --run-name my-run \
  --owner <name> \
  --experiment-name interp-experiment \
  --image-uri us-central1-docker.pkg.dev/project-25ea6636-1c58-40fa-88b/genai/obfuscation-prompting:latest
```

**3. Submit**

```bash
scripts/submit_vertex_job.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --region us-central1 \
  --display-name <name> \
  --config vertex_jobs/runs/<generated>.yaml
```

**4. Download results**

```bash
scripts/download_vertex_results.sh \
  --project project-25ea6636-1c58-40fa-88b \
  --gcs-uri gs://genai-1010101/obfuscation-prompting/<experiment>/<display-name>/<run_id> \
  --dest-dir ./vertex_downloads
```

**Docker image** (Artifact Registry, linux/amd64, CUDA):

```bash
docker buildx build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/project-25ea6636-1c58-40fa-88b/genai/obfuscation-prompting:latest \
  --push .
```

---

## Project structure

```
src/                     # Reusable library code (import as src.*; scripts stay thin)
  clients/               #   TogetherClient (API) + HFClient (local, activation capture)
  dataset/               #   Synthetic dataset generator (ShippingDomain + others)
  framing/               #   18 framing conditions + FramingLoader
  loaders/               #   ConcealmentLoader, FramingLoader, BasePromptLoader
  monitors/              #   RegexMonitor, KeywordMonitor, LLMMonitor
  pipeline/              #   Pipeline, PipelineStep, PromptResult
  interp/                #   ActivationStore, linear probes, patching, SAE utils
  steps/                 #   BaseModelStep, MonitorStep
  storage/               #   ResultStorage

scripts/                 # Pipeline drivers + plotting/eval CLIs (run from repo root)
  run_blackbox.py        #   Black-box A0/A1/A2 experiment
  run_framing.py         #   18-condition implicit framing experiment
  run_interp.py          #   Activation capture + probe + causal analysis
  run_patching.py        #   Logit patching experiment
  main.py                #   Generic pipeline entrypoint (GPQA/MedQA/concealment)
  make_dataset.py        #   Standalone dataset generation
  plot_framing.py        #   Framing experiment figures
  plot_results.py        #   Black-box experiment figures
  plot_interp.py         #   Interp experiment figures
  eval_last_token_concealment.py                   # Causal evaluation script
  eval_full_sequence_concealment.py                # Full-sequence evaluation
  run_concealment_local.py                         # Reuse a dataset JSONL locally
  filter_triplets.py                               # Filter result triplets
  run_vertex_last_token_concealment_experiment.py  # Vertex entrypoint for interp
  create_vertex_run_config.py                      # Per-run YAML generator
  submit_vertex_job.sh                             # Vertex job submission
  download_vertex_results.sh                       # GCS artifact download
  vertex_job_runner.py                             # In-container wrapper + GCS upload
  docker_build_push.sh                             # Build + push the experiment image

scratch/                 # One-off / AI-generated code; nothing imports from it
  vertex_smoke_test/     #   Minimal image + loop script for Vertex wiring checks
  notes/                 #   Private analysis writeups (git-ignored)

vertex_jobs/
  experiment_l4.yaml     # Template for L4 GPU experiment jobs
  smoke_l4.yaml          # Template for infrastructure smoke tests
  runs/                  # Generated per-run YAMLs (one per submission)

docs/                    # Detailed experiment and architecture documentation
saved_experiments/       # Manually archived experiment snapshots
data/                    # Generated datasets
results/                 # Run outputs: JSON + plots (git-ignored)
activations/             # Saved .npz activation files
```

**Run everything from the repository root.** The experiment drivers import
`src.*`, so invoke them as modules — `python3.11 -m scripts.run_framing ...`,
not `python3.11 scripts/run_framing.py` (inside the Docker image
`PYTHONPATH=/app` is set, so both forms work there).

---

## Results summary

| Experiment | Model | Key result |
|------------|-------|------------|
| Black-box | Qwen2.5-7B (Together) | A2: 5% disclosure, A0: 98%, A1: 74–83% |
| Framing (18 conditions) | Qwen2.5-7B (Together) | EXPLICIT: 15%, all implicit: 61–100%; no implicit condition reliably suppresses |
| Interp (A0/A2) | Qwen2.5-1.5B (L4 GPU) | Condition separable layer 3; disclosure peak layer 16; 8-dim subspace Cohen's d ≈ 2.2 |
| Framing × activations | Qwen2.5-1.5B (L4 GPU) | Implicit/explicit concealment cosine sim 0.994–0.998 — same circuit |

---

## Documentation

- [`docs/FRAMING_EXPERIMENT.md`](docs/FRAMING_EXPERIMENT.md) — framing conditions reference and design
- [`docs/CONDITIONS_REFERENCE.md`](docs/CONDITIONS_REFERENCE.md) — all 18 condition templates
- [`docs/PIPELINE_ARCHITECTURE.md`](docs/PIPELINE_ARCHITECTURE.md) — pipeline internals
- [`docs/RUNNING.md`](docs/RUNNING.md) — detailed CLI reference for all entrypoints
- [`docs/VERTEX_EXPERIMENT_RUNBOOK.md`](docs/VERTEX_EXPERIMENT_RUNBOOK.md) — Vertex AI workflow
- [`scratch/notes/interp-experiment-analysis.org`](scratch/notes/interp-experiment-analysis.org) — A0/A2 interp findings
- [`scratch/notes/framing-interp-analysis.org`](scratch/notes/framing-interp-analysis.org) — implicit vs explicit activation analysis
- [`scratch/notes/experiment-report.org`](scratch/notes/experiment-report.org) — full research report (black-box + framing)
