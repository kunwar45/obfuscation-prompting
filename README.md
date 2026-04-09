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

### 1. Black-box experiment (`run_blackbox.py`)

Establishes a causal baseline across three conditions:
- **A0** — transparent ("be accurate and honest")
- **A1** — implicit pressure (performance evaluated on retention, avoid alarming statements)
- **A2** — explicit suppression ("do not mention the delivery timeline under any circumstances")

```bash
# Smoke test (3 scenarios)
python run_blackbox.py --smoke-only

# Full run (50 scenarios, all three conditions)
python run_blackbox.py --skip-smoke --n-scenarios 50

# Custom model
python run_blackbox.py --skip-smoke --n-scenarios 50 \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo
```

Requires `TOGETHER_API_KEY`. Saves results to `results/` and generates 5 plots.

**Key finding:** A2 achieves ~95% concealment; A0 achieves ~98% disclosure; A1 sits in between at ~74–83%.

---

### 2. Framing experiment (`run_framing.py`)

Maps the space between transparent and suppressed across **18 framing conditions** in 5 dimensions (motivation, incentive, audience, baseline, and control). No explicit suppression instruction in most conditions — tests whether implicit framing alone induces concealment.

```bash
# Local model, smoke test
python3.11 run_framing.py --local --smoke-only --smoke-scenarios 2 \
  --conditions BASE,M_inst_s,I_strong_s,SEMI,EXPLICIT

# Local model, full run (all 18 conditions)
python3.11 run_framing.py --local --skip-smoke --n-scenarios 30 \
  --local-model Qwen/Qwen2.5-1.5B-Instruct --max-tokens 256

# Cloud (Together AI), full run with LLM monitor
python run_framing.py --skip-smoke --n-scenarios 30

# With activation capture (for interpretability analysis downstream)
python3.11 run_framing.py --local --skip-smoke --n-scenarios 30 \
  --conditions BASE,I_social,M_inst_s,I_strong_s,SEMI,EXPLICIT \
  --capture-activations last_token
```

Saves results to `results/` and generates 5 figures including an empirical explicitness spectrum and Cohen's h effect sizes vs baseline.

**Key finding:** Implicit framings rarely achieve <70% disclosure. Only EXPLICIT (15%) and SEMI (73%) reliably suppress. Audience framings (A_naive, A_none) can reach 100% disclosure — higher than the transparent baseline.

---

### 3. Interpretability experiment (`run_interp.py`)

Runs the pipeline with activation capture enabled, then probes the hidden states layer-by-layer to find where the model encodes (a) which condition it received, and (b) the disclosure decision. Also runs causal patching to test whether identified dimensions causally mediate concealment.

```bash
# Recommended: Qwen 1.5B on local GPU
python3.11 run_interp.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --n-scenarios 30

# Smaller smoke run
python3.11 run_interp.py \
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
run_blackbox.py          # Black-box A0/A1/A2 experiment
run_framing.py           # 18-condition implicit framing experiment
run_interp.py            # Activation capture + probe + causal analysis
run_patching.py          # Logit patching experiment

src/
  clients/               # TogetherClient (API) + HFClient (local, activation capture)
  dataset/               # Synthetic dataset generator (ShippingDomain + others)
  framing/               # 18 framing conditions + FramingLoader
  loaders/               # ConcealmentLoader, FramingLoader, BasePromptLoader
  monitors/              # RegexMonitor, KeywordMonitor, LLMMonitor
  pipeline/              # Pipeline, PipelineStep, PromptResult
  interp/                # ActivationStore, linear probes, patching, SAE utils
  steps/                 # BaseModelStep, MonitorStep
  storage/               # ResultStorage

scripts/
  run_vertex_last_token_concealment_experiment.py  # Vertex entrypoint for interp
  eval_last_token_concealment.py                   # Causal evaluation script
  plot_framing.py                                  # Framing experiment figures
  plot_results.py                                  # Black-box experiment figures
  create_vertex_run_config.py                      # Per-run YAML generator
  submit_vertex_job.sh                             # Vertex job submission
  download_vertex_results.sh                       # GCS artifact download
  vertex_job_runner.py                             # In-container wrapper + GCS upload

vertex_jobs/
  experiment_l4.yaml     # Template for L4 GPU experiment jobs
  smoke_l4.yaml          # Template for infrastructure smoke tests
  runs/                  # Generated per-run YAMLs (one per submission)

docs/                    # Detailed experiment and architecture documentation
local/                   # Local analysis writeups and notes (not committed)
saved_experiments/       # Manually archived experiment snapshots
data/                    # Generated datasets (git-ignored)
results/                 # Run outputs: JSON + plots (git-ignored)
activations/             # Saved .npz activation files (git-ignored)
```

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
- [`local/interp-experiment-analysis.org`](local/interp-experiment-analysis.org) — A0/A2 interp findings
- [`local/framing-interp-analysis.org`](local/framing-interp-analysis.org) — implicit vs explicit activation analysis
- [`local/experiment-report.org`](local/experiment-report.org) — full research report (black-box + framing)
