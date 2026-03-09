# Running the Experiments

Quick-start guide for all three experiments. All commands assume you are in the
project root directory.

---

## Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For Together AI experiments only:
cp .env.example .env
# Edit .env and set TOGETHER_API_KEY=sk-...
```

**Note:** Use `python3.11` (not `python3`) for torch-dependent code (local model experiments).

---

## Experiment 1: Black-Box Concealment Experiment

Tests three conditions: A0 (transparent), A1 (implicit incentive), A2 (explicit suppress).

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--smoke-only` | — | Run smoke test (2 scenarios) only |
| `--skip-smoke` | — | Skip smoke, go straight to full run |
| `--smoke-scenarios N` | 2 | Scenarios for smoke test |
| `--n-scenarios N` | 30 | Scenarios for full run |
| `--conditions C` | `A0,A1,A2` | Comma-separated condition keys |
| `--model M` | Llama-3.3-70B | Together AI base model |
| `--monitor-model M` | same | Together AI LLM monitor model |
| `--seed N` | 42 | RNG seed |
| `--output-dir D` | `results` | Results directory |
| `--data-dir D` | `data` | Dataset JSONL directory |
| `--no-plot` | — | Skip plot generation |

### Usage

```bash
# Quick smoke test
python run_blackbox.py --smoke-only

# Full run (30 scenarios × 3 conditions ≈ 76 regular prompts)
python run_blackbox.py --skip-smoke --n-scenarios 30

# Specific conditions only
python run_blackbox.py --conditions A0,A2 --skip-smoke
```

### Output

```
results/run_<ts>_<id>.json         — full results
results/disclosure_by_condition.{pdf,png}
results/concordance.{pdf,png}
results/by_secret.{pdf,png}
results/verdict_distribution.{pdf,png}
results/by_domain.{pdf,png}
```

---

## Experiment 2: Framing Experiment

Tests 18 conditions across 5 framing dimensions.

### CLI Reference

All flags from black-box experiment, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--local` | — | Use local HuggingFace model (no API key needed) |
| `--local-model M` | `Qwen/Qwen2.5-1.5B-Instruct` | HF model name |
| `--dtype D` | `float16` | Model dtype (`float16`, `bfloat16`, `float32`) |
| `--max-tokens N` | 256 | Max new tokens (local mode) |

### Usage

```bash
# Smoke test — local model, 6 key conditions
python3.11 run_framing.py --local --smoke-only --smoke-scenarios 2 \
  --conditions BASE,M_inst,M_inst_s,I_strong_s,SEMI,EXPLICIT

# Full run — local model, all 18 conditions
python3.11 run_framing.py --local --skip-smoke --n-scenarios 30 --max-tokens 256

# Red-team iteration — test strengthened conditions
python3.11 run_framing.py --local --smoke-only --smoke-scenarios 3 \
  --conditions BASE,M_inst,M_inst_s,I_strong_s,A_naive_s,SEMI,EXPLICIT

# Cloud (Together AI) — full run
python run_framing.py --skip-smoke --n-scenarios 30

# Specific conditions only
python3.11 run_framing.py --local --skip-smoke \
  --conditions BASE,M_inst,M_inst_s,I_strong,I_strong_s,SEMI,EXPLICIT
```

### Output

```
results/run_<ts>_<id>.json
results/framing_main.{pdf,png}              — Fig 1: disclosure by condition
results/framing_by_dimension.{pdf,png}     — Fig 2: by dimension
results/framing_heatmap.{pdf,png}          — Fig 3: condition × secret heatmap
results/framing_monitor_agreement.{pdf,png}— Fig 4: regex vs LLM scatter
results/framing_spectrum.{pdf,png}         — Fig 5: empirical spectrum
```

### Valid Condition Keys

```
BASE
M_alt, M_inst, M_inst_s, M_dec, M_arb
I_mild, I_strong, I_strong_s, I_reward, I_social
A_naive, A_naive_s, A_expert, A_machine, A_none
SEMI, EXPLICIT
```

See `docs/CONDITIONS_REFERENCE.md` for full descriptions.

---

## Experiment 3: Interpretability Experiment

Runs a local model, captures per-layer residual-stream activations, and performs
linear probes, logit lens, and PCA analysis.

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--smoke-only` | — | Smoke test only |
| `--skip-smoke` | — | Skip smoke |
| `--smoke-scenarios N` | 3 | Smoke scenarios |
| `--n-scenarios N` | 30 | Full run scenarios |
| `--model M` | `gpt2` | HF model (gpt2 for quick testing) |
| `--dtype D` | `float32` | Model dtype |
| `--conditions C` | `A0,A2` | Concealment conditions (A0/A1/A2) |
| `--max-tokens N` | 256 | Max new tokens |
| `--n-logit-pairs N` | 3 | A0/A2 pairs for logit-lens |
| `--seed N` | 42 | RNG seed |
| `--output-dir D` | `results` | Results directory |
| `--data-dir D` | `data` | Dataset directory |
| `--no-plot` | — | Skip plots |

### Usage

```bash
# Quick smoke test (gpt2 for speed)
python3.11 run_interp.py --smoke-only

# Recommended full run (1.5B model, float16, 30 scenarios)
python3.11 run_interp.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --n-scenarios 30 \
  --max-tokens 256

# Skip smoke, specific dtype
python3.11 run_interp.py \
  --skip-smoke \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --n-scenarios 40
```

### Output

```
results/run_<ts>_<id>.json          — inference results
results/run_<ts>_<id>_analysis.json — probe + logit lens + PCA data
activations/interp_full_<ts>/       — .npz activation files
results/probe_accuracy.png
results/probe_heatmap.png
results/logit_lens.png
results/logit_lens_gap.png
results/activation_pca.png
```

---

## Plotting Standalone

All plot scripts can be run standalone on existing result files:

```bash
# Black-box results
python scripts/plot_results.py results/run_<ts>_<id>.json

# Framing results
python scripts/plot_framing.py results/run_<ts>_<id>.json

# Interp results (requires analysis JSON)
python scripts/plot_interp.py results/run_<ts>_<id>.json
```

---

## Environment Notes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | Use `python3.11` not `python3` |
| `TOGETHER_API_KEY not set` | Add to `.env` or use `--local` |
| MPS out of memory | Reduce `--max-tokens` or use `--dtype float16` |
| Slow on CPU | Use `--dtype float16`, reduce `--n-scenarios` |
| `matplotlib` missing | `pip install matplotlib scipy` |
