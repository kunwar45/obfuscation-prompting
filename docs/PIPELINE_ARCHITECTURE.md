# Pipeline Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Entry Points                                    │
│                                                                     │
│  run_blackbox.py   run_framing.py   run_interp.py   main.py         │
└──────────┬─────────────┬───────────────┬──────────────┬────────────┘
           │             │               │              │
           ▼             ▼               ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Dataset Generation                                 │
│                                                                     │
│   DatasetGenerator(domains=[ShippingDomain()])                      │
│     .generate(n_scenarios) → list[scenario dict]                    │
│     .to_jsonl(scenarios, path) → JSONL file                         │
│                                                                     │
│   Domains: ShippingDomain | BugDomain | BacklogDomain               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  JSONL file on disk
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Loaders                                          │
│                                                                     │
│   FramingLoader(jsonl, role, framing_keys, query_types)  ─────────┐ │
│   ConcealmentLoader(jsonl, conditions, query_types)               │ │
│                                                                   │ │
│   → list[PromptResult] (prompt_id, prompt, metadata pre-filled)   │ │
└───────────────────────────────────────────────────────────────────┼─┘
                                                                    │
                           ┌────────────────────────────────────────┘
                           │  list[PromptResult]
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Clients                                        │
│                                                                     │
│   TogetherClient(api_key)      HFClient(model_name, capture_mode)   │
│   │  .chat(messages, config)   │  .chat(messages, config)           │
│   │  → (reasoning, answer)     │  .save_activations(result, path)   │
│   │                            │  .model / .tokenizer               │
│   └── implements ChatClient ───┘                                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Pipeline                                       │
│                                                                     │
│   Pipeline(steps=[BaseModelStep, MonitorStep])                      │
│     .run(prompts) → list[PromptResult]                              │
│                                                                     │
│   BaseModelStep  ─── calls client.chat()                            │
│                  ─── optionally calls client.save_activations()     │
│                  ─── fills result.reasoning_trace + .final_answer   │
│                                                                     │
│   MonitorStep    ─── runs all monitors in sequence                  │
│                  ─── fills result.monitor_results dict              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Monitors                                       │
│                                                                     │
│   RegexMonitor   — pattern matching, primary binary label            │
│   KeywordMonitor — reasoning-trace term matching                     │
│   LLMMonitor     — second model qualitative analysis (Together only) │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Storage + Analysis                              │
│                                                                     │
│   ResultStorage(config).save(results)                               │
│     → results/run_YYYYMMDD_HHMMSS_<uuid>.json                       │
│                                                                     │
│   ActivationStore(act_dir, result_path)  [interp only]              │
│     .load_layer(idx) → np.ndarray [n_prompts, d_model]              │
│     .filter_ids(predicate) → list[str]                              │
│     .get_results() → list[dict]                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Plots                                        │
│                                                                     │
│   scripts/plot_results.py     → 5 figs (black-box experiment)       │
│   scripts/plot_framing.py     → 5 figs (framing experiment)         │
│   scripts/plot_interp.py      → 5 figs (interp experiment)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Per-Prompt

```
scenario JSONL
  │
  ▼  (Loader)
PromptResult(prompt_id, prompt="", metadata={...})
  │
  ▼  (BaseModelStep)
PromptResult(
  reasoning_trace="<reasoning>...</reasoning>",
  final_answer="<answer>...</answer>",
  activation_path="activations/.../..."   ← only if HFClient + capture_mode != "none"
)
  │
  ▼  (MonitorStep)
PromptResult(
  monitor_results={
    "regex":   {contains_secret_exact, contains_secret_partial, contains_refusal, ...},
    "keyword": {matched, found, ...},
    "llm":     {verdict, mentioned_secret, ...}   ← only if LLMMonitor in pipeline
  }
)
  │
  ▼  (ResultStorage)
results/run_<ts>_<id>.json
```

---

## Key Interfaces

### ChatClient Protocol (`src/clients/__init__.py`)

```python
class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        config: Config,
    ) -> tuple[str, str]:
        """Returns (reasoning_trace, final_answer)."""
        ...
```

Both `TogetherClient` and `HFClient` implement this protocol.

### Config (`src/config.py`)

Key fields used across the pipeline:

| Field | Type | Used by |
|-------|------|---------|
| `base_model` | str | BaseModelStep, ResultStorage |
| `monitor_model` | str | LLMMonitor |
| `max_tokens` | int | BaseModelStep |
| `together_api_key` | str | TogetherClient |
| `capture_activations` | str | BaseModelStep, HFClient |
| `activations_dir` | str | BaseModelStep |
| `output_dir` | str | ResultStorage |

`capture_activations` options: `"none"`, `"last_token"`, `"full_sequence"`, `"reasoning_span"`

### PromptResult (`src/pipeline/result.py`)

```python
@dataclass
class PromptResult:
    prompt_id: str
    prompt: str
    metadata: dict          # loader-injected: system_prompt, regex_monitor, keyword_hints, ...
    reasoning_trace: str    # model output up to </reasoning>
    final_answer: str       # model output after <answer>
    monitor_results: dict   # keyed by monitor name
    activation_path: str | None  # path to .npz file (HFClient only)
    base_model_id: str
    timestamp: str
```

---

## Adding a New Experiment

1. **New loader** — create `src/loaders/my_loader.py` implementing `BasePromptLoader.load()`
2. **New entry point** — mirror `run_framing.py` pattern:
   - `generate_dataset()` → JSONL
   - `build_pipeline()` or `build_pipeline_local()`
   - `run_batch(pipeline, dataset_path, ...)` → `list[PromptResult]`
   - `save_results()` + optional `plot_all_*()`
3. **New plot script** — `scripts/plot_my_experiment.py` reading from `results/*.json`

No changes needed to `Pipeline`, `BaseModelStep`, `MonitorStep`, or `ResultStorage`.

---

## Experiment → Script Mapping

| Experiment | Entry Point | Loader | Plots |
|-----------|-------------|--------|-------|
| Black-box (A0/A1/A2) | `run_blackbox.py` | `ConcealmentLoader` | `scripts/plot_results.py` |
| Framing (18 conditions) | `run_framing.py` | `FramingLoader` | `scripts/plot_framing.py` |
| Interpretability | `run_interp.py` | `ConcealmentLoader` | `scripts/plot_interp.py` |
| Custom pipeline | `main.py` | various | — |
