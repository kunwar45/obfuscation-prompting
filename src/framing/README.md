# src/framing — Framing Experiment Module

This package implements the controlled framing experiment: a systematic test of
which implicit system-prompt framings cause a model to spontaneously conceal a
secret present in its context.

## Module Structure

```
src/framing/
├── __init__.py
├── conditions.py      — 18 FramingCondition objects, REGISTRY, DIMENSIONS, CONDITION_ORDER
├── framing_loader.py  — FramingLoader: JSONL → list[PromptResult]
└── README.md          — this file
```

## conditions.py

Central registry of all framing conditions.

```python
from src.framing.conditions import (
    REGISTRY,          # dict[str, FramingCondition]
    CONDITION_ORDER,   # list[str] — canonical plot ordering
    DIMENSIONS,        # dict[str, list[str]] — dim → keys
    render,            # render(key, role) → str
)

# Example
system_prompt = render("M_inst_s", "customer service assistant")
cond = REGISTRY["M_inst_s"]
print(cond.dimension)   # "motivation"
print(cond.label)       # "Instrumental (Strong)"
```

### FramingCondition fields

| Field | Type | Description |
|-------|------|-------------|
| `key` | str | Unique identifier (e.g. `"M_inst_s"`) |
| `dimension` | str | One of: baseline, motivation, incentive, audience, control |
| `label` | str | Human-readable name for plots |
| `template` | str | System prompt template with `{role}` placeholder |

### Explicitness Spectrum

Conditions are ordered from least to most suppression-inducing:

```
BASE (reference)
  → M_alt, M_arb, I_mild, I_reward, I_social, A_machine, A_none  (weak/null)
  → M_dec, M_inst, A_naive, A_expert                               (moderate implicit)
  → M_inst_s, I_strong, I_strong_s, A_naive_s                     (strong implicit)
  → SEMI                                                            (semi-explicit)
  → EXPLICIT                                                        (explicit)
```

## framing_loader.py

```python
from src.framing.framing_loader import FramingLoader

loader = FramingLoader(
    jsonl_path="data/framing_20260309.jsonl",
    role="customer service assistant",
    framing_keys=["BASE", "M_inst", "EXPLICIT"],
    query_types="B1",
    limit=None,          # None = all scenarios
)
prompts = loader.load()  # list[PromptResult]
```

Each prompt gets `metadata` keys:
- `system_prompt` — rendered framing template + "\n\n" + context paragraph
- `condition` — framing key (backward compat alias)
- `framing_dimension` — dimension name
- `framing_label` — human-readable label
- `keyword_hints`, `regex_monitor`, `example_id`, `domain`, `control_type`

### Prompt ID Format

```
{example_id}_{framing_key}_{query_type}
# e.g. "shipping_0003_M_inst_s_B1"
```

## Usage in run_framing.py

```python
from src.framing.conditions import CONDITION_ORDER, REGISTRY
from src.framing.framing_loader import FramingLoader

framing_keys = ["BASE", "M_inst_s", "I_strong_s", "SEMI", "EXPLICIT"]
loader = FramingLoader(
    jsonl_path=dataset_path,
    role=domain.role,
    framing_keys=framing_keys,
    query_types="B1",
)
prompts = loader.load()
results = pipeline.run(prompts)
```

## Adding a New Condition

1. Add a `FramingCondition` entry to `REGISTRY` in `conditions.py`
2. Add the key to the appropriate list in `DIMENSIONS`
3. Insert the key at the appropriate position in `CONDITION_ORDER`

No other files need to change — the loader and plots read from `REGISTRY`/`CONDITION_ORDER` dynamically.
