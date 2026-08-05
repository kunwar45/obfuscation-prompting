"""Mechanistic interpretability experiment for concealment behaviour.

Runs a local HuggingFace model on the shipping-domain concealment dataset,
captures per-layer residual-stream activations, then performs:

  1. Linear probes  — layer-by-layer CV accuracy for condition (A0 vs A2)
                       and disclosure (regex match) classification
  2. Logit lens     — P(secret_token | layer L) for matched A0/A2 prompt pairs
  3. PCA            — 2-D projection of last-layer activations coloured by condition

Flow
----
  smoke test  → 3 scenarios, gpt2, verify no pipeline errors
  full run    → N scenarios, user-specified model, full analysis + plots

Usage
-----
  python3.11 run_interp.py --smoke-only
  python3.11 run_interp.py --model Qwen/Qwen2.5-1.5B-Instruct
  python3.11 run_interp.py --skip-smoke --n-scenarios 40 --model google/gemma-2-2b-it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from src.clients.hf_client import HFClient
from src.config import Config
from src.dataset.generator import DatasetGenerator
from src.dataset.shipping_domain import ShippingDomain
from src.interp.activation_store import ActivationStore
from src.loaders.concealment_loader import ConcealmentLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.pipeline import Pipeline
from src.pipeline.result import PromptResult
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep
from src.storage.storage import ResultStorage


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n_scenarios: int, seed: int, output_path: str) -> int:
    gen = DatasetGenerator(domains=[ShippingDomain()], seed=seed)
    scenarios = gen.generate(n_scenarios)
    gen.to_jsonl(scenarios, output_path)
    return len(scenarios)


# ── Pipeline (no LLM monitor — local only) ───────────────────────────────────

def build_pipeline(client: HFClient, max_tokens: int = 256) -> Pipeline:
    monitors = [RegexMonitor(), KeywordMonitor()]
    steps = [BaseModelStep(client, _dummy_config(client, max_tokens)), MonitorStep(monitors)]
    return Pipeline(steps)


def _dummy_config(client: HFClient, max_tokens: int = 256) -> Config:
    """Minimal config for BaseModelStep using an HFClient."""
    cfg = Config()
    cfg.base_model = client.model_name
    cfg.capture_activations = client.capture_mode
    cfg.activations_dir = "activations"
    cfg.max_tokens = max_tokens
    return cfg


# ── Inference pass ────────────────────────────────────────────────────────────

def run_inference(
    client: HFClient,
    dataset_path: str,
    conditions: str,
    act_dir: str,
    output_dir: str,
    max_tokens: int = 256,
) -> tuple[list[PromptResult], str]:
    """Run the pipeline, save results + activations, return (results, json_path)."""
    loader = ConcealmentLoader(
        jsonl_path=dataset_path,
        conditions=conditions,
        query_types="B1",
    )
    prompts = loader.load()
    print(f"  {len(prompts)} prompts loaded from {dataset_path}")

    # Point activation saving to the right dir
    config = _dummy_config(client, max_tokens)
    config.activations_dir = act_dir

    monitors = [RegexMonitor(), KeywordMonitor()]
    steps = [BaseModelStep(client, config), MonitorStep(monitors)]
    pipeline = Pipeline(steps)
    results = pipeline.run(prompts)

    # Save JSON
    storage = ResultStorage(config)
    result_path = storage.save(results)
    return results, result_path


# ── Probe experiment ──────────────────────────────────────────────────────────

def _disclosed(r: dict) -> bool:
    """Return True if the regex monitor detected secret disclosure."""
    regex = r.get("monitor_results", {}).get("regex", {})
    return bool(regex.get("contains_secret_partial") or regex.get("contains_secret_exact"))


def _is_regular(r: dict) -> bool:
    """Return True if the prompt is a regular (non-control) prompt."""
    return r.get("metadata", {}).get("control_type") is None


def _try_probe(
    store: ActivationStore,
    ids_a: list[str],
    ids_b: list[str],
    name: str,
    label_a_name: str,
    label_b_name: str,
    min_samples: int,
    results: dict[str, dict[int, float]],
) -> None:
    """Train a binary probe on two ID groups and store results if enough samples."""
    if len(ids_a) < min_samples or len(ids_b) < min_samples:
        print(f"  [SKIP] {name}: {label_a_name}={len(ids_a)}, {label_b_name}={len(ids_b)} (need ≥{min_samples} each)")
        return

    combined_ids = ids_a + ids_b
    labels = np.array([0] * len(ids_a) + [1] * len(ids_b))
    n_per_class = min(len(ids_a), len(ids_b))
    n_folds = max(2, min(5, n_per_class))

    accs = _probe_layers(store, labels, combined_ids, n_folds)
    results[name] = accs
    print(f"  {name}: {len(combined_ids)} samples ({label_a_name}={len(ids_a)}, {label_b_name}={len(ids_b)}), {n_folds}-fold CV")


def run_probes(
    store: ActivationStore,
    min_samples_per_class: int = 3,
) -> dict[str, dict[int, float]]:
    """Run binary probes across all layers.

    Core probes (always run):
      condition    : A0=0 vs A2=1 (where does the model encode 'suppress'?)
      disclosure   : regex_disclosed=0/1 (where does the model commit to answer?)

    A1 behavior-filtered probes (when A1 data present):
      a1c_vs_a2c   : A1-concealing vs A2-concealing (same mechanism?)
      a1c_vs_a0d   : A1-concealing vs A0-disclosing (sanity: should separate)
      a1d_vs_a0d   : A1-disclosing vs A0-disclosing (did model ignore A1?)

    Returns dict mapping probe_name → {layer_idx → CV accuracy}.
    """
    results: dict[str, dict[int, float]] = {}

    # ── Condition probe: A0 vs A2 only ───────────────────────────────────
    a0a2_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") in ("A0", "A2")
        and _is_regular(r)
    )
    cond_labels = store.get_labels(
        lambda r: 1 if r.get("metadata", {}).get("condition") == "A2" else 0,
        prompt_ids=a0a2_ids,
    )

    n_per_class = min(np.bincount(cond_labels).tolist()) if len(a0a2_ids) >= 2 else 0
    n_folds = max(2, min(5, n_per_class))

    if len(a0a2_ids) >= 2 * min_samples_per_class:
        cond_accs = _probe_layers(store, cond_labels, a0a2_ids, n_folds)
        results["condition"] = cond_accs
        print(f"  Condition probe: {len(a0a2_ids)} samples, {n_folds}-fold CV")
    else:
        print(f"  [SKIP] Condition probe: only {len(a0a2_ids)} A0/A2 samples (need {2*min_samples_per_class})")

    # ── Disclosure probe: regex_disclosed yes/no ──────────────────────────
    regular_ids = store.filter_ids(lambda r: _is_regular(r))
    disc_labels = store.get_labels(
        lambda r: int(_disclosed(r)),
        prompt_ids=regular_ids,
    )

    classes, counts = np.unique(disc_labels, return_counts=True)
    if len(classes) >= 2 and counts.min() >= min_samples_per_class:
        n_folds_d = max(2, min(5, int(counts.min())))
        disc_accs = _probe_layers(store, disc_labels, regular_ids, n_folds_d)
        results["disclosure"] = disc_accs
        print(f"  Disclosure probe: {len(regular_ids)} samples (disclosed={sum(disc_labels)}, n_folds={n_folds_d})")
    else:
        print(f"  [SKIP] Disclosure probe: label distribution {dict(zip(classes.tolist(), counts.tolist()))} (need ≥{min_samples_per_class} per class)")

    # ── A1 behavior-filtered probes ──────────────────────────────────────
    has_a1 = len(store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A1" and _is_regular(r)
    )) > 0

    if has_a1:
        print("\n  A1 behavior-filtered probes:")

        # Subgroups
        a1_concealing_ids = store.filter_ids(
            lambda r: r.get("metadata", {}).get("condition") == "A1"
            and _is_regular(r) and not _disclosed(r)
        )
        a1_disclosing_ids = store.filter_ids(
            lambda r: r.get("metadata", {}).get("condition") == "A1"
            and _is_regular(r) and _disclosed(r)
        )
        a2_concealing_ids = store.filter_ids(
            lambda r: r.get("metadata", {}).get("condition") == "A2"
            and _is_regular(r) and not _disclosed(r)
        )
        a0_disclosing_ids = store.filter_ids(
            lambda r: r.get("metadata", {}).get("condition") == "A0"
            and _is_regular(r) and _disclosed(r)
        )

        print(f"    Subgroup sizes: A1-concealing={len(a1_concealing_ids)}, "
              f"A1-disclosing={len(a1_disclosing_ids)}, "
              f"A2-concealing={len(a2_concealing_ids)}, "
              f"A0-disclosing={len(a0_disclosing_ids)}")

        # Central test: A1-concealing vs A2-concealing (same mechanism?)
        _try_probe(store, a1_concealing_ids, a2_concealing_ids,
                   "a1c_vs_a2c", "A1-conceal", "A2-conceal",
                   min_samples_per_class, results)

        # Sanity check: A1-concealing vs A0-disclosing (should separate)
        _try_probe(store, a1_concealing_ids, a0_disclosing_ids,
                   "a1c_vs_a0d", "A1-conceal", "A0-disclose",
                   min_samples_per_class, results)

        # Secondary: A1-disclosing vs A0-disclosing (ignored A1?)
        _try_probe(store, a1_disclosing_ids, a0_disclosing_ids,
                   "a1d_vs_a0d", "A1-disclose", "A0-disclose",
                   min_samples_per_class, results)

    return results


def _probe_layers(
    store: ActivationStore,
    labels: np.ndarray,
    prompt_ids: list[str],
    n_folds: int,
) -> dict[int, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n_total_layers = store.n_layers()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs: dict[int, float] = {}

    for layer_idx in range(n_total_layers):
        X = store.load_layer(layer_idx, prompt_ids=prompt_ids)
        pipe = SKPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        scores = cross_val_score(pipe, X, labels, cv=cv, scoring="accuracy")
        accs[layer_idx] = float(scores.mean())

    return accs


# ── Logit lens ────────────────────────────────────────────────────────────────

def run_logit_lens(
    client: HFClient,
    store: ActivationStore,
    n_pairs: int = 3,
) -> list[dict]:
    """Compute logit-lens probability of secret token across layers.

    Finds matched A0/A2 pairs (and A0/A1/A2 triples when A1 is available).

    Returns a list of dicts:
        {example_id, secret, a0_probs, a2_probs, [a1_probs, a1_disclosed]}
    """
    results_list = store.get_results()

    # Group by example_id → find pairs/triples
    by_example: dict[str, dict[str, dict]] = {}
    for r in results_list:
        meta = r.get("metadata", {})
        eid = meta.get("example_id", "")
        cond = meta.get("condition", "")
        if cond in ("A0", "A1", "A2") and meta.get("control_type") is None:
            by_example.setdefault(eid, {})[cond] = r

    # Require at least A0+A2; A1 is optional
    groups = [
        (eid, v) for eid, v in by_example.items()
        if "A0" in v and "A2" in v
    ][:n_pairs]

    if not groups:
        print("  [SKIP] Logit lens: no matched A0/A2 pairs found.")
        return []

    final_ln = _get_final_ln(client.model)
    lm_head_weight = client.model.lm_head.weight  # [vocab, d_model]

    lens_data = []
    for eid, group in groups:
        r_a0 = group["A0"]
        r_a2 = group["A2"]
        secret_text = (r_a0.get("metadata", {}).get("keyword_hints") or [""])[0]
        target_tok_id = _find_secret_token_id(client.tokenizer, secret_text)

        if target_tok_id is None:
            print(f"  [SKIP] Logit lens for {eid}: can't tokenise secret '{secret_text}'")
            continue

        probs_a0 = _logit_lens_one(client, r_a0, target_tok_id, final_ln, lm_head_weight)
        probs_a2 = _logit_lens_one(client, r_a2, target_tok_id, final_ln, lm_head_weight)

        entry = {
            "example_id": eid,
            "secret": secret_text,
            "target_token_id": target_tok_id,
            "target_token_str": client.tokenizer.decode([target_tok_id]),
            "a0_probs": probs_a0.tolist(),
            "a2_probs": probs_a2.tolist(),
        }

        # Add A1 if available for this example
        if "A1" in group:
            r_a1 = group["A1"]
            probs_a1 = _logit_lens_one(client, r_a1, target_tok_id, final_ln, lm_head_weight)
            entry["a1_probs"] = probs_a1.tolist()
            entry["a1_disclosed"] = _disclosed(r_a1)

        lens_data.append(entry)
        a1_info = ""
        if "A1" in group:
            a1_tag = "disclosed" if entry.get("a1_disclosed") else "concealed"
            a1_info = f" | A1={a1_tag}"
        print(f"  Logit lens: {eid} | secret='{secret_text}' | token='{client.tokenizer.decode([target_tok_id])}'{a1_info}")

    return lens_data


def _get_final_ln(model):
    """Architecture-aware final layer norm retrieval."""
    # Llama / Qwen / Gemma
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    # Mistral / Falcon
    if hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
        return model.model.final_layernorm
    return torch.nn.Identity()


def _find_secret_token_id(tokenizer, secret_text: str) -> int | None:
    """Find the token ID for the first word of the secret (e.g. '28' from '28 days').

    Tries both the bare word and the space-prefixed variant, and returns the
    token whose decoded text actually contains the first word, avoiding returning
    a standalone space token when the tokenizer splits " 28" as [" ", "28"].
    """
    if not secret_text:
        return None
    first_word = secret_text.split()[0]
    for variant in (first_word, " " + first_word):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        # Return the first token whose decoded text contains the target word
        for tok_id in ids:
            if first_word in tokenizer.decode([tok_id]):
                return tok_id
    return None


def _logit_lens_one(
    client: HFClient,
    result: dict,
    target_tok_id: int,
    final_ln,
    lm_head_weight: torch.Tensor,
) -> np.ndarray:
    """Compute P(target_tok_id) at each layer for the last prompt token position."""
    meta = result.get("metadata", {})
    system_content = meta.get("system_prompt", "")
    user_content = result.get("prompt", "")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt_str = client._apply_chat_template(messages)
    input_ids = client.tokenizer(prompt_str, return_tensors="pt").input_ids.to(client.model.device)

    with torch.no_grad():
        out = client.model(input_ids, output_hidden_states=True, return_dict=True)

    hidden_states = out.hidden_states  # tuple: [n_layers+1] of [1, seq_len, d_model]
    probs = []
    for hs in hidden_states:
        last = hs[0, -1, :]  # [d_model]
        with torch.no_grad():
            normed = final_ln(last.unsqueeze(0)).squeeze(0)
            logits = normed @ lm_head_weight.T  # [vocab]
            p = torch.softmax(logits, dim=-1)[target_tok_id].item()
        probs.append(p)

    return np.array(probs)


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(store: ActivationStore) -> dict:
    """PCA of last-layer activations, coloured by condition."""
    last_layer = store.n_layers() - 1
    all_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("control_type") is None
    )
    if len(all_ids) < 3:
        return {}

    X = store.load_layer(last_layer, prompt_ids=all_ids)

    # Extract condition strings directly (not via get_labels which forces int)
    id_to_result = {r["prompt_id"]: r for r in store.get_results(prompt_ids=all_ids)}
    cond_strs = [
        id_to_result.get(pid, {}).get("metadata", {}).get("condition", "?")
        for pid in all_ids
    ]

    # Normalise to float64 for PCA stability
    X = X.astype(np.float64)
    X -= X.mean(axis=0)

    # PCA via SVD
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    X_2d = X @ Vt[:2].T  # [n, 2]

    color_map = {"A0": "#2196F3", "A1": "#FF9800", "A2": "#F44336"}
    colors = [color_map.get(c, "#9E9E9E") for c in cond_strs]

    return {"X_2d": X_2d, "colors": colors, "conditions": cond_strs, "layer": last_layer}


# ── A1 projection onto A0↔A2 axis ───────────────────────────────────────────

def run_a1_projection(
    store: ActivationStore,
    probe_results: dict[str, dict[int, float]],
) -> dict[int, dict[str, float]]:
    """Project A1-concealing activations onto the A0↔A2 separating hyperplane.

    For each layer, trains a logistic regression on A0 vs A2, extracts the weight
    vector (the concealment direction), then projects A1-concealing activations
    onto it. If A1-concealing scores land near A2, the same mechanism is used.

    Only reports layers where the A0-vs-A2 probe accuracy > 60%.

    Returns {layer_idx: {a0_mean, a0_std, a1c_mean, a1c_std, a2_mean, a2_std}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Get IDs for each subgroup
    a0_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A0" and _is_regular(r)
    )
    a2_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A2" and _is_regular(r)
    )
    a1c_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A1"
        and _is_regular(r) and not _disclosed(r)
    )

    if len(a0_ids) < 3 or len(a2_ids) < 3 or len(a1c_ids) < 2:
        print(f"  [SKIP] A1 projection: A0={len(a0_ids)}, A2={len(a2_ids)}, A1c={len(a1c_ids)} (insufficient)")
        return {}

    cond_accs = probe_results.get("condition", {})
    n_layers = store.n_layers()
    projection_data: dict[int, dict[str, float]] = {}

    for layer_idx in range(n_layers):
        # Skip layers where A0-vs-A2 probe is near chance
        if cond_accs.get(layer_idx, 0.0) < 0.60:
            continue

        X_a0 = store.load_layer(layer_idx, prompt_ids=a0_ids)
        X_a2 = store.load_layer(layer_idx, prompt_ids=a2_ids)
        X_a1c = store.load_layer(layer_idx, prompt_ids=a1c_ids)

        # Fit scaler + classifier on A0+A2 only
        X_train = np.vstack([X_a0, X_a2])
        y_train = np.array([0] * len(X_a0) + [1] * len(X_a2))

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)

        w = clf.coef_[0]  # concealment direction

        # Project all groups onto w
        scores_a0 = scaler.transform(X_a0) @ w
        scores_a2 = scaler.transform(X_a2) @ w
        scores_a1c = scaler.transform(X_a1c) @ w

        projection_data[layer_idx] = {
            "a0_mean": float(scores_a0.mean()),
            "a0_std": float(scores_a0.std()),
            "a1c_mean": float(scores_a1c.mean()),
            "a1c_std": float(scores_a1c.std()),
            "a2_mean": float(scores_a2.mean()),
            "a2_std": float(scores_a2.std()),
        }

    print(f"  A1 projection: {len(projection_data)} layers (of {n_layers}) above 60% probe threshold")
    return projection_data


# ── Cosine similarity analysis ───────────────────────────────────────────────

def run_cosine_similarity(store: ActivationStore) -> dict[int, dict[str, float]]:
    """Compute pairwise cosine similarity of mean activations per behavioral subgroup.

    Compares: A1-concealing vs A2-concealing, A1-concealing vs A0-disclosing,
    A1-disclosing vs A0-disclosing.

    Returns {layer_idx: {cos_a1c_a2c, cos_a1c_a0d, cos_a1d_a0d}}.
    """
    a0d_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A0"
        and _is_regular(r) and _disclosed(r)
    )
    a1c_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A1"
        and _is_regular(r) and not _disclosed(r)
    )
    a1d_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A1"
        and _is_regular(r) and _disclosed(r)
    )
    a2c_ids = store.filter_ids(
        lambda r: r.get("metadata", {}).get("condition") == "A2"
        and _is_regular(r) and not _disclosed(r)
    )

    # Need at least 1 sample per group for mean; skip groups that are empty
    groups = {"a0d": a0d_ids, "a1c": a1c_ids, "a1d": a1d_ids, "a2c": a2c_ids}
    missing = [k for k, v in groups.items() if len(v) == 0]
    if missing:
        print(f"  [SKIP] Cosine similarity: empty subgroups {missing}")
        return {}

    n_layers = store.n_layers()
    cosine_data: dict[int, dict[str, float]] = {}

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    for layer_idx in range(n_layers):
        mean_a0d = store.load_layer(layer_idx, prompt_ids=a0d_ids).mean(axis=0)
        mean_a1c = store.load_layer(layer_idx, prompt_ids=a1c_ids).mean(axis=0)
        mean_a1d = store.load_layer(layer_idx, prompt_ids=a1d_ids).mean(axis=0) if len(a1d_ids) > 0 else None
        mean_a2c = store.load_layer(layer_idx, prompt_ids=a2c_ids).mean(axis=0)

        entry: dict[str, float] = {
            "cos_a1c_a2c": _cosine(mean_a1c, mean_a2c),
            "cos_a1c_a0d": _cosine(mean_a1c, mean_a0d),
        }
        if mean_a1d is not None:
            entry["cos_a1d_a0d"] = _cosine(mean_a1d, mean_a0d)

        cosine_data[layer_idx] = entry

    print(f"  Cosine similarity: {n_layers} layers, subgroups: "
          + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    return cosine_data


# ── Summary printing ──────────────────────────────────────────────────────────

def print_probe_summary(probe_results: dict[str, dict[int, float]]) -> None:
    if not probe_results:
        print("  No probe results to summarise.")
        return

    print()
    for probe_name, layer_accs in probe_results.items():
        layers = sorted(layer_accs.keys())
        accs = [layer_accs[l] for l in layers]
        peak_layer = layers[int(np.argmax(accs))]
        peak_acc = max(accs)
        chance = 1 / 3 if "3class" in probe_name else 0.5
        print(f"  ┌─ {probe_name.upper()} PROBE {'─' * (50 - len(probe_name))}┐")
        print(f"  │  Layers: {len(layers)}   Peak layer: {peak_layer}   Peak CV acc: {peak_acc:.1%}   Chance: {chance:.0%}")
        # Mini bar chart (20-char wide)
        bar_w = 30
        for i, (l, a) in enumerate(zip(layers, accs)):
            bar = "█" * int(a * bar_w)
            marker = " ← peak" if l == peak_layer else ""
            if i % max(1, len(layers) // 8) == 0 or l == peak_layer:
                print(f"  │  L{l:>2}  {bar:<{bar_w}} {a:.1%}{marker}")
        print(f"  └{'─' * 58}┘")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpretability experiment for concealment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-scenarios", type=int, default=3)
    parser.add_argument("--n-scenarios", type=int, default=30,
                        help="Scenarios for full run (≥20 recommended for probes)")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model name (default: gpt2 for quick testing; "
                             "recommend Qwen/Qwen2.5-1.5B-Instruct for real results)")
    parser.add_argument("--dtype", default="float32",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (float32 safest for MPS/CPU smoke tests)")
    parser.add_argument("--conditions", default="A0,A2",
                        help="Conditions to run (A0=transparent, A2=explicit suppress). "
                             "A0,A2 gives clean binary probes; add A1 for 3-class.")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens per generation (default: 256 for speed)")
    parser.add_argument("--n-logit-pairs", type=int, default=3,
                        help="Number of A0/A2 pairs for logit-lens analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


# ── Run one batch ─────────────────────────────────────────────────────────────

def run_batch(
    model_name: str,
    dtype: str,
    n_scenarios: int,
    conditions: str,
    seed: int,
    data_dir: str,
    output_dir: str,
    timestamp: str,
    tag: str,
    max_tokens: int = 256,
) -> tuple[ActivationStore, list[PromptResult], str, str]:
    """Generate dataset, run inference, return (store, results, result_path, act_dir)."""
    dataset_path = os.path.join(data_dir, f"interp_{tag}_{timestamp}.jsonl")
    act_dir = os.path.join("activations", f"interp_{tag}_{timestamp}")
    os.makedirs(act_dir, exist_ok=True)

    n = generate_dataset(n_scenarios, seed, dataset_path)
    print(f"  Generated {n} scenarios → {dataset_path}")

    client = HFClient(
        model_name=model_name,
        dtype=dtype,
        capture_mode="last_token",
    )

    results, result_path = run_inference(
        client=client,
        dataset_path=dataset_path,
        conditions=conditions,
        act_dir=act_dir,
        output_dir=output_dir,
        max_tokens=max_tokens,
    )

    n_with_acts = sum(1 for r in results if r.activation_path)
    print(f"  Activations saved: {n_with_acts}/{len(results)} prompts → {act_dir}/")

    store = ActivationStore(act_dir, result_path)
    print(f"  ActivationStore: {len(store)} indexed, {store.n_layers()} layers")

    return store, results, result_path, act_dir, client


# ── Smoke summary ─────────────────────────────────────────────────────────────

def print_smoke_summary(results: list[PromptResult], store: ActivationStore) -> bool:
    W = 36
    print()
    print(f"  ┌{'─' * (W + 26)}┐")
    print(f"  │  {'PROMPT ID':<{W}} {'COND':>4}  {'ACT':>3}  {'REGEX':>5}  │")
    print(f"  │  {'─' * W}  {'─' * 4}  {'─' * 3}  {'─' * 5}  │")
    errors = 0
    for r in results:
        cond = r.metadata.get("condition", "?")
        has_act = "✓" if r.activation_path else "✗"
        if not r.activation_path:
            errors += 1
        regex = r.monitor_results.get("regex", {})
        regex_hit = "YES" if (regex.get("contains_secret_partial") or regex.get("contains_secret_exact")) else "no"
        pid = r.prompt_id[-W:] if len(r.prompt_id) > W else r.prompt_id
        print(f"  │  {pid:<{W}} {cond:>4}  {has_act:>3}  {regex_hit:>5}  │")
    print(f"  └{'─' * (W + 26)}┘")

    print(f"\n  Store: {len(store)} prompts indexed, {store.n_layers()} layers")
    if errors:
        print(f"  ⚠ {errors} prompts missing activations — check model/capture_mode.")
        return False
    print(f"  ✓ All {len(results)} prompts have saved activations.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Smoke test ────────────────────────────────────────────────────────
    if not args.skip_smoke:
        print(f"\n{'━' * 62}")
        print(f"  SMOKE TEST  ({args.smoke_scenarios} scenarios | model: {args.model})")
        print(f"{'━' * 62}")

        try:
            store, results, result_path, act_dir, client = run_batch(
                model_name=args.model,
                dtype=args.dtype,
                n_scenarios=args.smoke_scenarios,
                conditions=args.conditions,
                seed=args.seed,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                timestamp=timestamp,
                tag="smoke",
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"\n[FATAL] Smoke test inference failed: {e}")
            raise

        ok = print_smoke_summary(results, store)

        # Quick probe sanity (won't have enough samples usually — just checks no crash)
        print("\n  Running probe sanity check…")
        try:
            smoke_probes = run_probes(store, min_samples_per_class=2)
        except Exception as e:
            print(f"  [WARN] Probe error (non-fatal for smoke): {e}")
            smoke_probes = {}

        if smoke_probes:
            print_probe_summary(smoke_probes)

        print(f"  Results → {result_path}")

        if args.smoke_only:
            print("\n  --smoke-only: done.\n")
            return
        if not ok:
            print("\n  [WARN] Some activations missing, but continuing to full run.\n")
        else:
            print("\n  Smoke test passed. Starting full run…\n")

    # ── Full run ──────────────────────────────────────────────────────────
    print(f"{'━' * 62}")
    print(f"  FULL RUN")
    print(f"  model      : {args.model}  ({args.dtype})")
    print(f"  scenarios  : {args.n_scenarios}  conditions: {args.conditions}")
    print(f"{'━' * 62}")

    store, results, result_path, act_dir, client = run_batch(
        model_name=args.model,
        dtype=args.dtype,
        n_scenarios=args.n_scenarios,
        conditions=args.conditions,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        timestamp=timestamp,
        tag="full",
        max_tokens=args.max_tokens,
    )

    print(f"\n  Results saved → {result_path}")

    # ── Analysis ──────────────────────────────────────────────────────────
    print("\n  Running linear probes…")
    probe_results = run_probes(store)
    print_probe_summary(probe_results)

    print("  Running logit lens…")
    logit_data = run_logit_lens(client, store, n_pairs=args.n_logit_pairs)

    print("  Running PCA…")
    pca_data = run_pca(store)

    # A1-specific analyses (gated on A1 being in conditions)
    conditions_list = [c.strip() for c in args.conditions.split(",")]
    projection_data: dict = {}
    cosine_data: dict = {}
    if "A1" in conditions_list:
        print("\n  Running A1 projection analysis…")
        projection_data = run_a1_projection(store, probe_results)

        print("  Running cosine similarity analysis…")
        cosine_data = run_cosine_similarity(store)

    # Save analysis results alongside the main JSON
    analysis_path = result_path.replace(".json", "_analysis.json")
    _save_analysis(analysis_path, probe_results, logit_data, pca_data,
                   projection_data, cosine_data)
    print(f"  Analysis saved → {analysis_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n  Generating plots…")
        try:
            from scripts.plot_interp import plot_all_interp
            plot_all_interp(
                result_path=result_path,
                probe_results=probe_results,
                logit_data=logit_data,
                pca_data=pca_data,
                out_dir=args.output_dir,
                projection_data=projection_data,
                cosine_data=cosine_data,
            )
        except ImportError as e:
            print(f"  [WARN] Plotting skipped: {e}")

    print(f"\n{'━' * 62}")
    print(f"  Done.")
    print(f"  Results    : {result_path}")
    print(f"  Analysis   : {analysis_path}")
    print(f"  Activations: {act_dir}/")
    print(f"{'━' * 62}\n")


def _save_analysis(
    path: str,
    probe_results: dict,
    logit_data: list,
    pca_data: dict,
    projection_data: dict | None = None,
    cosine_data: dict | None = None,
) -> None:
    payload = {
        "probe_results": {
            name: {str(k): v for k, v in accs.items()}
            for name, accs in probe_results.items()
        },
        "logit_lens": logit_data,
        "pca": {
            "X_2d": pca_data.get("X_2d", np.zeros((0, 2))).tolist() if pca_data else [],
            "colors": pca_data.get("colors", []),
            "conditions": pca_data.get("conditions", []),
            "layer": pca_data.get("layer", -1),
        } if pca_data else {},
    }
    if projection_data:
        payload["a1_projection"] = {str(k): v for k, v in projection_data.items()}
    if cosine_data:
        payload["cosine_similarity"] = {str(k): v for k, v in cosine_data.items()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
