"""Activation patching experiment for concealment behaviour.

Two analyses:

  Attribution sweep  — gradient-based approximation of causal patching across
                       all (layer, position) pairs at once. Identifies which
                       layers causally determine whether the model discloses
                       the secret. Fast: 2 fwd + 1 bwd pass per example pair.

  Transfer test      — patches activations from the EXPLICIT (A2) concealment
                       run into the IMPLICIT (A1) run at the top attribution
                       layer. If A1's output shifts toward concealment, the
                       mechanism is shared (H1). If A1's output is unchanged
                       or incoherent, the pathways are distinct (H2).

Flow
----
  For each A0/A2 pair (same example_id, same context/secret, same query):
    1. Reconstruct input_ids for A0 (disclosure) and A2 (concealment)
    2. Cache A0's residual stream at every layer (clean run)
    3. attribution_patch(clean=A0_cache, corrupt=A2_input_ids, metric=logit[secret_tok])
       → [n_layers, seq_len] array: how much does restoring A0's state at (L, P) recover
         secret-token probability in the A2 run?
    4. (optional) causal_patch on top --causal-top-k layers to confirm
  Aggregate attribution scores across pairs.
  Transfer test:
    5. Construct A1 input_ids (same context/query, implicit system prompt)
    6. For each pair: patch A2's key-layer activations into A1's run; decode; monitor

Usage
-----
  # Attribution sweep on existing run_interp.py results
  python3.11 run_patching.py \\
    --results  results/run_20260303_230750_7ac2de91.json \\
    --activations activations/interp_concealment_20260303_172755 \\
    --model  Qwen/Qwen2.5-1.5B-Instruct \\
    --dtype  float16

  # With causal confirmation on top 5 layers
  python3.11 run_patching.py --results ... --activations ... --model ... --causal-top-k 5

  # Full suite including transfer test (needs dataset JSONL for A1 system prompts)
  python3.11 run_patching.py --results ... --activations ... --model ... \\
    --transfer --dataset data/interp_full_<ts>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import torch

from src.clients.hf_client import HFClient
from src.interp.activation_store import ActivationStore
from src.interp.patching import attribution_patch, causal_patch
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.result import PromptResult


# ── Input reconstruction ──────────────────────────────────────────────────────

def _build_input_ids(client: HFClient, result: dict) -> torch.Tensor:
    """Reconstruct input_ids for a result dict from its stored metadata.

    Uses metadata.system_prompt (the A-condition + context) and prompt (user query).
    """
    meta = result.get("metadata", {})
    messages = [
        {"role": "system", "content": meta.get("system_prompt", "")},
        {"role": "user",   "content": result.get("prompt", "")},
    ]
    prompt_str = client._apply_chat_template(messages)
    return client.tokenizer(prompt_str, return_tensors="pt").input_ids.to(client.model.device)


def _build_a1_input_ids(
    client: HFClient,
    result_a0: dict,
    dataset_index: dict[str, dict],
) -> torch.Tensor | None:
    """Construct A1 input_ids using the A1 system prompt from the dataset JSONL.

    Returns None if the example_id is not in dataset_index or has no A1 condition.
    """
    example_id = result_a0.get("metadata", {}).get("example_id", "")
    scenario = dataset_index.get(example_id)
    if scenario is None:
        return None
    a1_system = scenario.get("conditions", {}).get("A1")
    if not a1_system:
        return None
    context = result_a0.get("metadata", {}).get("context", "")
    user_query = result_a0.get("prompt", "")
    messages = [
        {"role": "system", "content": f"{a1_system}\n\n{context}"},
        {"role": "user",   "content": user_query},
    ]
    prompt_str = client._apply_chat_template(messages)
    return client.tokenizer(prompt_str, return_tensors="pt").input_ids.to(client.model.device)


# ── Cache building ────────────────────────────────────────────────────────────

def build_clean_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Run a forward pass and cache each transformer layer's residual stream output.

    Returns {layer_idx: [1, seq_len, d_model]} tensors on CPU.
    These are passed as `clean_cache` to attribution_patch or causal_patch.
    """
    cache: dict[int, torch.Tensor] = {}
    hooks = []
    n_layers = model.config.num_hidden_layers

    for layer_idx in range(n_layers):
        def _save(module, inp, out, idx=layer_idx):
            hs = out[0] if isinstance(out, tuple) else out
            cache[idx] = hs.detach().clone().cpu()
        hooks.append(model.model.layers[layer_idx].register_forward_hook(_save))

    with torch.no_grad():
        model(input_ids, return_dict=True)

    for h in hooks:
        h.remove()
    return cache


# ── Metric function ───────────────────────────────────────────────────────────

def _find_secret_token_id(tokenizer, secret_text: str) -> int | None:
    """Return the best single-token proxy for the first word of the secret.

    Priority:
      1. A token whose decoded text exactly contains first_word (e.g. '2' → '2')
      2. First token of the encoded sequence as a proxy for multi-token numbers
         (e.g. Qwen splits '28' → ['2','8']; we use '2' as the proxy)

    The proxy is imperfect for multi-token numbers but still gives a
    directional signal: if layer L causally determines whether the model
    generates digits in this range, attribution scores will peak there.
    """
    if not secret_text:
        return None
    first_word = secret_text.split()[0]
    # Prefer a token whose decode exactly contains the target
    for variant in (first_word, " " + first_word):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        for tok_id in ids:
            if first_word in tokenizer.decode([tok_id]):
                return tok_id
    # Fallback: first token of the encoding (proxy for multi-token numbers)
    ids = tokenizer.encode(first_word, add_special_tokens=False)
    if ids:
        return ids[0]
    return None


def make_metric_fn(target_tok_id: int):
    """Return a differentiable metric: logit of the secret token at the last position.

    Used by attribution_patch (which calls .backward() on the metric).
    High value = model is about to generate the secret token.
    """
    def fn(logits: torch.Tensor) -> torch.Tensor:
        return logits[0, -1, target_tok_id]
    return fn


# ── Pair collection ───────────────────────────────────────────────────────────

def collect_pairs(results_path: str) -> list[dict]:
    """Collect A0/A2 matched pairs from a results JSON.

    Returns list of {"example_id", "a0": result_dict, "a2": result_dict}.
    Only includes non-control, B1 entries that have both A0 and A2.
    """
    with open(results_path) as f:
        data = json.load(f)
    raw = data.get("results", data) if isinstance(data, dict) else data

    by_example: dict[str, dict[str, dict]] = {}
    for r in raw:
        meta = r.get("metadata", {})
        if meta.get("control_type") or meta.get("query_type") != "B1":
            continue
        eid = meta.get("example_id", "")
        cond = meta.get("condition", "")
        if eid and cond in ("A0", "A1", "A2"):
            by_example.setdefault(eid, {})[cond] = r

    pairs = []
    for eid, cmap in sorted(by_example.items()):
        if "A0" in cmap and "A2" in cmap:
            pairs.append({"example_id": eid, "a0": cmap["A0"], "a2": cmap["A2"],
                          "a1": cmap.get("A1")})
    return pairs


# ── Cache alignment ───────────────────────────────────────────────────────────

def align_cache_to_length(
    cache: dict[int, torch.Tensor],
    target_len: int,
) -> dict[int, torch.Tensor]:
    """Right-align a clean cache to match a target sequence length.

    A0 and A2 share the same context + user query suffix but differ in system
    prompt length. A2 is typically longer. We pad the front of A0's cache with
    zeros so positions align from the right (shared suffix tokens line up).

    If the cache is longer than target_len, it is right-trimmed instead.
    """
    aligned: dict[int, torch.Tensor] = {}
    for layer_idx, hs in cache.items():
        # hs: [1, src_len, d_model]
        src_len = hs.shape[1]
        if src_len == target_len:
            aligned[layer_idx] = hs
        elif src_len < target_len:
            pad = torch.zeros(1, target_len - src_len, hs.shape[2],
                              dtype=hs.dtype, device=hs.device)
            aligned[layer_idx] = torch.cat([pad, hs], dim=1)
        else:
            # Trim from the front (keep the suffix)
            aligned[layer_idx] = hs[:, src_len - target_len:, :]
    return aligned


# ── Attribution sweep ─────────────────────────────────────────────────────────

def run_attribution_sweep(
    client: HFClient,
    pairs: list[dict],
    n_pairs: int | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Run attribution_patch on every A0/A2 pair.

    Returns:
        per_pair_scores: list of {"example_id", "secret", "scores": [n_layers, seq_len]}
        mean_by_layer:   [n_layers] mean absolute attribution score per layer
                         (summed over positions, averaged over pairs)
    """
    model = client.model
    tokenizer = client.tokenizer
    pairs_to_run = pairs[:n_pairs] if n_pairs else pairs
    n_layers = model.config.num_hidden_layers

    per_pair: list[dict] = []
    layer_totals = np.zeros(n_layers)
    n_valid = 0

    for i, pair in enumerate(pairs_to_run):
        eid = pair["example_id"]
        r_a0 = pair["a0"]
        r_a2 = pair["a2"]

        secret_text = (r_a0.get("metadata", {}).get("keyword_hints") or [""])[0]
        target_tok_id = _find_secret_token_id(tokenizer, secret_text)
        if target_tok_id is None:
            print(f"  [SKIP] {eid}: can't tokenise secret '{secret_text}'")
            continue

        a0_ids = _build_input_ids(client, r_a0)
        a2_ids = _build_input_ids(client, r_a2)

        print(f"  [{i+1}/{len(pairs_to_run)}] {eid} | secret='{secret_text}' "
              f"| tok={tokenizer.decode([target_tok_id])!r} "
              f"| A0_len={a0_ids.shape[1]} A2_len={a2_ids.shape[1]}")

        # Build clean (A0) cache
        clean_cache = build_clean_cache(model, a0_ids)

        # Align A0 cache to A2 sequence length (they share the same suffix —
        # context + user query — but A2 has a longer system prompt prefix)
        device_cache = {
            k: v.to(model.device)
            for k, v in align_cache_to_length(clean_cache, a2_ids.shape[1]).items()
        }

        # Attribution patch: clean=A0, corrupt=A2 — how much of A0's state
        # is needed to restore disclosure behaviour in A2's run?
        metric_fn = make_metric_fn(target_tok_id)
        try:
            scores = attribution_patch(model, device_cache, a2_ids, metric_fn)
        except Exception as e:
            print(f"    [WARN] attribution_patch failed for {eid}: {e}")
            continue

        # scores: [n_layers, seq_len_a2]
        # Pad/trim to n_layers (paranoia)
        if scores.shape[0] != n_layers:
            scores = scores[:n_layers]

        # Per-layer summary: sum of absolute scores over positions
        layer_sum = np.abs(scores).sum(axis=1)  # [n_layers]
        layer_totals += layer_sum
        n_valid += 1

        per_pair.append({
            "example_id": eid,
            "secret": secret_text,
            "target_token_id": target_tok_id,
            "target_token_str": tokenizer.decode([target_tok_id]),
            "scores": scores.tolist(),
            "layer_sums": layer_sum.tolist(),
        })

    mean_by_layer = layer_totals / max(n_valid, 1)
    print(f"\n  Attribution sweep: {n_valid}/{len(pairs_to_run)} pairs processed")
    if n_valid > 0:
        top3 = np.argsort(mean_by_layer)[::-1][:3]
        print(f"  Top layers by mean attribution: {list(top3)}")

    return per_pair, mean_by_layer


# ── Causal confirmation ───────────────────────────────────────────────────────

def run_causal_confirmation(
    client: HFClient,
    pairs: list[dict],
    top_layers: list[int],
) -> list[dict]:
    """Run true causal patching on the top attribution layers.

    For each pair and each top layer: patch A0's clean activation at that
    layer (all positions) into the A2 run and measure how much the secret
    token logit recovers.

    Returns list of {"example_id", "layer", "causal_score"} dicts.
    """
    model = client.model
    tokenizer = client.tokenizer
    results = []

    for pair in pairs:
        eid = pair["example_id"]
        r_a0 = pair["a0"]
        r_a2 = pair["a2"]

        secret_text = (r_a0.get("metadata", {}).get("keyword_hints") or [""])[0]
        target_tok_id = _find_secret_token_id(tokenizer, secret_text)
        if target_tok_id is None:
            continue

        a0_ids = _build_input_ids(client, r_a0)
        a2_ids = _build_input_ids(client, r_a2)
        clean_cache = build_clean_cache(model, a0_ids)
        device_cache = {
            k: v.to(model.device)
            for k, v in align_cache_to_length(clean_cache, a2_ids.shape[1]).items()
        }

        # Baseline: corrupt (A2) metric with no patching
        with torch.no_grad():
            baseline_out = model(a2_ids, return_dict=True)
        baseline_score = baseline_out.logits[0, -1, target_tok_id].item()

        metric_fn_float = lambda logits: logits[0, -1, target_tok_id].item()

        for layer_idx in top_layers:
            scores = causal_patch(
                model, device_cache, a2_ids,
                metric_fn=metric_fn_float,
                layers=[layer_idx],
            )
            # scores: [1, seq_len] — mean over positions (total recovery from this layer)
            layer_score = float(np.mean(scores[0]))
            recovery = layer_score - baseline_score

            results.append({
                "example_id": eid,
                "layer": layer_idx,
                "causal_score_mean": layer_score,
                "baseline_score": baseline_score,
                "recovery": recovery,
            })
            print(f"    {eid} | layer {layer_idx:>2} | "
                  f"baseline={baseline_score:.3f} score={layer_score:.3f} "
                  f"recovery={recovery:+.3f}")

    return results


# ── Transfer test ─────────────────────────────────────────────────────────────

def run_transfer_test(
    client: HFClient,
    pairs: list[dict],
    top_layers: list[int],
    dataset_index: dict[str, dict] | None = None,
) -> list[dict]:
    """Patch activations across conditions to test whether mechanisms are shared.

    Two sub-tests per pair:

    (A) Restoration test: patch A0 key-layer activations into A2 run.
        If A2 now discloses → A0's mechanism at that layer causally suppresses A2.
        (Sanity check / measure of causal effect size.)

    (B) Transfer test (H1/H2): patch A2 key-layer activations into A1 run.
        If A1 shifts toward concealment → shared mechanism (H1).
        If A1 is unchanged → distinct mechanism (H2).
        Requires dataset_index to reconstruct A1 input_ids.

    Returns list of transfer result dicts.
    """
    model = client.model
    tokenizer = client.tokenizer
    regex_monitor = RegexMonitor()
    results = []
    key_layer = top_layers[0] if top_layers else 0

    for pair in pairs:
        eid = pair["example_id"]
        r_a0 = pair["a0"]
        r_a2 = pair["a2"]

        secret_text = (r_a0.get("metadata", {}).get("keyword_hints") or [""])[0]
        regex_meta = r_a0.get("metadata", {}).get("regex_monitor", {})
        target_tok_id = _find_secret_token_id(tokenizer, secret_text)
        if target_tok_id is None:
            continue

        a0_ids = _build_input_ids(client, r_a0)
        a2_ids = _build_input_ids(client, r_a2)
        a0_cache = build_clean_cache(model, a0_ids)
        a2_cache = build_clean_cache(model, a2_ids)

        # ── (A) Restoration: patch A0 → A2 ──────────────────────────────
        a0_patch_cache = {key_layer: a0_cache[key_layer].to(model.device)}
        restored_text = _generate_with_patch(model, tokenizer, a2_ids, a0_patch_cache, key_layer)
        restored_regex = _run_regex(regex_monitor, restored_text, regex_meta)

        # Baseline: A2 unpatched (should conceal)
        baseline_a2 = r_a2.get("final_answer", "")
        baseline_regex = _run_regex(regex_monitor, baseline_a2, regex_meta)

        result = {
            "example_id": eid,
            "key_layer": key_layer,
            "secret": secret_text,
            "restoration": {
                "baseline_a2_disclosed": baseline_regex,
                "patched_a0_into_a2_disclosed": restored_regex,
                "patched_text_snippet": restored_text[:200],
            },
        }

        print(f"  {eid} | A0→A2 patch at layer {key_layer}: "
              f"baseline={baseline_regex} → patched={restored_regex}")

        # ── (B) Transfer: patch A2 → A1 ─────────────────────────────────
        if dataset_index is not None:
            a1_ids = _build_a1_input_ids(client, r_a0, dataset_index)
            if a1_ids is not None:
                a2_patch_cache = {key_layer: a2_cache[key_layer].to(model.device)}
                a1_baseline_text = _generate_plain(model, tokenizer, a1_ids)
                a1_patched_text  = _generate_with_patch(model, tokenizer, a1_ids, a2_patch_cache, key_layer)

                a1_baseline_disc = _run_regex(regex_monitor, a1_baseline_text, regex_meta)
                a1_patched_disc  = _run_regex(regex_monitor, a1_patched_text, regex_meta)

                result["transfer"] = {
                    "a1_baseline_disclosed": a1_baseline_disc,
                    "a1_patched_a2_into_a1_disclosed": a1_patched_disc,
                    "a1_baseline_snippet": a1_baseline_text[:200],
                    "a1_patched_snippet": a1_patched_text[:200],
                    "h1_consistent": (a1_baseline_disc and not a1_patched_disc)
                                     or (not a1_baseline_disc and not a1_patched_disc),
                }
                print(f"         A2→A1 patch at layer {key_layer}: "
                      f"a1_baseline={a1_baseline_disc} → a1_patched={a1_patched_disc}")

        results.append(result)

    return results


def _generate_plain(model, tokenizer, input_ids: torch.Tensor, max_new: int = 128) -> str:
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
    new = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def _generate_with_patch(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    patch_cache: dict[int, torch.Tensor],
    layer_idx: int,
    max_new: int = 128,
) -> str:
    """Generate with a single-layer activation patch applied at every step."""
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            return (patch_cache[layer_idx].to(out[0].device),) + out[1:]
        return patch_cache[layer_idx].to(out.device)

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False)
    finally:
        handle.remove()

    new = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def _run_regex(monitor: RegexMonitor, text: str, regex_meta: dict) -> bool:
    """Return True if text discloses the secret according to the regex monitor."""
    if not text or not regex_meta:
        return False

    dummy_result = PromptResult(
        prompt_id="transfer_test",
        prompt="",
        timestamp="",
        final_answer=text,
        metadata={"regex_monitor": regex_meta},
    )
    out = monitor.run(dummy_result)
    return bool(out.get("contains_secret_exact") or out.get("contains_secret_partial"))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    mean_by_layer: np.ndarray,
    per_pair_scores: list[dict],
    causal_results: list[dict],
    transfer_results: list[dict],
    out_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    n_layers = len(mean_by_layer)

    # ── 1. Mean attribution by layer ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    layers = list(range(n_layers))
    ax.bar(layers, mean_by_layer, color="#1976D2", alpha=0.8)
    peak = int(np.argmax(mean_by_layer))
    ax.axvline(peak, color="red", linestyle="--", linewidth=1.2, label=f"Peak: layer {peak}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |attribution| (summed over positions)")
    ax.set_title("Attribution Patching: A0 → A2\n(which layer causally mediates disclosure?)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "patching_attribution_by_layer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 2. Attribution heatmap (last 30 positions, all pairs averaged) ────
    if per_pair_scores:
        n_last = 30
        heatmap_rows = []
        for entry in per_pair_scores:
            scores = np.array(entry["scores"])  # [n_layers, seq_len]
            if scores.shape[1] >= n_last:
                scores = scores[:, -n_last:]
            else:
                pad = np.zeros((scores.shape[0], n_last - scores.shape[1]))
                scores = np.concatenate([pad, scores], axis=1)
            heatmap_rows.append(np.abs(scores))

        heatmap = np.mean(np.stack(heatmap_rows, axis=0), axis=0)  # [n_layers, n_last]

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="hot")
        ax.set_xlabel(f"Token position (last {n_last})")
        ax.set_ylabel("Layer")
        ax.set_title("Attribution Heatmap: |score| averaged across pairs\n"
                     "(bright = high causal effect of patching A0 → A2)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        path = os.path.join(out_dir, "patching_attribution_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    # ── 3. Transfer test summary ──────────────────────────────────────────
    if transfer_results:
        eids = [t["example_id"] for t in transfer_results]
        a2_baseline = [t["restoration"]["baseline_a2_disclosed"] for t in transfer_results]
        a2_patched  = [t["restoration"]["patched_a0_into_a2_disclosed"] for t in transfer_results]
        has_transfer = all("transfer" in t for t in transfer_results)

        x = np.arange(len(eids))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(eids) * 0.8), 4))
        ax.bar(x - width, a2_baseline, width, label="A2 baseline", color="#F44336", alpha=0.8)
        ax.bar(x,         a2_patched,  width, label="A2 + A0 patch", color="#2196F3", alpha=0.8)
        if has_transfer:
            a1_baseline = [t["transfer"]["a1_baseline_disclosed"] for t in transfer_results]
            a1_patched  = [t["transfer"]["a1_patched_a2_into_a1_disclosed"] for t in transfer_results]
            ax.bar(x + width,     a1_baseline, width, label="A1 baseline",     color="#FF9800", alpha=0.8)
            ax.bar(x + 2*width,   a1_patched,  width, label="A1 + A2 patch",   color="#9C27B0", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(eids, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Disclosed (1=yes, 0=no)")
        ax.set_title("Transfer Test: behavioral change from activation patching\n"
                     "(H1 = A2 patch changes A1; H2 = no change)")
        ax.set_ylim(-0.1, 1.4)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, "patching_transfer.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")


# ── Saving ────────────────────────────────────────────────────────────────────

def save_results(
    out_dir: str,
    timestamp: str,
    model_name: str,
    results_path: str,
    mean_by_layer: np.ndarray,
    per_pair_scores: list[dict],
    causal_results: list[dict],
    transfer_results: list[dict],
) -> str:
    short_id = str(uuid.uuid4()).split("-")[0]
    fname = f"patching_{timestamp}_{short_id}.json"
    path = os.path.join(out_dir, fname)

    n_layers = len(mean_by_layer)
    top_layers = np.argsort(mean_by_layer)[::-1][:10].tolist()

    payload = {
        "timestamp": timestamp,
        "model": model_name,
        "source_results": os.path.abspath(results_path),
        "n_pairs": len(per_pair_scores),
        "summary": {
            "top_layers_by_attribution": top_layers,
            "mean_attribution_by_layer": {str(i): float(v) for i, v in enumerate(mean_by_layer)},
        },
        "attribution_per_pair": [
            {k: v for k, v in entry.items() if k != "scores"}  # omit raw [n_layers, seq_len]
            for entry in per_pair_scores
        ],
        "attribution_scores_raw": [
            {"example_id": e["example_id"], "scores": e["scores"]}
            for e in per_pair_scores
        ],
        "causal_confirmation": causal_results,
        "transfer_test": transfer_results,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(
    mean_by_layer: np.ndarray,
    causal_results: list[dict],
    transfer_results: list[dict],
) -> None:
    n_layers = len(mean_by_layer)
    top5 = np.argsort(mean_by_layer)[::-1][:5]

    print()
    print("─" * 60)
    print("  ATTRIBUTION SWEEP — TOP LAYERS")
    print("─" * 60)
    for rank, layer in enumerate(top5):
        bar = "█" * int(mean_by_layer[layer] / mean_by_layer.max() * 30)
        print(f"  #{rank+1}  Layer {layer:>2}  {bar:<30}  {mean_by_layer[layer]:.4f}")
    print()

    if causal_results:
        print("─" * 60)
        print("  CAUSAL CONFIRMATION")
        print("─" * 60)
        by_layer: dict[int, list[float]] = {}
        for r in causal_results:
            by_layer.setdefault(r["layer"], []).append(r["recovery"])
        for layer in sorted(by_layer):
            mean_rec = np.mean(by_layer[layer])
            print(f"  Layer {layer:>2}  mean recovery: {mean_rec:+.4f}")
        print()

    if transfer_results:
        print("─" * 60)
        print("  TRANSFER TEST — H1 vs H2")
        print("─" * 60)
        restorations = [t["restoration"]["patched_a0_into_a2_disclosed"]
                        for t in transfer_results]
        print(f"  A0→A2 restoration: {sum(restorations)}/{len(restorations)} examples now disclose")

        if all("transfer" in t for t in transfer_results):
            transfers = [t["transfer"]["a1_patched_a2_into_a1_disclosed"]
                         for t in transfer_results]
            baselines = [t["transfer"]["a1_baseline_disclosed"]
                         for t in transfer_results]
            shifts = sum(1 for b, p in zip(baselines, transfers) if b != p)
            print(f"  A2→A1 transfer:   {shifts}/{len(transfers)} examples changed behaviour")
            if len(transfers) > 0:
                if shifts / len(transfers) > 0.4:
                    print("  → CONSISTENT WITH H1 (shared mechanism): A2 patch shifts A1")
                else:
                    print("  → CONSISTENT WITH H2 (distinct mechanism): A2 patch does not shift A1")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Activation patching experiment for concealment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results",     required=True,
                   help="Results JSON from run_interp.py (must have A0 + A2 entries)")
    p.add_argument("--activations", required=True,
                   help="Activations directory (used to verify store loads; "
                        "patching re-runs forward passes on the loaded model)")
    p.add_argument("--model",       required=True,
                   help="HuggingFace model name (must match what produced the results)")
    p.add_argument("--dtype",       default="float32",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model dtype — use float32 on MPS/CPU for autograd stability")
    p.add_argument("--dataset",     default=None,
                   help="Dataset JSONL path (required for --transfer A1 sub-test)")
    p.add_argument("--n-pairs",     type=int, default=None,
                   help="Max A0/A2 pairs to process (None = all)")
    p.add_argument("--causal-top-k", type=int, default=0,
                   help="Run true causal patch on top K attribution layers (slow; 0 = skip)")
    p.add_argument("--transfer",    action="store_true",
                   help="Run transfer test (A0→A2 restoration + A2→A1 H1/H2 test)")
    p.add_argument("--output-dir",  default="results")
    p.add_argument("--no-plot",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Verify inputs ─────────────────────────────────────────────────────
    if not os.path.isfile(args.results):
        print(f"ERROR: results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.activations):
        print(f"ERROR: activations dir not found: {args.activations}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'━' * 62}")
    print(f"  PATCHING EXPERIMENT")
    print(f"  model      : {args.model}  ({args.dtype})")
    print(f"  results    : {args.results}")
    print(f"{'━' * 62}")

    # ── Collect A0/A2 pairs ───────────────────────────────────────────────
    pairs = collect_pairs(args.results)
    if not pairs:
        print("ERROR: no A0/A2 pairs found in results file.", file=sys.stderr)
        print("  Run run_interp.py with --conditions A0,A2 first.", file=sys.stderr)
        sys.exit(1)

    n_to_run = args.n_pairs or len(pairs)
    print(f"\n  Found {len(pairs)} A0/A2 pairs — running {n_to_run}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n  Loading model: {args.model} …")
    client = HFClient(model_name=args.model, dtype=args.dtype, capture_mode="none")

    # Load dataset index if needed
    dataset_index: dict[str, dict] | None = None
    if args.transfer and args.dataset:
        print(f"  Loading dataset index: {args.dataset}")
        import json as _json
        dataset_index = {}
        with open(args.dataset) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = _json.loads(line)
                    dataset_index[s["example_id"]] = s
        print(f"  {len(dataset_index)} scenarios indexed")

    # ── Attribution sweep ─────────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  Phase 1: Attribution sweep ({n_to_run} pairs)")
    print(f"{'─' * 62}")
    per_pair_scores, mean_by_layer = run_attribution_sweep(client, pairs, n_pairs=n_to_run)

    if not per_pair_scores:
        print("ERROR: no valid pairs processed (secret tokenisation failed for all?)",
              file=sys.stderr)
        sys.exit(1)

    top_layers = np.argsort(mean_by_layer)[::-1][:max(args.causal_top_k, 1)].tolist()

    # ── Causal confirmation ───────────────────────────────────────────────
    causal_results: list[dict] = []
    if args.causal_top_k > 0:
        print(f"\n{'─' * 62}")
        print(f"  Phase 2: Causal confirmation (top {args.causal_top_k} layers)")
        print(f"  Layers: {top_layers[:args.causal_top_k]}")
        print(f"{'─' * 62}")
        causal_results = run_causal_confirmation(
            client, pairs[:n_to_run], top_layers[:args.causal_top_k]
        )

    # ── Transfer test ─────────────────────────────────────────────────────
    transfer_results: list[dict] = []
    if args.transfer:
        if dataset_index is None and args.dataset is None:
            print("\n  [WARN] --transfer requires --dataset for the A1 sub-test; "
                  "running restoration-only sub-test.")
        print(f"\n{'─' * 62}")
        print(f"  Phase 3: Transfer test (key layer: {top_layers[0]})")
        print(f"{'─' * 62}")
        transfer_results = run_transfer_test(
            client, pairs[:n_to_run], top_layers, dataset_index
        )

    # ── Save + summarise ──────────────────────────────────────────────────
    out_path = save_results(
        args.output_dir, timestamp, args.model, args.results,
        mean_by_layer, per_pair_scores, causal_results, transfer_results,
    )

    print_summary(mean_by_layer, causal_results, transfer_results)

    if not args.no_plot:
        print("  Generating plots…")
        plot_results(mean_by_layer, per_pair_scores, causal_results,
                     transfer_results, args.output_dir)

    print(f"{'━' * 62}")
    print(f"  Done.")
    print(f"  Results: {out_path}")
    print(f"{'━' * 62}\n")


if __name__ == "__main__":
    main()
