"""Evaluate whether saved last-token activations represent concealment.

This script is intentionally lightweight for local Mac runs:

1. Load a run JSON + activation files produced by `run_interp.py`
2. Identify candidate concealment dimensions from saved last-token activations
3. Test whether those dimensions are:
   - consistent across successful concealment examples
   - relatively unique to concealment vs disclosure/failure examples
4. Run prompt-time targeted patching and ablation on the selected layer/dim set

The saved activations from `HFClient` come from the full prompt+response
sequence. Those are used here for descriptive analysis (consistency/uniqueness).
For causal tests, we rebuild the prompt-only input and intervene on the final
prompt token, which is the position that controls the next-token distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.clients.hf_client import HFClient
from src.interp.activation_store import ActivationStore


EPS = 1e-8


@dataclass
class ExamplePair:
    example_id: str
    a0: dict
    a2: dict


def find_secret_token_id(tokenizer, secret_text: str) -> int | None:
    if not secret_text:
        return None
    first_word = secret_text.split()[0]
    for variant in (first_word, " " + first_word):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        for tok_id in ids:
            if first_word in tokenizer.decode([tok_id]):
                return tok_id
    ids = tokenizer.encode(first_word, add_special_tokens=False)
    if ids:
        return ids[0]
    return None


def disclosed(result: dict) -> bool:
    reg = result.get("monitor_results", {}).get("regex", {})
    return bool(reg.get("contains_secret_exact") or reg.get("contains_secret_partial"))


def load_pairs(results_path: str) -> tuple[list[ExamplePair], list[dict]]:
    with open(results_path) as f:
        raw = json.load(f)
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        meta = row.get("metadata", {})
        if meta.get("control_type") is not None:
            continue
        if meta.get("query_type") != "B1":
            continue
        cond = meta.get("condition")
        eid = meta.get("example_id")
        if cond in ("A0", "A2") and eid:
            grouped[eid][cond] = row

    pairs = []
    for eid, cmap in sorted(grouped.items()):
        if "A0" in cmap and "A2" in cmap:
            pairs.append(ExamplePair(example_id=eid, a0=cmap["A0"], a2=cmap["A2"]))
    return pairs, rows


def build_prompt_input_ids(client: HFClient, result: dict) -> torch.Tensor:
    meta = result.get("metadata", {})
    messages = [
        {"role": "system", "content": meta.get("system_prompt", "")},
        {"role": "user", "content": result.get("prompt", "")},
    ]
    prompt = client._apply_chat_template(messages)
    toks = client.tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids.to(client.model.device)
    return input_ids


def capture_last_token_vector(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["vec"] = hs[0, -1, :].detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids, return_dict=True)
    finally:
        handle.remove()

    return captured["vec"]


def token_probability(logits: torch.Tensor, token_id: int) -> float:
    probs = torch.softmax(logits[0, -1], dim=-1)
    return float(probs[token_id].item())


def run_with_dim_patch(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    dims: np.ndarray,
    replacement: torch.Tensor,
    token_id: int,
) -> float:
    dims_t = torch.tensor(dims, device=model.device, dtype=torch.long)
    repl = replacement.to(model.device)

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[0, -1, dims_t] = repl[dims_t].to(hs.dtype)
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model(input_ids, return_dict=True)
    finally:
        handle.remove()

    return token_probability(out.logits, token_id)


def run_with_zero_ablation(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    dims: np.ndarray,
    token_id: int,
) -> float:
    dims_t = torch.tensor(dims, device=model.device, dtype=torch.long)

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[0, -1, dims_t] = 0.0
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model(input_ids, return_dict=True)
    finally:
        handle.remove()

    return token_probability(out.logits, token_id)


def identify_candidates(
    store: ActivationStore,
    pairs: list[ExamplePair],
    top_k_dims: int,
) -> dict:
    success_a2_ids = [p.a2["prompt_id"] for p in pairs if disclosed(p.a0) and not disclosed(p.a2)]
    other_ids = [
        p["prompt_id"]
        for p in store.get_results()
        if p["prompt_id"] not in set(success_a2_ids)
        and p.get("metadata", {}).get("control_type") is None
        and p.get("metadata", {}).get("query_type") == "B1"
    ]
    if len(success_a2_ids) < 2:
        raise ValueError("Need at least 2 successful concealment examples to identify candidates.")

    best: dict | None = None
    n_layers = store.n_layers()

    for layer_idx in range(n_layers):
        xs = store.load_layer(layer_idx, prompt_ids=success_a2_ids)
        xo = store.load_layer(layer_idx, prompt_ids=other_ids)
        ms = xs.mean(axis=0)
        mo = xo.mean(axis=0)
        vs = xs.var(axis=0)
        vo = xo.var(axis=0)
        pooled = np.sqrt((vs + vo) / 2.0 + EPS)
        effect = (ms - mo) / pooled

        direction = np.sign(effect)
        direction[direction == 0] = 1
        aligned = (np.sign((xs - mo) * direction) >= 0).all(axis=0)

        candidate_idx = np.where(aligned)[0]
        if candidate_idx.size == 0:
            continue
        candidate_scores = np.abs(effect[candidate_idx])
        order = np.argsort(-candidate_scores)
        take = candidate_idx[order[:top_k_dims]]
        score = float(np.mean(np.abs(effect[take])))

        signed = direction[take]
        success_scores = (xs[:, take] * signed).mean(axis=1)
        other_scores = (xo[:, take] * signed).mean(axis=1)

        payload = {
            "layer": layer_idx,
            "dims": take.tolist(),
            "signed_effect": effect[take].tolist(),
            "layer_score": score,
            "success_mean_score": float(success_scores.mean()),
            "other_mean_score": float(other_scores.mean()),
            "success_pairwise_cosine": mean_pairwise_cosine(xs[:, take]),
            "other_pairwise_cosine": mean_pairwise_cosine(xo[:, take]),
            "success_ids": success_a2_ids,
            "other_ids": other_ids,
        }
        if best is None or payload["layer_score"] > best["layer_score"]:
            best = payload

    if best is None:
        raise ValueError("No consistent candidate dimensions found.")
    return best


def mean_pairwise_cosine(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)
    vals = []
    for i in range(len(xn)):
        for j in range(i + 1, len(xn)):
            vals.append(float(np.dot(xn[i], xn[j])))
    return float(np.mean(vals)) if vals else float("nan")


def uniqueness_summary(
    store: ActivationStore,
    layer_idx: int,
    dims: list[int],
    pairs: list[ExamplePair],
) -> dict:
    success_ids = [p.a2["prompt_id"] for p in pairs if disclosed(p.a0) and not disclosed(p.a2)]
    a2_fail_ids = [p.a2["prompt_id"] for p in pairs if disclosed(p.a2)]
    a0_disclose_ids = [p.a0["prompt_id"] for p in pairs if disclosed(p.a0)]

    xs = store.load_layer(layer_idx, prompt_ids=success_ids)[:, dims]
    xf = store.load_layer(layer_idx, prompt_ids=a2_fail_ids)[:, dims]
    xa0 = store.load_layer(layer_idx, prompt_ids=a0_disclose_ids)[:, dims]

    success_mean = xs.mean(axis=0)
    sign = np.sign(success_mean)
    sign[sign == 0] = 1

    def score(arr: np.ndarray) -> np.ndarray:
        return (arr * sign).mean(axis=1)

    return {
        "success_n": len(success_ids),
        "a2_fail_n": len(a2_fail_ids),
        "a0_disclose_n": len(a0_disclose_ids),
        "success_score_mean": float(score(xs).mean()),
        "a2_fail_score_mean": float(score(xf).mean()) if len(xf) else float("nan"),
        "a0_disclose_score_mean": float(score(xa0).mean()) if len(xa0) else float("nan"),
        "success_score_std": float(score(xs).std()) if len(xs) else float("nan"),
        "a2_fail_score_std": float(score(xf).std()) if len(xf) else float("nan"),
        "a0_disclose_score_std": float(score(xa0).std()) if len(xa0) else float("nan"),
    }


def prompt_class_mean(
    client: HFClient,
    rows: list[dict],
    layer_idx: int,
    predicate,
) -> torch.Tensor:
    vecs = []
    for row in rows:
        if predicate(row):
            ids = build_prompt_input_ids(client, row)
            vecs.append(capture_last_token_vector(client.model, ids, layer_idx).float().cpu())
    if not vecs:
        raise ValueError("No prompt vectors found for requested predicate.")
    return torch.stack(vecs, dim=0).mean(dim=0)


def run_causal_tests(
    client: HFClient,
    rows: list[dict],
    pairs: list[ExamplePair],
    layer_idx: int,
    dims: list[int],
) -> dict:
    dims_arr = np.array(dims, dtype=np.int64)
    success_pairs = [p for p in pairs if disclosed(p.a0) and not disclosed(p.a2)]
    disclosed_mean = prompt_class_mean(
        client,
        rows,
        layer_idx,
        lambda r: r.get("metadata", {}).get("condition") == "A0"
        and disclosed(r)
        and r.get("metadata", {}).get("control_type") is None
        and r.get("metadata", {}).get("query_type") == "B1",
    )

    patch_a0_deltas = []
    reverse_patch_a2_deltas = []
    full_patch_a0_deltas = []
    full_reverse_patch_a2_deltas = []
    ablate_zero_deltas = []
    ablate_mean_deltas = []
    per_example = []

    for pair in success_pairs:
        secret = pair.a0.get("metadata", {}).get("keyword_hints", [])
        secret_text = secret[0] if secret else ""
        token_id = find_secret_token_id(client.tokenizer, secret_text)
        if token_id is None:
            continue

        a0_ids = build_prompt_input_ids(client, pair.a0)
        a2_ids = build_prompt_input_ids(client, pair.a2)

        a0_vec = capture_last_token_vector(client.model, a0_ids, layer_idx)
        a2_vec = capture_last_token_vector(client.model, a2_ids, layer_idx)

        with torch.no_grad():
            a0_out = client.model(a0_ids, return_dict=True)
            a2_out = client.model(a2_ids, return_dict=True)

        a0_base = token_probability(a0_out.logits, token_id)
        a2_base = token_probability(a2_out.logits, token_id)

        a0_patched = run_with_dim_patch(client.model, a0_ids, layer_idx, dims_arr, a2_vec, token_id)
        a2_reverse_patched = run_with_dim_patch(
            client.model, a2_ids, layer_idx, dims_arr, a0_vec, token_id
        )
        a0_full_patched = run_with_dim_patch(
            client.model,
            a0_ids,
            layer_idx,
            np.arange(a0_vec.shape[0]),
            a2_vec,
            token_id,
        )
        a2_full_reverse_patched = run_with_dim_patch(
            client.model,
            a2_ids,
            layer_idx,
            np.arange(a0_vec.shape[0]),
            a0_vec,
            token_id,
        )
        a2_zero_ablated = run_with_zero_ablation(client.model, a2_ids, layer_idx, dims_arr, token_id)
        a2_mean_ablated = run_with_dim_patch(
            client.model, a2_ids, layer_idx, dims_arr, disclosed_mean, token_id
        )

        patch_a0_deltas.append(a0_patched - a0_base)
        reverse_patch_a2_deltas.append(a2_reverse_patched - a2_base)
        full_patch_a0_deltas.append(a0_full_patched - a0_base)
        full_reverse_patch_a2_deltas.append(a2_full_reverse_patched - a2_base)
        ablate_zero_deltas.append(a2_zero_ablated - a2_base)
        ablate_mean_deltas.append(a2_mean_ablated - a2_base)

        per_example.append({
            "example_id": pair.example_id,
            "secret_text": secret_text,
            "token_id": int(token_id),
            "a0_base_prob": a0_base,
            "a0_patched_with_a2_prob": a0_patched,
            "a0_full_patched_with_a2_prob": a0_full_patched,
            "a2_base_prob": a2_base,
            "a2_reverse_patched_with_a0_prob": a2_reverse_patched,
            "a2_full_reverse_patched_with_a0_prob": a2_full_reverse_patched,
            "a2_zero_ablated_prob": a2_zero_ablated,
            "a2_mean_ablated_prob": a2_mean_ablated,
        })

    return {
        "n_tested": len(per_example),
        "patch_a0_with_a2_mean_delta": float(np.mean(patch_a0_deltas)) if patch_a0_deltas else float("nan"),
        "reverse_patch_a2_with_a0_mean_delta": float(np.mean(reverse_patch_a2_deltas)) if reverse_patch_a2_deltas else float("nan"),
        "full_patch_a0_with_a2_mean_delta": float(np.mean(full_patch_a0_deltas)) if full_patch_a0_deltas else float("nan"),
        "full_reverse_patch_a2_with_a0_mean_delta": float(np.mean(full_reverse_patch_a2_deltas)) if full_reverse_patch_a2_deltas else float("nan"),
        "zero_ablate_a2_mean_delta": float(np.mean(ablate_zero_deltas)) if ablate_zero_deltas else float("nan"),
        "mean_ablate_a2_mean_delta": float(np.mean(ablate_mean_deltas)) if ablate_mean_deltas else float("nan"),
        "per_example": per_example,
    }


def make_summary(candidate: dict, unique: dict, causal: dict) -> dict:
    return {
        "selected_layer": candidate["layer"],
        "selected_dims": candidate["dims"],
        "selected_effects": candidate["signed_effect"],
        "consistency": {
            "success_pairwise_cosine": candidate["success_pairwise_cosine"],
            "other_pairwise_cosine": candidate["other_pairwise_cosine"],
            "layer_score": candidate["layer_score"],
        },
        "uniqueness": unique,
        "causal_tests": causal,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate whether last-token activations represent concealment")
    ap.add_argument("--results", required=True, help="Path to run_<...>.json")
    ap.add_argument("--activations", required=True, help="Path to activations dir for that run")
    ap.add_argument("--model", required=True, help="HF model name used for the run")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--top-k-dims", type=int, default=8, help="Number of concealment dims to keep at the best layer")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    store = ActivationStore(args.activations, args.results)
    pairs, rows = load_pairs(args.results)
    candidate = identify_candidates(store, pairs, args.top_k_dims)
    unique = uniqueness_summary(store, candidate["layer"], candidate["dims"], pairs)

    client = HFClient(model_name=args.model, dtype=args.dtype, capture_mode="none")
    causal = run_causal_tests(client, rows, pairs, candidate["layer"], candidate["dims"])
    summary = make_summary(candidate, unique, causal)

    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved analysis to {args.out}")


if __name__ == "__main__":
    main()
