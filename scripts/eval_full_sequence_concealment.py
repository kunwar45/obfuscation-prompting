"""Evaluate whether prompt-aligned full-sequence activations better capture concealment.

This script uses saved `full_sequence` activation files, but only compares the
prompt portion of each sequence. Prompt positions are aligned from the right so
that `rel_pos=1` is the final prompt token, `rel_pos=2` the token before that,
and so on.
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


def disclosed(result: dict) -> bool:
    reg = result.get("monitor_results", {}).get("regex", {})
    return bool(reg.get("contains_secret_exact") or reg.get("contains_secret_partial"))


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
    return ids[0] if ids else None


def load_pairs(results_path: str) -> tuple[list[ExamplePair], list[dict]]:
    with open(results_path) as f:
        raw = json.load(f)
    rows = raw.get("results", raw) if isinstance(raw, dict) else raw

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        meta = row.get("metadata", {})
        if meta.get("control_type") is not None or meta.get("query_type") != "B1":
            continue
        eid = meta.get("example_id")
        cond = meta.get("condition")
        if eid and cond in ("A0", "A2"):
            grouped[eid][cond] = row

    pairs = []
    for eid, cmap in sorted(grouped.items()):
        if "A0" in cmap and "A2" in cmap:
            pairs.append(ExamplePair(eid, cmap["A0"], cmap["A2"]))
    return pairs, rows


def build_prompt_input_ids(client: HFClient, result: dict) -> torch.Tensor:
    meta = result.get("metadata", {})
    messages = [
        {"role": "system", "content": meta.get("system_prompt", "")},
        {"role": "user", "content": result.get("prompt", "")},
    ]
    prompt = client._apply_chat_template(messages)
    toks = client.tokenizer(prompt, return_tensors="pt")
    return toks.input_ids.to(client.model.device)


def load_prompt_aligned_position(
    store: ActivationStore,
    prompt_id: str,
    layer_idx: int,
    prompt_len: int,
    rel_pos: int,
) -> np.ndarray:
    data = store.load(prompt_id)
    arr = data[f"layer_{layer_idx}"]
    prompt_arr = arr[:prompt_len]
    return prompt_arr[-rel_pos]


def mean_pairwise_cosine(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)
    vals = []
    for i in range(len(xn)):
        for j in range(i + 1, len(xn)):
            vals.append(float(np.dot(xn[i], xn[j])))
    return float(np.mean(vals)) if vals else float("nan")


def capture_prompt_position_vector(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    rel_pos: int,
) -> torch.Tensor:
    box: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        box["v"] = hs[0, -rel_pos, :].detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids, return_dict=True)
    finally:
        handle.remove()
    return box["v"]


def token_probability(logits: torch.Tensor, token_id: int) -> float:
    probs = torch.softmax(logits[0, -1], dim=-1)
    return float(probs[token_id].item())


def run_with_patch(
    model,
    input_ids: torch.Tensor,
    layer_idx: int,
    rel_pos: int,
    dims: np.ndarray,
    replacement: torch.Tensor,
    token_id: int,
) -> float:
    dims_t = torch.tensor(dims, device=model.device, dtype=torch.long)
    repl = replacement.to(model.device)

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[0, -rel_pos, dims_t] = repl[dims_t].to(hs.dtype)
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
    rel_pos: int,
    dims: np.ndarray,
    token_id: int,
) -> float:
    dims_t = torch.tensor(dims, device=model.device, dtype=torch.long)

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs.clone()
        hs[0, -rel_pos, dims_t] = 0.0
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
    prompt_lens: dict[str, int],
    top_k_dims: int,
    max_rel_pos: int,
) -> dict:
    success_ids = [p.a2["prompt_id"] for p in pairs if disclosed(p.a0) and not disclosed(p.a2)]
    other_ids = [
        r["prompt_id"]
        for r in store.get_results()
        if r["prompt_id"] not in set(success_ids)
        and r.get("metadata", {}).get("control_type") is None
        and r.get("metadata", {}).get("query_type") == "B1"
    ]
    if len(success_ids) < 2:
        raise ValueError("Need at least 2 successful concealment examples.")

    max_valid_rel = min(min(prompt_lens[pid] for pid in success_ids), max_rel_pos)
    best = None

    for layer_idx in range(store.n_layers()):
        for rel_pos in range(1, max_valid_rel + 1):
            xs = np.stack([
                load_prompt_aligned_position(store, pid, layer_idx, prompt_lens[pid], rel_pos)
                for pid in success_ids
            ], axis=0)
            xo = np.stack([
                load_prompt_aligned_position(store, pid, layer_idx, prompt_lens[pid], rel_pos)
                for pid in other_ids
                if prompt_lens[pid] >= rel_pos
            ], axis=0)
            if len(xo) < 2:
                continue

            ms = xs.mean(axis=0)
            mo = xo.mean(axis=0)
            effect = (ms - mo) / np.sqrt((xs.var(axis=0) + xo.var(axis=0)) / 2.0 + EPS)
            direction = np.sign(effect)
            direction[direction == 0] = 1
            aligned = (np.sign((xs - mo) * direction) >= 0).all(axis=0)
            candidate_idx = np.where(aligned)[0]
            if candidate_idx.size == 0:
                continue
            order = np.argsort(-np.abs(effect[candidate_idx]))
            take = candidate_idx[order[:top_k_dims]]
            score = float(np.mean(np.abs(effect[take])))

            payload = {
                "layer": layer_idx,
                "rel_pos": rel_pos,
                "dims": take.tolist(),
                "signed_effect": effect[take].tolist(),
                "layer_pos_score": score,
                "success_pairwise_cosine": mean_pairwise_cosine(xs[:, take]),
                "other_pairwise_cosine": mean_pairwise_cosine(xo[:, take]),
            }
            if best is None or payload["layer_pos_score"] > best["layer_pos_score"]:
                best = payload

    if best is None:
        raise ValueError("No full-sequence candidate found.")
    return best


def uniqueness_summary(
    store: ActivationStore,
    prompt_lens: dict[str, int],
    layer_idx: int,
    rel_pos: int,
    dims: list[int],
    pairs: list[ExamplePair],
) -> dict:
    success_ids = [p.a2["prompt_id"] for p in pairs if disclosed(p.a0) and not disclosed(p.a2)]
    a2_fail_ids = [p.a2["prompt_id"] for p in pairs if disclosed(p.a2)]
    a0_disclose_ids = [p.a0["prompt_id"] for p in pairs if disclosed(p.a0)]

    def stack(ids: list[str]) -> np.ndarray:
        kept = [pid for pid in ids if prompt_lens[pid] >= rel_pos]
        return np.stack([
            load_prompt_aligned_position(store, pid, layer_idx, prompt_lens[pid], rel_pos)
            for pid in kept
        ], axis=0)[:, dims]

    xs = stack(success_ids)
    xf = stack(a2_fail_ids)
    xa0 = stack(a0_disclose_ids)

    sign = np.sign(xs.mean(axis=0))
    sign[sign == 0] = 1

    def score(arr: np.ndarray) -> np.ndarray:
        return (arr * sign).mean(axis=1)

    return {
        "success_n": len(xs),
        "a2_fail_n": len(xf),
        "a0_disclose_n": len(xa0),
        "success_score_mean": float(score(xs).mean()),
        "a2_fail_score_mean": float(score(xf).mean()),
        "a0_disclose_score_mean": float(score(xa0).mean()),
        "success_score_std": float(score(xs).std()),
        "a2_fail_score_std": float(score(xf).std()),
        "a0_disclose_score_std": float(score(xa0).std()),
    }


def run_causal_tests(
    client: HFClient,
    pairs: list[ExamplePair],
    layer_idx: int,
    rel_pos: int,
    dims: list[int],
) -> dict:
    dims_arr = np.array(dims, dtype=np.int64)
    success_pairs = [p for p in pairs if disclosed(p.a0) and not disclosed(p.a2)]

    patch_a0_deltas = []
    reverse_patch_a2_deltas = []
    full_patch_a0_deltas = []
    full_reverse_patch_a2_deltas = []
    zero_ablate_a2_deltas = []
    per_example = []

    for pair in success_pairs:
        secret_text = pair.a0.get("metadata", {}).get("keyword_hints", [""])[0]
        token_id = find_secret_token_id(client.tokenizer, secret_text)
        if token_id is None:
            continue

        a0_ids = build_prompt_input_ids(client, pair.a0)
        a2_ids = build_prompt_input_ids(client, pair.a2)
        if a0_ids.shape[1] < rel_pos or a2_ids.shape[1] < rel_pos:
            continue

        a0_vec = capture_prompt_position_vector(client.model, a0_ids, layer_idx, rel_pos)
        a2_vec = capture_prompt_position_vector(client.model, a2_ids, layer_idx, rel_pos)

        with torch.no_grad():
            a0_out = client.model(a0_ids, return_dict=True)
            a2_out = client.model(a2_ids, return_dict=True)

        a0_base = token_probability(a0_out.logits, token_id)
        a2_base = token_probability(a2_out.logits, token_id)

        a0_patch = run_with_patch(client.model, a0_ids, layer_idx, rel_pos, dims_arr, a2_vec, token_id)
        a2_rev = run_with_patch(client.model, a2_ids, layer_idx, rel_pos, dims_arr, a0_vec, token_id)
        a0_full_patch = run_with_patch(
            client.model, a0_ids, layer_idx, rel_pos, np.arange(a0_vec.shape[0]), a2_vec, token_id
        )
        a2_full_rev = run_with_patch(
            client.model, a2_ids, layer_idx, rel_pos, np.arange(a0_vec.shape[0]), a0_vec, token_id
        )
        a2_zero = run_with_zero_ablation(client.model, a2_ids, layer_idx, rel_pos, dims_arr, token_id)

        patch_a0_deltas.append(a0_patch - a0_base)
        reverse_patch_a2_deltas.append(a2_rev - a2_base)
        full_patch_a0_deltas.append(a0_full_patch - a0_base)
        full_reverse_patch_a2_deltas.append(a2_full_rev - a2_base)
        zero_ablate_a2_deltas.append(a2_zero - a2_base)

        per_example.append({
            "example_id": pair.example_id,
            "a0_base_prob": a0_base,
            "a0_patch_prob": a0_patch,
            "a0_full_patch_prob": a0_full_patch,
            "a2_base_prob": a2_base,
            "a2_reverse_patch_prob": a2_rev,
            "a2_full_reverse_patch_prob": a2_full_rev,
            "a2_zero_ablate_prob": a2_zero,
        })

    return {
        "n_tested": len(per_example),
        "patch_a0_with_a2_mean_delta": float(np.mean(patch_a0_deltas)) if patch_a0_deltas else float("nan"),
        "reverse_patch_a2_with_a0_mean_delta": float(np.mean(reverse_patch_a2_deltas)) if reverse_patch_a2_deltas else float("nan"),
        "full_patch_a0_with_a2_mean_delta": float(np.mean(full_patch_a0_deltas)) if full_patch_a0_deltas else float("nan"),
        "full_reverse_patch_a2_with_a0_mean_delta": float(np.mean(full_reverse_patch_a2_deltas)) if full_reverse_patch_a2_deltas else float("nan"),
        "zero_ablate_a2_mean_delta": float(np.mean(zero_ablate_a2_deltas)) if zero_ablate_a2_deltas else float("nan"),
        "per_example": per_example,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate full-sequence prompt-aligned concealment signal")
    ap.add_argument("--results", required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--top-k-dims", type=int, default=8)
    ap.add_argument("--max-rel-pos", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    store = ActivationStore(args.activations, args.results)
    pairs, rows = load_pairs(args.results)
    client = HFClient(model_name=args.model, dtype=args.dtype, capture_mode="none")

    prompt_lens = {}
    for row in rows:
        if row.get("metadata", {}).get("control_type") is None and row.get("metadata", {}).get("query_type") == "B1":
            prompt_lens[row["prompt_id"]] = int(build_prompt_input_ids(client, row).shape[1])

    candidate = identify_candidates(store, pairs, prompt_lens, args.top_k_dims, args.max_rel_pos)
    unique = uniqueness_summary(
        store, prompt_lens, candidate["layer"], candidate["rel_pos"], candidate["dims"], pairs
    )
    causal = run_causal_tests(client, pairs, candidate["layer"], candidate["rel_pos"], candidate["dims"])

    payload = {
        "selected_layer": candidate["layer"],
        "selected_rel_pos": candidate["rel_pos"],
        "selected_dims": candidate["dims"],
        "selected_effects": candidate["signed_effect"],
        "consistency": {
            "success_pairwise_cosine": candidate["success_pairwise_cosine"],
            "other_pairwise_cosine": candidate["other_pairwise_cosine"],
            "layer_pos_score": candidate["layer_pos_score"],
        },
        "uniqueness": unique,
        "causal_tests": causal,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved analysis to {args.out}")


if __name__ == "__main__":
    main()
