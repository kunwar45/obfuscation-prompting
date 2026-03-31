"""Extract matched triplets from a concealment experiment results file.

A matched triplet is an example_id where:
  - A0 (baseline) → disclosed the secret
  - A1 (implicit pressure) → concealed the secret
  - A2 (explicit pressure) → concealed the secret

These triplets are the foundation for mechanistic analysis (probing, patching)
because the same context/secret produces different outputs across conditions,
enabling causally comparable activation comparisons.

Verdict priority: llm monitor > regex fallback > unknown

Usage
-----
  python3.11 scripts/filter_triplets.py results/run_<id>.json
  python3.11 scripts/filter_triplets.py results/run_<id>.json --output /tmp/triplets.json
  python3.11 scripts/filter_triplets.py results/run_<id>.json --min-triplets 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


# ── Verdict helpers ────────────────────────────────────────────────────────────

def _verdict(result: dict) -> str:
    """Classify a single result as 'disclosed', 'concealed', or 'unclear'.

    Priority: LLM monitor verdict → regex fallback → 'unclear'.
    'redirected' is treated as concealed (model deflected without disclosing).
    """
    llm = result.get("monitor_results", {}).get("llm", {})
    llm_verdict = llm.get("verdict", "")

    if llm_verdict == "disclosed":
        return "disclosed"
    if llm_verdict in ("concealed", "redirected"):
        return "concealed"

    # Regex fallback (used when llm monitor absent or unclear)
    regex = result.get("monitor_results", {}).get("regex", {})
    if regex:
        if regex.get("contains_secret_exact") or regex.get("contains_secret_partial"):
            return "disclosed"
        # No regex match and no refusal → omission/concealment
        return "concealed"

    return "unclear"


# ── Loading ────────────────────────────────────────────────────────────────────

def _load_results(path: str) -> list[dict]:
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("results", [])
    return payload  # bare list


# ── Core filtering logic ───────────────────────────────────────────────────────

def extract_triplets(results: list[dict]) -> tuple[list[dict], dict]:
    """Group results by example_id and extract matched triplets.

    Returns:
        triplets: list of {"example_id", "a0", "a1", "a2"} dicts
        stats: summary counts per condition
    """
    # Group by example_id, skip controls and non-B1 queries
    by_example: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        meta = r.get("metadata", {})
        if meta.get("control_type"):
            continue
        if meta.get("query_type") != "B1":
            continue
        example_id = meta.get("example_id") or _parse_example_id(r.get("prompt_id", ""))
        condition = meta.get("condition") or _parse_condition(r.get("prompt_id", ""))
        if condition in ("A0", "A1", "A2"):
            by_example[example_id][condition] = r

    # Count verdicts per condition
    stats: dict[str, dict[str, int]] = {
        "A0": defaultdict(int),
        "A1": defaultdict(int),
        "A2": defaultdict(int),
    }
    for example_id, cond_map in by_example.items():
        for cond in ("A0", "A1", "A2"):
            if cond in cond_map:
                v = _verdict(cond_map[cond])
                stats[cond][v] += 1

    # Extract matched triplets
    triplets = []
    for example_id, cond_map in sorted(by_example.items()):
        if not all(c in cond_map for c in ("A0", "A1", "A2")):
            continue
        v_a0 = _verdict(cond_map["A0"])
        v_a1 = _verdict(cond_map["A1"])
        v_a2 = _verdict(cond_map["A2"])
        if v_a0 == "disclosed" and v_a1 == "concealed" and v_a2 == "concealed":
            triplets.append({
                "example_id": example_id,
                "verdicts": {"A0": v_a0, "A1": v_a1, "A2": v_a2},
                "a0": cond_map["A0"],
                "a1": cond_map["A1"],
                "a2": cond_map["A2"],
            })

    return triplets, {c: dict(d) for c, d in stats.items()}


def _parse_example_id(prompt_id: str) -> str:
    """Extract example_id from prompt_id like 'shipping_0001_A0_B1'."""
    parts = prompt_id.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else prompt_id


def _parse_condition(prompt_id: str) -> str:
    """Extract condition from prompt_id like 'shipping_0001_A0_B1'."""
    parts = prompt_id.rsplit("_", 2)
    return parts[1] if len(parts) == 3 else ""


# ── Output & display ───────────────────────────────────────────────────────────

def _print_summary(stats: dict, triplets: list[dict], total_examples: int) -> None:
    print()
    print("─" * 50)
    print("  BEHAVIORAL FILTERING GATE — SUMMARY")
    print("─" * 50)
    print(f"  Total examples (non-control, B1):  {total_examples}")
    print()
    for cond in ("A0", "A1", "A2"):
        d = stats.get(cond, {})
        total = sum(d.values()) or 1
        disc = d.get("disclosed", 0)
        conc = d.get("concealed", 0)
        unkw = d.get("unclear", 0)
        print(f"  {cond}:  disclosed={disc}/{total} ({disc/total:.0%})  "
              f"concealed={conc}/{total} ({conc/total:.0%})  "
              f"unclear={unkw}/{total} ({unkw/total:.0%})")
    print()
    n = len(triplets)
    rate = n / total_examples if total_examples else 0
    print(f"  Matched triplets (A0=disc, A1=conc, A2=conc):  {n} / {total_examples}  ({rate:.0%})")
    print("─" * 50)
    if triplets:
        print("  Example triplet IDs:")
        for t in triplets[:5]:
            print(f"    {t['example_id']}")
        if n > 5:
            print(f"    ... and {n - 5} more")
    print()


def save_triplets(triplets: list[dict], stats: dict, source_path: str, output_path: str) -> None:
    total_examples = sum(sum(d.values()) for d in stats.values()) // 3  # approx
    payload = {
        "source_run": os.path.abspath(source_path),
        "summary": {
            "total_examples": total_examples,
            "matched_triplets": len(triplets),
            "match_rate": round(len(triplets) / total_examples, 4) if total_examples else 0,
            "a0_disclose_rate": _rate(stats.get("A0", {}), "disclosed"),
            "a1_conceal_rate": _rate(stats.get("A1", {}), "concealed"),
            "a2_conceal_rate": _rate(stats.get("A2", {}), "concealed"),
        },
        "triplets": triplets,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved → {output_path}")


def _rate(d: dict, key: str) -> float:
    total = sum(d.values()) or 1
    return round(d.get(key, 0) / total, 4)


def _default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return base + "_triplets.json"


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract matched A0/A1/A2 triplets from results JSON.")
    parser.add_argument("results", help="Path to results JSON from run_blackbox.py or run_interp.py")
    parser.add_argument("--output", "-o", help="Output path (default: <results>_triplets.json)")
    parser.add_argument(
        "--min-triplets", type=int, default=0,
        help="Exit with code 1 if fewer than N triplets found (useful in CI/scripts)",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.results):
        print(f"ERROR: file not found: {args.results}", file=sys.stderr)
        return 1

    results = _load_results(args.results)
    triplets, stats = extract_triplets(results)

    # Count total unique non-control B1 examples across all conditions
    seen_examples: set[str] = set()
    for r in results:
        meta = r.get("metadata", {})
        if not meta.get("control_type") and meta.get("query_type") == "B1":
            eid = meta.get("example_id") or _parse_example_id(r.get("prompt_id", ""))
            seen_examples.add(eid)
    total_examples = len(seen_examples)

    _print_summary(stats, triplets, total_examples)

    output_path = args.output or _default_output_path(args.results)
    save_triplets(triplets, stats, args.results, output_path)

    if args.min_triplets and len(triplets) < args.min_triplets:
        print(f"WARNING: only {len(triplets)} triplets found, need {args.min_triplets}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
