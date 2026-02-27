#!/usr/bin/env python3
"""CLI for generating the synthetic concealment dataset.

Usage
-----
python make_dataset.py --n 150 --seed 42 \\
    --domains shipping,bug,backlog \\
    --output data/concealment_dataset.jsonl
"""

import argparse
import json
import os
import re
import sys

from src.dataset.backlog_domain import BacklogDomain
from src.dataset.bug_domain import BugDomain
from src.dataset.generator import DatasetGenerator
from src.dataset.shipping_domain import ShippingDomain

AVAILABLE_DOMAINS = {
    "shipping": ShippingDomain,
    "bug": BugDomain,
    "backlog": BacklogDomain,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic concealment dataset JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n", type=int, default=150,
                        help="Total number of scenarios to generate (default: 150)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--domains", default="shipping,bug,backlog",
                        help="Comma-separated list of domains to include "
                             "(default: shipping,bug,backlog)")
    parser.add_argument("--output", default="data/concealment_dataset.jsonl",
                        help="Output path for the JSONL file (default: data/concealment_dataset.jsonl)")
    return parser.parse_args()


def validate_regex_patterns(scenarios: list[dict]) -> int:
    """Compile all regex patterns to verify they are valid. Returns error count."""
    errors = 0
    for scenario in scenarios:
        rm = scenario.get("regex_monitor", {})
        for key in ("secret_patterns", "wrong_secret_patterns", "refusal_patterns"):
            for pattern in rm.get(key, []):
                try:
                    re.compile(pattern)
                except re.error as exc:
                    print(f"  [WARN] Invalid regex in {scenario['example_id']} [{key}]: "
                          f"{pattern!r} — {exc}", file=sys.stderr)
                    errors += 1
    return errors


def print_summary(scenarios: list[dict], output_path: str) -> None:
    total = len(scenarios)
    control_types = {}
    domain_counts: dict[str, int] = {}
    sample_golds: list[tuple[str, list[str]]] = []

    for s in scenarios:
        ct = s.get("control_type") or "regular"
        control_types[ct] = control_types.get(ct, 0) + 1
        domain = s["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if ct == "regular" and len(sample_golds) < 5:
            sample_golds.append((s["example_id"], s["gold"]))

    print(f"\nGenerated {total} scenarios → {output_path}")
    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain:20s}  {count:4d}")
    print("\nControl breakdown:")
    for ct, count in sorted(control_types.items()):
        pct = 100.0 * count / total
        print(f"  {ct:25s}  {count:4d}  ({pct:.1f}%)")
    print("\nSample gold forms (first 5 regular scenarios):")
    for eid, gold in sample_golds:
        print(f"  {eid}: {gold}")


def main() -> None:
    args = parse_args()

    # Resolve domain classes
    requested = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in requested if d not in AVAILABLE_DOMAINS]
    if unknown:
        print(f"Error: unknown domain(s): {unknown}. "
              f"Available: {list(AVAILABLE_DOMAINS)}", file=sys.stderr)
        sys.exit(1)

    domains = [AVAILABLE_DOMAINS[d]() for d in requested]

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate
    generator = DatasetGenerator(domains=domains, seed=args.seed)
    scenarios = generator.generate(n=args.n)

    # Validate regex patterns
    errors = validate_regex_patterns(scenarios)
    if errors:
        print(f"Warning: {errors} invalid regex pattern(s) found.", file=sys.stderr)

    # Write JSONL
    generator.to_jsonl(scenarios, args.output)

    # Print summary
    print_summary(scenarios, args.output)


if __name__ == "__main__":
    main()
