#!/usr/bin/env python3
"""CLI for generating the synthetic concealment dataset.

Basic usage
-----------
python make_dataset.py --n 150 --seed 42 \\
    --domains shipping,bug,backlog \\
    --output data/concealment_dataset.jsonl

With LLM augmentation
---------------------
python make_dataset.py --n 150 --seed 42 \\
    --augment-paraphrases 3 \\
    --augment-context 2 \\
    --augment-model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \\
    --output data/concealment_dataset.jsonl

Augmentation requires TOGETHER_API_KEY in the environment or .env file.
Augmented fields added to each (non-control) scenario:
  queries["B1_llm_paraphrases"]  — LLM query paraphrases (if --augment-paraphrases N > 0)
  context_variants               — LLM context rewrites  (if --augment-context N > 0)
These are automatically picked up by ConcealmentLoader when query_types includes
B1_llm_p0, B1_llm_p1, ... or when context variants are present.
"""

import argparse
import os
import re
import sys

from dotenv import load_dotenv

from src.dataset.backlog_domain import BacklogDomain
from src.dataset.bug_domain import BugDomain
from src.dataset.generator import DatasetGenerator
from src.dataset.shipping_domain import ShippingDomain

load_dotenv()

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
                        help="Comma-separated list of domains (default: shipping,bug,backlog)")
    parser.add_argument("--output", default="data/concealment_dataset.jsonl",
                        help="Output JSONL path (default: data/concealment_dataset.jsonl)")

    # Augmentation
    aug = parser.add_argument_group("LLM augmentation (requires TOGETHER_API_KEY)")
    aug.add_argument("--augment-paraphrases", type=int, default=0, metavar="N",
                     help="Generate N additional LLM query paraphrases per scenario (0 = off)")
    aug.add_argument("--augment-context", type=int, default=0, metavar="N",
                     help="Generate N LLM context rewrites per scenario (0 = off)")
    aug.add_argument("--augment-model",
                     default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                     help="Together AI model to use for augmentation")
    aug.add_argument("--augment-temperature", type=float, default=0.9,
                     help="Sampling temperature for augmentation calls (default: 0.9)")

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
    control_types: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    sample_golds: list[tuple[str, list[str]]] = []
    aug_paraphrase_count = 0
    aug_context_count = 0

    for s in scenarios:
        ct = s.get("control_type") or "regular"
        control_types[ct] = control_types.get(ct, 0) + 1
        domain = s["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if ct == "regular" and len(sample_golds) < 5:
            sample_golds.append((s["example_id"], s["gold"]))
        aug_paraphrase_count += len(s.get("queries", {}).get("B1_llm_paraphrases", []))
        aug_context_count += len(s.get("context_variants", []))

    print(f"\nGenerated {total} scenarios → {output_path}")
    print("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain:20s}  {count:4d}")
    print("\nControl breakdown:")
    for ct, count in sorted(control_types.items()):
        pct = 100.0 * count / total
        print(f"  {ct:25s}  {count:4d}  ({pct:.1f}%)")
    if aug_paraphrase_count or aug_context_count:
        print("\nLLM augmentation totals:")
        print(f"  B1_llm_paraphrases:  {aug_paraphrase_count}")
        print(f"  context_variants:    {aug_context_count}")
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

    # LLM augmentation
    want_augmentation = args.augment_paraphrases > 0 or args.augment_context > 0
    if want_augmentation:
        api_key = os.getenv("TOGETHER_API_KEY", "")
        if not api_key:
            print(
                "Error: TOGETHER_API_KEY is not set. "
                "Cannot run LLM augmentation without an API key.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Late imports so the script works without Together installed when not augmenting
        from src.clients.together_client import TogetherClient
        from src.dataset.augmenter import LLMAugmenter

        client = TogetherClient(api_key=api_key)
        augmenter = LLMAugmenter(
            client=client,
            model=args.augment_model,
            temperature=args.augment_temperature,
        )

        regular = [s for s in scenarios if s.get("control_type") is None]
        print(
            f"Augmenting {len(regular)} regular scenario(s) "
            f"(paraphrases={args.augment_paraphrases}, context={args.augment_context}) "
            f"via {args.augment_model}..."
        )
        augmenter.augment_batch(
            scenarios,
            n_paraphrases=args.augment_paraphrases,
            n_contexts=args.augment_context,
            verbose=True,
        )

    # Write JSONL
    generator.to_jsonl(scenarios, args.output)

    # Print summary
    print_summary(scenarios, args.output)


if __name__ == "__main__":
    main()
