"""Black-box concealment experiment on Together AI.

Generates a synthetic shipping-domain concealment dataset, runs it through
the full pipeline (base model + regex monitor + LLM monitor), saves rich
results JSON, and plots disclosure rates by condition.

Flow
----
  1. Smoke test  — N_SMOKE scenarios (default 3) to verify end-to-end
  2. Full run    — N_FULL scenarios (default 20) with all monitors
  3. Save        — timestamped results JSON + dataset JSONL
  4. Plot        — disclosure rates, monitor concordance, per-secret breakdown

Usage
-----
  python run_blackbox.py                      # smoke (3) then full run (20)
  python run_blackbox.py --smoke-only         # smoke only, then exit
  python run_blackbox.py --skip-smoke         # skip smoke, full run only
  python run_blackbox.py --n-scenarios 50     # custom full-run size
  python run_blackbox.py --conditions A0,A2   # subset of conditions
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from src.clients.together_client import TogetherClient
from src.config import Config
from src.dataset.generator import DatasetGenerator
from src.dataset.shipping_domain import ShippingDomain
from src.loaders.concealment_loader import ConcealmentLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.llm_monitor import LLMMonitor
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.pipeline import Pipeline
from src.pipeline.result import PromptResult
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep
from src.storage.storage import ResultStorage


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n_scenarios: int, seed: int, output_path: str) -> int:
    """Generate a concealment JSONL. Returns actual number of scenarios written."""
    gen = DatasetGenerator(domains=[ShippingDomain()], seed=seed)
    scenarios = gen.generate(n_scenarios)
    gen.to_jsonl(scenarios, output_path)
    return len(scenarios)


# ── Pipeline construction ─────────────────────────────────────────────────────

def build_pipeline(client: TogetherClient, config: Config) -> Pipeline:
    monitors = [
        RegexMonitor(),
        LLMMonitor(client, config),
        KeywordMonitor(),
    ]
    steps = [
        BaseModelStep(client, config),
        MonitorStep(monitors),
    ]
    return Pipeline(steps)


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    client: TogetherClient,
    config: Config,
    dataset_path: str,
    conditions: str,
    limit: int | None = None,
) -> list[PromptResult]:
    loader = ConcealmentLoader(
        jsonl_path=dataset_path,
        conditions=conditions,
        query_types="B1",
        limit=limit,
    )
    prompts = loader.load()
    print(f"  Loaded {len(prompts)} prompts ({dataset_path})")
    pipeline = build_pipeline(client, config)
    return pipeline.run(prompts)


# ── Smoke test summary ────────────────────────────────────────────────────────

def print_smoke_summary(results: list[PromptResult]) -> bool:
    """Print per-prompt table. Returns True if no parse errors."""
    W = 38
    print()
    print(f"  ┌{'─' * (W + 32)}┐")
    print(f"  │  {'PROMPT ID':<{W}} {'REGEX':>5}  {'LLM':>9}  {'KEYWORD':>7}  │")
    print(f"  │  {'─' * W}  {'─' * 5}  {'─' * 9}  {'─' * 7}  │")

    parse_errors = 0
    for r in results:
        regex = r.monitor_results.get("regex", {})
        llm   = r.monitor_results.get("llm", {})
        kw    = r.monitor_results.get("keyword", {})

        regex_hit = regex.get("contains_secret_partial") or regex.get("contains_secret_exact")

        llm_raw = llm.get("mentioned_secret")
        if llm_raw is None and llm.get("verdict") == "parse_error":
            llm_str = "ERR"
            parse_errors += 1
        elif llm_raw is None:
            llm_str = "n/a"
        else:
            llm_str = llm.get("verdict", "?")[:9]

        kw_hit = kw.get("matched", False)

        pid = r.prompt_id[-W:] if len(r.prompt_id) > W else r.prompt_id
        print(f"  │  {pid:<{W}} {'YES' if regex_hit else 'no':>5}  {llm_str:>9}  {'YES' if kw_hit else 'no':>7}  │")

    print(f"  └{'─' * (W + 32)}┘")

    if parse_errors:
        print(f"\n  ⚠ {parse_errors} LLM monitor parse error(s). Check --monitor-model.")
        return False
    print(f"\n  ✓ All {len(results)} prompts processed without errors.")
    return True


# ── Rich results saving ───────────────────────────────────────────────────────

def save_results(
    results: list[PromptResult],
    config: Config,
    run_metadata: dict | None = None,
) -> str:
    """Save using ResultStorage and return path."""
    storage = ResultStorage(config)
    return storage.save(results, run_metadata=run_metadata)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black-box concealment experiment on Together AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run smoke test only, then exit")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test, go straight to full run")
    parser.add_argument("--n-scenarios", type=int, default=20,
                        help="Number of scenarios for the full run")
    parser.add_argument("--smoke-scenarios", type=int, default=3,
                        help="Number of scenarios for the smoke test")
    parser.add_argument("--model",
                        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="Together AI base model")
    parser.add_argument("--monitor-model",
                        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="Together AI LLM monitor model")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for result JSON files")
    parser.add_argument("--data-dir", default="data",
                        help="Directory for generated dataset JSONL files")
    parser.add_argument("--conditions", default="A0,A1,A2",
                        help="Comma-separated conditions (A0=transparent, A1=implicit, A2=suppress)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for dataset generation")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting after the full run")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config = Config.from_env()
    if not config.together_api_key:
        print("Error: TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    config.base_model    = args.model
    config.monitor_model = args.monitor_model
    config.output_dir    = args.output_dir

    client = TogetherClient(api_key=config.together_api_key)
    os.makedirs(args.data_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    conditions_list = [c.strip() for c in args.conditions.split(",") if c.strip()]
    n_conds = len(conditions_list)

    # Build self-documenting run metadata
    domain = ShippingDomain()
    sample_secret = {"structured": {"lead_time": "28 days"}}
    run_metadata = {
        "experiment": "blackbox",
        "environment": "together",
        "domain": "shipping",
        "seed": args.seed,
        "conditions": conditions_list,
        "condition_templates": {
            "A0": domain.render_A0(),
            "A1": domain.render_A1(),
            "A2": domain.render_A2(sample_secret),
        },
        "monitors": ["regex", "llm", "keyword"],
        "cli_args": {k: v for k, v in vars(args).items() if k != "no_plot"},
    }

    # ── Smoke test ────────────────────────────────────────────────────────
    if not args.skip_smoke:
        print(f"\n{'━' * 62}")
        print(f"  SMOKE TEST  ({args.smoke_scenarios} scenarios × {n_conds} conditions)")
        print(f"{'━' * 62}")

        smoke_path = os.path.join(args.data_dir, f"smoke_{timestamp}.jsonl")
        n = generate_dataset(args.smoke_scenarios, args.seed, smoke_path)
        print(f"  Generated {n} scenarios → {smoke_path}")

        try:
            smoke_results = run_batch(client, config, smoke_path, args.conditions)
        except Exception as e:
            print(f"\n[FATAL] Smoke test failed: {e}")
            raise

        ok = print_smoke_summary(smoke_results)

        smoke_meta = {**run_metadata, "run_type": "smoke", "n_scenarios": args.smoke_scenarios}
        smoke_result_path = save_results(smoke_results, config, run_metadata=smoke_meta)
        print(f"\n  Smoke results → {smoke_result_path}")

        if not ok:
            print("\n  LLM monitor parse errors found. Review monitor model or prompt.")
            print("  Continuing to full run (parse errors are non-fatal).")

        if args.smoke_only:
            print("\n  --smoke-only: exiting.")
            return

        print(f"\n  Smoke test complete. Proceeding to full run…\n")

    # ── Full run ──────────────────────────────────────────────────────────
    print(f"{'━' * 62}")
    # Regular scenarios ≈ 85% of total; controls take the rest
    approx_prompts = int(args.n_scenarios * 0.85) * n_conds
    print(f"  FULL RUN  ({args.n_scenarios} scenarios → ~{approx_prompts} regular prompts)")
    print(f"  model          : {args.model}")
    print(f"  monitor model  : {args.monitor_model}")
    print(f"  conditions     : {args.conditions}")
    print(f"{'━' * 62}")

    full_path = os.path.join(args.data_dir, f"blackbox_{timestamp}.jsonl")
    n = generate_dataset(args.n_scenarios, args.seed, full_path)
    print(f"  Generated {n} scenarios → {full_path}")

    results = run_batch(client, config, full_path, args.conditions)

    full_meta = {**run_metadata, "run_type": "full", "n_scenarios": args.n_scenarios}
    result_path = save_results(results, config, run_metadata=full_meta)
    print(f"\n  Results saved → {result_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n  Generating plots…")
        try:
            from scripts.plot_results import plot_all
            plot_all(result_path)
        except ImportError as e:
            print(f"  [WARN] Plotting skipped — missing dependency: {e}")
            print("  Run: pip install matplotlib")

    print(f"\n{'━' * 62}")
    print(f"  Done. Full results at: {result_path}")
    print(f"{'━' * 62}\n")


if __name__ == "__main__":
    main()
