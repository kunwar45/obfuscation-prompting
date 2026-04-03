"""Controlled framing experiment — which implicit prompts elicit concealment?

Generates a synthetic shipping-domain concealment dataset, runs it through
the full pipeline with framing conditions (no existing files modified),
saves rich results JSON, and plots disclosure rates.

Supports both Together AI (cloud) and local HuggingFace models.

Flow
----
  1. Smoke test  — N_SMOKE scenarios to verify end-to-end
  2. Full run    — N_FULL scenarios with all (or selected) framing conditions
  3. Save        — timestamped results JSON + dataset JSONL
  4. Plot        — publication figures via scripts/plot_framing.py

Usage
-----
  python run_framing.py                          # smoke (2) then full run (30)
  python run_framing.py --smoke-only             # smoke only, then exit
  python run_framing.py --skip-smoke             # skip smoke, full run only
  python run_framing.py --n-scenarios 30
  python run_framing.py --conditions BASE,M_inst,EXPLICIT
  python run_framing.py --smoke-only --smoke-scenarios 2 --conditions BASE,M_inst,EXPLICIT

  # Local model (no API key required):
  python3.11 run_framing.py --local --smoke-only --smoke-scenarios 2 --conditions BASE,M_inst,EXPLICIT
  python3.11 run_framing.py --local --skip-smoke --n-scenarios 30 --max-tokens 256
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from src.config import Config
from src.dataset.generator import DatasetGenerator
from src.dataset.shipping_domain import ShippingDomain
from src.framing.conditions import CONDITION_ORDER, REGISTRY
from src.framing.framing_loader import FramingLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.pipeline import Pipeline
from src.pipeline.result import PromptResult
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep
from src.storage.storage import ResultStorage


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset(n_scenarios: int, seed: int, output_path: str) -> int:
    """Generate a concealment JSONL. Returns actual number of scenarios written."""
    gen = DatasetGenerator(domains=[ShippingDomain()], seed=seed)
    scenarios = gen.generate(n_scenarios)
    gen.to_jsonl(scenarios, output_path)
    return len(scenarios)


# ── Pipeline construction (Together AI) ───────────────────────────────────────

def build_pipeline(client, config: Config) -> Pipeline:
    from src.clients.together_client import TogetherClient
    from src.monitors.llm_monitor import LLMMonitor
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


# ── Pipeline construction (local HuggingFace) ─────────────────────────────────

def _dummy_config(model_name: str, max_tokens: int) -> Config:
    """Minimal Config for local runs (no API key, no activation capture)."""
    cfg = Config()
    cfg.base_model = model_name
    cfg.capture_activations = "none"
    cfg.max_tokens = max_tokens
    return cfg


def build_pipeline_local(model_name: str, max_tokens: int, dtype: str) -> tuple[Config, Pipeline]:
    """Build a pipeline backed by a local HuggingFace model (no LLM monitor)."""
    from src.clients.hf_client import HFClient
    client = HFClient(model_name=model_name, capture_mode="none", dtype=dtype)
    monitors = [RegexMonitor(), KeywordMonitor()]
    config = _dummy_config(model_name, max_tokens)
    steps = [BaseModelStep(client, config), MonitorStep(monitors)]
    return config, Pipeline(steps)


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_batch(
    pipeline: Pipeline,
    dataset_path: str,
    framing_keys: list[str],
    role: str,
    limit: int | None = None,
) -> list[PromptResult]:
    loader = FramingLoader(
        jsonl_path=dataset_path,
        role=role,
        framing_keys=framing_keys,
        query_types="B1",
        limit=limit,
    )
    prompts = loader.load()
    print(f"  Loaded {len(prompts)} prompts ({dataset_path})")
    return pipeline.run(prompts)


# ── Smoke test summary ─────────────────────────────────────────────────────────

def print_smoke_summary(results: list[PromptResult]) -> bool:
    """Print per-prompt table. Returns True if no LLM parse errors."""
    W = 48
    DIM = 10
    print()
    print(f"  ┌{'─' * (W + DIM + 22)}┐")
    print(f"  │  {'PROMPT ID':<{W}} {'DIM':<{DIM}} {'REGEX':>5}  {'LLM':>9}  │")
    print(f"  │  {'─' * W}  {'─' * DIM}  {'─' * 5}  {'─' * 9}  │")

    parse_errors = 0
    for r in results:
        regex = r.monitor_results.get("regex", {})
        llm   = r.monitor_results.get("llm", {})

        regex_hit = regex.get("contains_secret_partial") or regex.get("contains_secret_exact")

        llm_raw = llm.get("mentioned_secret")
        if llm_raw is None and llm.get("verdict") == "parse_error":
            llm_str = "ERR"
            parse_errors += 1
        elif llm_raw is None:
            llm_str = "n/a"
        else:
            llm_str = llm.get("verdict", "?")[:9]

        dim = r.metadata.get("framing_dimension", "")[:DIM]
        pid = r.prompt_id[-(W):] if len(r.prompt_id) > W else r.prompt_id
        print(
            f"  │  {pid:<{W}} {dim:<{DIM}} {'YES' if regex_hit else 'no':>5}  {llm_str:>9}  │"
        )

    print(f"  └{'─' * (W + DIM + 22)}┘")

    if parse_errors:
        print(f"\n  ⚠ {parse_errors} LLM monitor parse error(s). Check --monitor-model.")
        return False
    print(f"\n  ✓ All {len(results)} prompts processed without errors.")
    return True


# ── Results saving ─────────────────────────────────────────────────────────────

def save_results(
    results: list[PromptResult],
    config: Config,
    run_metadata: dict | None = None,
) -> str:
    storage = ResultStorage(config)
    return storage.save(results, run_metadata=run_metadata)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    all_keys = ",".join(CONDITION_ORDER)
    parser = argparse.ArgumentParser(
        description="Controlled framing experiment (Together AI or local HF model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run smoke test only, then exit")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test, go straight to full run")
    parser.add_argument("--smoke-scenarios", type=int, default=2,
                        help="Number of scenarios for the smoke test")
    parser.add_argument("--n-scenarios", type=int, default=30,
                        help="Number of scenarios for the full run")
    parser.add_argument("--model",
                        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="Together AI base model (ignored when --local is set)")
    parser.add_argument("--monitor-model",
                        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        help="Together AI LLM monitor model (ignored when --local is set)")
    parser.add_argument("--conditions", default=all_keys,
                        help="Comma-separated framing condition keys (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for dataset generation")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for result JSON files")
    parser.add_argument("--data-dir", default="data",
                        help="Directory for generated dataset JSONL files")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting after the full run")

    # ── Local model args ───────────────────────────────────────────────────────
    local_group = parser.add_argument_group("local model (no API key required)")
    local_group.add_argument("--local", action="store_true",
                             help="Use a local HuggingFace model instead of Together AI")
    local_group.add_argument("--local-model",
                             default="Qwen/Qwen2.5-1.5B-Instruct",
                             help="HuggingFace model name for local runs")
    local_group.add_argument("--dtype",
                             default="float16",
                             choices=["bfloat16", "float16", "float32"],
                             help="Model dtype for local runs")
    local_group.add_argument("--max-tokens", type=int, default=256,
                             help="Max new tokens per generation (local mode; 256 is fast)")

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Parse and validate condition keys first (same for both modes)
    framing_keys = [k.strip() for k in args.conditions.split(",") if k.strip()]
    unknown = [k for k in framing_keys if k not in REGISTRY]
    if unknown:
        print(f"Error: Unknown framing keys: {unknown!r}")
        print(f"  Valid keys: {list(REGISTRY.keys())}")
        sys.exit(1)

    domain = ShippingDomain()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_keys = len(framing_keys)

    # ── Build pipeline ─────────────────────────────────────────────────────────
    if args.local:
        print(f"\n  Mode: LOCAL  ({args.local_model}  dtype={args.dtype})")
        config, pipeline = build_pipeline_local(args.local_model, args.max_tokens, args.dtype)
    else:
        if not os.environ.get("TOGETHER_API_KEY"):
            # Try loading from .env
            try:
                config_tmp = Config.from_env()
                if not config_tmp.together_api_key:
                    raise ValueError("empty key")
            except Exception:
                print("Error: TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key.")
                print("  Alternatively, use --local for a local HuggingFace model.")
                sys.exit(1)
        from src.clients.together_client import TogetherClient
        config = Config.from_env()
        config.base_model    = args.model
        config.monitor_model = args.monitor_model
        config.output_dir    = args.output_dir
        client = TogetherClient(api_key=config.together_api_key)
        pipeline = build_pipeline(client, config)
        print(f"\n  Mode: TOGETHER  ({args.model})")

    # ── Run metadata (self-documenting output) ──────────────────────────────
    run_metadata: dict = {
        "experiment": "framing",
        "environment": "local" if args.local else "together",
        "domain": "shipping",
        "seed": args.seed,
        "framing_keys": framing_keys,
        "framing_conditions": {
            k: {
                "dimension": REGISTRY[k].dimension,
                "label": REGISTRY[k].label,
                "template": REGISTRY[k].template,
            }
            for k in framing_keys
        },
        "monitors": ["regex", "keyword"] if args.local else ["regex", "llm", "keyword"],
        "cli_args": {k: v for k, v in vars(args).items() if k != "no_plot"},
    }
    if args.local:
        run_metadata["local_model"] = args.local_model
        run_metadata["dtype"] = args.dtype
        run_metadata["max_tokens"] = args.max_tokens

    # ── Smoke test ─────────────────────────────────────────────────────────────
    if not args.skip_smoke:
        print(f"\n{'━' * 66}")
        print(f"  SMOKE TEST  ({args.smoke_scenarios} scenarios × {n_keys} conditions)")
        print(f"{'━' * 66}")

        smoke_path = os.path.join(args.data_dir, f"framing_smoke_{timestamp}.jsonl")
        n = generate_dataset(args.smoke_scenarios, args.seed, smoke_path)
        print(f"  Generated {n} scenarios → {smoke_path}")

        try:
            smoke_results = run_batch(pipeline, smoke_path, framing_keys, domain.role)
        except Exception as e:
            print(f"\n[FATAL] Smoke test failed: {e}")
            raise

        ok = print_smoke_summary(smoke_results)

        smoke_meta = {**run_metadata, "run_type": "smoke", "n_scenarios": args.smoke_scenarios}
        smoke_result_path = save_results(smoke_results, config, run_metadata=smoke_meta)
        print(f"\n  Smoke results → {smoke_result_path}")

        if not ok:
            print("\n  LLM monitor parse errors found. Review monitor model or prompt.")
            print("  Continuing (parse errors are non-fatal).")

        if args.smoke_only:
            print("\n  --smoke-only: exiting.")
            return

        print(f"\n  Smoke test complete. Proceeding to full run…\n")

    # ── Full run ───────────────────────────────────────────────────────────────
    print(f"{'━' * 66}")
    approx_regular = int(args.n_scenarios * 0.85)
    approx_prompts = approx_regular * n_keys
    print(f"  FULL RUN  ({args.n_scenarios} scenarios → ~{approx_prompts} regular prompts)")
    if args.local:
        print(f"  model          : {args.local_model}  ({args.dtype})")
        print(f"  max tokens     : {args.max_tokens}")
    else:
        print(f"  model          : {args.model}")
        print(f"  monitor model  : {args.monitor_model}")
    print(f"  conditions     : {', '.join(framing_keys)}")
    print(f"{'━' * 66}")

    full_path = os.path.join(args.data_dir, f"framing_{timestamp}.jsonl")
    n = generate_dataset(args.n_scenarios, args.seed, full_path)
    print(f"  Generated {n} scenarios → {full_path}")

    results = run_batch(pipeline, full_path, framing_keys, domain.role)

    full_meta = {**run_metadata, "run_type": "full", "n_scenarios": args.n_scenarios}
    result_path = save_results(results, config, run_metadata=full_meta)
    print(f"\n  Results saved → {result_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if not args.no_plot:
        print("\n  Generating plots…")
        try:
            from scripts.plot_framing import plot_all_framing
            plot_all_framing(result_path)
        except ImportError as e:
            print(f"  [WARN] Plotting skipped — missing dependency: {e}")
            print("  Run: pip install matplotlib scipy")

    print(f"\n{'━' * 66}")
    print(f"  Done. Full results at: {result_path}")
    print(f"{'━' * 66}\n")


if __name__ == "__main__":
    main()
