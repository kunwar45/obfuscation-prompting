#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ACTIVATIONS_DIR = ROOT / "activations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the last-token concealment interpretability experiment end-to-end "
            "for Vertex: generate a fresh interpretability run, then evaluate it."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model name to use for the interpretability run and evaluator.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--n-scenarios", type=int, default=30)
    parser.add_argument("--conditions", default="A0,A2")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-dims", type=int, default=8)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--smoke-mode",
        action="store_true",
        help=(
            "Local lightweight validation mode. Still runs the evaluator, "
            "but treats evaluator failure as non-fatal so small smoke models "
            "like gpt2 can validate the workflow before a real Vertex run."
        ),
    )
    parser.add_argument(
        "--eval-output",
        default="",
        help=(
            "Optional explicit evaluator output path. Defaults to "
            "<results_json>_last_token_eval.json in the output dir."
        ),
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, allow_failure: bool = False) -> int:
    print(f"RUN {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0 and not allow_failure:
        raise SystemExit(proc.returncode)
    return proc.returncode


def newest_result_json(output_dir: Path) -> Path:
    candidates = sorted(
        (
            p for p in output_dir.glob("run_*.json")
            if not p.name.endswith("_analysis.json")
            and not p.name.endswith("_last_token_eval.json")
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No run_*.json found in {output_dir}")
    return candidates[-1]


def activations_dir_for_result(results_path: Path) -> Path:
    with results_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("results", payload)
    act_paths = [
        Path(row["activation_path"])
        for row in rows
        if isinstance(row, dict) and row.get("activation_path")
    ]
    if not act_paths:
        raise FileNotFoundError(f"No activation_path entries found in {results_path}")
    first = act_paths[0]
    if first.is_absolute():
        return first.parent
    return (ROOT / first).parent


def default_eval_output(results_path: Path, explicit: str) -> Path:
    if explicit:
        return ROOT / explicit if not Path(explicit).is_absolute() else Path(explicit)
    stem = results_path.stem + "_last_token_eval.json"
    return results_path.with_name(stem)


def main() -> int:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    before = {p.resolve() for p in output_dir.glob("run_*.json")}

    run_command(
        [
            sys.executable,
            "run_interp.py",
            "--skip-smoke",
            "--n-scenarios",
            str(args.n_scenarios),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--conditions",
            args.conditions,
            "--max-tokens",
            str(args.max_tokens),
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
            "--data-dir",
            args.data_dir,
            "--no-plot",
        ]
    )

    after = {p.resolve() for p in output_dir.glob("run_*.json")}
    new_results = sorted(
        (
            p for p in (after - before)
            if not p.name.endswith("_analysis.json")
            and not p.name.endswith("_last_token_eval.json")
        ),
        key=lambda p: p.stat().st_mtime,
    )
    results_path = new_results[-1] if new_results else newest_result_json(output_dir)
    activations_path = activations_dir_for_result(results_path)
    eval_output = default_eval_output(results_path, args.eval_output)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    eval_exit = run_command(
        [
            sys.executable,
            "scripts/eval_last_token_concealment.py",
            "--results",
            str(results_path),
            "--activations",
            str(activations_path),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--top-k-dims",
            str(args.top_k_dims),
            "--out",
            str(eval_output),
        ],
        allow_failure=args.smoke_mode,
    )

    if eval_exit != 0:
        print(
            "SMOKE_MODE_EVAL_SKIPPED evaluator exited non-zero during local smoke. "
            "This is allowed in --smoke-mode because tiny models or tiny runs may "
            "not produce enough successful concealment examples.",
            flush=True,
        )

    print(f"RESULTS_JSON {results_path}", flush=True)
    print(f"ACTIVATIONS_DIR {activations_path}", flush=True)
    if eval_exit == 0:
        print(f"EVAL_JSON {eval_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
