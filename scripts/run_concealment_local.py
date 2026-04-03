"""Run the concealment dataset locally with configurable activation capture.

This is a lightweight alternative to `run_interp.py` when you want to reuse an
existing dataset JSONL and compare capture modes on the exact same prompts.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.clients.hf_client import HFClient
from src.config import Config
from src.loaders.concealment_loader import ConcealmentLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.pipeline import Pipeline
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep
from src.storage.storage import ResultStorage


def main() -> None:
    ap = argparse.ArgumentParser(description="Run concealment dataset with local HF model")
    ap.add_argument("--dataset", required=True, help="Path to concealment JSONL")
    ap.add_argument("--model", required=True, help="HF model name")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--capture-mode", default="last_token",
                    choices=["none", "last_token", "full_sequence", "reasoning_span"])
    ap.add_argument("--conditions", default="A0,A2")
    ap.add_argument("--query-types", default="B1")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--activations-dir", default="activations/compare_capture")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.base_model = args.model
    cfg.monitor_model = "regex+keyword"
    cfg.output_dir = args.output_dir
    cfg.max_tokens = args.max_tokens
    cfg.temperature = args.temperature
    cfg.capture_activations = args.capture_mode
    cfg.activations_dir = args.activations_dir

    loader = ConcealmentLoader(
        jsonl_path=args.dataset,
        conditions=args.conditions,
        query_types=args.query_types,
    )
    prompts = loader.load()
    print(f"Loaded {len(prompts)} prompt(s) from {args.dataset}")

    client = HFClient(
        model_name=args.model,
        dtype=args.dtype,
        capture_mode=args.capture_mode,
    )
    pipeline = Pipeline([
        BaseModelStep(client, cfg),
        MonitorStep([RegexMonitor(), KeywordMonitor()]),
    ])
    results = pipeline.run(prompts)
    output_path = ResultStorage(cfg).save(results)
    print(f"Results saved to: {output_path}")
    print(f"Activations dir base: {os.path.join(args.activations_dir, args.model.replace('/', '_'))}")


if __name__ == "__main__":
    main()
