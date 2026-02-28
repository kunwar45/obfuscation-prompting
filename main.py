import argparse
import sys

from src.clients.together_client import TogetherClient
from src.config import Config
from src.loaders.concealment_loader import ConcealmentLoader
from src.loaders.gpqa_loader import GPQALoader
from src.loaders.json_loader import JSONFilePromptLoader
from src.loaders.medqa_obfuscation_loader import MedQAObfuscationLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.llm_monitor import LLMMonitor
from src.monitors.regex_monitor import RegexMonitor
from src.pipeline.pipeline import Pipeline
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep
from src.storage.storage import ResultStorage

DATASETS = ["gpqa", "medqa-obfuscation", "concealment"]


def build_loader(config: Config):
    if config.prompts_file:
        return JSONFilePromptLoader(config.prompts_file), config.prompts_file

    limit = config.gpqa_limit or None

    if config.dataset == "gpqa":
        loader = GPQALoader(subset=config.gpqa_subset, limit=limit, seed=config.gpqa_seed)
        desc = f"GPQA {config.gpqa_subset}" + (f" (limit={limit})" if limit else "")
    elif config.dataset == "medqa-obfuscation":
        loader = MedQAObfuscationLoader(limit=limit, seed=config.gpqa_seed)
        desc = "MedQA obfuscation task" + (f" (limit={limit})" if limit else "")
    elif config.dataset == "concealment":
        if not config.concealment_file:
            print("Error: --concealment-file is required when using --dataset concealment")
            sys.exit(1)
        loader = ConcealmentLoader(
            jsonl_path=config.concealment_file,
            conditions=config.concealment_conditions,
            query_types=config.concealment_query_types,
            limit=limit,
        )
        desc = (
            f"Concealment dataset ({config.concealment_file})"
            f" conditions={config.concealment_conditions}"
            f" query_types={config.concealment_query_types}"
            + (f" (limit={limit} scenarios)" if limit else "")
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset!r}. Choose from: {DATASETS}")

    return loader, desc


def parse_args(config: Config) -> Config:
    parser = argparse.ArgumentParser(description="Together AI Reasoning Pipeline")
    parser.add_argument("--prompts", dest="prompts_file", default="",
                        help="Path to a JSON prompts file (overrides --dataset)")
    parser.add_argument("--dataset", dest="dataset", default=config.dataset,
                        choices=DATASETS,
                        help="Dataset to use (default: gpqa)")
    parser.add_argument("--gpqa-subset", dest="gpqa_subset", default=config.gpqa_subset,
                        choices=["gpqa_main", "gpqa_diamond", "gpqa_extended"],
                        help="GPQA subset (only used with --dataset gpqa)")
    parser.add_argument("--limit", dest="gpqa_limit", type=int, default=config.gpqa_limit,
                        help="Max number of questions to run (0 = all)")
    parser.add_argument("--base-model", dest="base_model", default=config.base_model,
                        help="Together AI model for base generation")
    parser.add_argument("--monitor-model", dest="monitor_model", default=config.monitor_model,
                        help="Together AI model for monitoring")
    parser.add_argument("--output-dir", dest="output_dir", default=config.output_dir,
                        help="Directory to write results")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=config.max_tokens,
                        help="Max tokens per generation")
    parser.add_argument("--temperature", dest="temperature", type=float, default=config.temperature,
                        help="Sampling temperature")
    parser.add_argument("--concealment-file", dest="concealment_file",
                        default=config.concealment_file,
                        help="Path to concealment JSONL dataset (required for --dataset concealment)")
    parser.add_argument("--concealment-conditions", dest="concealment_conditions",
                        default=config.concealment_conditions,
                        help="Comma-separated conditions to expand, e.g. A0,A1,A2")
    parser.add_argument("--concealment-query-types", dest="concealment_query_types",
                        default=config.concealment_query_types,
                        help="Comma-separated query types to expand, e.g. B1")
    args = parser.parse_args()

    config.prompts_file = args.prompts_file
    config.dataset = args.dataset
    config.gpqa_subset = args.gpqa_subset
    config.gpqa_limit = args.gpqa_limit
    config.base_model = args.base_model
    config.monitor_model = args.monitor_model
    config.output_dir = args.output_dir
    config.max_tokens = args.max_tokens
    config.temperature = args.temperature
    config.concealment_file = args.concealment_file
    config.concealment_conditions = args.concealment_conditions
    config.concealment_query_types = args.concealment_query_types
    return config


def main():
    config = Config.from_env()
    config = parse_args(config)

    if not config.together_api_key:
        print("Error: TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    loader, source_desc = build_loader(config)
    prompts = loader.load()
    print(f"Loaded {len(prompts)} prompt(s) from {source_desc}")

    client = TogetherClient(api_key=config.together_api_key)
    monitors = [
        LLMMonitor(client, config),
        # Static keywords (checked on every prompt) can be added here.
        # Loaders may also inject per-prompt terms via metadata["keyword_hints"]
        # — KeywordMonitor merges both sources automatically.
        KeywordMonitor(),
        RegexMonitor(),
    ]
    steps = [
        BaseModelStep(client, config),
        MonitorStep(monitors),
    ]
    pipeline = Pipeline(steps)

    print(f"Running pipeline with base_model={config.base_model}, monitor_model={config.monitor_model}")
    results = pipeline.run(prompts)

    storage = ResultStorage(config)
    output_path = storage.save(results)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
