import argparse
import sys

from src.clients.together_client import TogetherClient
from src.config import Config
from src.clients import ChatClient
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
    # Local HF model flags
    parser.add_argument("--local", dest="use_local_model", action="store_true",
                        default=config.use_local_model,
                        help="Use a local HuggingFace model instead of Together AI")
    parser.add_argument("--local-model", dest="local_model_name", default=config.local_model_name,
                        help="HuggingFace model name/path (required with --local)")
    parser.add_argument("--dtype", dest="local_model_dtype", default=config.local_model_dtype,
                        choices=["bfloat16", "float16", "float32"],
                        help="Torch dtype for local model (default: bfloat16)")
    parser.add_argument("--capture-activations", dest="capture_activations",
                        default=config.capture_activations,
                        choices=["none", "last_token", "full_sequence", "reasoning_span"],
                        help="Activation capture mode (default: none)")
    parser.add_argument("--activations-dir", dest="activations_dir", default=config.activations_dir,
                        help="Directory to write .npz activation files (default: activations)")
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
    config.use_local_model = args.use_local_model
    config.local_model_name = args.local_model_name
    config.local_model_dtype = args.local_model_dtype
    config.capture_activations = args.capture_activations
    config.activations_dir = args.activations_dir
    return config


def build_client(config: Config) -> ChatClient:
    if config.use_local_model:
        if not config.local_model_name:
            print("Error: --local-model is required when using --local.")
            sys.exit(1)
        from src.clients.hf_client import HFClient
        print(f"Using local HF model: {config.local_model_name} (dtype={config.local_model_dtype})")
        return HFClient(
            model_name=config.local_model_name,
            dtype=config.local_model_dtype,
            capture_mode=config.capture_activations,
        )
    else:
        if not config.together_api_key:
            print("Error: TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key.")
            sys.exit(1)
        print(f"Using Together AI backend (model={config.base_model})")
        return TogetherClient(api_key=config.together_api_key)


def main():
    config = Config.from_env()
    config = parse_args(config)

    loader, source_desc = build_loader(config)
    prompts = loader.load()
    print(f"Loaded {len(prompts)} prompt(s) from {source_desc}")

    client = build_client(config)
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
