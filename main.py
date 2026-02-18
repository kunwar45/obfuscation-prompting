import argparse
import sys

from src.clients.together_client import TogetherClient
from src.config import Config
from src.loaders.gpqa_loader import GPQALoader
from src.loaders.json_loader import JSONFilePromptLoader
from src.monitors.keyword_monitor import KeywordMonitor
from src.monitors.llm_monitor import LLMMonitor
from src.pipeline.pipeline import Pipeline
from src.steps.base_model_step import BaseModelStep
from src.steps.monitor_step import MonitorStep
from src.storage.storage import ResultStorage


def parse_args(config: Config) -> Config:
    parser = argparse.ArgumentParser(description="Together AI Reasoning Pipeline")
    parser.add_argument("--prompts", dest="prompts_file", default="",
                        help="Path to a JSON prompts file (skips GPQA when supplied)")
    parser.add_argument("--gpqa-subset", dest="gpqa_subset", default=config.gpqa_subset,
                        choices=["gpqa_main", "gpqa_diamond", "gpqa_extended"],
                        help="GPQA subset to use (default: gpqa_main)")
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
    args = parser.parse_args()

    config.prompts_file = args.prompts_file
    config.gpqa_subset = args.gpqa_subset
    config.gpqa_limit = args.gpqa_limit
    config.base_model = args.base_model
    config.monitor_model = args.monitor_model
    config.output_dir = args.output_dir
    config.max_tokens = args.max_tokens
    config.temperature = args.temperature
    return config


def main():
    config = Config.from_env()
    config = parse_args(config)

    if not config.together_api_key:
        print("Error: TOGETHER_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    if config.prompts_file:
        loader = JSONFilePromptLoader(config.prompts_file)
        source_desc = config.prompts_file
    else:
        limit = config.gpqa_limit or None
        loader = GPQALoader(subset=config.gpqa_subset, limit=limit, seed=config.gpqa_seed)
        source_desc = f"GPQA {config.gpqa_subset}" + (f" (limit={limit})" if limit else "")

    prompts = loader.load()
    print(f"Loaded {len(prompts)} prompt(s) from {source_desc}")

    client = TogetherClient(api_key=config.together_api_key)
    monitors = [
        LLMMonitor(client, config),
        KeywordMonitor(keywords=["therefore", "because", "however", "thus", "since"]),
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
