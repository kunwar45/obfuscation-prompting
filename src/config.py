import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    monitor_model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    together_api_key: str = ""
    output_dir: str = "results"
    max_tokens: int = 2048
    temperature: float = 0.7
    # Dataset selection (used when --prompts is not supplied)
    dataset: str = "gpqa"          # "gpqa" | "medqa-obfuscation"
    gpqa_subset: str = "gpqa_main" # only for dataset=gpqa
    gpqa_limit: int = 0            # 0 = no limit (shared across datasets)
    gpqa_seed: int = 42            # shared RNG seed
    # JSON fallback (used only when --prompts is supplied)
    prompts_file: str = ""
    # Concealment dataset
    concealment_file: str = ""
    concealment_conditions: str = "A0,A1,A2"
    concealment_query_types: str = "B1"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
        )
