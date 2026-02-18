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
    # GPQA options (used when --prompts is not supplied)
    gpqa_subset: str = "gpqa_main"
    gpqa_limit: int = 0   # 0 = no limit
    gpqa_seed: int = 42
    # JSON fallback (used only when --prompts is supplied)
    prompts_file: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
        )
