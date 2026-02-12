import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    monitor_model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    together_api_key: str = ""
    prompts_file: str = "prompts/sample_prompts.json"
    output_dir: str = "results"
    max_tokens: int = 2048
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
        )
