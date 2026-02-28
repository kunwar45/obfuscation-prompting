import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    provider: str = "together"  # "together" | "local-openai" | "mock"
    base_model: str = "ServiceNow-AI/Apriel-1.5-15b-Thinker"
    monitor_model: str = "ServiceNow-AI/Apriel-1.5-15b-Thinker"
    together_api_key: str = ""
    local_api_base: str = "http://127.0.0.1:8000/v1"
    local_api_key: str = "local"
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
    disable_llm_monitor: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            provider=os.getenv("PROVIDER", "together"),
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
            local_api_base=os.getenv("LOCAL_API_BASE", "http://127.0.0.1:8000/v1"),
            local_api_key=os.getenv("LOCAL_API_KEY", os.getenv("OPENAI_API_KEY", "local")),
        )
