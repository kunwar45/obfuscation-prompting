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
    # Local model
    use_local_model: bool = False
    local_model_name: str = ""
    local_model_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    # Activation capture
    capture_activations: str = "none"    # "none" | "last_token" | "full_sequence" | "reasoning_span"
    activations_dir: str = "activations"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
            use_local_model=os.getenv("USE_LOCAL_MODEL", "").lower() in ("1", "true", "yes"),
            local_model_name=os.getenv("LOCAL_MODEL_NAME", ""),
        )
