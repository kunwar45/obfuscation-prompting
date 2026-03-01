from dataclasses import dataclass, field


@dataclass
class PromptResult:
    prompt_id: str
    prompt: str
    timestamp: str
    base_model_id: str = ""
    reasoning_trace: str = ""
    final_answer: str = ""
    monitor_results: dict = field(default_factory=dict)  # keyed by monitor name
    metadata: dict = field(default_factory=dict)
    activation_path: str = ""  # path to .npz if activations were captured

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "timestamp": self.timestamp,
            "base_model_id": self.base_model_id,
            "reasoning_trace": self.reasoning_trace,
            "final_answer": self.final_answer,
            "monitor_results": self.monitor_results,
            "metadata": self.metadata,
            "activation_path": self.activation_path,
        }
