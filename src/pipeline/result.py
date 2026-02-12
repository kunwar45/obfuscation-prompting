from dataclasses import dataclass, field


@dataclass
class PromptResult:
    prompt_id: str
    prompt: str
    timestamp: str
    base_model_id: str = ""
    reasoning_trace: str = ""
    final_answer: str = ""
    monitor_model_id: str = ""
    monitor_analysis: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "timestamp": self.timestamp,
            "base_model_id": self.base_model_id,
            "reasoning_trace": self.reasoning_trace,
            "final_answer": self.final_answer,
            "monitor_model_id": self.monitor_model_id,
            "monitor_analysis": self.monitor_analysis,
            "metadata": self.metadata,
        }
