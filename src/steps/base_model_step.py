import os
import re

from src.clients import ChatClient
from src.config import Config
from src.pipeline.result import PromptResult
from src.pipeline.step import PipelineStep

SYSTEM_PROMPT = """\
You are a helpful assistant. Think through the problem step by step, then provide your final answer.

Format your response exactly as follows:
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<answer>
Your final answer here.
</answer>"""


class BaseModelStep(PipelineStep):
    def __init__(self, client: ChatClient, config: Config):
        self.client = client
        self.config = config

    @property
    def name(self) -> str:
        return "base_model"

    def run(self, result: PromptResult) -> PromptResult:
        system_content = result.metadata.get("system_prompt") or SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": result.prompt},
        ]
        raw = self.client.chat(
            model=self.config.base_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        result.base_model_id = self.config.base_model
        result.reasoning_trace = self._extract_tag(raw, "reasoning")
        result.final_answer = self._extract_tag(raw, "answer")

        if (
            self.config.capture_activations != "none"
            and hasattr(self.client, "save_activations")
        ):
            act_dir = os.path.join(self.config.activations_dir, self.config.base_model.replace("/", "_"))
            path = self.client.save_activations(result.prompt_id, act_dir)
            result.activation_path = path

        return result

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
