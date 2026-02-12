import re

from src.clients.together_client import TogetherClient
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
    def __init__(self, client: TogetherClient, config: Config):
        self.client = client
        self.config = config

    @property
    def name(self) -> str:
        return "base_model"

    def run(self, result: PromptResult) -> PromptResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        return result

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
