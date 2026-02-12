from datetime import datetime, timezone

from tqdm import tqdm

from src.display import box, rule

from .result import PromptResult
from .step import PipelineStep

PREVIEW_PROMPT = 220
PREVIEW_ANSWER = 120
PREVIEW_REASONING = 180
PREVIEW_ANALYSIS = 200


def _preview(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


class Pipeline:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, prompts: list[dict]) -> list[PromptResult]:
        results = []
        total = len(prompts)
        for i, prompt in enumerate(tqdm(prompts, desc="Questions", unit="q")):
            prompt_id = prompt["id"]
            result = PromptResult(
                prompt_id=prompt_id,
                prompt=prompt["text"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            for step in self.steps:
                result = step.run(result)
            results.append(result)

            # Output: rule, question header, then boxes (use tqdm.write so progress bar is fine)
            prompt_preview = _preview(
                prompt["text"].replace("\n", " ").strip(), PREVIEW_PROMPT
            )
            base_content = (
                f"Answer: {_preview(result.final_answer, PREVIEW_ANSWER)}\n\n"
                f"Reasoning: {_preview(result.reasoning_trace, PREVIEW_REASONING)}"
            )
            monitor_content = _preview(result.monitor_analysis, PREVIEW_ANALYSIS)

            tqdm.write("")
            tqdm.write(rule())
            tqdm.write(f"Question {i + 1} of {total}")
            tqdm.write(rule())
            tqdm.write(box("Prompt", prompt_preview))
            tqdm.write("")
            tqdm.write(box("Base model", base_content))
            tqdm.write("")
            tqdm.write(box("Monitor", monitor_content))
        return results
