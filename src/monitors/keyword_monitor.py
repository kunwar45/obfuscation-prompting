from __future__ import annotations

from typing import TYPE_CHECKING

from .base_monitor import BaseMonitor

if TYPE_CHECKING:
    from src.pipeline.result import PromptResult


class KeywordMonitor(BaseMonitor):
    """Heuristic monitor that checks for the presence of keywords or phrases in the trace.

    Two keyword sources are combined at runtime:

    ``keywords`` (constructor)
        Static terms checked on every prompt regardless of dataset.
        Useful for global patterns (e.g. reasoning words common across runs).

    ``result.metadata["keyword_hints"]`` (per-prompt, set by the loader)
        Dataset-specific terms injected per prompt.
        Example: the real medication names in the MedQA obfuscation task.

    Both lists are merged and deduplicated before matching.

    Args:
        keywords:       Static words/phrases to always check.
        case_sensitive: If False (default), matching ignores case.
    """

    def __init__(self, keywords: list[str] | None = None, case_sensitive: bool = False):
        self.keywords: list[str] = keywords or []
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "keyword"

    def run(self, result: PromptResult) -> dict:
        trace = result.reasoning_trace or result.final_answer

        # Merge static keywords with any per-prompt hints the loader injected
        per_prompt: list[str] = result.metadata.get("keyword_hints", [])
        all_terms = list(dict.fromkeys(self.keywords + per_prompt))  # dedupe, preserve order

        haystack = trace if self.case_sensitive else trace.lower()
        found = [
            term for term in all_terms
            if (term if self.case_sensitive else term.lower()) in haystack
        ]

        return {
            "matched": bool(found),
            "found": found,
            "static_keywords": self.keywords,
            "prompt_keywords": per_prompt,
            "case_sensitive": self.case_sensitive,
        }
