from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.result import PromptResult


class BaseMonitor(ABC):
    """A monitor receives a completed PromptResult and returns a dict of findings.

    Monitors have access to the full result object, including:
    - result.reasoning_trace  — the base model's chain-of-thought
    - result.final_answer     — the base model's stated answer
    - result.metadata         — loader-supplied per-prompt context

    Loader metadata conventions
    ---------------------------
    Loaders may populate result.metadata with keys that monitors recognise:

    ``keyword_hints`` : list[str]
        Terms the KeywordMonitor should check for on top of its static keyword list.
        Use this when the terms to watch for are only known at load time
        (e.g. the real medication names in the MedQA obfuscation task).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used as the key in PromptResult.monitor_results."""

    @abstractmethod
    def run(self, result: PromptResult) -> dict:
        """Analyse result and return a JSON-serialisable dict of findings."""
